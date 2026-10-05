use crate::{
    Result, TrainError,
    checkpoint::{self, Provenance, Snapshot},
    loss::masked_mse,
};
use candle_core::{Device, Tensor};
use ferrots_core::config::Config;
use ferrots_data::{
    batch::{Dataset, HostBatch},
    cache::write_synced,
};
use ferrots_nn::model::{ForecastModel, ModelInput};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub struct Supervision {
    pub target: Tensor,
    pub mask: Tensor,
}
pub struct DeviceBatch {
    pub input: ModelInput,
    pub supervision: Supervision,
}
impl DeviceBatch {
    pub fn from_host(h: &HostBatch, d: &Device) -> Result<Self> {
        Ok(Self {
            input: ModelInput {
                history: Tensor::from_slice(
                    &h.input.history,
                    (h.batch_size, h.lookback, h.channels),
                    d,
                )?,
                history_time: Some(Tensor::from_slice(
                    &h.input.history_time,
                    (h.batch_size, h.lookback, h.time_width),
                    d,
                )?),
                future_time: Some(Tensor::from_slice(
                    &h.input.future_time,
                    (h.batch_size, h.horizon, h.time_width),
                    d,
                )?),
            },
            supervision: Supervision {
                target: Tensor::from_slice(
                    &h.supervision.targets,
                    (h.batch_size, h.horizon, h.target_count),
                    d,
                )?,
                mask: Tensor::from_slice(
                    &h.supervision.valid,
                    (h.batch_size, h.horizon, h.target_count),
                    d,
                )?,
            },
        })
    }
}
pub fn select_targets(prediction: &Tensor, indices: &[usize]) -> Result<Tensor> {
    if indices.iter().copied().eq(0..prediction.dim(2)?) {
        return Ok(prediction.clone());
    }
    let indices: Vec<u32> = indices.iter().map(|&j| j as u32).collect();
    Ok(prediction.index_select(
        &Tensor::from_slice(&indices, indices.len(), prediction.device())?,
        2,
    )?)
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitAccess {
    pub batches: [usize; 3],
    pub windows: [usize; 3],
}
pub fn tracked_batch(
    data: &Dataset,
    split: u8,
    entries: &[usize],
    access: &mut SplitAccess,
) -> Result<HostBatch> {
    if split > 2
        || entries
            .iter()
            .any(|&i| data.index.entries.get(i).is_none_or(|e| e.split != split))
    {
        return Err(TrainError::InvalidState("split access mismatch".into()));
    }
    let batch = data.batch(entries)?;
    access.batches[split as usize] += 1;
    access.windows[split as usize] += batch.batch_size;
    Ok(batch)
}
pub fn validation_loss(
    model: &ForecastModel,
    data: &Dataset,
    config: &Config,
    device: &Device,
    access: &mut SplitAccess,
) -> Result<f64> {
    if !model.is_frozen() {
        return Err(TrainError::InvalidState(
            "validation requires an independent frozen model".into(),
        ));
    }
    let mut sum = 0f64;
    let mut count = 0u64;
    for chunk in data.index.order(1).chunks(config.train.batch_size) {
        let host = tracked_batch(data, 1, chunk, access)?;
        let batch = DeviceBatch::from_host(&host, device)?;
        let pred = select_targets(&model.forward(&batch.input)?, &data.target_indices)?;
        if pred.track_op() {
            return Err(TrainError::InvalidState(
                "validation output tracks a graph".into(),
            ));
        }
        let loss = masked_mse(&pred, &batch.supervision.target, &batch.supervision.mask)?;
        sum += loss.sum.to_scalar::<f32>()? as f64;
        count += loss.valid_count;
    }
    if count == 0 {
        return Err(TrainError::InvalidState(
            "validation has no valid labels".into(),
        ));
    }
    Ok(sum / count as f64)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EarlyStopping {
    pub best_epoch: usize,
    pub best: Option<f64>,
    pub stale: usize,
    patience: usize,
}
impl EarlyStopping {
    pub fn patience(&self) -> usize {
        self.patience
    }
    pub fn new(patience: usize) -> Result<Self> {
        if patience == 0 {
            return Err(TrainError::InvalidState("patience must be positive".into()));
        }
        Ok(Self {
            best_epoch: 0,
            best: None,
            stale: 0,
            patience,
        })
    }
    pub fn observe(&mut self, epoch: usize, value: f64) -> Result<bool> {
        if !value.is_finite() {
            return Err(TrainError::InvalidState(
                "non-finite validation loss".into(),
            ));
        }
        if self.best.is_none_or(|best| value < best) {
            self.best = Some(value);
            self.best_epoch = epoch;
            self.stale = 0;
            Ok(true)
        } else {
            self.stale += 1;
            Ok(false)
        }
    }
    pub fn should_stop(&self) -> bool {
        self.stale >= self.patience
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpochRecord {
    pub epoch: usize,
    pub learning_rate: f64,
    pub train_loss: f64,
    pub validation_loss: f64,
    pub valid_labels: u64,
    pub skipped_empty_batches: usize,
    pub optimizer_steps: u64,
    #[serde(default)]
    pub next_forward_id: u64,
    pub seconds: f64,
    #[serde(default)]
    pub train_seconds: f64,
    #[serde(default)]
    pub validation_seconds: f64,
    #[serde(default)]
    pub training_windows: usize,
    #[serde(default)]
    pub steady_seconds: f64,
    #[serde(default)]
    pub steady_windows: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainSummary {
    pub format_version: u32,
    pub run_id: String,
    pub run_dir: PathBuf,
    pub model: String,
    pub best_epoch: usize,
    pub best_validation_loss: f64,
    pub optimizer_steps: u64,
    #[serde(default)]
    pub next_forward_id: u64,
    pub epochs: Vec<EpochRecord>,
    pub access: SplitAccess,
    pub data_sha256: String,
    pub source_sha256: String,
    pub seconds: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauseSummary {
    pub format_version: u32,
    pub status: String,
    pub run_id: String,
    pub run_dir: PathBuf,
    pub model: String,
    pub best_epoch: Option<usize>,
    pub best_validation_loss: Option<f64>,
    pub optimizer_steps: u64,
    pub next_forward_id: u64,
    pub phase: crate::session::Phase,
    pub epochs: Vec<EpochRecord>,
    pub access: SplitAccess,
    pub data_sha256: String,
    pub source_sha256: String,
    pub seconds: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RunOutcome {
    Paused(PauseSummary),
    Complete(TrainSummary),
}
pub struct Trainer<'a> {
    pub config: &'a Config,
    pub data: &'a Dataset,
    pub device: &'a Device,
    pub provenance: &'a Provenance,
}
impl<'a> Trainer<'a> {
    pub fn fit(&self) -> Result<TrainSummary> {
        match self.fit_with_limit(None)? {
            RunOutcome::Complete(summary) => Ok(summary),
            RunOutcome::Paused(_) => Err(TrainError::InvalidState("unlimited fit paused".into())),
        }
    }
    pub fn fit_with_limit(&self, stop_after_steps: Option<u64>) -> Result<RunOutcome> {
        self.validate_limit(stop_after_steps)?;
        self.config.validate_model_inputs()?;
        if self.data.index.counts[0] == 0 || self.data.index.counts[1] == 0 {
            return Err(TrainError::InvalidState(
                "training and validation require legal windows".into(),
            ));
        }
        let model = ForecastModel::from_config(self.config, self.device)?;
        if model.named_parameters().is_empty() {
            if stop_after_steps.is_some() {
                return Err(TrainError::InvalidState(
                    "stop-after-steps requires a trainable model".into(),
                ));
            }
            return self.fit_baseline(model);
        }
        let session = crate::session::TrainingSession::from_model(
            self.config,
            self.data,
            self.device,
            model,
        )?;
        let (run_id, run) = self.create_run(false)?;
        self.run_session(session, run_id, run, stop_after_steps, None)
    }
    pub fn resume(
        &self,
        path: impl AsRef<std::path::Path>,
        stop_after_steps: Option<u64>,
    ) -> Result<RunOutcome> {
        self.validate_limit(stop_after_steps)?;
        let session = checkpoint::load_session(
            path.as_ref(),
            self.config,
            self.data,
            self.device,
            self.provenance,
        )?;
        let source = checkpoint::load(path, self.device)?;
        let parent = checkpoint::ParentGeneration {
            generation: source
                .directory
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| TrainError::InvalidState("invalid parent generation name".into()))?
                .into(),
            manifest_sha256: ferrots_data::cache::fingerprint(
                source.directory.join("manifest.json"),
            )?,
        };
        let (run_id, run) = self.create_run(true)?;
        checkpoint::inherit_best(&run, &source, self.device)?;
        write_synced(
            run.join("continuation.json"),
            &serde_json::to_vec_pretty(&serde_json::json!({
                "format_version":1,"parent_run":source.directory.parent().and_then(|p|p.parent()),
                "parent_generation":parent.generation,"parent_manifest_sha256":parent.manifest_sha256,
                "original_code_sha":source.manifest.provenance.code_sha,"continuation_provenance":self.provenance,
            }))?,
        )?;
        self.run_session(session, run_id, run, stop_after_steps, Some(parent))
    }
    fn validate_limit(&self, limit: Option<u64>) -> Result<()> {
        if limit == Some(0) {
            return Err(TrainError::InvalidState(
                "stop-after-steps must be positive".into(),
            ));
        }
        Ok(())
    }
    fn create_run(&self, continuation: bool) -> Result<(String, PathBuf)> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| TrainError::InvalidState("system clock before epoch".into()))?
            .as_nanos();
        let generated = format!(
            "{}-{}-{}-{nonce}",
            if continuation { "continuation" } else { "run" },
            self.config.train.seed,
            std::process::id()
        );
        let run_id = if continuation {
            generated
        } else {
            self.config.output.run_id.clone().unwrap_or(generated)
        };
        let run = self.config.output.root.join(&run_id);
        fs::create_dir_all(&self.config.output.root)?;
        fs::create_dir(&run)?;
        write_synced(
            run.join("resolved-config.toml"),
            toml::to_string_pretty(self.config)?.as_bytes(),
        )?;
        Ok((run_id, run))
    }
    fn fit_baseline(&self, model: ForecastModel) -> Result<RunOutcome> {
        let start = Instant::now();
        let (run_id, run) = self.create_run(false)?;
        let mut access = SplitAccess::default();
        let frozen = model.freeze_for_eval()?;
        let validation =
            validation_loss(&frozen, self.data, self.config, self.device, &mut access)?;
        let snap = Snapshot {
            model: &frozen,
            config: self.config,
            scaler: &self.data.scaler,
            provenance: self.provenance,
            data_sha256: &self.data.cache.source_sha256,
            epoch: 0,
            validation_loss: validation,
        };
        checkpoint::publish(&run, "best", &snap)?;
        checkpoint::publish(&run, "last", &snap)?;
        write_synced(run.join("epochs.jsonl"), b"")?;
        let summary = TrainSummary {
            format_version: 1,
            run_id,
            run_dir: run.clone(),
            model: self.config.model.name().into(),
            best_epoch: 0,
            best_validation_loss: validation,
            optimizer_steps: 0,
            next_forward_id: 0,
            epochs: vec![],
            access,
            data_sha256: self.data.cache.source_sha256.clone(),
            source_sha256: self.provenance.source_sha256.clone(),
            seconds: start.elapsed().as_secs_f64(),
        };
        write_synced(
            run.join("train-summary.json"),
            &serde_json::to_vec_pretty(&summary)?,
        )?;
        Ok(RunOutcome::Complete(summary))
    }
    fn run_session(
        &self,
        mut session: crate::session::TrainingSession<'a>,
        run_id: String,
        run: PathBuf,
        stop_after_steps: Option<u64>,
        parent: Option<checkpoint::ParentGeneration>,
    ) -> Result<RunOutcome> {
        use crate::session::{Phase, Progress};
        let start = Instant::now();
        let initial_steps = session.state.global_optimizer_step;
        let mut performance = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(run.join("performance.jsonl"))?;
        let mut log = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(run.join("epochs.jsonl"))?;
        for record in &session.state.epochs {
            serde_json::to_writer(&mut log, record)?;
            writeln!(log)?;
        }
        log.flush()?;
        loop {
            if session.state.phase == Phase::Complete {
                if !run.join("last").exists() {
                    checkpoint::publish_training(
                        &run,
                        "last",
                        &mut session,
                        self.provenance,
                        parent.clone(),
                    )?;
                }
                break;
            }
            match session.advance()? {
                Progress::Complete => break,
                Progress::Training { updated } => {
                    if updated
                        && stop_after_steps.is_some_and(|n| {
                            session.state.global_optimizer_step - initial_steps >= n
                        })
                    {
                        checkpoint::publish_training(
                            &run,
                            "last",
                            &mut session,
                            self.provenance,
                            parent.clone(),
                        )?;
                        let s = &session.state;
                        let summary = PauseSummary {
                            format_version: 2,
                            status: "paused".into(),
                            run_id,
                            run_dir: run.clone(),
                            model: self.config.model.name().into(),
                            best_epoch: s.early.best.map(|_| s.early.best_epoch),
                            best_validation_loss: s.early.best,
                            optimizer_steps: s.global_optimizer_step,
                            next_forward_id: s.next_forward_id,
                            phase: s.phase,
                            epochs: s.epochs.clone(),
                            access: s.access.clone(),
                            data_sha256: self.data.cache.source_sha256.clone(),
                            source_sha256: self.provenance.source_sha256.clone(),
                            seconds: start.elapsed().as_secs_f64(),
                        };
                        write_synced(
                            run.join("train-summary.json"),
                            &serde_json::to_vec_pretty(&summary)?,
                        )?;
                        return Ok(RunOutcome::Paused(summary));
                    }
                }
                Progress::Epoch { record, improved } => {
                    self.device.synchronize()?;
                    let checkpoint_start = Instant::now();
                    if improved {
                        checkpoint::publish_training(
                            &run,
                            "best",
                            &mut session,
                            self.provenance,
                            parent.clone(),
                        )?;
                    }
                    checkpoint::publish_training(
                        &run,
                        "last",
                        &mut session,
                        self.provenance,
                        parent.clone(),
                    )?;
                    self.device.synchronize()?;
                    serde_json::to_writer(
                        &mut performance,
                        &serde_json::json!({
                            "epoch":record.epoch,"train_seconds":record.train_seconds,
                            "validation_seconds":record.validation_seconds,
                            "checkpoint_seconds":checkpoint_start.elapsed().as_secs_f64(),
                            "training_windows":record.training_windows,
                            "steady_seconds":record.steady_seconds,"steady_windows":record.steady_windows,
                            "steady_samples_per_second":if record.steady_seconds>0. {Some(record.steady_windows as f64/record.steady_seconds)} else {None},
                            "steady_definition":"exclude first optimizer group of each epoch",
                        }),
                    )?;
                    writeln!(performance)?;
                    performance.flush()?;
                    serde_json::to_writer(&mut log, &record)?;
                    writeln!(log)?;
                    log.flush()?;
                    eprintln!(
                        "epoch={} train_mse={} validation_mse={} optimizer_steps={} seconds={:.3}",
                        record.epoch,
                        record.train_loss,
                        record.validation_loss,
                        record.optimizer_steps,
                        record.seconds
                    );
                }
            }
        }
        session.validate_state()?;
        let selected = checkpoint::load(run.join("best"), self.device)?;
        if selected.manifest.epoch != session.state.early.best_epoch {
            return Err(TrainError::InvalidState(
                "selected best epoch mismatch".into(),
            ));
        }
        let s = session.state;
        let summary = TrainSummary {
            format_version: 2,
            run_id,
            run_dir: run.clone(),
            model: self.config.model.name().into(),
            best_epoch: s.early.best_epoch,
            best_validation_loss: s.early.best.expect("validated labels"),
            optimizer_steps: s.global_optimizer_step,
            next_forward_id: s.next_forward_id,
            epochs: s.epochs,
            access: s.access,
            data_sha256: self.data.cache.source_sha256.clone(),
            source_sha256: self.provenance.source_sha256.clone(),
            seconds: start.elapsed().as_secs_f64(),
        };
        write_synced(
            run.join("train-summary.json"),
            &serde_json::to_vec_pretty(&summary)?,
        )?;
        Ok(RunOutcome::Complete(summary))
    }
}
