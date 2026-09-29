use crate::{
    NamedParams, ParamGroup, Result, StatefulAdam, TrainError,
    accumulation::GradientAccumulator,
    checkpoint::{self, Provenance, Snapshot},
    loss::masked_mse,
    schedule::epoch_end_rate,
};
use candle_core::{Device, Tensor};
use ferrots_core::config::Config;
use ferrots_data::{
    batch::{Dataset, HostBatch},
    cache::write_synced,
};
use ferrots_nn::baselines::{Baseline, ModelInput};
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
    model: &Baseline,
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
pub struct EarlyStopping {
    pub best_epoch: usize,
    pub best: Option<f64>,
    pub stale: usize,
    patience: usize,
}
impl EarlyStopping {
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
pub struct EpochRecord {
    pub epoch: usize,
    pub learning_rate: f64,
    pub train_loss: f64,
    pub validation_loss: f64,
    pub valid_labels: u64,
    pub skipped_empty_batches: usize,
    pub optimizer_steps: u64,
    pub seconds: f64,
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
    pub epochs: Vec<EpochRecord>,
    pub access: SplitAccess,
    pub data_sha256: String,
    pub source_sha256: String,
    pub seconds: f64,
}
pub struct Trainer<'a> {
    pub config: &'a Config,
    pub data: &'a Dataset,
    pub device: &'a Device,
    pub provenance: &'a Provenance,
}
impl Trainer<'_> {
    pub fn fit(&self) -> Result<TrainSummary> {
        let start = Instant::now();
        let config = self.config;
        config.validate_model_inputs()?;
        if self.data.index.counts[0] == 0 || self.data.index.counts[1] == 0 {
            return Err(TrainError::InvalidState(
                "training and validation require legal windows".into(),
            ));
        }
        let run_id = config.output.run_id.clone().unwrap_or_else(|| {
            format!(
                "run-{}-{}",
                config.train.seed,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system clock")
                    .as_nanos()
            )
        });
        let run = config.output.root.join(&run_id);
        fs::create_dir_all(&config.output.root)?;
        fs::create_dir(&run)?;
        write_synced(
            run.join("resolved-config.toml"),
            toml::to_string_pretty(config)?.as_bytes(),
        )?;
        let mut log = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(run.join("epochs.jsonl"))?;
        let model = Baseline::new(
            &config.model,
            &config.window,
            self.data.series.width(),
            config.train.seed,
            self.device,
        )?;
        let mut access = SplitAccess::default();
        let mut epochs = vec![];
        let mut steps = 0u64;
        let mut early = EarlyStopping::new(config.train.patience)?;
        if model.named_parameters().is_empty() {
            let frozen = model.freeze_for_eval()?;
            let validation = validation_loss(&frozen, self.data, config, self.device, &mut access)?;
            early.observe(0, validation)?;
            let snap = Snapshot {
                model: &frozen,
                config,
                scaler: &self.data.scaler,
                provenance: self.provenance,
                data_sha256: &self.data.cache.source_sha256,
                epoch: 0,
                validation_loss: validation,
            };
            checkpoint::publish(&run, "best", &snap)?;
            checkpoint::publish(&run, "last", &snap)?;
        } else {
            let params = NamedParams::new(model.named_parameters())?;
            let mut group = ParamGroup::adam(
                params.iter().map(|(n, _)| n.to_string()).collect(),
                config.train.learning_rate,
            );
            group.beta1 = config.train.beta1;
            group.beta2 = config.train.beta2;
            group.eps = config.train.eps;
            group.weight_decay = config.train.weight_decay;
            let mut optimizer = StatefulAdam::new(params.clone(), vec![group])?;
            let mut accumulator = GradientAccumulator::new(params);
            for epoch in 1..=config.train.max_epochs {
                let epoch_start = Instant::now();
                let used_rate = optimizer.groups()[0].lr;
                let mut sum = 0f64;
                let mut count = 0u64;
                let mut skipped = 0;
                let order = self
                    .data
                    .index
                    .training_order(model.sampling_rng(), epoch as u64 - 1);
                for (batch_index, chunk) in order.chunks(config.train.batch_size).enumerate() {
                    let host = tracked_batch(self.data, 0, chunk, &mut access)?;
                    let batch = DeviceBatch::from_host(&host, self.device)?;
                    let pred =
                        select_targets(&model.forward(&batch.input)?, &self.data.target_indices)?;
                    let loss =
                        masked_mse(&pred, &batch.supervision.target, &batch.supervision.mask)?;
                    sum += loss.sum.to_scalar::<f32>()? as f64;
                    count += loss.valid_count;
                    if loss.valid_count == 0 {
                        skipped += 1;
                    }
                    accumulator.add(&loss)?;
                    if (batch_index + 1) % config.train.accumulation_steps == 0 {
                        steps += u64::from(accumulator.step(&mut optimizer)?);
                    }
                }
                steps += u64::from(accumulator.step(&mut optimizer)?);
                if count == 0 {
                    return Err(TrainError::InvalidState(
                        "training epoch has no valid labels".into(),
                    ));
                }
                let frozen = model.freeze_for_eval()?;
                let validation =
                    validation_loss(&frozen, self.data, config, self.device, &mut access)?;
                let improved = early.observe(epoch, validation)?;
                let snap = Snapshot {
                    model: &frozen,
                    config,
                    scaler: &self.data.scaler,
                    provenance: self.provenance,
                    data_sha256: &self.data.cache.source_sha256,
                    epoch,
                    validation_loss: validation,
                };
                checkpoint::publish(&run, "last", &snap)?;
                if improved {
                    checkpoint::publish(&run, "best", &snap)?;
                }
                self.device.synchronize()?;
                let record = EpochRecord {
                    epoch,
                    learning_rate: used_rate,
                    train_loss: sum / count as f64,
                    validation_loss: validation,
                    valid_labels: count,
                    skipped_empty_batches: skipped,
                    optimizer_steps: steps,
                    seconds: epoch_start.elapsed().as_secs_f64(),
                };
                serde_json::to_writer(&mut log, &record)?;
                writeln!(log)?;
                log.flush()?;
                eprintln!(
                    "epoch={epoch} train_mse={} validation_mse={validation} optimizer_steps={steps} seconds={:.3}",
                    record.train_loss, record.seconds
                );
                epochs.push(record);
                if early.should_stop() {
                    break;
                }
                optimizer.set_learning_rate(epoch_end_rate(
                    config.train.learning_rate,
                    epoch,
                    config.train.lr_schedule,
                )?)?;
            }
        }
        let selected = checkpoint::load(run.join("best"), self.device)?;
        if selected.manifest.epoch != early.best_epoch {
            return Err(TrainError::InvalidState(
                "selected best epoch mismatch".into(),
            ));
        }
        let summary = TrainSummary {
            format_version: 1,
            run_id,
            run_dir: run.clone(),
            model: config.model.name().into(),
            best_epoch: early.best_epoch,
            best_validation_loss: early.best.expect("validated labels"),
            optimizer_steps: steps,
            epochs,
            access,
            data_sha256: self.data.cache.source_sha256.clone(),
            source_sha256: self.provenance.source_sha256.clone(),
            seconds: start.elapsed().as_secs_f64(),
        };
        write_synced(
            run.join("train-summary.json"),
            &serde_json::to_vec_pretty(&summary)?,
        )?;
        Ok(summary)
    }
}
