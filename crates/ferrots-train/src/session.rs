//! One optimizer-boundary state machine for fresh and resumed training.
use crate::{
    NamedParams, ParamGroup, Result, StatefulAdam, TrainError,
    accumulation::GradientAccumulator,
    loss::masked_mse,
    optim::OptimizerSnapshot,
    schedule::epoch_end_rate,
    trainer::{
        DeviceBatch, EarlyStopping, EpochRecord, SplitAccess, select_targets, tracked_batch,
        validation_loss,
    },
};
use candle_core::{Device, Tensor};
use ferrots_core::{
    config::{Config, Schedule},
    reference_rng::{ReferenceRng, ReferenceRngState},
};
use ferrots_data::{batch::Dataset, cache::hash_bytes};
use ferrots_nn::{
    Mode,
    itransformer::DROPOUT_SITE_VERSION,
    model::{ForecastModel, ForwardContext},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, time::Instant};

fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(TrainError::InvalidState(message.into()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Train,
    Validate,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndReason {
    MaxEpochs,
    EarlyStopping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingState {
    pub format_version: u32,
    pub phase: Phase,
    pub epoch: usize,
    /// Number of training windows already consumed in the current permutation.
    pub next_batch_offset: usize,
    pub order_sha256: String,
    pub global_optimizer_step: u64,
    pub next_forward_id: u64,
    pub train_sum: f64,
    pub valid_count: u64,
    pub skipped_empty_batches: usize,
    pub access: SplitAccess,
    pub epochs: Vec<EpochRecord>,
    pub early: EarlyStopping,
    pub best_generation: Option<String>,
    pub end_reason: Option<EndReason>,
    pub epoch_seconds: f64,
    #[serde(default)]
    pub steady_seconds: f64,
    #[serde(default)]
    pub steady_windows: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerState {
    pub format_version: u32,
    pub kind: Schedule,
    pub initial_lr: f64,
    pub applied_epoch: usize,
    pub current_lr: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RngState {
    pub format_version: u32,
    pub sampling_policy: String,
    pub seed: u64,
    pub post_init: ReferenceRngState,
    pub next_forward_id: u64,
    pub dropout_site_version: u32,
}

pub struct SessionSnapshot {
    pub config_sha256: String,
    pub weights: BTreeMap<String, Tensor>,
    pub buffers: BTreeMap<String, Tensor>,
    pub optimizer: OptimizerSnapshot,
    pub trainer: TrainingState,
    pub scheduler: SchedulerState,
    pub rng: RngState,
}
pub enum Progress {
    Training { updated: bool },
    Epoch { record: EpochRecord, improved: bool },
    Complete,
}
pub struct TrainingSession<'a> {
    pub model: ForecastModel,
    pub optimizer: StatefulAdam,
    pub state: TrainingState,
    pub scheduler: SchedulerState,
    config: &'a Config,
    data: &'a Dataset,
    device: &'a Device,
    sampler: ReferenceRng,
    accumulator: GradientAccumulator,
    order: Vec<usize>,
}
impl<'a> TrainingSession<'a> {
    pub fn config(&self) -> &Config {
        self.config
    }
    pub fn data(&self) -> &Dataset {
        self.data
    }
    pub fn device(&self) -> &Device {
        self.device
    }
    pub fn new(config: &'a Config, data: &'a Dataset, device: &'a Device) -> Result<Self> {
        Self::from_model(
            config,
            data,
            device,
            ForecastModel::from_config(config, device)?,
        )
    }
    pub fn from_model(
        config: &'a Config,
        data: &'a Dataset,
        device: &'a Device,
        model: ForecastModel,
    ) -> Result<Self> {
        config.validate_model_inputs()?;
        require(
            data.index.counts[0] > 0 && data.index.counts[1] > 0,
            "training and validation require legal windows",
        )?;
        let params = NamedParams::new(model.named_parameters())?;
        let mut group = ParamGroup::adam(
            params.iter().map(|(n, _)| n.into()).collect(),
            config.train.learning_rate,
        );
        group.beta1 = config.train.beta1;
        group.beta2 = config.train.beta2;
        group.eps = config.train.eps;
        group.weight_decay = config.train.weight_decay;
        let optimizer = StatefulAdam::new(params.clone(), vec![group])?;
        let sampler = model.sampling_rng().clone();
        let mut result = Self {
            model,
            optimizer,
            config,
            data,
            device,
            sampler,
            order: vec![],
            accumulator: GradientAccumulator::new(params),
            state: TrainingState {
                format_version: 1,
                phase: Phase::Train,
                epoch: 1,
                next_batch_offset: 0,
                order_sha256: String::new(),
                global_optimizer_step: 0,
                next_forward_id: 0,
                train_sum: 0.,
                valid_count: 0,
                skipped_empty_batches: 0,
                access: SplitAccess::default(),
                epochs: vec![],
                early: EarlyStopping::new(config.train.patience)?,
                best_generation: None,
                end_reason: None,
                epoch_seconds: 0.,
                steady_seconds: 0.,
                steady_windows: 0,
            },
            scheduler: SchedulerState {
                format_version: 1,
                kind: config.train.lr_schedule,
                initial_lr: config.train.learning_rate,
                applied_epoch: 0,
                current_lr: config.train.learning_rate,
            },
        };
        result.rebuild_order(false)?;
        Ok(result)
    }
    fn rebuild_order(&mut self, verify: bool) -> Result<()> {
        let mut order = self
            .data
            .index
            .training_order(&self.sampler, self.state.epoch as u64 - 1);
        if self.config.train.drop_last {
            order.truncate(
                order.len() / self.config.train.batch_size * self.config.train.batch_size,
            );
        }
        require(!order.is_empty(), "drop_last leaves no training batches")?;
        let bytes: Vec<u8> = order
            .iter()
            .flat_map(|&n| (n as u64).to_le_bytes())
            .collect();
        let digest = hash_bytes(&bytes);
        if verify {
            require(
                digest == self.state.order_sha256,
                "saved training permutation mismatch",
            )?;
        }
        self.state.order_sha256 = digest;
        self.order = order;
        Ok(())
    }
    pub fn rng_state(&self) -> RngState {
        RngState {
            format_version: 1,
            sampling_policy: "pytorch-five-draws-per-epoch-v1".into(),
            seed: self.config.train.seed,
            post_init: self.sampler.snapshot(),
            next_forward_id: self.state.next_forward_id,
            dropout_site_version: DROPOUT_SITE_VERSION,
        }
    }
    pub fn at_boundary(&self) -> bool {
        self.accumulator.is_empty()
    }
    pub fn next_entries(&self) -> &[usize] {
        if self.state.phase != Phase::Train {
            return &[];
        }
        &self.order[self.state.next_batch_offset
            ..(self.state.next_batch_offset + self.config.train.batch_size).min(self.order.len())]
    }
    pub fn validate_state(&self) -> Result<()> {
        let s = &self.state;
        let t = &self.config.train;
        require(
            self.at_boundary(),
            "checkpoint requires empty gradient accumulation",
        )?;
        require(
            s.format_version == 1
                && s.epoch > 0
                && s.epoch <= t.max_epochs
                && s.early.patience() == t.patience,
            "trainer version/epoch/patience mismatch",
        )?;
        require(
            s.train_sum.is_finite()
                && s.train_sum >= 0.
                && s.epoch_seconds.is_finite()
                && s.epoch_seconds >= 0.,
            "non-finite or negative trainer counters",
        )?;
        require(
            s.steady_seconds.is_finite()
                && s.steady_seconds >= 0.
                && s.steady_seconds <= s.epoch_seconds
                && s.steady_windows <= s.next_batch_offset,
            "invalid steady-state timing counters",
        )?;
        require(
            s.next_batch_offset <= self.order.len(),
            "invalid training cursor",
        )?;
        let batch = t.batch_size;
        let consumed = s.next_batch_offset.div_ceil(batch);
        require(
            (s.next_batch_offset == self.order.len() || s.next_batch_offset.is_multiple_of(batch))
                && (s.next_batch_offset == self.order.len()
                    || consumed.is_multiple_of(t.accumulation_steps)),
            "cursor is not an optimizer-group boundary",
        )?;
        require(
            (s.phase == Phase::Train && s.next_batch_offset < self.order.len())
                || (s.phase != Phase::Train && s.next_batch_offset == self.order.len()),
            "trainer phase/cursor mismatch",
        )?;
        let expected_forwards = (s.epoch - 1)
            .checked_mul(self.order.len().div_ceil(batch))
            .and_then(|n| n.checked_add(consumed))
            .ok_or_else(|| TrainError::InvalidState("forward count overflow".into()))?
            as u64;
        require(
            s.next_forward_id == expected_forwards
                && s.global_optimizer_step <= s.next_forward_id
                && s.skipped_empty_batches <= consumed,
            "forward/update/empty-batch count mismatch",
        )?;
        let completed = if s.phase == Phase::Complete {
            s.epoch
        } else {
            s.epoch - 1
        };
        require(
            s.epochs.len() == completed
                && s.epochs.iter().enumerate().all(|(i, e)| {
                    e.epoch == i + 1
                        && e.train_loss.is_finite()
                        && e.validation_loss.is_finite()
                        && e.seconds.is_finite()
                        && e.train_loss >= 0.
                        && e.validation_loss >= 0.
                        && e.seconds >= 0.
                        && e.valid_labels > 0
                }),
            "epoch history mismatch",
        )?;
        require(
            (s.phase == Phase::Complete) == s.end_reason.is_some(),
            "termination phase mismatch",
        )?;
        require(
            s.access.windows[0] == (s.epoch - 1) * self.order.len() + s.next_batch_offset
                && s.access.batches[0] as u64 == s.next_forward_id
                && s.access.windows[1] == completed * self.data.index.counts[1]
                && s.access.batches[1] == completed * self.data.index.counts[1].div_ceil(batch)
                && s.access.windows[2] == 0
                && s.access.batches[2] == 0,
            "split access counters mismatch",
        )?;
        if completed == 0 {
            require(
                s.early.best.is_none()
                    && s.early.best_epoch == 0
                    && s.early.stale == 0
                    && s.best_generation.is_none(),
                "best state exists before validation",
            )?;
        } else {
            let mut early = EarlyStopping::new(t.patience)?;
            for record in &s.epochs {
                early.observe(record.epoch, record.validation_loss)?;
            }
            require(
                early.best == s.early.best
                    && early.best_epoch == s.early.best_epoch
                    && early.stale == s.early.stale,
                "early stopping state mismatch",
            )?;
        }
        let rate = if s.epoch == 1 {
            t.learning_rate
        } else {
            epoch_end_rate(t.learning_rate, s.epoch - 1, t.lr_schedule)?
        };
        require(
            self.scheduler.format_version == 1
                && self.scheduler.kind == t.lr_schedule
                && self.scheduler.initial_lr == t.learning_rate
                && self.scheduler.applied_epoch == s.epoch - 1
                && self.scheduler.current_lr == rate,
            "scheduler state mismatch",
        )?;
        require(
            self.optimizer.groups().len() == 1
                && self.optimizer.groups().iter().all(|g| {
                    g.lr == rate
                        && g.beta1 == t.beta1
                        && g.beta2 == t.beta2
                        && g.eps == t.eps
                        && g.weight_decay == t.weight_decay
                }),
            "optimizer groups differ from scheduler/config",
        )?;
        require(
            self.optimizer.params().iter().all(|(n, _)| {
                self.optimizer
                    .state(n)
                    .is_some_and(|p| p.step == s.global_optimizer_step)
            }),
            "optimizer step counters mismatch",
        )?;
        if s.phase != Phase::Train {
            require(
                s.valid_count > 0,
                "completed training epoch has no valid labels",
            )?;
        }
        if s.next_batch_offset == 0 {
            require(
                s.train_sum == 0. && s.valid_count == 0 && s.skipped_empty_batches == 0,
                "fresh epoch has partial aggregates",
            )?;
        }
        match s.end_reason {
            Some(EndReason::EarlyStopping) => {
                require(s.early.should_stop(), "invalid early-stop reason")?
            }
            Some(EndReason::MaxEpochs) => require(
                s.epoch == t.max_epochs && !s.early.should_stop(),
                "invalid max-epoch reason",
            )?,
            None => require(
                !s.early.should_stop(),
                "unfinished trainer exceeded patience",
            )?,
        }
        Ok(())
    }
    pub fn snapshot(&self) -> Result<SessionSnapshot> {
        self.validate_state()?;
        Ok(SessionSnapshot {
            config_sha256: hash_bytes(toml::to_string_pretty(self.config)?.as_bytes()),
            weights: self
                .model
                .weights()
                .iter()
                .map(|(n, w)| Ok((n.clone(), w.detach().copy()?)))
                .collect::<Result<_>>()?,
            buffers: self
                .model
                .buffers()?
                .into_iter()
                .map(|(n, buffer)| Ok((n, buffer.detach().copy()?)))
                .collect::<Result<_>>()?,
            optimizer: self.optimizer.snapshot()?,
            trainer: self.state.clone(),
            scheduler: self.scheduler.clone(),
            rng: self.rng_state(),
        })
    }
    pub fn resume(
        config: &'a Config,
        data: &'a Dataset,
        device: &'a Device,
        snapshot: SessionSnapshot,
    ) -> Result<Self> {
        require(
            snapshot.config_sha256 == hash_bytes(toml::to_string_pretty(config)?.as_bytes()),
            "resume configuration mismatch",
        )?;
        let mut result = Self::new(config, data, device)?;
        let r = &snapshot.rng;
        require(
            r.format_version == 1
                && r.seed == config.train.seed
                && r.next_forward_id == snapshot.trainer.next_forward_id
                && r.dropout_site_version == DROPOUT_SITE_VERSION
                && r.sampling_policy == "pytorch-five-draws-per-epoch-v1"
                && r.post_init == result.sampler.snapshot(),
            "RNG identity/state mismatch",
        )?;
        result.sampler = ReferenceRng::from_state(&r.post_init)
            .map_err(|e| TrainError::InvalidState(e.to_string()))?;
        result.state = snapshot.trainer;
        result.scheduler = snapshot.scheduler;
        require(
            result.state.epoch > 0 && result.state.epoch <= config.train.max_epochs,
            "invalid restored epoch",
        )?;
        result.rebuild_order(true)?;
        result.optimizer.restore_snapshot(&snapshot.optimizer)?;
        result.validate_state()?;
        result.model.load_buffers(&snapshot.buffers)?;
        result.model.load_weights(&snapshot.weights)?;
        Ok(result)
    }
    pub fn advance(&mut self) -> Result<Progress> {
        match self.state.phase {
            Phase::Complete => Ok(Progress::Complete),
            Phase::Train => self.train_group(),
            Phase::Validate => self.validate_epoch(),
        }
    }
    fn train_group(&mut self) -> Result<Progress> {
        require(
            self.at_boundary(),
            "training group begins with pending gradients",
        )?;
        self.device.synchronize()?;
        let begin_offset = self.state.next_batch_offset;
        let start = Instant::now();
        for _ in 0..self.config.train.accumulation_steps {
            if self.state.next_batch_offset == self.order.len() {
                break;
            }
            let end =
                (self.state.next_batch_offset + self.config.train.batch_size).min(self.order.len());
            let host = tracked_batch(
                self.data,
                0,
                &self.order[self.state.next_batch_offset..end],
                &mut self.state.access,
            )?;
            let batch = DeviceBatch::from_host(&host, self.device)?;
            let prediction = self.model.forward_with_context(
                &batch.input,
                ForwardContext {
                    mode: Mode::Train,
                    forward_id: self.state.next_forward_id,
                },
            )?;
            self.state.next_forward_id = self
                .state
                .next_forward_id
                .checked_add(1)
                .ok_or_else(|| TrainError::InvalidState("forward counter overflow".into()))?;
            let pred = select_targets(&prediction, &self.data.target_indices)?;
            let loss = masked_mse(&pred, &batch.supervision.target, &batch.supervision.mask)?;
            self.state.train_sum += loss.sum.to_scalar::<f32>()? as f64;
            self.state.valid_count = self
                .state
                .valid_count
                .checked_add(loss.valid_count)
                .ok_or_else(|| TrainError::InvalidState("label count overflow".into()))?;
            if loss.valid_count == 0 {
                self.state.skipped_empty_batches += 1;
            }
            self.accumulator.add(&loss)?;
            self.state.next_batch_offset = end;
        }
        let updated = self.accumulator.step(&mut self.optimizer)?;
        self.state.global_optimizer_step += u64::from(updated);
        self.device.synchronize()?;
        let seconds = start.elapsed().as_secs_f64();
        self.state.epoch_seconds += seconds;
        if begin_offset > 0 {
            self.state.steady_seconds += seconds;
            self.state.steady_windows += self.state.next_batch_offset - begin_offset;
        }
        if self.state.next_batch_offset == self.order.len() {
            require(
                self.state.valid_count > 0,
                "training epoch has no valid labels",
            )?;
            self.state.phase = Phase::Validate;
        }
        Ok(Progress::Training { updated })
    }
    fn validate_epoch(&mut self) -> Result<Progress> {
        self.device.synchronize()?;
        let start = Instant::now();
        let frozen = self.model.freeze_for_eval()?;
        let validation = validation_loss(
            &frozen,
            self.data,
            self.config,
            self.device,
            &mut self.state.access,
        )?;
        let improved = self.state.early.observe(self.state.epoch, validation)?;
        self.device.synchronize()?;
        let validation_seconds = start.elapsed().as_secs_f64();
        let record = EpochRecord {
            epoch: self.state.epoch,
            learning_rate: self.scheduler.current_lr,
            train_loss: self.state.train_sum / self.state.valid_count as f64,
            validation_loss: validation,
            valid_labels: self.state.valid_count,
            skipped_empty_batches: self.state.skipped_empty_batches,
            optimizer_steps: self.state.global_optimizer_step,
            next_forward_id: self.state.next_forward_id,
            seconds: self.state.epoch_seconds + validation_seconds,
            train_seconds: self.state.epoch_seconds,
            validation_seconds,
            training_windows: self.state.next_batch_offset,
            steady_seconds: self.state.steady_seconds,
            steady_windows: self.state.steady_windows,
        };
        self.state.epochs.push(record.clone());
        if self.state.early.should_stop() || self.state.epoch == self.config.train.max_epochs {
            self.state.phase = Phase::Complete;
            self.state.end_reason = Some(if self.state.early.should_stop() {
                EndReason::EarlyStopping
            } else {
                EndReason::MaxEpochs
            });
        } else {
            self.scheduler.current_lr = epoch_end_rate(
                self.scheduler.initial_lr,
                self.state.epoch,
                self.scheduler.kind,
            )?;
            self.scheduler.applied_epoch = self.state.epoch;
            self.optimizer
                .set_learning_rate(self.scheduler.current_lr)?;
            self.state.epoch += 1;
            self.state.next_batch_offset = 0;
            self.state.train_sum = 0.;
            self.state.valid_count = 0;
            self.state.skipped_empty_batches = 0;
            self.state.epoch_seconds = 0.;
            self.state.steady_seconds = 0.;
            self.state.steady_windows = 0;
            self.state.phase = Phase::Train;
            self.rebuild_order(false)?;
        }
        Ok(Progress::Epoch { record, improved })
    }
}
