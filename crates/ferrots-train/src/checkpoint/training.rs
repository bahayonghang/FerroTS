//! Complete training transactions. No live session is changed by a failed load.
use super::*;
use crate::{
    NamedParams, ParamGroup, StatefulAdam,
    session::{Phase, RngState, SchedulerState, SessionSnapshot, TrainingSession, TrainingState},
};
use candle_core::DeviceLocation;
use ferrots_data::batch::Dataset;
use ferrots_nn::itransformer::DROPOUT_SITE_VERSION;

pub(super) const FILES: [&str; 11] = [
    "model.safetensors",
    "optimizer.safetensors",
    "optimizer-state.json",
    "buffers.safetensors",
    "trainer-state.json",
    "scheduler-state.json",
    "rng-state.json",
    "scaler.json",
    "data-schema.json",
    "resolved-config.toml",
    "environment.json",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParentGeneration {
    pub generation: String,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingIdentity {
    pub mapping_version: u32,
    pub schema_version: u32,
    pub rng_version: u32,
    pub dropout_site_version: u32,
    pub boundary: String,
    pub lookback: usize,
    pub horizon: usize,
    pub channels: usize,
    pub time_width: usize,
    pub parent: Option<ParentGeneration>,
}
impl TrainingIdentity {
    fn new(config: &Config, parent: Option<ParentGeneration>) -> Self {
        Self {
            mapping_version: 1,
            schema_version: 1,
            rng_version: 1,
            dropout_site_version: DROPOUT_SITE_VERSION,
            boundary: "empty_accumulation".into(),
            lookback: config.window.lookback,
            horizon: config.window.horizon,
            channels: config.data.input_columns.len(),
            time_width: if config.data.time_features == "t" {
                5
            } else {
                4
            },
            parent,
        }
    }
}

/// Hooks run before each commit transition. A failure keeps staged or orphaned files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishStage {
    StagingCreated,
    PayloadsWritten,
    ManifestWritten,
    Validated,
    GenerationCommitted,
    ReferencePending,
}
fn valid_generation(name: &str) -> bool {
    name.starts_with("generation-") && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}
fn digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
fn device_name(device: &Device) -> Result<String> {
    match device.location() {
        DeviceLocation::Cpu => Ok("cpu".into()),
        DeviceLocation::Cuda { gpu_id } => Ok(format!("cuda:{gpu_id}")),
        _ => Err(TrainError::InvalidState("unsupported resume device".into())),
    }
}
fn json<T: serde::de::DeserializeOwned>(directory: &Path, name: &str) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(directory.join(name))?)?)
}

pub(super) fn read_state(checkpoint: &Checkpoint, device: &Device) -> Result<SessionSnapshot> {
    let m = &checkpoint.manifest;
    require(
        m.format_version == 2 && !m.inference_only,
        "v1 checkpoint is inference-only; resume requires v2",
    )?;
    let c = &checkpoint.config;
    let identity = m
        .training
        .as_ref()
        .ok_or_else(|| TrainError::InvalidState("missing training identity".into()))?;
    require(
        *identity == TrainingIdentity::new(c, identity.parent.clone()),
        "training model/mapping/schema/RNG/boundary identity mismatch",
    )?;
    if let Some(parent) = &identity.parent {
        require(
            valid_generation(&parent.generation) && digest(&parent.manifest_sha256),
            "invalid parent generation",
        )?;
    }
    require(
        digest(&m.data_sha256) && digest(&m.provenance.source_sha256),
        "invalid training provenance digest",
    )?;
    let environment: BTreeMap<String, String> = json(&checkpoint.directory, "environment.json")?;
    require(
        environment == m.provenance.environment,
        "checkpoint environment mismatch",
    )?;
    let trainer: TrainingState = json(&checkpoint.directory, "trainer-state.json")?;
    let scheduler: SchedulerState = json(&checkpoint.directory, "scheduler-state.json")?;
    let rng: RngState = json(&checkpoint.directory, "rng-state.json")?;
    require(
        trainer.format_version == 1 && trainer.epoch > 0 && trainer.epoch <= c.train.max_epochs,
        "invalid trainer version/epoch",
    )?;
    require(
        trainer.train_sum.is_finite()
            && trainer.train_sum >= 0.
            && trainer.epoch_seconds.is_finite()
            && trainer.epoch_seconds >= 0.,
        "invalid trainer aggregates",
    )?;
    require(
        trainer
            .best_generation
            .as_deref()
            .is_none_or(valid_generation)
            && trainer.early.best.is_some() == trainer.best_generation.is_some(),
        "invalid best generation binding",
    )?;
    require(
        m.epoch == trainer.epochs.last().map_or(0, |e| e.epoch)
            && m.validation_loss == trainer.epochs.last().map(|e| e.validation_loss),
        "manifest observation mismatch",
    )?;
    require(
        rng.format_version == 1
            && rng.dropout_site_version == DROPOUT_SITE_VERSION
            && rng.seed == c.train.seed
            && rng.next_forward_id == trainer.next_forward_id
            && rng.sampling_policy == "pytorch-five-draws-per-epoch-v1"
            && rng.post_init == checkpoint.model.sampling_rng().snapshot(),
        "RNG identity/state mismatch",
    )?;
    ferrots_core::reference_rng::ReferenceRng::from_state(&rng.post_init)
        .map_err(|e| TrainError::InvalidState(e.to_string()))?;
    let lr = if trainer.epoch == 1 {
        c.train.learning_rate
    } else {
        crate::schedule::epoch_end_rate(
            c.train.learning_rate,
            trainer.epoch - 1,
            c.train.lr_schedule,
        )?
    };
    require(
        scheduler.format_version == 1
            && scheduler.kind == c.train.lr_schedule
            && scheduler.initial_lr == c.train.learning_rate
            && scheduler.applied_epoch == trainer.epoch - 1
            && scheduler.current_lr == lr,
        "scheduler identity/rate mismatch",
    )?;
    let model = ForecastModel::from_config(c, device)?;
    let params = NamedParams::new(model.named_parameters())?;
    require(
        params.iter().next().is_some(),
        "parameter-free model cannot resume",
    )?;
    let group = ParamGroup::adam(params.iter().map(|(n, _)| n.to_owned()).collect(), lr);
    let mut optimizer = StatefulAdam::new(params, vec![group])?;
    optimizer.load(&checkpoint.directory)?;
    require(
        optimizer.groups().len() == 1
            && optimizer.groups().iter().all(|g| {
                g.lr == lr
                    && g.beta1 == c.train.beta1
                    && g.beta2 == c.train.beta2
                    && g.eps == c.train.eps
                    && g.weight_decay == c.train.weight_decay
            }),
        "optimizer group/config mismatch",
    )?;
    require(
        optimizer.params().iter().all(|(n, _)| {
            optimizer
                .state(n)
                .is_some_and(|s| s.step == trainer.global_optimizer_step)
        }),
        "optimizer step mismatch",
    )?;
    Ok(SessionSnapshot {
        config_sha256: m.config_sha256.clone(),
        weights: checkpoint
            .model
            .weights()
            .iter()
            .map(|(n, t)| Ok((n.clone(), t.copy()?)))
            .collect::<Result<_>>()?,
        buffers: checkpoint
            .model
            .buffers()?
            .into_iter()
            .map(|(n, t)| Ok((n, t.detach().copy()?)))
            .collect::<Result<_>>()?,
        optimizer: optimizer.snapshot()?,
        trainer,
        scheduler,
        rng,
    })
}

fn verify_best(checkpoint: &Checkpoint, state: &TrainingState, device: &Device) -> Result<()> {
    if let Some(name) = &state.best_generation {
        let directory = if checkpoint.directory.file_name().and_then(|s| s.to_str()) == Some(name) {
            checkpoint.directory.clone()
        } else {
            checkpoint
                .directory
                .parent()
                .ok_or_else(|| TrainError::InvalidState("generation parent missing".into()))?
                .join(name)
        };
        let best = if directory == checkpoint.directory {
            None
        } else {
            Some(super::load(&directory, device)?)
        };
        let best = best.as_ref().unwrap_or(checkpoint);
        require(
            best.manifest.format_version == 2
                && best.manifest.epoch == state.early.best_epoch
                && best.manifest.validation_loss == state.early.best
                && best.manifest.config_sha256 == checkpoint.manifest.config_sha256
                && best.manifest.data_sha256 == checkpoint.manifest.data_sha256
                && best.manifest.provenance.source_sha256
                    == checkpoint.manifest.provenance.source_sha256
                && best.manifest.provenance.environment
                    == checkpoint.manifest.provenance.environment,
            "best generation identity mismatch",
        )?;
    }
    Ok(())
}

/// Copy the selected historical best unchanged so the continuation is self-contained.
pub fn inherit_best(run: &Path, source: &Checkpoint, device: &Device) -> Result<()> {
    let state = read_state(source, device)?.trainer;
    verify_best(source, &state, device)?;
    let Some(name) = state.best_generation else {
        return Ok(());
    };
    require(
        !run.join("best").exists(),
        "continuation already contains a best reference",
    )?;
    let original = source
        .directory
        .parent()
        .ok_or_else(|| TrainError::InvalidState("checkpoint parent missing".into()))?
        .join(&name);
    let stage = run.join(".staging").join(&name);
    fs::create_dir_all(run.join(".staging"))?;
    fs::create_dir(&stage)?;
    for file in FILES
        .iter()
        .copied()
        .chain(std::iter::once("manifest.json"))
    {
        write_synced(stage.join(file), &fs::read(original.join(file))?)?;
    }
    super::load(&stage, device)?;
    require(
        fingerprint(stage.join("manifest.json"))? == fingerprint(original.join("manifest.json"))?,
        "inherited best manifest changed",
    )?;
    fs::create_dir_all(run.join("checkpoints"))?;
    let destination = run.join("checkpoints").join(name);
    fs::rename(stage, &destination)?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| TrainError::InvalidState("system clock before epoch".into()))?
        .as_nanos();
    publish_reference(run, "best", &destination, nonce, &mut |_| Ok(()))
}

pub fn load_session<'a>(
    path: impl AsRef<Path>,
    config: &'a Config,
    data: &'a Dataset,
    device: &'a Device,
    provenance: &Provenance,
) -> Result<TrainingSession<'a>> {
    let checkpoint = super::load(path, device)?;
    require(
        checkpoint.manifest.format_version == 2,
        "v1 checkpoint is inference-only; resume requires v2",
    )?;
    require(
        checkpoint.manifest.config_sha256 == hash_bytes(toml::to_string_pretty(config)?.as_bytes())
            && checkpoint.manifest.data_sha256 == data.cache.source_sha256
            && checkpoint.manifest.data_sha256 == fingerprint(&config.data.path)?
            && checkpoint.schema == DataSchema::from_config(config)?
            && serde_json::to_value(&checkpoint.scaler)? == serde_json::to_value(&data.scaler)?,
        "resume config/data/scaler mismatch",
    )?;
    require(
        checkpoint.manifest.provenance.source_sha256 == provenance.source_sha256
            && checkpoint.manifest.provenance.environment == provenance.environment
            && checkpoint.manifest.device == device_name(device)?,
        "resume source/environment/device mismatch",
    )?;
    let snapshot = read_state(&checkpoint, device)?;
    verify_best(&checkpoint, &snapshot.trainer, device)?;
    TrainingSession::resume(config, data, device, snapshot)
}

pub fn publish_training(
    run: &Path,
    selector: &str,
    session: &mut TrainingSession<'_>,
    provenance: &Provenance,
    parent: Option<ParentGeneration>,
) -> Result<PathBuf> {
    publish_training_with_hook(run, selector, session, provenance, parent, |_| Ok(()))
}

pub fn publish_training_with_hook(
    run: &Path,
    selector: &str,
    session: &mut TrainingSession<'_>,
    provenance: &Provenance,
    parent: Option<ParentGeneration>,
    mut hook: impl FnMut(PublishStage) -> Result<()>,
) -> Result<PathBuf> {
    require(
        matches!(selector, "best" | "last"),
        "checkpoint selector must be best or last",
    )?;
    let mut snapshot = session.snapshot()?;
    let config = session.config();
    require(
        config.train.device == device_name(session.device())?,
        "checkpoint device mismatch",
    )?;
    if selector == "best" {
        require(
            snapshot.trainer.epochs.last().is_some_and(|e| {
                e.epoch == snapshot.trainer.early.best_epoch
                    && Some(e.validation_loss) == snapshot.trainer.early.best
            }) && (snapshot.trainer.phase == Phase::Complete
                || snapshot.trainer.next_batch_offset == 0),
            "best publication requires a validation improvement boundary",
        )?;
        if run.join("best").exists() && !references(&run.join("best"))?.is_empty() {
            let old = super::load(run.join("best"), session.device())?;
            require(
                old.manifest
                    .validation_loss
                    .zip(snapshot.trainer.early.best)
                    .is_some_and(|(old, new)| new < old),
                "best publication requires strict improvement",
            )?;
        }
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| TrainError::InvalidState("system clock before epoch".into()))?
        .as_nanos();
    let generation = format!(
        "generation-{}-{}-{nonce}",
        snapshot.trainer.epoch,
        std::process::id()
    );
    if selector == "best" {
        snapshot.trainer.best_generation = Some(generation.clone());
    }
    let stage = run.join(".staging").join(&generation);
    fs::create_dir_all(run.join(".staging"))?;
    fs::create_dir(&stage)?;
    hook(PublishStage::StagingCreated)?;
    let weights: HashMap<_, _> = snapshot
        .weights
        .iter()
        .map(|(n, t)| (n.clone(), t.clone()))
        .collect();
    candle_core::safetensors::save(&weights, stage.join("model.safetensors"))?;
    session.optimizer.save(&stage)?;
    let buffers: HashMap<_, _> = snapshot
        .buffers
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    candle_core::safetensors::save(&buffers, stage.join("buffers.safetensors"))?;
    let schema = DataSchema::from_config(config)?;
    for (name, bytes) in [
        (
            "trainer-state.json",
            serde_json::to_vec_pretty(&snapshot.trainer)?,
        ),
        (
            "scheduler-state.json",
            serde_json::to_vec_pretty(&snapshot.scheduler)?,
        ),
        ("rng-state.json", serde_json::to_vec_pretty(&snapshot.rng)?),
        (
            "scaler.json",
            serde_json::to_vec_pretty(&session.data().scaler)?,
        ),
        ("data-schema.json", serde_json::to_vec_pretty(&schema)?),
        (
            "environment.json",
            serde_json::to_vec_pretty(&provenance.environment)?,
        ),
        (
            "resolved-config.toml",
            toml::to_string_pretty(config)?.into_bytes(),
        ),
    ] {
        write_synced(stage.join(name), &bytes)?;
    }
    let mut files = BTreeMap::new();
    for name in FILES {
        let p = stage.join(name);
        fs::OpenOptions::new().write(true).open(&p)?.sync_all()?;
        files.insert(
            name.into(),
            FileReceipt {
                bytes: fs::metadata(&p)?.len(),
                sha256: fingerprint(&p)?,
            },
        );
    }
    hook(PublishStage::PayloadsWritten)?;
    let manifest = Manifest {
        format_version: 2,
        model_version: 1,
        inference_only: false,
        model: config.model.name().into(),
        dtype: "f32".into(),
        upstream_sha: model_upstream(config).into(),
        data_sha256: session.data().cache.source_sha256.clone(),
        config_sha256: snapshot.config_sha256.clone(),
        protocol_fingerprint: schema.protocol_fingerprint,
        seed: config.train.seed,
        device: config.train.device.clone(),
        epoch: snapshot.trainer.epochs.last().map_or(0, |e| e.epoch),
        validation_loss: snapshot.trainer.epochs.last().map(|e| e.validation_loss),
        training: Some(TrainingIdentity::new(config, parent)),
        provenance: provenance.clone(),
        parameters: weights
            .iter()
            .map(|(n, t)| (n.clone(), t.dims().to_vec()))
            .collect(),
        files,
    };
    write_synced(
        stage.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    hook(PublishStage::ManifestWritten)?;
    let checked = super::load(&stage, session.device())?;
    TrainingSession::resume(
        config,
        session.data(),
        session.device(),
        read_state(&checked, session.device())?,
    )?;
    if selector == "last"
        && let Some(best) = &snapshot.trainer.best_generation
    {
        let mut target = checked;
        target.directory = run.join("checkpoints").join(&generation);
        require(
            run.join("checkpoints").join(best).is_dir(),
            "best generation missing from run",
        )?;
        verify_best(&target, &snapshot.trainer, session.device())?;
    }
    hook(PublishStage::Validated)?;
    let destination = run.join("checkpoints").join(&generation);
    fs::create_dir_all(run.join("checkpoints"))?;
    fs::rename(&stage, &destination)?;
    hook(PublishStage::GenerationCommitted)?;
    publish_reference(run, selector, &destination, nonce, &mut hook)?;
    if selector == "best" {
        session.state.best_generation = Some(generation);
    }
    Ok(destination)
}

pub(super) fn publish_reference(
    run: &Path,
    selector: &str,
    directory: &Path,
    nonce: u128,
    hook: &mut impl FnMut(PublishStage) -> Result<()>,
) -> Result<()> {
    let selector_dir = run.join(selector);
    fs::create_dir_all(&selector_dir)?;
    let refs = references(&selector_dir)?;
    let sequence = refs.last().map(|(n, _)| n + 1).unwrap_or(1);
    let reference = Reference {
        version: 1,
        generation: directory
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| TrainError::InvalidState("invalid generation name".into()))?
            .into(),
        manifest_sha256: fingerprint(directory.join("manifest.json"))?,
    };
    let pending = selector_dir.join(format!(".{sequence:020}-{nonce}.pending"));
    write_synced(&pending, &serde_json::to_vec(&reference)?)?;
    hook(PublishStage::ReferencePending)?;
    fs::rename(pending, selector_dir.join(format!("{sequence:020}.json")))?;
    Ok(())
}
