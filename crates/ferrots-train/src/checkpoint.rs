//! Immutable inference bundles and append-only best/last references.
use crate::{Result, TrainError};
use candle_core::{DType, Device};
use ferrots_core::config::{Config, Overrides};
use ferrots_data::{
    cache::{fingerprint, hash_bytes, protocol_hash, write_synced},
    scaler::Scaler,
};
use ferrots_nn::baselines::{Baseline, UPSTREAM_SHA};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const FILES: [&str; 4] = [
    "model.safetensors",
    "scaler.json",
    "data-schema.json",
    "resolved-config.toml",
];
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(TrainError::InvalidState(message.into()))
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub code_sha: String,
    pub dirty: bool,
    pub source_sha256: String,
    pub environment: BTreeMap<String, String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataSchema {
    pub format_version: u32,
    pub protocol_id: String,
    pub protocol_fingerprint: String,
    pub input_columns: Vec<String>,
    pub target_columns: Vec<String>,
    pub future_known_columns: Vec<String>,
    pub timestamp_format: String,
    pub interval_seconds: i64,
}
impl DataSchema {
    pub fn from_config(config: &Config) -> Result<Self> {
        let d = &config.data;
        Ok(Self {
            format_version: 1,
            protocol_id: d.protocol_id.clone(),
            protocol_fingerprint: protocol_hash(d)?,
            input_columns: d.input_columns.clone(),
            target_columns: d.target_columns.clone(),
            future_known_columns: d.future_known_columns.clone(),
            timestamp_format: d.timestamp_format.clone(),
            interval_seconds: d.interval_seconds,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileReceipt {
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format_version: u32,
    pub model_version: u32,
    pub inference_only: bool,
    pub model: String,
    pub dtype: String,
    pub upstream_sha: String,
    pub data_sha256: String,
    pub config_sha256: String,
    pub protocol_fingerprint: String,
    pub seed: u64,
    pub device: String,
    pub epoch: usize,
    pub validation_loss: f64,
    pub provenance: Provenance,
    pub parameters: BTreeMap<String, Vec<usize>>,
    pub files: BTreeMap<String, FileReceipt>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    version: u32,
    generation: String,
    manifest_sha256: String,
}
pub struct Snapshot<'a> {
    pub model: &'a Baseline,
    pub config: &'a Config,
    pub scaler: &'a Scaler,
    pub provenance: &'a Provenance,
    pub data_sha256: &'a str,
    pub epoch: usize,
    pub validation_loss: f64,
}
pub struct Checkpoint {
    pub model: Baseline,
    pub config: Config,
    pub scaler: Scaler,
    pub schema: DataSchema,
    pub manifest: Manifest,
    pub directory: PathBuf,
}
fn references(path: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut result = vec![];
    if path.exists() {
        for entry in fs::read_dir(path)? {
            let p = entry?.path();
            if p.extension().is_some_and(|s| s == "json") {
                let seq = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .ok_or_else(|| {
                        TrainError::InvalidState(
                            "invalid committed checkpoint reference name".into(),
                        )
                    })?;
                result.push((seq, p));
            }
        }
    }
    result.sort_by_key(|(seq, _)| *seq);
    Ok(result)
}
pub fn resolve(path: &Path) -> Result<PathBuf> {
    if path.join("manifest.json").is_file() {
        return Ok(path.to_path_buf());
    }
    let refs = references(path)?;
    let (_, latest) = refs
        .last()
        .ok_or_else(|| TrainError::InvalidState("no committed checkpoint reference".into()))?;
    let r: Reference = serde_json::from_slice(&fs::read(latest)?)?;
    require(
        r.version == 1
            && r.generation.starts_with("generation-")
            && r.generation
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-'),
        "invalid checkpoint reference",
    )?;
    let bundle = path
        .parent()
        .ok_or_else(|| TrainError::InvalidState("reference has no run directory".into()))?
        .join("checkpoints")
        .join(r.generation);
    require(
        fingerprint(bundle.join("manifest.json"))? == r.manifest_sha256,
        "checkpoint manifest checksum mismatch",
    )?;
    Ok(bundle)
}
pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Checkpoint> {
    let directory = resolve(path.as_ref())?;
    let manifest: Manifest = serde_json::from_slice(&fs::read(directory.join("manifest.json"))?)?;
    require(
        manifest.format_version == 1
            && manifest.model_version == 1
            && manifest.inference_only
            && manifest.dtype == "f32"
            && manifest.upstream_sha == UPSTREAM_SHA,
        "unsupported checkpoint version/capability/dtype/upstream",
    )?;
    require(
        manifest.validation_loss.is_finite()
            && manifest.data_sha256.len() == 64
            && manifest.provenance.source_sha256.len() == 64,
        "invalid checkpoint provenance",
    )?;
    require(
        manifest.files.len() == FILES.len()
            && FILES.iter().all(|n| manifest.files.contains_key(*n)),
        "checkpoint file set mismatch",
    )?;
    for name in FILES {
        let p = directory.join(name);
        let r = &manifest.files[name];
        require(
            fs::metadata(&p)?.len() == r.bytes && fingerprint(&p)? == r.sha256,
            "checkpoint file checksum mismatch",
        )?;
    }
    require(
        manifest.config_sha256 == manifest.files["resolved-config.toml"].sha256,
        "checkpoint config checksum mismatch",
    )?;
    let config = Config::load(directory.join("resolved-config.toml"), Overrides::default())?;
    config.validate_model_inputs()?;
    require(
        config.model.name() == manifest.model
            && config.train.seed == manifest.seed
            && config.train.device == manifest.device,
        "checkpoint configuration identity mismatch",
    )?;
    let schema: DataSchema =
        serde_json::from_slice(&fs::read(directory.join("data-schema.json"))?)?;
    require(
        schema == DataSchema::from_config(&config)?
            && schema.protocol_fingerprint == manifest.protocol_fingerprint,
        "checkpoint data schema/protocol mismatch",
    )?;
    let scaler: Scaler = serde_json::from_slice(&fs::read(directory.join("scaler.json"))?)?;
    scaler.validate_for(&config.data, &schema.input_columns)?;
    let model = Baseline::new(
        &config.model,
        &config.window,
        schema.input_columns.len(),
        config.train.seed,
        device,
    )?;
    let expected: BTreeMap<_, _> = model
        .weights()
        .iter()
        .map(|(n, v)| (n.clone(), v.dims().to_vec()))
        .collect();
    require(
        expected == manifest.parameters,
        "checkpoint parameter schema mismatch",
    )?;
    let weights = candle_core::safetensors::load(directory.join("model.safetensors"), device)?;
    require(
        weights.len() == expected.len()
            && weights.iter().all(|(n, t)| {
                t.dtype() == DType::F32 && expected.get(n).is_some_and(|s| s == t.dims())
            }),
        "checkpoint tensor names/shapes/dtypes mismatch",
    )?;
    model.load_weights(&weights.into_iter().collect())?;
    Ok(Checkpoint {
        model: model.freeze_for_eval()?,
        config,
        scaler,
        schema,
        manifest,
        directory,
    })
}

pub fn publish(run: &Path, selector: &str, snapshot: &Snapshot<'_>) -> Result<PathBuf> {
    require(
        matches!(selector, "best" | "last"),
        "checkpoint selector must be best or last",
    )?;
    snapshot.config.validate_model_inputs()?;
    snapshot
        .scaler
        .validate_for(&snapshot.config.data, &snapshot.config.data.input_columns)?;
    require(
        snapshot.validation_loss.is_finite(),
        "non-finite checkpoint validation loss",
    )?;
    require(
        serde_json::to_value(&snapshot.model.spec)?
            == serde_json::to_value(&snapshot.config.model)?
            && serde_json::to_value(&snapshot.model.window)?
                == serde_json::to_value(&snapshot.config.window)?
            && snapshot.model.channels == snapshot.config.data.input_columns.len(),
        "snapshot model/config mismatch",
    )?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| TrainError::InvalidState("system clock before epoch".into()))?
        .as_nanos();
    let generation = format!(
        "generation-{}-{}-{nonce}",
        snapshot.epoch,
        std::process::id()
    );
    let stage = run.join(".staging").join(&generation);
    fs::create_dir_all(&stage)?;
    let tensors: HashMap<_, _> = snapshot
        .model
        .weights()
        .iter()
        .map(|(n, v)| (n.clone(), v.detach()))
        .collect();
    let model_path = stage.join("model.safetensors");
    candle_core::safetensors::save(&tensors, &model_path)?;
    fs::OpenOptions::new()
        .write(true)
        .open(&model_path)?
        .sync_all()?;
    write_synced(
        stage.join("scaler.json"),
        &serde_json::to_vec_pretty(snapshot.scaler)?,
    )?;
    let schema = DataSchema::from_config(snapshot.config)?;
    write_synced(
        stage.join("data-schema.json"),
        &serde_json::to_vec_pretty(&schema)?,
    )?;
    let config = toml::to_string_pretty(snapshot.config)?;
    write_synced(stage.join("resolved-config.toml"), config.as_bytes())?;
    let files = FILES
        .iter()
        .map(|name| {
            let p = stage.join(name);
            Ok((
                name.to_string(),
                FileReceipt {
                    bytes: fs::metadata(&p)?.len(),
                    sha256: fingerprint(p)?,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let manifest = Manifest {
        format_version: 1,
        model_version: 1,
        inference_only: true,
        model: snapshot.config.model.name().into(),
        dtype: "f32".into(),
        upstream_sha: UPSTREAM_SHA.into(),
        data_sha256: snapshot.data_sha256.into(),
        config_sha256: hash_bytes(config.as_bytes()),
        protocol_fingerprint: schema.protocol_fingerprint,
        seed: snapshot.config.train.seed,
        device: snapshot.config.train.device.clone(),
        epoch: snapshot.epoch,
        validation_loss: snapshot.validation_loss,
        provenance: snapshot.provenance.clone(),
        parameters: tensors
            .iter()
            .map(|(n, v)| (n.clone(), v.dims().to_vec()))
            .collect(),
        files,
    };
    write_synced(
        stage.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    load(&stage, &Device::Cpu)?;
    let destination = run.join("checkpoints").join(&generation);
    fs::create_dir_all(run.join("checkpoints"))?;
    fs::rename(&stage, &destination)?;
    let selector_dir = run.join(selector);
    fs::create_dir_all(&selector_dir)?;
    let refs = references(&selector_dir)?;
    let sequence = refs.last().map(|(n, _)| n + 1).unwrap_or(1);
    let reference = Reference {
        version: 1,
        generation,
        manifest_sha256: fingerprint(destination.join("manifest.json"))?,
    };
    let pending = selector_dir.join(format!(".{sequence:020}-{nonce}.pending"));
    write_synced(&pending, &serde_json::to_vec(&reference)?)?;
    fs::rename(pending, selector_dir.join(format!("{sequence:020}.json")))?;
    Ok(destination)
}
