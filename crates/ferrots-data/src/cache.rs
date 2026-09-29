//! Content-addressed, immutable prepared tables. Cache hits do not parse CSV.
use crate::{
    PreparedSeries, RawReport, Result, Segment, Span, ensure, prepare, preprocess::PreparedReport,
    read_csv,
};
use ferrots_core::config::DataSpec;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub const CACHE_VERSION: u32 = 1;
pub fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn fingerprint(path: impl AsRef<Path>) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub fn protocol_hash(spec: &DataSpec) -> Result<String> {
    let mut normalized = spec.clone();
    normalized.path = PathBuf::new();
    normalized.cache_dir = PathBuf::new();
    normalized.cache_bypass = false;
    Ok(hash_bytes(&serde_json::to_vec(&normalized)?))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub key: String,
    pub source_sha256: String,
    pub hit: bool,
    pub parse_count: usize,
    pub hash_seconds: f64,
    pub total_seconds: f64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    version: u32,
    key: String,
    columns: Vec<String>,
    rows: usize,
    segments: Vec<Segment>,
    spans: Vec<Span>,
    raw: RawReport,
    report: PreparedReport,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    key: String,
    table_sha256: String,
    metadata_sha256: String,
}
pub fn write_synced(path: impl AsRef<Path>, bytes: &[u8]) -> Result<()> {
    let mut file = File::options().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn save(directory: &Path, key: &str, p: &PreparedSeries) -> Result<()> {
    let times: Vec<u8> = p.timestamps.iter().flat_map(|v| v.to_le_bytes()).collect();
    let values: Vec<u8> = p.values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "timestamps",
        TensorView::new(Dtype::I64, vec![p.rows()], &times)?,
    );
    tensors.insert(
        "values",
        TensorView::new(Dtype::F64, vec![p.rows(), p.width()], &values)?,
    );
    for (name, data) in [
        ("observed", &p.observed),
        ("available", &p.available),
        ("imputed", &p.imputed),
    ] {
        tensors.insert(
            name,
            TensorView::new(Dtype::U8, vec![p.rows(), p.width()], data)?,
        );
    }
    tensors.insert(
        "splits",
        TensorView::new(Dtype::U8, vec![p.rows()], &p.splits)?,
    );
    let serialized = safetensors::serialize(tensors, None)?;
    write_synced(directory.join("table.safetensors"), &serialized)?;
    let metadata = Metadata {
        version: CACHE_VERSION,
        key: key.into(),
        columns: p.columns.clone(),
        rows: p.rows(),
        segments: p.segments.clone(),
        spans: p.spans.clone(),
        raw: p.raw.clone(),
        report: p.report.clone(),
    };
    let json = serde_json::to_vec(&metadata)?;
    write_synced(directory.join("metadata.json"), &json)?;
    write_synced(
        directory.join("manifest.json"),
        &serde_json::to_vec(&Manifest {
            version: CACHE_VERSION,
            key: key.into(),
            table_sha256: hash_bytes(&serialized),
            metadata_sha256: hash_bytes(&json),
        })?,
    )
}
fn load(directory: &Path, key: &str) -> Result<PreparedSeries> {
    let manifest: Manifest = serde_json::from_slice(&fs::read(directory.join("manifest.json"))?)?;
    ensure(
        manifest.version == CACHE_VERSION && manifest.key == key,
        "cache version/key mismatch",
    )?;
    let json = fs::read(directory.join("metadata.json"))?;
    let bytes = fs::read(directory.join("table.safetensors"))?;
    ensure(
        hash_bytes(&json) == manifest.metadata_sha256
            && hash_bytes(&bytes) == manifest.table_sha256,
        "cache checksum mismatch; use explicit bypass to rebuild",
    )?;
    let meta: Metadata = serde_json::from_slice(&json)?;
    ensure(
        meta.version == CACHE_VERSION
            && meta.key == key
            && meta.rows > 0
            && !meta.columns.is_empty(),
        "invalid cache metadata",
    )?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    ensure(tensors.len() == 6, "unexpected cache tensor keys")?;
    let get = |name: &str, dtype: Dtype, shape: &[usize]| -> Result<Vec<u8>> {
        let view = tensors.tensor(name)?;
        ensure(
            view.dtype() == dtype && view.shape() == shape,
            format!("cache tensor schema mismatch: {name}"),
        )?;
        Ok(view.data().to_vec())
    };
    let timestamps = get("timestamps", Dtype::I64, &[meta.rows])?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b))
        .collect();
    let values: Vec<f64> = get("values", Dtype::F64, &[meta.rows, meta.columns.len()])?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    let observed = get("observed", Dtype::U8, &[meta.rows, meta.columns.len()])?;
    let available = get("available", Dtype::U8, &[meta.rows, meta.columns.len()])?;
    let imputed = get("imputed", Dtype::U8, &[meta.rows, meta.columns.len()])?;
    let splits = get("splits", Dtype::U8, &[meta.rows])?;
    ensure(
        values.iter().all(|v| v.is_finite())
            && observed
                .iter()
                .chain(&available)
                .chain(&imputed)
                .all(|&m| m <= 1)
            && splits.iter().all(|&s| s <= 3),
        "invalid cached values/masks",
    )?;
    ensure(
        meta.segments
            .iter()
            .enumerate()
            .all(|(id, s)| s.id == id && s.start < s.end && s.end <= meta.rows)
            && meta.spans.iter().all(|s| {
                meta.segments
                    .get(s.segment)
                    .is_some_and(|seg| s.start >= seg.start && s.end <= seg.end && s.start < s.end)
            }),
        "invalid cached segments",
    )?;
    Ok(PreparedSeries {
        columns: meta.columns,
        timestamps,
        values,
        observed,
        available,
        imputed,
        splits,
        segments: meta.segments,
        spans: meta.spans,
        raw: meta.raw,
        report: meta.report,
    })
}
pub fn load_prepared(spec: &DataSpec) -> Result<(PreparedSeries, CacheStats)> {
    load_mode(spec, false)
}
pub fn load_inference(spec: &DataSpec) -> Result<(PreparedSeries, CacheStats)> {
    load_mode(spec, true)
}
fn load_mode(spec: &DataSpec, inference: bool) -> Result<(PreparedSeries, CacheStats)> {
    let start = Instant::now();
    let source_sha256 = fingerprint(&spec.path)?;
    let hash_seconds = start.elapsed().as_secs_f64();
    let key = hash_bytes(
        format!(
            "{CACHE_VERSION}:{inference}:{source_sha256}:{}",
            protocol_hash(spec)?
        )
        .as_bytes(),
    );
    let directory = spec.cache_dir.join(&key);
    let hit = directory.exists() && !spec.cache_bypass;
    let p = if hit {
        load(&directory, &key)?
    } else {
        let frame = read_csv(spec)?;
        let p = if inference {
            crate::preprocess::prepare_inference(frame, spec)?
        } else {
            prepare(frame, spec)?
        };
        ensure(
            fingerprint(&spec.path)? == source_sha256,
            "source changed while parsing",
        )?;
        if !spec.cache_bypass {
            fs::create_dir_all(&spec.cache_dir)?;
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after epoch")
                .as_nanos();
            let stage = spec
                .cache_dir
                .join(format!(".staging-{}-{nonce}", std::process::id()));
            fs::create_dir(&stage)?;
            save(&stage, &key, &p)?;
            load(&stage, &key)?;
            fs::rename(&stage, &directory)?;
        }
        p
    };
    let stats = CacheStats {
        key,
        source_sha256,
        hit,
        parse_count: usize::from(!hit),
        hash_seconds,
        total_seconds: start.elapsed().as_secs_f64(),
    };
    Ok((p, stats))
}
