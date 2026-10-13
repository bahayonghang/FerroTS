//! Streaming metrics and optional long-table prediction output.
use crate::{
    Result, TrainError,
    checkpoint::Checkpoint,
    trainer::{DeviceBatch, SplitAccess, select_targets, tracked_batch},
};
use candle_core::Device;
use ferrots_core::metrics::{MetricGrid, MetricReport};
use ferrots_data::{
    batch::{Dataset, future_batch},
    cache::{fingerprint, load_inference},
    csv_source::timestamp_text,
};
use ferrots_nn::model::ForwardContext;
use flate2::{Compression, write::GzEncoder};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

enum Output {
    Plain(BufWriter<File>),
    Gzip(GzEncoder<BufWriter<File>>),
}
impl Write for Output {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(w) => w.write(b),
            Self::Gzip(w) => w.write(b),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(w) => w.flush(),
            Self::Gzip(w) => w.flush(),
        }
    }
}
impl Output {
    fn finish(self) -> std::io::Result<()> {
        let mut w = match self {
            Self::Plain(w) => w,
            Self::Gzip(w) => w.finish()?,
        };
        w.flush()?;
        w.get_ref().sync_all()
    }
}
#[derive(Serialize)]
struct PredictionRow<'a> {
    run_id: &'a str,
    series_id: &'a str,
    forecast_origin: &'a str,
    target_time: &'a str,
    horizon: usize,
    target_name: &'a str,
    y_true: Option<f64>,
    y_pred: f64,
    valid: bool,
}
struct PredictionWriter {
    writer: csv::Writer<Output>,
    columns: Vec<usize>,
    rows: u64,
}
impl PredictionWriter {
    fn new(path: &Path, names: &[String], filter: &[String]) -> Result<Self> {
        let columns = if filter.is_empty() {
            (0..names.len()).collect()
        } else {
            filter
                .iter()
                .map(|n| {
                    names.iter().position(|v| v == n).ok_or_else(|| {
                        TrainError::InvalidState(format!("unknown export target: {n}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?
        };
        if columns
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != columns.len()
        {
            return Err(TrainError::InvalidState("duplicate export target".into()));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = BufWriter::new(File::options().write(true).create_new(true).open(path)?);
        let output = if path.extension().is_some_and(|s| s == "gz") {
            Output::Gzip(GzEncoder::new(file, Compression::fast()))
        } else {
            Output::Plain(file)
        };
        Ok(Self {
            writer: csv::Writer::from_writer(output),
            columns,
            rows: 0,
        })
    }
    fn row(&mut self, column: usize, row: PredictionRow<'_>) -> Result<()> {
        if self.columns.contains(&column) {
            self.writer.serialize(row)?;
            self.rows += 1;
        }
        Ok(())
    }
    fn finish(self) -> Result<u64> {
        let rows = self.rows;
        self.writer
            .into_inner()
            .map_err(|e| e.into_error())?
            .finish()?;
        Ok(rows)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    pub format_version: u32,
    pub run_id: String,
    pub protocol_id: String,
    pub model: String,
    pub split: u8,
    pub seed: u64,
    pub sampling_policy: String,
    pub first_forward_id: u64,
    pub batch_size: usize,
    pub data_sha256: String,
    pub checkpoint_sha256: String,
    pub metrics: MetricReport,
    pub windows: usize,
    pub prediction_rows: u64,
    pub access: SplitAccess,
    pub seconds: f64,
}
pub struct EvaluationOptions<'a> {
    pub run_id: &'a str,
    pub split: u8,
    pub batch_size: usize,
    pub export: Option<&'a Path>,
    pub columns: &'a [String],
}
pub fn evaluate(
    checkpoint: &Checkpoint,
    data: &Dataset,
    device: &Device,
    options: &EvaluationOptions<'_>,
) -> Result<Evaluation> {
    if options.split > 2 || options.batch_size == 0 || !checkpoint.model.is_frozen() {
        return Err(TrainError::InvalidState(
            "invalid evaluation options/model".into(),
        ));
    }
    let start = Instant::now();
    let mut metrics = MetricGrid::new(data.spec.target_columns.clone(), data.window.horizon)?;
    let mut writer = options
        .export
        .map(|p| PredictionWriter::new(p, &data.spec.target_columns, options.columns))
        .transpose()?;
    let mut access = SplitAccess::default();
    let c = data.target_indices.len();
    // Separate validation (01), evaluation (10) and future prediction (11)
    // in the high two bits. Evaluation also binds the split in bits 60..61.
    let first_forward_id = (2 << 62) | ((options.split as u64) << 60);
    for (batch_id, chunk) in data
        .index
        .order(options.split)
        .chunks(options.batch_size)
        .enumerate()
    {
        let host = tracked_batch(data, options.split, chunk, &mut access)?;
        let batch = DeviceBatch::from_host(&host, device)?;
        let pred = select_targets(
            &checkpoint.model.forward_with_context(
                &batch.input,
                ForwardContext::evaluation_with_id(first_forward_id | batch_id as u64),
            )?,
            &data.target_indices,
        )?;
        if pred.track_op() {
            return Err(TrainError::InvalidState(
                "evaluation output tracks a graph".into(),
            ));
        }
        let values = pred.flatten_all()?.to_vec1::<f32>()?;
        for (b, origin) in host.origins.iter().enumerate() {
            let origin_text = if writer.is_some() {
                timestamp_text(origin.time)?
            } else {
                String::new()
            };
            for h in 0..data.window.horizon {
                let row = origin.row + data.window.gap + h;
                let target_time = if writer.is_some() {
                    timestamp_text(data.series.timestamps[row])?
                } else {
                    String::new()
                };
                for (j, &col) in data.target_indices.iter().enumerate() {
                    let k = (b * data.window.horizon + h) * c + j;
                    let valid = host.supervision.valid[k] != 0;
                    let pred = f64::from(values[k]);
                    let target = f64::from(host.supervision.targets[k]);
                    let original_prediction = data.scaler.inverse(col, pred);
                    let original_target = data.series.values[row * data.series.width() + col];
                    metrics.add(
                        h,
                        j,
                        (pred, target),
                        (original_prediction, original_target),
                        valid,
                    )?;
                    if let Some(writer) = &mut writer {
                        writer.row(
                            j,
                            PredictionRow {
                                run_id: options.run_id,
                                series_id: "0",
                                forecast_origin: &origin_text,
                                target_time: &target_time,
                                horizon: h + 1,
                                target_name: &data.spec.target_columns[j],
                                y_true: valid.then_some(original_target),
                                y_pred: original_prediction,
                                valid,
                            },
                        )?;
                    }
                }
            }
        }
    }
    let metrics = metrics.finish();
    if metrics.standardized.overall.count == 0 {
        return Err(TrainError::InvalidState(
            "evaluation has no valid labels".into(),
        ));
    }
    let prediction_rows = writer
        .map(PredictionWriter::finish)
        .transpose()?
        .unwrap_or(0);
    Ok(Evaluation {
        format_version: 1,
        run_id: options.run_id.into(),
        protocol_id: data.spec.protocol_id.clone(),
        model: checkpoint.config.model.name().into(),
        split: options.split,
        seed: checkpoint.config.train.seed,
        sampling_policy: "evaluation-split-batch-v1".into(),
        first_forward_id,
        batch_size: options.batch_size,
        data_sha256: data.cache.source_sha256.clone(),
        checkpoint_sha256: fingerprint(checkpoint.directory.join("manifest.json"))?,
        metrics,
        windows: access.windows[options.split as usize],
        prediction_rows,
        access,
        seconds: start.elapsed().as_secs_f64(),
    })
}
#[derive(Serialize)]
pub struct FuturePrediction {
    pub format_version: u32,
    pub run_id: String,
    pub data_sha256: String,
    pub checkpoint_sha256: String,
    pub seed: u64,
    pub sampling_policy: String,
    pub forward_id: u64,
    pub rows: u64,
    pub output: PathBuf,
}
pub fn predict_future(
    checkpoint: &Checkpoint,
    input: Option<&Path>,
    device: &Device,
    run_id: &str,
    output: &Path,
    columns: &[String],
) -> Result<FuturePrediction> {
    if !checkpoint.model.is_frozen() {
        return Err(TrainError::InvalidState(
            "prediction requires an independent frozen model".into(),
        ));
    }
    let mut spec = checkpoint.config.data.clone();
    if let Some(input) = input {
        spec.path = input.into();
    }
    let (series, cache) = load_inference(&spec)?;
    let host = future_batch(
        &series,
        &spec,
        &checkpoint.config.window,
        &checkpoint.scaler,
    )?;
    let batch = DeviceBatch::from_host(&host, device)?;
    let indices: Vec<usize> = spec
        .target_columns
        .iter()
        .map(|n| {
            series
                .columns
                .iter()
                .position(|c| c == n)
                .expect("validated targets")
        })
        .collect();
    let forward_id = 3 << 62;
    let prediction = checkpoint
        .model
        .forward_with_context(&batch.input, ForwardContext::evaluation_with_id(forward_id))?;
    if prediction.track_op() {
        return Err(TrainError::InvalidState(
            "prediction output tracks a graph".into(),
        ));
    }
    let pred = select_targets(&prediction, &indices)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut writer = PredictionWriter::new(output, &spec.target_columns, columns)?;
    let origin = &host.origins[0];
    let origin_text = timestamp_text(origin.time)?;
    for h in 0..host.horizon {
        let target_time = timestamp_text(
            origin.time
                + (checkpoint.config.window.gap + h) as i64 * spec.interval_seconds * 1_000_000,
        )?;
        for (j, &col) in indices.iter().enumerate() {
            let y = checkpoint
                .scaler
                .inverse(col, f64::from(pred[h * indices.len() + j]));
            if !y.is_finite() {
                return Err(TrainError::InvalidState(
                    "non-finite future prediction".into(),
                ));
            }
            writer.row(
                j,
                PredictionRow {
                    run_id,
                    series_id: "0",
                    forecast_origin: &origin_text,
                    target_time: &target_time,
                    horizon: h + 1,
                    target_name: &spec.target_columns[j],
                    y_true: None,
                    y_pred: y,
                    valid: false,
                },
            )?;
        }
    }
    Ok(FuturePrediction {
        format_version: 1,
        run_id: run_id.into(),
        data_sha256: cache.source_sha256,
        checkpoint_sha256: fingerprint(checkpoint.directory.join("manifest.json"))?,
        seed: checkpoint.config.train.seed,
        sampling_policy: "future-fixed-v1".into(),
        forward_id,
        rows: writer.finish()?,
        output: output.into(),
    })
}
