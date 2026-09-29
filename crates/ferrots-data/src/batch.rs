use crate::{
    PreparedSeries, Result,
    cache::{CacheStats, load_prepared},
    ensure,
    scaler::Scaler,
    time_features,
    windows::WindowIndex,
};
use ferrots_core::config::{DataSpec, WindowSpec};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Origin {
    pub row: usize,
    pub time: i64,
    pub segment: usize,
}
pub struct HostInput {
    pub history: Vec<f32>,
    pub history_time: Vec<f32>,
    pub future_time: Vec<f32>,
}
pub struct HostSupervision {
    pub targets: Vec<f32>,
    pub valid: Vec<u8>,
}
pub struct HostBatch {
    pub batch_size: usize,
    pub lookback: usize,
    pub horizon: usize,
    pub channels: usize,
    pub target_count: usize,
    pub time_width: usize,
    pub input: HostInput,
    pub supervision: HostSupervision,
    pub origins: Vec<Origin>,
}
pub struct Dataset {
    pub series: PreparedSeries,
    pub scaler: Scaler,
    pub index: WindowIndex,
    pub cache: CacheStats,
    pub spec: DataSpec,
    pub window: WindowSpec,
    pub target_indices: Vec<usize>,
    normalized: Vec<f32>,
    features: Vec<f32>,
    time_width: usize,
}

/// Build one future-only batch with stored training scales. No labels are read.
pub fn future_batch(
    series: &PreparedSeries,
    spec: &DataSpec,
    window: &WindowSpec,
    scaler: &Scaler,
) -> Result<HostBatch> {
    scaler.validate_for(spec, &series.columns)?;
    let n = series.rows();
    let c = series.width();
    let l = window.lookback;
    let h = window.horizon;
    let span = series
        .spans
        .last()
        .ok_or_else(|| crate::DataError::Invalid("no complete tail segment".into()))?;
    ensure(
        span.end == n && n - span.start >= l,
        "tail history is outside a valid segment",
    )?;
    ensure(
        series.available[(n - l) * c..n * c].iter().all(|&m| m != 0),
        "tail history has missing inputs",
    )?;
    let history = series.values[(n - l) * c..n * c]
        .iter()
        .enumerate()
        .map(|(k, &v)| scaler.transform(k % c, v))
        .collect();
    let delta = spec.interval_seconds * 1_000_000;
    let origin = series.timestamps[n - 1] + delta;
    let future: Vec<i64> = (0..h)
        .map(|k| origin + (window.gap + k) as i64 * delta)
        .collect();
    Ok(HostBatch {
        batch_size: 1,
        lookback: l,
        horizon: h,
        channels: c,
        target_count: spec.target_columns.len(),
        time_width: time_features::width(&spec.time_features)?,
        input: HostInput {
            history,
            history_time: time_features::compute(&series.timestamps[n - l..], &spec.time_features)?,
            future_time: time_features::compute(&future, &spec.time_features)?,
        },
        supervision: HostSupervision {
            targets: vec![0.; h * spec.target_columns.len()],
            valid: vec![0; h * spec.target_columns.len()],
        },
        origins: vec![Origin {
            row: n,
            time: origin,
            segment: span.segment,
        }],
    })
}
impl Dataset {
    pub fn load(spec: &DataSpec, window: &WindowSpec, scaler: Option<Scaler>) -> Result<Self> {
        let (series, cache) = load_prepared(spec)?;
        Self::new(series, cache, spec, window, scaler)
    }
    pub fn new(
        series: PreparedSeries,
        cache: CacheStats,
        spec: &DataSpec,
        window: &WindowSpec,
        scaler: Option<Scaler>,
    ) -> Result<Self> {
        let scaler = match scaler {
            Some(s) => {
                s.validate_for(spec, &series.columns)?;
                s
            }
            None => Scaler::fit(&series, spec)?,
        };
        let index = WindowIndex::build(&series, window)?;
        let target_indices =
            spec.target_columns
                .iter()
                .map(|n| {
                    series.columns.iter().position(|c| c == n).ok_or_else(|| {
                        crate::DataError::Invalid(format!("target column absent: {n}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
        let normalized: Vec<_> = series
            .values
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                if series.available[k] == 0 {
                    0.0
                } else {
                    scaler.transform(k % series.width(), v)
                }
            })
            .collect();
        ensure(
            normalized.iter().all(|v| v.is_finite()),
            "normalized data exceeds FP32 range",
        )?;
        let features = time_features::compute(&series.timestamps, &spec.time_features)?;
        let time_width = time_features::width(&spec.time_features)?;
        Ok(Self {
            series,
            scaler,
            index,
            cache,
            spec: spec.clone(),
            window: window.clone(),
            target_indices,
            normalized,
            features,
            time_width,
        })
    }
    pub fn batch(&self, entries: &[usize]) -> Result<HostBatch> {
        ensure(!entries.is_empty(), "empty batch")?;
        let (b, l, h, c, ct, f) = (
            entries.len(),
            self.window.lookback,
            self.window.horizon,
            self.series.width(),
            self.target_indices.len(),
            self.time_width,
        );
        let mut input = HostInput {
            history: Vec::with_capacity(b * l * c),
            history_time: Vec::with_capacity(b * l * f),
            future_time: Vec::with_capacity(b * h * f),
        };
        let mut supervision = HostSupervision {
            targets: Vec::with_capacity(b * h * ct),
            valid: Vec::with_capacity(b * h * ct),
        };
        let mut origins = Vec::with_capacity(b);
        for &idx in entries {
            let w = self
                .index
                .entries
                .get(idx)
                .ok_or_else(|| crate::DataError::Invalid("batch index out of bounds".into()))?;
            let o = w.origin;
            let a = o + self.window.gap;
            let z = a + h;
            input
                .history
                .extend_from_slice(&self.normalized[(o - l) * c..o * c]);
            input
                .history_time
                .extend_from_slice(&self.features[(o - l) * f..o * f]);
            input
                .future_time
                .extend_from_slice(&self.features[a * f..z * f]);
            for i in a..z {
                for &j in &self.target_indices {
                    let k = i * c + j;
                    let valid = if self.spec.missing.filled_targets_valid {
                        self.series.available[k]
                    } else {
                        self.series.observed[k]
                    };
                    supervision.valid.push(valid);
                    supervision
                        .targets
                        .push(if valid != 0 { self.normalized[k] } else { 0.0 });
                }
            }
            origins.push(Origin {
                row: o,
                time: self.series.timestamps[o],
                segment: w.segment,
            });
        }
        Ok(HostBatch {
            batch_size: b,
            lookback: l,
            horizon: h,
            channels: c,
            target_count: ct,
            time_width: f,
            input,
            supervision,
            origins,
        })
    }
}
