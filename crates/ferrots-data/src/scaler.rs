use crate::{PreparedSeries, Result, ensure, preprocess::TRAIN};
use ferrots_core::config::DataSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnScale {
    pub name: String,
    pub count: u64,
    pub mean: f64,
    pub scale: f64,
    pub constant: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scaler {
    pub format_version: u32,
    pub protocol_id: String,
    pub protocol_fingerprint: String,
    pub fit_split: String,
    pub columns: Vec<ColumnScale>,
}
impl Scaler {
    pub fn fit(series: &PreparedSeries, spec: &DataSpec) -> Result<Self> {
        let c = series.width();
        let mut columns: Vec<_> = series
            .columns
            .iter()
            .map(|name| ColumnScale {
                name: name.clone(),
                count: 0,
                mean: 0.0,
                scale: 0.0,
                constant: false,
            })
            .collect();
        for span in &series.spans {
            for i in span.start..span.end {
                if series.splits[i] != TRAIN {
                    continue;
                }
                for (j, column) in columns.iter_mut().enumerate() {
                    let k = i * c + j;
                    let valid = if spec.missing.filled_targets_valid {
                        series.available[k]
                    } else {
                        series.observed[k]
                    };
                    if valid == 0 {
                        continue;
                    }
                    column.count += 1;
                    let delta = series.values[k] - column.mean;
                    column.mean += delta / column.count as f64;
                    column.scale += delta * (series.values[k] - column.mean);
                }
            }
        }
        for column in &mut columns {
            ensure(
                column.count > 0,
                format!("no training observations for {}", column.name),
            )?;
            column.constant = column.scale == 0.0;
            column.scale = if column.constant {
                1.0
            } else {
                (column.scale / column.count as f64).sqrt()
            };
            ensure(
                column.mean.is_finite() && column.scale.is_finite() && column.scale > 0.0,
                "invalid fitted scaler",
            )?;
        }
        Ok(Self {
            format_version: 1,
            protocol_id: spec.protocol_id.clone(),
            protocol_fingerprint: crate::cache::protocol_hash(spec)?,
            fit_split: "train".into(),
            columns,
        })
    }
    pub fn validate_for(&self, spec: &DataSpec, names: &[String]) -> Result<()> {
        ensure(
            self.format_version == 1
                && self.fit_split == "train"
                && self.protocol_id == spec.protocol_id
                && self.protocol_fingerprint == crate::cache::protocol_hash(spec)?,
            "scaler protocol mismatch",
        )?;
        ensure(
            self.columns.len() == names.len()
                && self.columns.iter().zip(names).all(|(s, n)| {
                    s.name == *n
                        && s.count > 0
                        && s.mean.is_finite()
                        && s.scale.is_finite()
                        && s.scale > 0.0
                }),
            "scaler column/schema mismatch",
        )
    }
    pub fn transform(&self, column: usize, value: f64) -> f32 {
        ((value - self.columns[column].mean) / self.columns[column].scale) as f32
    }
    pub fn inverse(&self, column: usize, value: f64) -> f64 {
        value * self.columns[column].scale + self.columns[column].mean
    }
}
