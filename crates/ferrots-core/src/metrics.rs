//! f64 element-weighted metrics. Missing labels never enter arithmetic.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Debug, thiserror::Error)]
#[error("invalid metrics: {0}")]
pub struct MetricError(pub &'static str);
#[derive(Debug, Clone, Copy, Default)]
pub struct Accumulator {
    pub count: u64,
    absolute: f64,
    squared: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub count: u64,
    pub mae: Option<f64>,
    pub mse: Option<f64>,
    pub rmse: Option<f64>,
}
impl Accumulator {
    pub fn add(&mut self, prediction: f64, target: f64, valid: bool) -> Result<(), MetricError> {
        if !valid {
            return Ok(());
        }
        let error = prediction - target;
        let absolute = self.absolute + error.abs();
        let squared = self.squared + error * error;
        if !prediction.is_finite()
            || !target.is_finite()
            || !absolute.is_finite()
            || !squared.is_finite()
        {
            return Err(MetricError("non-finite valid observation or sum"));
        }
        self.count = self
            .count
            .checked_add(1)
            .ok_or(MetricError("count overflow"))?;
        self.absolute = absolute;
        self.squared = squared;
        Ok(())
    }
    fn merge(&mut self, other: Self) {
        self.count += other.count;
        self.absolute += other.absolute;
        self.squared += other.squared;
    }
    pub fn finish(self) -> Metric {
        let mse = (self.count > 0).then(|| self.squared / self.count as f64);
        Metric {
            count: self.count,
            mae: (self.count > 0).then(|| self.absolute / self.count as f64),
            mse,
            rmse: mse.map(f64::sqrt),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardizedMetrics {
    pub overall: Metric,
    pub variables: BTreeMap<String, Metric>,
    pub horizons: Vec<Metric>,
    pub by_variable_horizon: BTreeMap<String, Vec<Metric>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginalMetrics {
    pub variables: BTreeMap<String, Metric>,
    pub by_variable_horizon: BTreeMap<String, Vec<Metric>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricReport {
    pub standardized: StandardizedMetrics,
    pub original: OriginalMetrics,
}
pub struct MetricGrid {
    names: Vec<String>,
    horizon: usize,
    standardized: Vec<Accumulator>,
    original: Vec<Accumulator>,
}
impl MetricGrid {
    pub fn new(names: Vec<String>, horizon: usize) -> Result<Self, MetricError> {
        if names.is_empty()
            || horizon == 0
            || names
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != names.len()
        {
            return Err(MetricError("invalid metric dimensions"));
        }
        let size = names
            .len()
            .checked_mul(horizon)
            .ok_or(MetricError("metric size overflow"))?;
        Ok(Self {
            names,
            horizon,
            standardized: vec![Accumulator::default(); size],
            original: vec![Accumulator::default(); size],
        })
    }
    pub fn add(
        &mut self,
        horizon: usize,
        column: usize,
        standardized: (f64, f64),
        original: (f64, f64),
        valid: bool,
    ) -> Result<(), MetricError> {
        if horizon >= self.horizon || column >= self.names.len() {
            return Err(MetricError("metric index out of range"));
        }
        let idx = horizon * self.names.len() + column;
        self.standardized[idx].add(standardized.0, standardized.1, valid)?;
        self.original[idx].add(original.0, original.1, valid)
    }
    pub fn finish(&self) -> MetricReport {
        let c = self.names.len();
        let mut total = Accumulator::default();
        let mut horizons = vec![Accumulator::default(); self.horizon];
        let mut variables = BTreeMap::new();
        let mut original_vars = BTreeMap::new();
        let mut detail = BTreeMap::new();
        let mut original_detail = BTreeMap::new();
        for (j, name) in self.names.iter().enumerate() {
            let mut s = Accumulator::default();
            let mut o = Accumulator::default();
            let mut sv = vec![];
            let mut ov = vec![];
            for (h, h_acc) in horizons.iter_mut().enumerate() {
                let a = self.standardized[h * c + j];
                let b = self.original[h * c + j];
                s.merge(a);
                o.merge(b);
                total.merge(a);
                h_acc.merge(a);
                sv.push(a.finish());
                ov.push(b.finish());
            }
            variables.insert(name.clone(), s.finish());
            original_vars.insert(name.clone(), o.finish());
            detail.insert(name.clone(), sv);
            original_detail.insert(name.clone(), ov);
        }
        MetricReport {
            standardized: StandardizedMetrics {
                overall: total.finish(),
                variables,
                horizons: horizons.into_iter().map(Accumulator::finish).collect(),
                by_variable_horizon: detail,
            },
            original: OriginalMetrics {
                variables: original_vars,
                by_variable_horizon: original_detail,
            },
        }
    }
}
