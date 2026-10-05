use crate::{NamedParams, Result, TrainError, params::validate_like};
use candle_core::{Tensor, backprop::GradStore};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::Path,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamGroup {
    pub names: Vec<String>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}
impl ParamGroup {
    pub fn adam(names: Vec<String>, lr: f64) -> Self {
        Self {
            names,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}
fn validate_groups(params: &NamedParams, groups: &[ParamGroup]) -> Result<()> {
    let mut names = BTreeSet::new();
    for g in groups {
        if g.names.is_empty()
            || !g.lr.is_finite()
            || g.lr < 0.0
            || !g.beta1.is_finite()
            || !(0.0..1.0).contains(&g.beta1)
            || !g.beta2.is_finite()
            || !(0.0..1.0).contains(&g.beta2)
            || !g.eps.is_finite()
            || g.eps <= 0.0
            || !g.weight_decay.is_finite()
            || g.weight_decay < 0.0
        {
            return Err(TrainError::InvalidState(
                "invalid optimizer parameter group".into(),
            ));
        }
        for name in &g.names {
            if params.get(name).is_none() || !names.insert(name.as_str()) {
                return Err(TrainError::InvalidState(format!(
                    "unknown or repeated group parameter {name}"
                )));
            }
        }
    }
    if names != params.iter().map(|(n, _)| n).collect() {
        return Err(TrainError::InvalidState(
            "parameter groups must cover each parameter exactly once".into(),
        ));
    }
    Ok(())
}

pub struct ParamState {
    pub step: u64,
    pub exp_avg: Tensor,
    pub exp_avg_sq: Tensor,
}
pub struct OptimizerSnapshot {
    pub kind: String,
    pub groups: Vec<ParamGroup>,
    pub state: BTreeMap<String, ParamState>,
}
/// Adam uses coupled L2 decay. AdamW uses decoupled parameter decay.
pub type StatefulAdam = StatefulOptimizer<false>;
pub type StatefulAdamW = StatefulOptimizer<true>;
pub struct StatefulOptimizer<const DECOUPLED: bool> {
    params: NamedParams,
    groups: Vec<ParamGroup>,
    state: BTreeMap<String, ParamState>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterMetadata {
    name: String,
    shape: Vec<usize>,
    step: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    format_version: u32,
    kind: String,
    dtype: String,
    parameters: Vec<ParameterMetadata>,
    groups: Vec<ParamGroup>,
}
impl<const DECOUPLED: bool> StatefulOptimizer<DECOUPLED> {
    pub fn snapshot(&self) -> Result<OptimizerSnapshot> {
        Ok(OptimizerSnapshot {
            kind: Self::kind().into(),
            groups: self.groups.clone(),
            state: self
                .state
                .iter()
                .map(|(name, s)| {
                    Ok((
                        name.clone(),
                        ParamState {
                            step: s.step,
                            exp_avg: s.exp_avg.detach().copy()?,
                            exp_avg_sq: s.exp_avg_sq.detach().copy()?,
                        },
                    ))
                })
                .collect::<Result<_>>()?,
        })
    }
    pub fn restore_snapshot(&mut self, snapshot: &OptimizerSnapshot) -> Result<()> {
        if snapshot.kind != Self::kind() || snapshot.state.len() != self.params.iter().count() {
            return Err(TrainError::InvalidState(
                "optimizer snapshot kind/parameter set mismatch".into(),
            ));
        }
        validate_groups(&self.params, &snapshot.groups)?;
        let mut pending = BTreeMap::new();
        for (name, var) in self.params.iter() {
            let s = snapshot.state.get(name).ok_or_else(|| {
                TrainError::InvalidState(format!("missing optimizer state {name}"))
            })?;
            validate_like(name, &s.exp_avg, var)?;
            validate_like(name, &s.exp_avg_sq, var)?;
            if s.exp_avg
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .any(|v| !v.is_finite())
                || s.exp_avg_sq
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .any(|&v| !v.is_finite() || v < 0.)
            {
                return Err(TrainError::InvalidState(
                    "non-finite optimizer state or negative second moment".into(),
                ));
            }
            pending.insert(
                name.into(),
                ParamState {
                    step: s.step,
                    exp_avg: s.exp_avg.detach().copy()?,
                    exp_avg_sq: s.exp_avg_sq.detach().copy()?,
                },
            );
        }
        self.state = pending;
        self.groups = snapshot.groups.clone();
        Ok(())
    }
    pub fn new(params: NamedParams, groups: Vec<ParamGroup>) -> Result<Self> {
        validate_groups(&params, &groups)?;
        let mut state = BTreeMap::new();
        for (name, var) in params.iter() {
            state.insert(
                name.to_owned(),
                ParamState {
                    step: 0,
                    exp_avg: var.zeros_like()?.detach(),
                    exp_avg_sq: var.zeros_like()?.detach(),
                },
            );
        }
        Ok(Self {
            params,
            groups,
            state,
        })
    }
    pub fn params(&self) -> &NamedParams {
        &self.params
    }
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }
    /// Change rates without replacing moments or per-parameter step counters.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err(TrainError::InvalidState("invalid learning rate".into()));
        }
        for group in &mut self.groups {
            group.lr = learning_rate;
        }
        Ok(())
    }
    pub fn state(&self, name: &str) -> Option<&ParamState> {
        self.state.get(name)
    }
    fn kind() -> &'static str {
        if DECOUPLED { "adamw" } else { "adam" }
    }

    /// Missing gradients skip decay, moment updates and the per-parameter counter.
    pub fn step(&mut self, gradients: &GradStore) -> Result<()> {
        let mut pending = Vec::new();
        for group in &self.groups {
            for name in &group.names {
                let var = self.params.get(name).expect("validated group");
                let Some(gradient) = gradients.get(var) else {
                    continue;
                };
                validate_like(name, gradient, var)?;
                let old = &self.state[name];
                let step = old
                    .step
                    .checked_add(1)
                    .ok_or_else(|| TrainError::InvalidState(format!("step overflow for {name}")))?;
                let param = var.detach();
                let gradient = if !DECOUPLED && group.weight_decay != 0.0 {
                    gradient
                        .detach()
                        .add(&param.affine(group.weight_decay, 0.0)?)?
                } else {
                    gradient.detach()
                };
                let exp_avg = old
                    .exp_avg
                    .affine(group.beta1, 0.0)?
                    .add(&gradient.affine(1.0 - group.beta1, 0.0)?)?
                    .detach();
                let exp_avg_sq = old
                    .exp_avg_sq
                    .affine(group.beta2, 0.0)?
                    .add(&gradient.sqr()?.affine(1.0 - group.beta2, 0.0)?)?
                    .detach();
                let correction1 = 1.0 - group.beta1.powf(step as f64);
                let correction2 = 1.0 - group.beta2.powf(step as f64);
                let denominator = exp_avg_sq
                    .affine(1.0 / correction2, 0.0)?
                    .sqrt()?
                    .affine(1.0, group.eps)?;
                let update = exp_avg
                    .div(&denominator)?
                    .affine(group.lr / correction1, 0.0)?;
                let param = if DECOUPLED {
                    param.affine(1.0 - group.lr * group.weight_decay, 0.0)?
                } else {
                    param
                };
                pending.push((
                    name.clone(),
                    param.sub(&update)?.detach(),
                    ParamState {
                        step,
                        exp_avg,
                        exp_avg_sq,
                    },
                ));
            }
        }
        self.params.device().synchronize()?;
        for (name, value, state) in pending {
            self.params
                .get(&name)
                .expect("validated parameter")
                .set(&value)?;
            self.state.insert(name, state);
        }
        Ok(())
    }

    /// Save into a checkpoint staging directory. The caller owns checkpoint publication.
    pub fn save(&self, directory: impl AsRef<Path>) -> Result<()> {
        let directory = directory.as_ref();
        let metadata = Metadata {
            format_version: 1,
            kind: Self::kind().into(),
            dtype: "f32".into(),
            groups: self.groups.clone(),
            parameters: self
                .params
                .iter()
                .map(|(name, var)| ParameterMetadata {
                    name: name.into(),
                    shape: var.dims().to_vec(),
                    step: self.state[name].step,
                })
                .collect(),
        };
        let json = serde_json::to_vec_pretty(&metadata)?;
        let mut tensors = HashMap::new();
        for (name, state) in &self.state {
            tensors.insert(format!("{name}.exp_avg"), state.exp_avg.clone());
            tensors.insert(format!("{name}.exp_avg_sq"), state.exp_avg_sq.clone());
        }
        fs::create_dir_all(directory)?;
        candle_core::safetensors::save(&tensors, directory.join("optimizer.safetensors"))?;
        fs::write(directory.join("optimizer-state.json"), json)?;
        Ok(())
    }

    /// Stage and validate all metadata and tensors before changing live state.
    pub fn load(&mut self, directory: impl AsRef<Path>) -> Result<()> {
        let directory = directory.as_ref();
        let metadata: Metadata =
            serde_json::from_slice(&fs::read(directory.join("optimizer-state.json"))?)?;
        if metadata.format_version != 1 || metadata.kind != Self::kind() || metadata.dtype != "f32"
        {
            return Err(TrainError::InvalidState(
                "unsupported optimizer format, kind or dtype".into(),
            ));
        }
        validate_groups(&self.params, &metadata.groups)?;
        let expected: Vec<_> = self.params.iter().map(|(n, v)| (n, v.dims())).collect();
        if metadata.parameters.len() != expected.len()
            || metadata
                .parameters
                .iter()
                .zip(&expected)
                .any(|(m, (name, shape))| m.name != *name || m.shape != *shape)
        {
            return Err(TrainError::InvalidState(
                "optimizer parameter names, order or shapes differ".into(),
            ));
        }
        let mut tensors = candle_core::safetensors::load(
            directory.join("optimizer.safetensors"),
            self.params.device(),
        )?;
        if tensors.len() != 2 * expected.len() {
            return Err(TrainError::InvalidState(
                "optimizer moment keys differ".into(),
            ));
        }
        let mut state = BTreeMap::new();
        for parameter in metadata.parameters {
            let name = parameter.name;
            let var = self.params.get(&name).expect("validated metadata");
            let mut moment = |suffix: &str| -> Result<Tensor> {
                let key = format!("{name}.{suffix}");
                let tensor = tensors
                    .remove(&key)
                    .ok_or_else(|| TrainError::InvalidState(format!("missing moment {key}")))?;
                validate_like(&key, &tensor, var)?;
                Ok(tensor.detach())
            };
            let exp_avg = moment("exp_avg")?;
            let exp_avg_sq = moment("exp_avg_sq")?;
            state.insert(
                name,
                ParamState {
                    step: parameter.step,
                    exp_avg,
                    exp_avg_sq,
                },
            );
        }
        self.params.device().synchronize()?;
        self.restore_snapshot(&OptimizerSnapshot {
            kind: metadata.kind,
            state,
            groups: metadata.groups,
        })
    }
}
