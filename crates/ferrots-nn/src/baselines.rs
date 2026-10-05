//! Linear/DLinear semantics ported from Time-Series-Library (MIT).
//! Source: models/DLinear.py at 4e938a1767106324dd753b2a44832bf870a0252e.
//! Copyright (c) 2021 THUML @ Tsinghua University. See docs/models/dlinear.md.
use candle_core::{DType, Device, Result, Tensor, Var};
use ferrots_core::config::{ModelSpec, WindowSpec};
use ferrots_core::reference_rng::ReferenceRng;
use std::collections::BTreeMap;

pub const UPSTREAM_SHA: &str = "4e938a1767106324dd753b2a44832bf870a0252e";
pub use crate::model::ModelInput;

/// Frozen models own detached copies and have no trainable variables.
pub struct Baseline {
    pub spec: ModelSpec,
    pub window: WindowSpec,
    pub channels: usize,
    weights: BTreeMap<String, Tensor>,
    parameters: Vec<(String, Var)>,
    frozen: bool,
    initialization_rng: ReferenceRng,
}
// Preserve upstream FP32 accumulation order. A dense averaging matrix changes
// tiny seasonal residuals, which Adam can amplify through its epsilon term.
fn smooth(history: &Tensor, kernel: usize) -> Result<Tensor> {
    let (batch, length, channels) = history.dims3()?;
    if length == 0 || kernel == 0 || kernel.is_multiple_of(2) {
        candle_core::bail!("invalid moving average dimensions");
    }
    let padding = kernel / 2;
    let padded = if padding == 0 {
        history.clone()
    } else {
        let first = history
            .narrow(1, 0, 1)?
            .broadcast_as((batch, padding, channels))?;
        let last = history
            .narrow(1, length - 1, 1)?
            .broadcast_as((batch, padding, channels))?;
        Tensor::cat(&[&first, history, &last], 1)?
    };
    let mut sum = padded.narrow(1, 0, length)?;
    for offset in 1..kernel {
        sum = sum.add(&padded.narrow(1, offset, length)?)?;
    }
    sum.broadcast_div(&Tensor::new(kernel as f32, history.device())?)
}
pub fn decompose(history: &Tensor, kernel: usize) -> Result<(Tensor, Tensor)> {
    let trend = smooth(history, kernel)?;
    Ok((history.sub(&trend)?, trend))
}
impl Baseline {
    pub fn new(
        spec: &ModelSpec,
        window: &WindowSpec,
        channels: usize,
        seed: u64,
        device: &Device,
    ) -> Result<Self> {
        if channels == 0 || window.lookback == 0 || window.horizon == 0 {
            candle_core::bail!("empty model dimensions");
        }
        if matches!(spec, ModelSpec::Itransformer { .. }) {
            candle_core::bail!("iTransformer requires the common model constructor");
        }
        if let ModelSpec::SeasonalNaive { period } = spec
            && (*period == 0 || *period > window.lookback)
        {
            candle_core::bail!("invalid seasonal period");
        }
        let mut model = Self {
            spec: spec.clone(),
            window: window.clone(),
            channels,
            weights: BTreeMap::new(),
            parameters: vec![],
            frozen: false,
            initialization_rng: ReferenceRng::new(seed),
        };
        let (prefixes, individual, constant) = match spec {
            ModelSpec::Linear { individual } => (vec!["Linear"], *individual, false),
            ModelSpec::Dlinear {
                individual,
                moving_avg,
            } => {
                if *moving_avg == 0 || moving_avg % 2 == 0 {
                    candle_core::bail!("invalid moving average dimensions");
                }
                (vec!["Linear_Seasonal", "Linear_Trend"], *individual, true)
            }
            _ => return Ok(model),
        };
        let rng = &mut model.initialization_rng;
        let bound = (1.0 / (window.lookback as f64).sqrt()) as f32;
        for j in 0..if individual { channels } else { 1 } {
            for prefix in &prefixes {
                let name = if individual {
                    format!("{prefix}.{j}")
                } else {
                    (*prefix).into()
                };
                let weight: Vec<f32> = (0..window.horizon * window.lookback)
                    .map(|_| {
                        let value = rng.uniform(-bound, bound);
                        if constant {
                            1.0 / window.lookback as f32
                        } else {
                            value
                        }
                    })
                    .collect();
                let bias: Vec<f32> = (0..window.horizon)
                    .map(|_| rng.uniform(-bound, bound))
                    .collect();
                for (name, var) in [
                    (
                        format!("{name}.weight"),
                        Var::from_vec(weight, (window.horizon, window.lookback), device)?,
                    ),
                    (
                        format!("{name}.bias"),
                        Var::from_vec(bias, window.horizon, device)?,
                    ),
                ] {
                    model.weights.insert(name.clone(), var.as_tensor().clone());
                    model.parameters.push((name, var));
                }
            }
        }
        model.parameters.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(model)
    }
    pub fn named_parameters(&self) -> Vec<(String, Var)> {
        self.parameters.clone()
    }
    pub fn sampling_rng(&self) -> &ReferenceRng {
        &self.initialization_rng
    }
    pub fn weights(&self) -> &BTreeMap<String, Tensor> {
        &self.weights
    }
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }
    pub fn load_weights(&self, weights: &BTreeMap<String, Tensor>) -> Result<()> {
        if self.frozen {
            candle_core::bail!("cannot change a frozen model");
        }
        if weights.len() != self.weights.len() {
            candle_core::bail!("model parameter names differ");
        }
        for (name, expected) in &self.weights {
            let Some(value) = weights.get(name) else {
                candle_core::bail!("missing parameter {name}");
            };
            if value.shape() != expected.shape()
                || value.dtype() != DType::F32
                || !value.device().same_device(expected.device())
            {
                candle_core::bail!("parameter schema mismatch: {name}");
            }
            if !value
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite())
            {
                candle_core::bail!("non-finite parameter: {name}");
            }
        }
        for (name, var) in &self.parameters {
            var.set(&weights[name])?;
        }
        Ok(())
    }
    pub fn freeze_for_eval(&self) -> Result<Self> {
        let weights = self
            .weights
            .iter()
            .map(|(n, w)| Ok((n.clone(), w.detach().copy()?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            spec: self.spec.clone(),
            window: self.window.clone(),
            channels: self.channels,
            weights,
            parameters: vec![],
            frozen: true,
            initialization_rng: self.initialization_rng.clone(),
        })
    }
    fn project(&self, x: &Tensor, prefix: &str, individual: bool) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let h = self.window.horizon;
        let x = x.transpose(1, 2)?.contiguous()?;
        if individual {
            let mut ys = Vec::with_capacity(c);
            for j in 0..c {
                let row = x.narrow(1, j, 1)?.reshape((b, l))?;
                let y = row
                    .matmul(&self.weights[&format!("{prefix}.{j}.weight")].t()?)?
                    .broadcast_add(&self.weights[&format!("{prefix}.{j}.bias")])?;
                ys.push(y.unsqueeze(1)?);
            }
            Tensor::cat(&ys, 1)?.transpose(1, 2)
        } else {
            x.reshape((b * c, l))?
                .matmul(&self.weights[&format!("{prefix}.weight")].t()?)?
                .broadcast_add(&self.weights[&format!("{prefix}.bias")])?
                .reshape((b, c, h))?
                .transpose(1, 2)
        }
    }
    pub fn forward(&self, input: &ModelInput) -> Result<Tensor> {
        let x = if self.frozen {
            input.history.detach()
        } else {
            input.history.clone()
        };
        let (b, l, c) = x.dims3()?;
        if b == 0 || l != self.window.lookback || c != self.channels || x.dtype() != DType::F32 {
            candle_core::bail!("model input shape/dtype mismatch");
        }
        match self.spec {
            ModelSpec::Itransformer { .. } => candle_core::bail!("iTransformer is not a baseline"),
            ModelSpec::LastValue => x
                .narrow(1, l - 1, 1)?
                .broadcast_as((b, self.window.horizon, c))?
                .contiguous(),
            ModelSpec::SeasonalNaive { period } => {
                let indices: Vec<u32> = (0..self.window.horizon)
                    .map(|k| (l - period + (self.window.gap + k) % period) as u32)
                    .collect();
                x.index_select(
                    &Tensor::from_vec(indices, self.window.horizon, x.device())?,
                    1,
                )
            }
            ModelSpec::Linear { individual } => self.project(&x, "Linear", individual),
            ModelSpec::Dlinear {
                individual,
                moving_avg,
            } => {
                let trend = smooth(&x, moving_avg)?;
                self.project(&x.sub(&trend)?, "Linear_Seasonal", individual)?
                    .add(&self.project(&trend, "Linear_Trend", individual)?)
            }
        }
    }
}
