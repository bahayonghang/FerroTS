//! iTransformer (MIT), thuml/iTransformer at c2426e68ca13f74aaec08045c5c724d8ad328124.
//! Copyright (c) 2022 THUML @ Tsinghua University. Conv1d weights use the equivalent Linear shape.
use crate::{
    Mode, RngStream, dropout,
    model::{ForwardContext, ModelInput},
    train_ops,
};
use candle_core::{DType, Device, Result, Tensor, Var};
use ferrots_core::{
    config::{Activation, ModelSpec, WindowSpec},
    reference_rng::ReferenceRng,
};
use std::collections::BTreeMap;

pub const UPSTREAM_SHA: &str = "c2426e68ca13f74aaec08045c5c724d8ad328124";
pub const MAPPING_VERSION: u32 = 1;
pub const DROPOUT_SITE_VERSION: u32 = 1;

pub struct ITransformer {
    pub spec: ModelSpec,
    pub window: WindowSpec,
    pub channels: usize,
    pub time_channels: usize,
    seed: u64,
    weights: BTreeMap<String, Tensor>,
    parameters: Vec<(String, Var)>,
    frozen: bool,
    initialization_rng: ReferenceRng,
}

fn op(result: train_ops::Result<Tensor>) -> Result<Tensor> {
    result.map_err(candle_core::Error::wrap)
}
fn record(trace: &mut Option<&mut BTreeMap<String, Tensor>>, name: &str, value: &Tensor) {
    if let Some(trace) = trace {
        trace.insert(name.into(), value.clone());
    }
}
impl ITransformer {
    pub fn new(
        spec: &ModelSpec,
        window: &WindowSpec,
        channels: usize,
        time_channels: usize,
        seed: u64,
        device: &Device,
    ) -> Result<Self> {
        let ModelSpec::Itransformer {
            d_model,
            n_heads,
            d_ff,
            e_layers,
            dropout,
            ..
        } = *spec
        else {
            candle_core::bail!("expected iTransformer configuration");
        };
        if channels == 0
            || window.lookback == 0
            || window.horizon == 0
            || d_model == 0
            || n_heads == 0
            || !d_model.is_multiple_of(n_heads)
            || d_ff == 0
            || e_layers == 0
            || !dropout.is_finite()
            || !(0.0..1.0).contains(&dropout)
            || !matches!(time_channels, 0 | 4 | 5)
        {
            candle_core::bail!("invalid iTransformer dimensions or dropout");
        }
        let mut result = Self {
            spec: spec.clone(),
            window: window.clone(),
            channels,
            time_channels,
            seed,
            weights: BTreeMap::new(),
            parameters: vec![],
            frozen: false,
            initialization_rng: ReferenceRng::new(seed),
        };
        result.add_linear(
            "enc_embedding.value_embedding",
            window.lookback,
            d_model,
            device,
        )?;
        for layer in 0..e_layers {
            let prefix = format!("encoder.attn_layers.{layer}");
            for projection in ["query", "key", "value", "out"] {
                result.add_linear(
                    &format!("{prefix}.attention.{projection}_projection"),
                    d_model,
                    d_model,
                    device,
                )?;
            }
            result.add_linear(&format!("{prefix}.conv1"), d_model, d_ff, device)?;
            result.add_linear(&format!("{prefix}.conv2"), d_ff, d_model, device)?;
            for norm in ["norm1", "norm2"] {
                result.add_norm(&format!("{prefix}.{norm}"), d_model, device)?;
            }
        }
        result.add_norm("encoder.norm", d_model, device)?;
        result.add_linear("projector", d_model, window.horizon, device)?;
        result.parameters.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(result)
    }
    fn add_parameter(&mut self, name: String, variable: Var) {
        self.weights
            .insert(name.clone(), variable.as_tensor().clone());
        self.parameters.push((name, variable));
    }
    fn add_linear(
        &mut self,
        prefix: &str,
        input: usize,
        output: usize,
        device: &Device,
    ) -> Result<()> {
        let count = input
            .checked_mul(output)
            .ok_or_else(|| candle_core::Error::Msg("linear dimension overflow".into()))?;
        let bound = (1.0 / (input as f64).sqrt()) as f32;
        let weight: Vec<_> = (0..count)
            .map(|_| self.initialization_rng.uniform(-bound, bound))
            .collect();
        let bias: Vec<_> = (0..output)
            .map(|_| self.initialization_rng.uniform(-bound, bound))
            .collect();
        self.add_parameter(
            format!("{prefix}.weight"),
            Var::from_vec(weight, (output, input), device)?,
        );
        self.add_parameter(
            format!("{prefix}.bias"),
            Var::from_vec(bias, output, device)?,
        );
        Ok(())
    }
    fn add_norm(&mut self, prefix: &str, width: usize, device: &Device) -> Result<()> {
        self.add_parameter(
            format!("{prefix}.weight"),
            Var::ones(width, DType::F32, device)?,
        );
        self.add_parameter(
            format!("{prefix}.bias"),
            Var::zeros(width, DType::F32, device)?,
        );
        Ok(())
    }
    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        op(train_ops::linear(
            x,
            &self.weights[&format!("{prefix}.weight")],
            &self.weights[&format!("{prefix}.bias")],
        ))
    }
    fn norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        op(train_ops::layer_norm(
            x,
            &self.weights[&format!("{prefix}.weight")],
            &self.weights[&format!("{prefix}.bias")],
            1e-5,
        ))
    }
    fn drop(
        &self,
        x: &Tensor,
        probability: f64,
        context: ForwardContext,
        site: u64,
    ) -> Result<Tensor> {
        op(dropout(
            x,
            probability,
            context.mode,
            RngStream {
                seed: self.seed,
                run_step: context.forward_id,
                layer_id: site,
            },
        ))
    }
    pub fn named_parameters(&self) -> Vec<(String, Var)> {
        self.parameters.clone()
    }
    pub fn weights(&self) -> &BTreeMap<String, Tensor> {
        &self.weights
    }
    /// Export independent tensors with the official pointwise Conv1d shapes.
    pub fn official_weights(&self) -> Result<BTreeMap<String, Tensor>> {
        self.weights
            .iter()
            .map(|(name, tensor)| {
                let tensor = if name.ends_with(".conv1.weight") || name.ends_with(".conv2.weight") {
                    tensor.unsqueeze(2)?
                } else {
                    tensor.clone()
                };
                Ok((name.clone(), tensor.detach().copy()?))
            })
            .collect()
    }
    /// Validate the complete official mapping before assigning any parameter.
    pub fn load_official_weights(&self, weights: &BTreeMap<String, Tensor>) -> Result<()> {
        let mapped = weights
            .iter()
            .map(|(name, tensor)| {
                let tensor = if name.ends_with(".conv1.weight") || name.ends_with(".conv2.weight") {
                    if tensor.rank() != 3 || tensor.dim(2)? != 1 {
                        candle_core::bail!("invalid official Conv1d shape: {name}");
                    }
                    tensor.squeeze(2)?
                } else {
                    tensor.clone()
                };
                Ok((name.clone(), tensor))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        self.load_weights(&mapped)
    }
    pub fn sampling_rng(&self) -> &ReferenceRng {
        &self.initialization_rng
    }
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }
    pub fn load_weights(&self, weights: &BTreeMap<String, Tensor>) -> Result<()> {
        if self.frozen || weights.len() != self.weights.len() {
            candle_core::bail!("invalid iTransformer weight set or frozen model");
        }
        for (name, expected) in &self.weights {
            let value = weights
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing parameter {name}")))?;
            if value.shape() != expected.shape()
                || value.dtype() != DType::F32
                || !value.device().same_device(expected.device())
                || !value
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|v| v.is_finite())
            {
                candle_core::bail!("parameter schema or value mismatch: {name}");
            }
        }
        for (name, variable) in &self.parameters {
            variable.set(&weights[name])?;
        }
        Ok(())
    }
    pub fn freeze_for_eval(&self) -> Result<Self> {
        Ok(Self {
            spec: self.spec.clone(),
            window: self.window.clone(),
            channels: self.channels,
            time_channels: self.time_channels,
            seed: self.seed,
            weights: self
                .weights
                .iter()
                .map(|(n, w)| Ok((n.clone(), w.detach().copy()?)))
                .collect::<Result<_>>()?,
            parameters: vec![],
            frozen: true,
            initialization_rng: self.initialization_rng.clone(),
        })
    }
    pub fn forward(&self, input: &ModelInput, context: ForwardContext) -> Result<Tensor> {
        self.forward_impl(input, context, None)
    }
    pub fn forward_with_trace(
        &self,
        input: &ModelInput,
        context: ForwardContext,
    ) -> Result<(Tensor, BTreeMap<String, Tensor>)> {
        let mut trace = BTreeMap::new();
        let output = self.forward_impl(input, context, Some(&mut trace))?;
        Ok((output, trace))
    }
    fn forward_impl(
        &self,
        input: &ModelInput,
        context: ForwardContext,
        mut trace: Option<&mut BTreeMap<String, Tensor>>,
    ) -> Result<Tensor> {
        if self.frozen && context.mode != Mode::Eval {
            candle_core::bail!("frozen model requires evaluation mode");
        }
        let ModelSpec::Itransformer {
            d_model,
            n_heads,
            e_layers,
            dropout,
            activation,
            use_norm,
            ..
        } = self.spec
        else {
            unreachable!()
        };
        let history = if self.frozen {
            input.history.detach()
        } else {
            input.history.clone()
        };
        let (batch, length, channels) = history.dims3()?;
        if batch == 0
            || length != self.window.lookback
            || channels != self.channels
            || history.dtype() != DType::F32
        {
            candle_core::bail!("iTransformer history shape/dtype mismatch");
        }
        let (normalized, statistics) = if use_norm {
            let mean = history.mean_keepdim(1)?.detach();
            let centered = history.broadcast_sub(&mean)?;
            let std = op(train_ops::pop_var(&centered, 1, true))?
                .affine(1.0, 1e-5)?
                .sqrt()?;
            (centered.broadcast_div(&std)?, Some((mean, std)))
        } else {
            (history, None)
        };
        let mut tokens = normalized.transpose(1, 2)?;
        match (&input.history_time, self.time_channels) {
            (None, 0) => {}
            (Some(times), width) if width > 0 => {
                if times.dims() != [batch, length, width] || times.dtype() != DType::F32 {
                    candle_core::bail!("iTransformer time feature binding mismatch");
                }
                let times = if self.frozen {
                    times.detach()
                } else {
                    times.clone()
                };
                tokens = Tensor::cat(&[tokens, times.transpose(1, 2)?], 1)?;
            }
            _ => candle_core::bail!("iTransformer time feature binding mismatch"),
        }
        let count = channels + self.time_channels;
        let embedded = self.linear(&tokens, "enc_embedding.value_embedding")?;
        record(&mut trace, "enc_embedding.value_embedding.0", &embedded);
        let mut x = self.drop(&embedded, dropout, context, 0)?;
        record(&mut trace, "enc_embedding.dropout.0", &x);
        record(&mut trace, "enc_embedding.0", &x);
        let head_width = d_model / n_heads;
        for layer in 0..e_layers {
            let prefix = format!("encoder.attn_layers.{layer}");
            let mut projected = vec![];
            for name in ["query", "key", "value"] {
                let name = format!("{prefix}.attention.{name}_projection");
                let tensor = self.linear(&x, &name)?;
                record(&mut trace, &format!("{name}.0"), &tensor);
                projected.push(
                    tensor
                        .reshape((batch, count, n_heads, head_width))?
                        .transpose(1, 2)?
                        .contiguous()?,
                );
            }
            let scores = projected[0]
                .matmul(&projected[1].transpose(2, 3)?.contiguous()?)?
                .affine(1.0 / (head_width as f64).sqrt(), 0.0)?;
            let site = 1 + 4 * layer as u64;
            let probabilities =
                self.drop(&op(train_ops::softmax(&scores, 3))?, dropout, context, site)?;
            record(
                &mut trace,
                &format!("{prefix}.attention.inner_attention.dropout.0"),
                &probabilities,
            );
            let attended = probabilities
                .matmul(&projected[2])?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, count, d_model))?;
            let attention =
                self.linear(&attended, &format!("{prefix}.attention.out_projection"))?;
            record(
                &mut trace,
                &format!("{prefix}.attention.out_projection.0"),
                &attention,
            );
            record(&mut trace, &format!("{prefix}.attention.0"), &attention);
            let residual = self.drop(&attention, dropout, context, site + 1)?;
            record(&mut trace, &format!("{prefix}.dropout.0"), &residual);
            x = self.norm(&x.add(&residual)?, &format!("{prefix}.norm1"))?;
            record(&mut trace, &format!("{prefix}.norm1.0"), &x);
            let ff = self
                .linear(&x, &format!("{prefix}.conv1"))?
                .transpose(1, 2)?;
            record(&mut trace, &format!("{prefix}.conv1.0"), &ff);
            let activated = match activation {
                Activation::Gelu => op(train_ops::gelu(&ff))?,
                Activation::Relu => ff.relu()?,
            };
            let ff = self.drop(&activated, dropout, context, site + 2)?;
            record(&mut trace, &format!("{prefix}.dropout.1"), &ff);
            let ff = self.linear(&ff.transpose(1, 2)?, &format!("{prefix}.conv2"))?;
            record(
                &mut trace,
                &format!("{prefix}.conv2.0"),
                &ff.transpose(1, 2)?,
            );
            let ff = self.drop(&ff, dropout, context, site + 3)?;
            record(&mut trace, &format!("{prefix}.dropout.2"), &ff);
            x = self.norm(&x.add(&ff)?, &format!("{prefix}.norm2"))?;
            record(&mut trace, &format!("{prefix}.norm2.0"), &x);
            record(&mut trace, &format!("{prefix}.0"), &x);
        }
        x = self.norm(&x, "encoder.norm")?;
        record(&mut trace, "encoder.norm.0", &x);
        let projected = self.linear(&x, "projector")?;
        record(&mut trace, "projector.0", &projected);
        let prediction = projected.transpose(1, 2)?.narrow(2, 0, channels)?;
        match statistics {
            Some((mean, std)) => prediction.broadcast_mul(&std)?.broadcast_add(&mean),
            None => Ok(prediction),
        }
    }
}
