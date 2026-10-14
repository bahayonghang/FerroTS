//! Trainable Informer, TSLib 4e938a1767106324dd753b2a44832bf870a0252e (MIT).
//! Includes the upstream ProbAttention output layout and unapplied attention dropout.
use crate::{
    Mode, RngStream, dropout, informer_ops,
    model::{ForwardContext, ModelInput},
    train_ops,
};
use candle_core::{DType, Device, Result, Tensor, Var};
use ferrots_core::{
    config::{Activation, ModelSpec, WindowSpec},
    reference_rng::ReferenceRng,
};
use std::{collections::BTreeMap, sync::Mutex};

pub const UPSTREAM_SHA: &str = "4e938a1767106324dd753b2a44832bf870a0252e";
pub const MAPPING_VERSION: u32 = 1;
pub const SAMPLING_SITE_VERSION: u32 = 1;

/// Geometry of the CPU `torch.randint` calls in one official Informer forward.
/// Index values stay on the isolated ProbSparse stream. Only their word count
/// is applied to the loader RNG, because that is what moves the next epoch's
/// train permutation. Validation and test counts follow TSLib: both loaders are
/// iterated after every train epoch, and neither drops the tail batch.
#[derive(Clone, Copy)]
pub struct InformerSamplingSchedule {
    pub lookback: usize,
    pub label_len: usize,
    pub horizon: usize,
    pub factor: usize,
    pub e_layers: usize,
    pub d_layers: usize,
    pub distil: bool,
    pub batch_size: usize,
    pub train_windows: usize,
    pub val_windows: usize,
    pub test_windows: usize,
    pub train_drop_last: bool,
}

fn prob_sample_count(factor: usize, length: usize) -> usize {
    (factor * (length as f64).ln().ceil() as usize)
        .max(1)
        .min(length)
}

/// Conv1d kernel 3 with circular padding 2, then MaxPool1d kernel 3, stride 2, padding 1.
fn distilled_encoder_length(length: usize) -> usize {
    (length + 2).div_ceil(2)
}

fn loader_batches(windows: usize, batch_size: usize, drop_last: bool) -> usize {
    if batch_size == 0 {
        return 0;
    }
    if drop_last {
        windows / batch_size
    } else {
        windows.div_ceil(batch_size)
    }
}

impl InformerSamplingSchedule {
    pub fn randint_sites(&self) -> Vec<(u64, usize)> {
        let mut sites = Vec::new();
        let mut length = self.lookback;
        for layer in 0..self.e_layers {
            push_site(&mut sites, length, length, self.factor);
            if self.distil && layer + 1 < self.e_layers {
                length = distilled_encoder_length(length);
            }
        }
        let decoder = self.label_len.saturating_add(self.horizon);
        for _ in 0..self.d_layers {
            push_site(&mut sites, decoder, decoder, self.factor);
            push_site(&mut sites, decoder, length, self.factor);
        }
        sites
    }

    /// Positions into the training-window list. Epoch is zero-based.
    pub fn permutation(&self, after_init: &ReferenceRng, epoch: u64) -> Vec<usize> {
        let sites = self.randint_sites();
        after_init.training_permutation_after_randint(
            self.train_windows,
            epoch,
            loader_batches(self.train_windows, self.batch_size, self.train_drop_last),
            loader_batches(self.val_windows, self.batch_size, false),
            loader_batches(self.test_windows, self.batch_size, false),
            &sites,
        )
    }
}

/// PyTorch 2.14 `NormalFill16` for float: eight `(u1, u2)` pairs from one block.
fn box_muller16(data: &mut [f32], std: f32) {
    let mut out = [0f32; 16];
    let two_pi = 2.0 * std::f32::consts::PI;
    for j in 0..8 {
        let u1 = 1.0 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = two_pi * u2;
        out[j] = radius * theta.cos() * std;
        out[j + 8] = radius * theta.sin() * std;
    }
    data[..16].copy_from_slice(&out);
}

fn push_site(sites: &mut Vec<(u64, usize)>, queries: usize, keys: usize, factor: usize) {
    let sample_k = prob_sample_count(factor, keys);
    sites.push((keys as u64, queries.saturating_mul(sample_k)));
}

pub struct Informer {
    pub spec: ModelSpec,
    pub window: WindowSpec,
    pub channels: usize,
    pub time_channels: usize,
    seed: u64,
    weights: BTreeMap<String, Tensor>,
    parameters: Vec<(String, Var)>,
    buffers: Mutex<BTreeMap<String, Tensor>>,
    frozen: bool,
    initialization_rng: ReferenceRng,
}
fn op(x: train_ops::Result<Tensor>) -> Result<Tensor> {
    x.map_err(candle_core::Error::wrap)
}
fn record(trace: &mut Option<&mut BTreeMap<String, Tensor>>, key: &str, x: &Tensor) {
    if let Some(trace) = trace {
        trace.insert(key.to_owned(), x.clone());
    }
}

impl Informer {
    pub fn new(
        spec: &ModelSpec,
        window: &WindowSpec,
        channels: usize,
        time_channels: usize,
        seed: u64,
        device: &Device,
    ) -> Result<Self> {
        let ModelSpec::Informer {
            d_model,
            n_heads,
            d_ff,
            e_layers,
            d_layers,
            factor,
            distil,
            label_len,
            dropout,
            ..
        } = *spec
        else {
            candle_core::bail!("expected Informer configuration")
        };
        if channels == 0
            || window.lookback == 0
            || window.horizon == 0
            || window.gap != 0
            || d_model == 0
            || !d_model.is_multiple_of(2)
            || n_heads == 0
            || !d_model.is_multiple_of(n_heads)
            || d_ff == 0
            || e_layers == 0
            || d_layers == 0
            || factor == 0
            || label_len == 0
            || label_len > window.lookback
            || !matches!(time_channels, 4 | 5)
            || !dropout.is_finite()
            || !(0.0..1.0).contains(&dropout)
        {
            candle_core::bail!("invalid Informer dimensions, time features or dropout");
        }
        let mut m = Self {
            spec: spec.clone(),
            window: window.clone(),
            channels,
            time_channels,
            seed,
            weights: BTreeMap::new(),
            parameters: vec![],
            buffers: Mutex::new(BTreeMap::new()),
            frozen: false,
            initialization_rng: ReferenceRng::new(seed),
        };
        for name in ["enc_embedding", "dec_embedding"] {
            m.add_affine(
                &format!("{name}.value_embedding.tokenConv"),
                channels,
                d_model,
                Some(3),
                false,
                true,
                device,
            )?;
            m.add_affine(
                &format!("{name}.temporal_embedding.embed"),
                time_channels,
                d_model,
                None,
                false,
                false,
                device,
            )?;
            let mut values = Vec::with_capacity(5000 * d_model);
            for p in 0..5000 {
                for j in 0..d_model {
                    let rate =
                        ((j / 2 * 2) as f32 * (-(10000_f64.ln()) / d_model as f64) as f32).exp();
                    let angle = p as f32 * rate;
                    values.push(if j % 2 == 0 { angle.sin() } else { angle.cos() });
                }
            }
            m.buffers
                .get_mut()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                .insert(
                    format!("{name}.position_embedding.pe"),
                    Tensor::from_vec(values, (1, 5000, d_model), device)?,
                );
        }
        for layer in 0..e_layers {
            let prefix = format!("encoder.attn_layers.{layer}");
            m.add_attention(&format!("{prefix}.attention"), d_model, device)?;
            m.add_ff(&prefix, d_model, d_ff, 2, device)?;
        }
        if distil {
            for layer in 0..e_layers - 1 {
                let prefix = format!("encoder.conv_layers.{layer}");
                m.add_affine(
                    &format!("{prefix}.downConv"),
                    d_model,
                    d_model,
                    Some(3),
                    true,
                    false,
                    device,
                )?;
                m.add_norm(&format!("{prefix}.norm"), d_model, device)?;
                let buffers = m
                    .buffers
                    .get_mut()
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                buffers.insert(
                    format!("{prefix}.norm.running_mean"),
                    Tensor::zeros(d_model, DType::F32, device)?,
                );
                buffers.insert(
                    format!("{prefix}.norm.running_var"),
                    Tensor::ones(d_model, DType::F32, device)?,
                );
                buffers.insert(
                    format!("{prefix}.norm.num_batches_tracked"),
                    Tensor::new(0_u32, device)?,
                );
            }
        }
        m.add_norm("encoder.norm", d_model, device)?;
        for layer in 0..d_layers {
            let prefix = format!("decoder.layers.{layer}");
            m.add_attention(&format!("{prefix}.self_attention"), d_model, device)?;
            m.add_attention(&format!("{prefix}.cross_attention"), d_model, device)?;
            m.add_ff(&prefix, d_model, d_ff, 3, device)?;
        }
        m.add_norm("decoder.norm", d_model, device)?;
        m.add_affine(
            "decoder.projection",
            d_model,
            channels,
            None,
            true,
            false,
            device,
        )?;
        m.parameters.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(m)
    }
    fn add_parameter(&mut self, name: String, value: Tensor) -> Result<()> {
        let var = Var::from_tensor(&value)?;
        self.weights.insert(name.clone(), var.as_tensor().clone());
        self.parameters.push((name, var));
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn add_affine(
        &mut self,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: Option<usize>,
        bias: bool,
        normal: bool,
        device: &Device,
    ) -> Result<()> {
        let fan = input * kernel.unwrap_or(1);
        let bound = (1.0 / (fan as f64).sqrt()) as f32;
        let mut values: Vec<f32> = (0..fan * output)
            .map(|_| self.initialization_rng.uniform(-bound, bound))
            .collect();
        if normal {
            // The uniform fill above is Conv1d::reset_parameters. TSLib then calls
            // kaiming_normal_(fan_in, leaky_relu, a=0). PyTorch 2.14 overwrites a
            // contiguous float tensor of at least 16 values with uniform(0, 1) and
            // applies Box–Muller in blocks of 16. A non-multiple of 16 redraws the
            // last block. This keeps the same uniform count as that path, so the
            // generator state after tokenConv still matches the official stream.
            let std = (2.0 / fan as f64).sqrt() as f32;
            let n = values.len();
            if n >= 16 {
                for value in &mut values {
                    *value = self.initialization_rng.uniform(0.0, 1.0);
                }
                let mut start = 0;
                while start + 16 <= n {
                    box_muller16(&mut values[start..start + 16], std);
                    start += 16;
                }
                if !n.is_multiple_of(16) {
                    let tail = n - 16;
                    for value in &mut values[tail..] {
                        *value = self.initialization_rng.uniform(0.0, 1.0);
                    }
                    box_muller16(&mut values[tail..], std);
                }
            } else {
                for pair in values.chunks_mut(2) {
                    let u = self.initialization_rng.uniform(f32::MIN_POSITIVE, 1.0) as f64;
                    let theta =
                        self.initialization_rng.uniform(0.0, 1.0) as f64 * std::f64::consts::TAU;
                    let radius = (-2.0 * u.ln()).sqrt() * std as f64;
                    pair[0] = (radius * theta.cos()) as f32;
                    if pair.len() == 2 {
                        pair[1] = (radius * theta.sin()) as f32;
                    }
                }
            }
        }
        let shape = if let Some(k) = kernel {
            vec![output, input, k]
        } else {
            vec![output, input]
        };
        self.add_parameter(
            format!("{prefix}.weight"),
            Tensor::from_vec(values, shape, device)?,
        )?;
        if bias {
            let values: Vec<_> = (0..output)
                .map(|_| self.initialization_rng.uniform(-bound, bound))
                .collect();
            self.add_parameter(
                format!("{prefix}.bias"),
                Tensor::from_vec(values, output, device)?,
            )?;
        }
        Ok(())
    }
    fn add_norm(&mut self, prefix: &str, width: usize, device: &Device) -> Result<()> {
        self.add_parameter(
            format!("{prefix}.weight"),
            Tensor::ones(width, DType::F32, device)?,
        )?;
        self.add_parameter(
            format!("{prefix}.bias"),
            Tensor::zeros(width, DType::F32, device)?,
        )
    }
    fn add_attention(&mut self, prefix: &str, width: usize, device: &Device) -> Result<()> {
        for name in ["query", "key", "value", "out"] {
            self.add_affine(
                &format!("{prefix}.{name}_projection"),
                width,
                width,
                None,
                true,
                false,
                device,
            )?;
        }
        Ok(())
    }
    fn add_ff(
        &mut self,
        prefix: &str,
        width: usize,
        ff: usize,
        norms: usize,
        device: &Device,
    ) -> Result<()> {
        self.add_affine(
            &format!("{prefix}.conv1"),
            width,
            ff,
            Some(1),
            true,
            false,
            device,
        )?;
        self.add_affine(
            &format!("{prefix}.conv2"),
            ff,
            width,
            Some(1),
            true,
            false,
            device,
        )?;
        for n in 1..=norms {
            self.add_norm(&format!("{prefix}.norm{n}"), width, device)?;
        }
        Ok(())
    }
    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w = &self.weights[&format!("{prefix}.weight")];
        let w = if w.rank() == 3 {
            w.squeeze(2)?
        } else {
            w.clone()
        };
        let mut shape = x.dims().to_vec();
        let last = shape.len() - 1;
        let input = shape[last];
        shape[last] = w.dim(0)?;
        let y = x
            .contiguous()?
            .reshape((x.elem_count() / input, input))?
            .matmul(&w.t()?)?;
        let y = match self.weights.get(&format!("{prefix}.bias")) {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        y.reshape(shape)
    }
    fn norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        op(train_ops::layer_norm(
            x,
            &self.weights[&format!("{prefix}.weight")],
            &self.weights[&format!("{prefix}.bias")],
            1e-5,
        ))
    }
    fn drop(&self, x: &Tensor, context: ForwardContext, site: u64) -> Result<Tensor> {
        let ModelSpec::Informer {
            dropout: probability,
            ..
        } = self.spec
        else {
            unreachable!()
        };
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
    // Circular convolution as three differentiable matrix products. No im2col or
    // overlapping-convolution backward dependency; padding=2 increases L by two.
    fn circular_conv(&self, x: &Tensor, prefix: &str, padding: usize) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let w = &self.weights[&format!("{prefix}.weight")];
        let output = w.dim(0)?;
        let out_len = l + 2 * padding - 2;
        let mut result = None;
        for k in 0..3 {
            let ids: Vec<u32> = (0..out_len)
                .map(|i| ((i + k + l - padding % l) % l) as u32)
                .collect();
            let source = x
                .index_select(&Tensor::from_vec(ids, out_len, x.device())?, 1)?
                .reshape((b * out_len, c))?;
            let weight = w.narrow(2, k, 1)?.squeeze(2)?.contiguous()?;
            let y = source.matmul(&weight.t()?)?;
            result = Some(match result {
                None => y,
                Some(sum) => y.add(&sum)?,
            });
        }
        let mut y = result.expect("three kernel taps");
        if let Some(bias) = self.weights.get(&format!("{prefix}.bias")) {
            y = y.broadcast_add(bias)?;
        }
        y.reshape((b, out_len, output))
    }
    fn embedding(
        &self,
        x: &Tensor,
        times: &Tensor,
        prefix: &str,
        context: ForwardContext,
        site: u64,
        trace: &mut Option<&mut BTreeMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let value = self.circular_conv(x, &format!("{prefix}.value_embedding.tokenConv"), 1)?;
        record(
            trace,
            &format!("{prefix}.value_embedding.tokenConv"),
            &value.transpose(1, 2)?,
        );
        record(trace, &format!("{prefix}.value_embedding"), &value);
        let temporal = self.linear(times, &format!("{prefix}.temporal_embedding.embed"))?;
        record(
            trace,
            &format!("{prefix}.temporal_embedding.embed"),
            &temporal,
        );
        record(trace, &format!("{prefix}.temporal_embedding"), &temporal);
        let position = self
            .buffers
            .lock()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            [&format!("{prefix}.position_embedding.pe")]
            .narrow(1, 0, x.dim(1)?)?;
        record(trace, &format!("{prefix}.position_embedding"), &position);
        let y = self.drop(
            &value.add(&temporal)?.broadcast_add(&position)?,
            context,
            site,
        )?;
        record(trace, &format!("{prefix}.dropout"), &y);
        record(trace, prefix, &y);
        Ok(y)
    }
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        prefix: &str,
        causal: bool,
        context: ForwardContext,
        site: u64,
        samples: Option<&BTreeMap<String, Tensor>>,
        trace: &mut Option<&mut BTreeMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let ModelSpec::Informer {
            d_model,
            n_heads,
            factor,
            ..
        } = self.spec
        else {
            unreachable!()
        };
        let (b, lq, _) = queries.dims3()?;
        let lk = keys.dim(1)?;
        let d = d_model / n_heads;
        let mut projected = vec![];
        for (name, x, l) in [
            ("query", queries, lq),
            ("key", keys, lk),
            ("value", keys, lk),
        ] {
            let y = self.linear(x, &format!("{prefix}.{name}_projection"))?;
            record(trace, &format!("{prefix}.{name}_projection"), &y);
            projected.push(
                y.reshape((b, l, n_heads, d))?
                    .transpose(1, 2)?
                    .contiguous()?,
            );
        }
        let (q, k, v) = (&projected[0], &projected[1], &projected[2]);
        let sample_k = prob_sample_count(factor, lk);
        let top = prob_sample_count(factor, lq);
        let indices = if let Some(samples) = samples {
            let ids = samples.get(prefix).ok_or_else(|| {
                candle_core::Error::Msg(format!("missing sample indices: {prefix}"))
            })?;
            if ids.dims() != [lq, sample_k]
                || ids.dtype() != DType::U32
                || !ids.device().same_device(q.device())
            {
                candle_core::bail!("invalid sample indices: {prefix}");
            }
            ids.clone()
        } else {
            let domain = if context.mode == Mode::Train {
                0x747261696e_u64
            } else {
                0x6576616c_u64
            };
            let seed = self.seed
                ^ context.forward_id.wrapping_mul(0x9e3779b97f4a7c15)
                ^ site.wrapping_mul(0xbf58476d1ce4e5b9)
                ^ domain;
            let mut rng = ReferenceRng::new(seed ^ (seed >> 32));
            let ids: Vec<u32> = (0..lq * sample_k)
                .map(|_| (rng.seed_draw() % lk as u64) as u32)
                .collect();
            Tensor::from_vec(ids, (lq, sample_k), q.device())?
        };
        // Selection is discrete. At most one B,H,Lq,D sampled-key slice at a time,
        // with O(Lq log Lk) scalar scores; never expanded B,H,Lq,Lk,D keys.
        let mut sample_scores = vec![];
        for s in 0..sample_k {
            let ids = indices.narrow(1, s, 1)?.contiguous()?.flatten_all()?;
            let selected = k.detach().index_select(&ids, 2)?;
            sample_scores.push(q.detach().mul(&selected)?.sum(3)?.unsqueeze(3)?);
        }
        let sample_scores = Tensor::cat(&sample_scores, 3)?;
        let importance = sample_scores
            .max(3)?
            .sub(&sample_scores.sum(3)?.affine(1.0 / lk as f64, 0.0)?)?;
        let selected = importance
            .contiguous()?
            .arg_sort_last_dim(false)?
            .narrow(2, 0, top)?
            .contiguous()?;
        let gather = selected
            .unsqueeze(3)?
            .broadcast_as((b, n_heads, top, d))?
            .contiguous()?;
        let reduced = q.gather(&gather, 2)?;
        let scores = reduced
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / (d as f64).sqrt(), 0.0)?;
        record(trace, &format!("{prefix}.sample_scores"), &sample_scores);
        record(trace, &format!("{prefix}.selected_queries"), &selected);
        record(trace, &format!("{prefix}.scores"), &scores);
        let probabilities = if causal {
            let positions = Tensor::arange(0_u32, lk as u32, q.device())?.reshape((1, 1, 1, lk))?;
            let valid = positions.broadcast_le(&selected.unsqueeze(3)?)?;
            record(trace, &format!("{prefix}.mask"), &valid);
            let excluded = Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?;
            op(train_ops::softmax(
                &valid.where_cond(&scores, &excluded)?,
                3,
            ))?
        } else {
            op(train_ops::softmax(&scores, 3))?
        };
        let base = if causal {
            informer_ops::prefix_sum(v, 2)?
        } else {
            v.mean_keepdim(2)?
                .broadcast_as((b, n_heads, lq, d))?
                .contiguous()?
        };
        let attended = informer_ops::replace_rows(&base, &probabilities.matmul(v)?, &selected)?;
        record(trace, &format!("{prefix}.inner_attention"), &attended);
        // Upstream deliberately views B,H,L,D as B,L,H*D without transposing.
        let output = self.linear(
            &attended.contiguous()?.reshape((b, lq, d_model))?,
            &format!("{prefix}.out_projection"),
        )?;
        record(trace, &format!("{prefix}.out_projection"), &output);
        record(trace, prefix, &output);
        Ok(output)
    }
    fn ff(
        &self,
        x: &Tensor,
        prefix: &str,
        context: ForwardContext,
        site: u64,
        trace: &mut Option<&mut BTreeMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let ModelSpec::Informer { activation, .. } = self.spec else {
            unreachable!()
        };
        let y = self.linear(x, &format!("{prefix}.conv1"))?;
        record(trace, &format!("{prefix}.conv1"), &y.transpose(1, 2)?);
        let y = match activation {
            Activation::Gelu => op(train_ops::gelu(&y))?,
            Activation::Relu => y.relu()?,
        };
        let y = self.drop(&y, context, site)?;
        let y = self.linear(&y, &format!("{prefix}.conv2"))?;
        record(trace, &format!("{prefix}.conv2"), &y.transpose(1, 2)?);
        let y = self.drop(&y, context, site + 1)?;
        record(trace, &format!("{prefix}.dropout"), &y);
        Ok(y)
    }
    fn distill(
        &self,
        x: &Tensor,
        prefix: &str,
        context: ForwardContext,
        trace: &mut Option<&mut BTreeMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let x = self.circular_conv(x, &format!("{prefix}.downConv"), 2)?;
        record(trace, &format!("{prefix}.downConv"), &x.transpose(1, 2)?);
        let (b, l, c) = x.dims3()?;
        let flat = x.reshape((b * l, c))?;
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mean_key = format!("{prefix}.norm.running_mean");
        let var_key = format!("{prefix}.norm.running_var");
        let count_key = format!("{prefix}.norm.num_batches_tracked");
        let (mean, var) = if context.mode == Mode::Train {
            if b * l <= 1 {
                candle_core::bail!("BatchNorm requires more than one value per channel");
            }
            let mean = flat.mean_keepdim(0)?;
            let var = op(train_ops::pop_var(&flat, 0, true))?;
            let running_mean = buffers[&mean_key]
                .affine(0.9, 0.0)?
                .add(&mean.detach().squeeze(0)?.affine(0.1, 0.0)?)?;
            let running_var = buffers[&var_key].affine(0.9, 0.0)?.add(
                &var.detach()
                    .squeeze(0)?
                    .affine(0.1 * (b * l) as f64 / (b * l - 1) as f64, 0.0)?,
            )?;
            let count = buffers[&count_key]
                .to_scalar::<u32>()?
                .checked_add(1)
                .ok_or_else(|| candle_core::Error::Msg("BatchNorm counter overflow".into()))?;
            buffers.insert(mean_key, running_mean);
            buffers.insert(var_key, running_var);
            buffers.insert(count_key, Tensor::new(count, x.device())?);
            (mean, var)
        } else {
            (
                buffers[&mean_key].unsqueeze(0)?,
                buffers[&var_key].unsqueeze(0)?,
            )
        };
        drop(buffers);
        let normalized = flat
            .broadcast_sub(&mean)?
            .broadcast_div(&var.affine(1.0, 1e-5)?.sqrt()?)?
            .broadcast_mul(&self.weights[&format!("{prefix}.norm.weight")])?
            .broadcast_add(&self.weights[&format!("{prefix}.norm.bias")])?
            .reshape((b, l, c))?;
        record(
            trace,
            &format!("{prefix}.norm"),
            &normalized.transpose(1, 2)?,
        );
        // Compose ELU because the generic Candle ELU backward is unsupported.
        let positive = normalized.ge(0_f32)?;
        // The unused negative branch must remain finite: otherwise exp(large
        // positive) overflows and its zero cotangent produces 0 * inf in bwd.
        let exponential_input = positive.where_cond(&normalized.zeros_like()?, &normalized)?;
        let activated =
            positive.where_cond(&normalized, &exponential_input.exp()?.affine(1.0, -1.0)?)?;
        record(
            trace,
            &format!("{prefix}.activation"),
            &activated.transpose(1, 2)?,
        );
        let pooled = informer_ops::max_pool1d(&activated.transpose(1, 2)?.contiguous()?)?;
        record(trace, &format!("{prefix}.maxPool"), &pooled);
        let y = pooled.transpose(1, 2)?.contiguous()?;
        record(trace, prefix, &y);
        Ok(y)
    }
    pub fn named_parameters(&self) -> Vec<(String, Var)> {
        self.parameters.clone()
    }
    pub fn weights(&self) -> &BTreeMap<String, Tensor> {
        &self.weights
    }
    pub fn sampling_rng(&self) -> &ReferenceRng {
        &self.initialization_rng
    }
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }
    pub fn buffers(&self) -> Result<BTreeMap<String, Tensor>> {
        self.buffers
            .lock()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            .iter()
            .map(|(n, v)| Ok((n.clone(), v.detach().copy()?)))
            .collect()
    }
    pub fn load_buffers(&self, values: &BTreeMap<String, Tensor>) -> Result<()> {
        if self.frozen {
            candle_core::bail!("cannot load frozen Informer");
        }
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        validate_mapping(&buffers, values)?;
        let replacement = values
            .iter()
            .map(|(n, v)| Ok((n.clone(), v.detach().copy()?)))
            .collect::<Result<_>>()?;
        *buffers = replacement;
        Ok(())
    }
    pub fn load_weights(&self, values: &BTreeMap<String, Tensor>) -> Result<()> {
        if self.frozen {
            candle_core::bail!("cannot load frozen Informer");
        }
        validate_mapping(&self.weights, values)?;
        for (name, var) in &self.parameters {
            var.set(&values[name])?;
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
                .map(|(n, v)| Ok((n.clone(), v.detach().copy()?)))
                .collect::<Result<_>>()?,
            parameters: vec![],
            buffers: Mutex::new(self.buffers()?),
            frozen: true,
            initialization_rng: self.initialization_rng.clone(),
        })
    }
    pub fn forward(&self, input: &ModelInput, context: ForwardContext) -> Result<Tensor> {
        self.forward_impl(input, context, None, None)
    }
    /// Fixed public/synthetic sample indices for upstream comparisons, never supervision.
    pub fn forward_with_samples(
        &self,
        input: &ModelInput,
        context: ForwardContext,
        samples: &BTreeMap<String, Tensor>,
    ) -> Result<Tensor> {
        self.forward_impl(input, context, Some(samples), None)
    }
    pub fn forward_with_trace(
        &self,
        input: &ModelInput,
        context: ForwardContext,
        samples: Option<&BTreeMap<String, Tensor>>,
    ) -> Result<(Tensor, BTreeMap<String, Tensor>)> {
        let mut trace = BTreeMap::new();
        let y = self.forward_impl(input, context, samples, Some(&mut trace))?;
        Ok((y, trace))
    }
    fn forward_impl(
        &self,
        input: &ModelInput,
        context: ForwardContext,
        samples: Option<&BTreeMap<String, Tensor>>,
        mut trace: Option<&mut BTreeMap<String, Tensor>>,
    ) -> Result<Tensor> {
        if self.frozen && context.mode != Mode::Eval {
            candle_core::bail!("frozen Informer requires evaluation mode");
        }
        let ModelSpec::Informer {
            e_layers,
            d_layers,
            distil,
            label_len,
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
        let (b, l, c) = history.dims3()?;
        if b == 0
            || l != self.window.lookback
            || c != self.channels
            || history.dtype() != DType::F32
        {
            candle_core::bail!("Informer history shape/dtype mismatch");
        }
        let times = input.history_time.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Informer requires history time features".into())
        })?;
        let future = input.future_time.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Informer requires future time features".into())
        })?;
        if times.dims() != [b, l, self.time_channels]
            || future.dims() != [b, self.window.horizon, self.time_channels]
            || times.dtype() != DType::F32
            || future.dtype() != DType::F32
        {
            candle_core::bail!("Informer time feature shape/dtype mismatch");
        }
        let times = if self.frozen {
            times.detach()
        } else {
            times.clone()
        };
        let future = if self.frozen {
            future.detach()
        } else {
            future.clone()
        };
        let decoder = Tensor::cat(
            &[
                history.narrow(1, l - label_len, label_len)?,
                Tensor::zeros((b, self.window.horizon, c), DType::F32, history.device())?,
            ],
            1,
        )?;
        let decoder_times = Tensor::cat(&[times.narrow(1, l - label_len, label_len)?, future], 1)?;
        record(&mut trace, "decoder_input", &decoder);
        let mut x = self.embedding(&history, &times, "enc_embedding", context, 0, &mut trace)?;
        let mut y = self.embedding(
            &decoder,
            &decoder_times,
            "dec_embedding",
            context,
            1,
            &mut trace,
        )?;
        for layer in 0..e_layers {
            let prefix = format!("encoder.attn_layers.{layer}");
            let site = 100 + layer as u64 * 10;
            let attended = self.attention(
                &x,
                &x,
                &format!("{prefix}.attention"),
                false,
                context,
                site,
                samples,
                &mut trace,
            )?;
            x = self.norm(
                &x.add(&self.drop(&attended, context, site + 1)?)?,
                &format!("{prefix}.norm1"),
            )?;
            record(&mut trace, &format!("{prefix}.norm1"), &x);
            let ff = self.ff(&x, &prefix, context, site + 2, &mut trace)?;
            x = self.norm(&x.add(&ff)?, &format!("{prefix}.norm2"))?;
            record(&mut trace, &format!("{prefix}.norm2"), &x);
            record(&mut trace, &prefix, &x);
            if distil && layer + 1 < e_layers {
                x = self.distill(
                    &x,
                    &format!("encoder.conv_layers.{layer}"),
                    context,
                    &mut trace,
                )?;
            }
        }
        x = self.norm(&x, "encoder.norm")?;
        record(&mut trace, "encoder.norm", &x);
        record(&mut trace, "encoder", &x);
        for layer in 0..d_layers {
            let prefix = format!("decoder.layers.{layer}");
            let site = 10000 + layer as u64 * 10;
            let attended = self.attention(
                &y,
                &y,
                &format!("{prefix}.self_attention"),
                true,
                context,
                site,
                samples,
                &mut trace,
            )?;
            y = self.norm(
                &y.add(&self.drop(&attended, context, site + 1)?)?,
                &format!("{prefix}.norm1"),
            )?;
            record(&mut trace, &format!("{prefix}.norm1"), &y);
            let attended = self.attention(
                &y,
                &x,
                &format!("{prefix}.cross_attention"),
                false,
                context,
                site + 2,
                samples,
                &mut trace,
            )?;
            y = self.norm(
                &y.add(&self.drop(&attended, context, site + 3)?)?,
                &format!("{prefix}.norm2"),
            )?;
            record(&mut trace, &format!("{prefix}.norm2"), &y);
            let ff = self.ff(&y, &prefix, context, site + 4, &mut trace)?;
            y = self.norm(&y.add(&ff)?, &format!("{prefix}.norm3"))?;
            record(&mut trace, &format!("{prefix}.norm3"), &y);
            record(&mut trace, &prefix, &y);
        }
        y = self.norm(&y, "decoder.norm")?;
        record(&mut trace, "decoder.norm", &y);
        y = self.linear(&y, "decoder.projection")?;
        record(&mut trace, "decoder.projection", &y);
        record(&mut trace, "decoder", &y);
        y.narrow(1, label_len, self.window.horizon)
    }
}
fn validate_mapping(
    expected: &BTreeMap<String, Tensor>,
    actual: &BTreeMap<String, Tensor>,
) -> Result<()> {
    if expected.len() != actual.len() {
        candle_core::bail!("Informer tensor names mismatch");
    }
    for (name, expected) in expected {
        let value = actual
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing Informer tensor {name}")))?;
        if value.shape() != expected.shape()
            || value.dtype() != expected.dtype()
            || !value.device().same_device(expected.device())
        {
            candle_core::bail!("Informer tensor schema mismatch: {name}");
        }
        if value.dtype() == DType::F32
            && !value
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite())
        {
            candle_core::bail!("nonfinite Informer tensor: {name}");
        }
    }
    Ok(())
}
