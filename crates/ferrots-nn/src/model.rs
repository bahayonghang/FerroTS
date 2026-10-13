//! Static model dispatch shared by training, checkpoints and evaluation.
use crate::{Mode, baselines::Baseline, informer::Informer, itransformer::ITransformer};
use candle_core::{Device, Result, Tensor, Var};
use ferrots_core::{
    config::{Config, ModelSpec, WindowSpec},
    reference_rng::ReferenceRng,
};
use std::collections::BTreeMap;

/// Supervision has no representation in the model input type.
pub struct ModelInput {
    pub history: Tensor,
    pub history_time: Option<Tensor>,
    pub future_time: Option<Tensor>,
}
impl ModelInput {
    pub fn history(history: Tensor) -> Self {
        Self {
            history,
            history_time: None,
            future_time: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ForwardContext {
    pub mode: Mode,
    pub forward_id: u64,
}
impl ForwardContext {
    pub const fn evaluation() -> Self {
        Self::evaluation_with_id(0)
    }
    pub const fn evaluation_with_id(forward_id: u64) -> Self {
        Self {
            mode: Mode::Eval,
            forward_id,
        }
    }
}

pub enum ForecastModel {
    Baseline(Baseline),
    ITransformer(ITransformer),
    Informer(Informer),
}
impl ForecastModel {
    pub fn new(
        spec: &ModelSpec,
        window: &WindowSpec,
        channels: usize,
        seed: u64,
        device: &Device,
    ) -> Result<Self> {
        if matches!(spec, ModelSpec::Itransformer { .. }) {
            return Ok(Self::ITransformer(ITransformer::new(
                spec, window, channels, 0, seed, device,
            )?));
        }
        if matches!(spec, ModelSpec::Informer { .. }) {
            return Ok(Self::Informer(Informer::new(
                spec, window, channels, 4, seed, device,
            )?));
        }
        Ok(Self::Baseline(Baseline::new(
            spec, window, channels, seed, device,
        )?))
    }
    pub fn from_config(config: &Config, device: &Device) -> Result<Self> {
        if crate::registry::find(config.model.name()).is_none() {
            candle_core::bail!("model is not implemented: {}", config.model.name());
        }
        config
            .validate_model_inputs()
            .map_err(candle_core::Error::wrap)?;
        if matches!(
            config.model,
            ModelSpec::Itransformer { .. } | ModelSpec::Informer { .. }
        ) {
            let width = match config.data.time_features.as_str() {
                "h" => 4,
                "t" => 5,
                _ => candle_core::bail!("unsupported time feature protocol"),
            };
            return match &config.model {
                ModelSpec::Informer { .. } => Ok(Self::Informer(Informer::new(
                    &config.model,
                    &config.window,
                    config.data.input_columns.len(),
                    width,
                    config.train.seed,
                    device,
                )?)),
                _ => Ok(Self::ITransformer(ITransformer::new(
                    &config.model,
                    &config.window,
                    config.data.input_columns.len(),
                    width,
                    config.train.seed,
                    device,
                )?)),
            };
        }
        Self::new(
            &config.model,
            &config.window,
            config.data.input_columns.len(),
            config.train.seed,
            device,
        )
    }
    pub fn spec(&self) -> &ModelSpec {
        match self {
            Self::Baseline(m) => &m.spec,
            Self::ITransformer(m) => &m.spec,
            Self::Informer(m) => &m.spec,
        }
    }
    pub fn window(&self) -> &WindowSpec {
        match self {
            Self::Baseline(m) => &m.window,
            Self::ITransformer(m) => &m.window,
            Self::Informer(m) => &m.window,
        }
    }
    pub fn channels(&self) -> usize {
        match self {
            Self::Baseline(m) => m.channels,
            Self::ITransformer(m) => m.channels,
            Self::Informer(m) => m.channels,
        }
    }
    pub fn named_parameters(&self) -> Vec<(String, Var)> {
        match self {
            Self::Baseline(m) => m.named_parameters(),
            Self::ITransformer(m) => m.named_parameters(),
            Self::Informer(m) => m.named_parameters(),
        }
    }
    pub fn weights(&self) -> &BTreeMap<String, Tensor> {
        match self {
            Self::Baseline(m) => m.weights(),
            Self::ITransformer(m) => m.weights(),
            Self::Informer(m) => m.weights(),
        }
    }
    pub fn sampling_rng(&self) -> &ReferenceRng {
        match self {
            Self::Baseline(m) => m.sampling_rng(),
            Self::ITransformer(m) => m.sampling_rng(),
            Self::Informer(m) => m.sampling_rng(),
        }
    }
    pub fn is_frozen(&self) -> bool {
        match self {
            Self::Baseline(m) => m.is_frozen(),
            Self::ITransformer(m) => m.is_frozen(),
            Self::Informer(m) => m.is_frozen(),
        }
    }
    pub fn load_weights(&self, weights: &BTreeMap<String, Tensor>) -> Result<()> {
        match self {
            Self::Baseline(m) => m.load_weights(weights),
            Self::ITransformer(m) => m.load_weights(weights),
            Self::Informer(m) => m.load_weights(weights),
        }
    }
    /// Persistent non-parameter state, copied independently from the live model.
    pub fn buffers(&self) -> Result<BTreeMap<String, Tensor>> {
        match self {
            Self::Informer(model) => model.buffers(),
            _ => Ok(BTreeMap::new()),
        }
    }
    /// The model owns the complete buffer schema and validates before applying it.
    pub fn load_buffers(&self, buffers: &BTreeMap<String, Tensor>) -> Result<()> {
        match self {
            Self::Informer(model) => model.load_buffers(buffers),
            _ if buffers.is_empty() => Ok(()),
            _ => candle_core::bail!("unexpected persistent model buffers"),
        }
    }
    pub fn freeze_for_eval(&self) -> Result<Self> {
        match self {
            Self::Baseline(m) => Ok(Self::Baseline(m.freeze_for_eval()?)),
            Self::ITransformer(m) => Ok(Self::ITransformer(m.freeze_for_eval()?)),
            Self::Informer(m) => Ok(Self::Informer(m.freeze_for_eval()?)),
        }
    }
    pub fn forward(&self, input: &ModelInput) -> Result<Tensor> {
        self.forward_with_context(input, ForwardContext::evaluation())
    }
    pub fn forward_with_context(
        &self,
        input: &ModelInput,
        context: ForwardContext,
    ) -> Result<Tensor> {
        match self {
            Self::Baseline(m) => m.forward(input),
            Self::ITransformer(m) => m.forward(input, context),
            Self::Informer(m) => m.forward(input, context),
        }
    }
}
