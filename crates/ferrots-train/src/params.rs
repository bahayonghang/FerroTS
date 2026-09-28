use crate::{Result, TrainError};
use candle_core::{DType, Device, Tensor, Var};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};

/// Parameter names are sorted and unique. Cloning shares the underlying variables.
#[derive(Clone)]
pub struct NamedParams {
    params: BTreeMap<String, Var>,
}
impl NamedParams {
    pub fn new(params: Vec<(String, Var)>) -> Result<Self> {
        if params.is_empty() {
            return Err(TrainError::InvalidState("parameter set is empty".into()));
        }
        let device = params[0].1.device().clone();
        let mut ordered = BTreeMap::new();
        let mut ids = HashSet::new();
        for (name, var) in params {
            if name.is_empty() || ordered.contains_key(&name) || !ids.insert(var.id()) {
                return Err(TrainError::InvalidState(
                    "parameter names and variable identities must be unique and names non-empty"
                        .into(),
                ));
            }
            require_f32(&var)?;
            if !device.same_device(var.device()) {
                return Err(TrainError::InvalidState(
                    "parameters must share one device".into(),
                ));
            }
            ordered.insert(name, var);
        }
        Ok(Self { params: ordered })
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Var)> {
        self.params.iter().map(|(n, v)| (n.as_str(), v))
    }
    pub fn get(&self, name: &str) -> Option<&Var> {
        self.params.get(name)
    }
    pub fn device(&self) -> &Device {
        self.params
            .first_key_value()
            .expect("non-empty parameter set")
            .1
            .device()
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let tensors: HashMap<_, _> = self
            .iter()
            .map(|(name, var)| (name.to_owned(), var.detach()))
            .collect();
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }
    /// Validate the complete file before modifying any variable.
    pub fn load(&self, path: impl AsRef<Path>) -> Result<()> {
        let tensors = candle_core::safetensors::load(path, self.device())?;
        if tensors.len() != self.params.len() {
            return Err(TrainError::InvalidState(
                "model parameter names differ".into(),
            ));
        }
        for (name, var) in self.iter() {
            let value = tensors.get(name).ok_or_else(|| {
                TrainError::InvalidState(format!("missing model parameter {name}"))
            })?;
            validate_like(name, value, var)?;
        }
        self.device().synchronize()?;
        for (name, var) in self.iter() {
            var.set(&tensors[name])?;
        }
        Ok(())
    }
}
pub(crate) fn require_f32(tensor: &Tensor) -> Result<()> {
    if tensor.dtype() != DType::F32 {
        return Err(ferrots_core::DTypeError::Unsupported {
            dtype: format!("{:?}", tensor.dtype()),
        }
        .into());
    }
    Ok(())
}
pub(crate) fn validate_like(name: &str, value: &Tensor, param: &Tensor) -> Result<()> {
    require_f32(value)?;
    if value.shape() != param.shape() || !value.device().same_device(param.device()) {
        return Err(TrainError::InvalidState(format!(
            "shape or device mismatch for {name}"
        )));
    }
    Ok(())
}
