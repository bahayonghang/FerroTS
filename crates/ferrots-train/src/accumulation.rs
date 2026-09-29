use crate::{NamedParams, Result, StatefulAdam, TrainError, loss::MaskedLoss};
use candle_core::{Tensor, backprop::GradStore};
use std::collections::BTreeMap;
pub struct GradientAccumulator {
    params: NamedParams,
    sums: BTreeMap<String, Tensor>,
    pub valid_count: u64,
}
impl GradientAccumulator {
    pub fn new(params: NamedParams) -> Self {
        Self {
            params,
            sums: BTreeMap::new(),
            valid_count: 0,
        }
    }
    pub fn add(&mut self, loss: &MaskedLoss) -> Result<()> {
        if loss.valid_count == 0 {
            return Ok(());
        }
        let gradients = loss.sum.backward()?;
        let mut pending = BTreeMap::new();
        for (name, var) in self.params.iter() {
            let gradient = gradients.get(var).ok_or_else(|| {
                TrainError::InvalidState(format!("missing expected gradient: {name}"))
            })?;
            if !gradient
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(TrainError::InvalidState(format!(
                    "non-finite gradient: {name}"
                )));
            }
            let gradient = gradient.detach();
            let sum = match self.sums.get(name) {
                Some(previous) => previous.add(&gradient)?,
                None => gradient,
            };
            pending.insert(name.to_string(), sum.detach());
        }
        self.valid_count = self
            .valid_count
            .checked_add(loss.valid_count)
            .ok_or_else(|| TrainError::InvalidState("gradient count overflow".into()))?;
        self.sums = pending;
        Ok(())
    }
    pub fn step(&mut self, optimizer: &mut StatefulAdam) -> Result<bool> {
        if self.valid_count == 0 {
            return Ok(false);
        }
        let mut gradients = GradStore::default();
        for (name, var) in self.params.iter() {
            let gradient = self.sums[name].affine(1.0 / self.valid_count as f64, 0.0)?;
            if !gradient
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(TrainError::InvalidState(format!(
                    "non-finite accumulated gradient: {name}"
                )));
            }
            gradients.insert(var, gradient);
        }
        optimizer.step(&gradients)?;
        self.sums.clear();
        self.valid_count = 0;
        Ok(true)
    }
}
