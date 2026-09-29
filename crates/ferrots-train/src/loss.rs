use crate::{Result, TrainError};
use candle_core::{DType, Tensor};
pub struct MaskedLoss {
    pub sum: Tensor,
    pub valid_count: u64,
}
pub fn masked_mse(prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<MaskedLoss> {
    if prediction.shape() != target.shape()
        || mask.shape() != target.shape()
        || prediction.dtype() != DType::F32
        || target.dtype() != DType::F32
        || mask.dtype() != DType::U8
    {
        return Err(TrainError::InvalidState("loss shape/dtype mismatch".into()));
    }
    let mask = mask.ne(0u8)?;
    let count = mask.to_dtype(DType::U32)?.sum_all()?.to_scalar::<u32>()? as u64;
    let zeros = target.zeros_like()?;
    let pred = mask.where_cond(prediction, &zeros)?;
    let target = mask.where_cond(target, &zeros)?;
    let sum = pred.sub(&target)?.sqr()?.sum_all()?;
    if !sum.to_scalar::<f32>()?.is_finite() {
        return Err(TrainError::InvalidState(
            "non-finite valid prediction/target or loss".into(),
        ));
    }
    Ok(MaskedLoss {
        sum,
        valid_count: count,
    })
}
