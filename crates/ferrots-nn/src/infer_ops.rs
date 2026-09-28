//! Fused operations for explicitly declared evaluation scopes.
use crate::train_ops::{OpError, Result, validate_float};
use candle_core::Tensor;

/// An explicit declaration that the caller requires no gradients.
/// Fused calls also reject every input tensor that records an operation graph.
pub struct InferenceOnly {
    _private: (),
}
impl InferenceOnly {
    pub fn evaluation() -> Self {
        Self { _private: () }
    }
}
fn validate_frozen(tensors: &[&Tensor]) -> Result<()> {
    for tensor in tensors {
        validate_float(tensor)?;
        if tensor.track_op() {
            return Err(OpError::InvalidArgument(
                "fused inference requires frozen tensors",
            ));
        }
    }
    Ok(())
}
pub fn softmax_fused(x: &Tensor, _: &InferenceOnly) -> Result<Tensor> {
    validate_frozen(&[x])?;
    Ok(candle_nn::ops::softmax_last_dim(x)?)
}
pub fn layer_norm_fused(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
    _: &InferenceOnly,
) -> Result<Tensor> {
    validate_frozen(&[x, gamma, beta])?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(OpError::InvalidArgument(
            "epsilon must be finite and positive",
        ));
    }
    Ok(candle_nn::ops::layer_norm(x, gamma, beta, eps)?)
}
