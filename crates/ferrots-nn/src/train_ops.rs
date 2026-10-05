//! Differentiable operations. F64 is limited to CPU numerical tests.
use candle_core::{DType, Tensor};

#[derive(Debug, thiserror::Error)]
pub enum OpError {
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error("an attention row has no valid positions")]
    FullyMaskedRow,
    #[error("invalid operation argument: {0}")]
    InvalidArgument(&'static str),
    #[error("unsupported operation dtype {0:?}; use f32, or CPU f64 for numerical tests")]
    UnsupportedDType(DType),
}
pub type Result<T> = std::result::Result<T, OpError>;

pub(crate) fn validate_float(x: &Tensor) -> Result<()> {
    match x.dtype() {
        DType::F32 => Ok(()),
        DType::F64 if x.device().is_cpu() => Ok(()),
        dtype => Err(OpError::UnsupportedDType(dtype)),
    }
}
fn validate_dim(x: &Tensor, dim: usize) -> Result<()> {
    validate_float(x)?;
    if dim >= x.rank() || x.dims()[dim] == 0 {
        return Err(OpError::InvalidArgument(
            "reduction dimension must exist and be non-empty",
        ));
    }
    Ok(())
}

/// Exact erf GELU, including the derivative of the erf expression.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    validate_float(x)?;
    // Candle's gelu_erf backward uses a rounded coefficient (0.398942).
    // Composing erf preserves the full coefficient in CPU F64 checks.
    Ok(x.mul(
        &x.affine(std::f64::consts::FRAC_1_SQRT_2, 0.0)?
            .erf()?
            .affine(0.5, 0.5)?,
    )?)
}

/// Apply x W^T + bias to the final axis, preserving all leading dimensions.
pub fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    validate_float(x)?;
    validate_float(weight)?;
    validate_float(bias)?;
    let Some(&input_width) = x.dims().last() else {
        return Err(OpError::InvalidArgument(
            "linear requires a nonempty final axis",
        ));
    };
    let (output_width, expected_width) = weight.dims2()?;
    if input_width == 0
        || output_width == 0
        || input_width != expected_width
        || bias.dims() != [output_width]
        || x.elem_count() == 0
    {
        return Err(OpError::InvalidArgument("linear shape mismatch"));
    }
    let mut shape = x.dims().to_vec();
    *shape.last_mut().expect("validated rank") = output_width;
    Ok(x.contiguous()?
        .reshape((x.elem_count() / input_width, input_width))?
        .matmul(&weight.t()?)?
        .broadcast_add(bias)?
        .reshape(shape)?)
}

pub fn softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    validate_dim(x, dim)?;
    let shifted = x.broadcast_sub(&x.max_keepdim(dim)?.detach())?;
    let exp = shifted.exp()?;
    Ok(exp.broadcast_div(&exp.sum_keepdim(dim)?)?)
}

/// Nonzero U8 mask entries are valid. The mask must broadcast to the input.
pub fn masked_softmax(x: &Tensor, mask: &Tensor, dim: usize) -> Result<Tensor> {
    validate_dim(x, dim)?;
    if mask.dtype() != DType::U8 {
        return Err(OpError::InvalidArgument("mask must have dtype u8"));
    }
    let mask = mask.ne(0_u8)?.broadcast_as(x.shape())?;
    let empty = mask.to_dtype(DType::F32)?.sum_keepdim(dim)?.eq(0_f32)?;
    if empty.flatten_all()?.to_vec1::<u8>()?.contains(&1) {
        return Err(OpError::FullyMaskedRow);
    }
    let excluded = Tensor::full(f64::NEG_INFINITY, x.shape(), x.device())?.to_dtype(x.dtype())?;
    softmax(&mask.where_cond(x, &excluded)?, dim)
}

pub fn pop_var(x: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    validate_dim(x, dim)?;
    let centered = x.broadcast_sub(&x.mean_keepdim(dim)?)?;
    let variance = centered.sqr()?.mean_keepdim(dim)?;
    Ok(if keepdim {
        variance
    } else {
        variance.squeeze(dim)?
    })
}

pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f64) -> Result<Tensor> {
    let dim = x.rank().checked_sub(1).ok_or(OpError::InvalidArgument(
        "layer norm input must have at least one dimension",
    ))?;
    validate_dim(x, dim)?;
    validate_float(gamma)?;
    validate_float(beta)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(OpError::InvalidArgument(
            "epsilon must be finite and positive",
        ));
    }
    if gamma.dims() != [x.dims()[dim]] || beta.dims() != gamma.dims() {
        return Err(OpError::InvalidArgument(
            "layer norm affine parameters must match the last dimension",
        ));
    }
    let centered = x.broadcast_sub(&x.mean_keepdim(dim)?)?;
    let variance = centered.sqr()?.mean_keepdim(dim)?;
    Ok(centered
        .broadcast_div(&variance.affine(1.0, eps)?.sqrt()?)?
        .broadcast_mul(gamma)?
        .broadcast_add(beta)?)
}

/// Q: [batch, heads, queries, width], K/V share batch, heads and key count.
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    dropout: Option<crate::Dropout>,
) -> Result<Tensor> {
    validate_float(q)?;
    validate_float(k)?;
    validate_float(v)?;
    let (b, h, lq, width) = q.dims4()?;
    let (bk, hk, lk, wk) = k.dims4()?;
    let (bv, hv, lv, dv) = v.dims4()?;
    if b == 0
        || h == 0
        || lq == 0
        || width == 0
        || lk == 0
        || dv == 0
        || (b, h, width) != (bk, hk, wk)
        || (b, h, lk) != (bv, hv, lv)
    {
        return Err(OpError::InvalidArgument(
            "incompatible or empty attention dimensions",
        ));
    }
    let scores = q
        .contiguous()?
        .matmul(&k.transpose(2, 3)?.contiguous()?)?
        .affine(1.0 / (width as f64).sqrt(), 0.0)?;
    let weights = match mask {
        Some(mask) => masked_softmax(&scores, mask, 3)?,
        None => softmax(&scores, 3)?,
    };
    let weights = match dropout {
        Some(dropout) => dropout.forward(&weights)?,
        None => weights,
    };
    Ok(weights.matmul(&v.contiguous()?)?)
}
