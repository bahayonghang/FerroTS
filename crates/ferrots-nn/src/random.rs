//! Rebuildable random streams. The key encoding is stable across platforms.
use crate::train_ops::{OpError, Result, validate_float};
use candle_core::Tensor;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RngStream {
    pub seed: u64,
    pub run_step: u64,
    pub layer_id: u64,
}
impl RngStream {
    fn generator(self) -> ChaCha8Rng {
        let mut key = [0_u8; 32];
        key[..8].copy_from_slice(&self.seed.to_le_bytes());
        key[8..16].copy_from_slice(&self.run_step.to_le_bytes());
        key[16..24].copy_from_slice(&self.layer_id.to_le_bytes());
        key[24..].copy_from_slice(b"FerroTS0");
        ChaCha8Rng::from_seed(key)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Train,
    Eval,
}
#[derive(Debug, Clone, Copy)]
pub struct Dropout {
    pub probability: f64,
    pub mode: Mode,
    pub stream: RngStream,
}
impl Dropout {
    pub fn forward(self, x: &Tensor) -> Result<Tensor> {
        dropout(x, self.probability, self.mode, self.stream)
    }
}
/// Eval preserves the tensor graph; inference callers must freeze their tensors separately.
pub fn dropout(x: &Tensor, p: f64, mode: Mode, stream: RngStream) -> Result<Tensor> {
    validate_float(x)?;
    if !p.is_finite() || !(0.0..1.0).contains(&p) {
        return Err(OpError::InvalidArgument(
            "dropout probability must be in [0, 1)",
        ));
    }
    if mode == Mode::Eval || p == 0.0 {
        return Ok(x.clone());
    }
    let mut rng = stream.generator();
    let mask: Vec<f32> = (0..x.elem_count())
        .map(|_| if rng.random::<f64>() >= p { 1.0 } else { 0.0 })
        .collect();
    let mask = Tensor::from_vec(mask, x.shape(), x.device())?.to_dtype(x.dtype())?;
    Ok(x.mul(&mask)?.affine(1.0 / (1.0 - p), 0.0)?)
}
