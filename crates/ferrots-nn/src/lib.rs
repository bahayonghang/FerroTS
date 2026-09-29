//! Tensor operations and neural networks for FerroTS.

mod device;
pub mod infer_ops;
pub use device::resolve_device;
pub mod baselines;
mod random;
pub mod train_ops;
pub use random::{Dropout, Mode, RngStream, dropout};
