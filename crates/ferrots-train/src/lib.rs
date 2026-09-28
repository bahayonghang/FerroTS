//! Training state and optimizers for FerroTS.

pub mod optim;
mod params;
pub use ferrots_nn::RngStream;
pub use optim::{ParamGroup, ParamState, StatefulAdam, StatefulAdamW, StatefulOptimizer};
pub use params::NamedParams;

#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    DType(#[from] ferrots_core::DTypeError),
    #[error("invalid training state: {0}")]
    InvalidState(String),
}
pub type Result<T> = std::result::Result<T, TrainError>;
