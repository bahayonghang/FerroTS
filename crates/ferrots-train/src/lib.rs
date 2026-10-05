//! Training state and optimizers for FerroTS.

pub mod accumulation;
pub mod checkpoint;
pub mod evaluation;
pub mod loss;
pub mod optim;
mod params;
pub mod schedule;
pub mod session;
pub mod trainer;
pub use ferrots_nn::RngStream;
pub use optim::{ParamGroup, ParamState, StatefulAdam, StatefulAdamW, StatefulOptimizer};
pub use params::NamedParams;

#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error(transparent)]
    Data(#[from] ferrots_data::DataError),
    #[error(transparent)]
    Config(#[from] ferrots_core::config::ConfigError),
    #[error(transparent)]
    Metrics(#[from] ferrots_core::metrics::MetricError),
    #[error(transparent)]
    Toml(#[from] toml::ser::Error),
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
