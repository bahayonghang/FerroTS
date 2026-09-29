//! Data loading and preprocessing for FerroTS.

pub mod batch;
pub mod cache;
pub mod csv_source;
pub mod preprocess;
pub mod scaler;
pub mod time_features;
pub mod windows;

pub use csv_source::{ColumnReport, Frame, RawReport, read_csv};
pub use preprocess::{PreparedSeries, Segment, Span, prepare};

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("data IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("data metadata: {0}")]
    Json(#[from] serde_json::Error),
    #[error("data tensors: {0}")]
    Tensors(#[from] safetensors::SafeTensorError),
    #[error("invalid data: {0}")]
    Invalid(String),
}
pub type Result<T> = std::result::Result<T, DataError>;
pub(crate) fn ensure(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(DataError::Invalid(message.into()))
    }
}
