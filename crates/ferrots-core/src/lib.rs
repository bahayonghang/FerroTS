//! Backend-independent types and contracts for FerroTS.

pub mod config;
mod device;
mod dtype;
pub mod metrics;
pub mod model;
pub mod reference_rng;

pub use device::{Device, DeviceError};
pub use dtype::{DTypeError, DTypePolicy};
