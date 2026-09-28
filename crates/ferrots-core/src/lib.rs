//! Backend-independent types and contracts for FerroTS.

mod device;
mod dtype;

pub use device::{Device, DeviceError};
pub use dtype::{DTypeError, DTypePolicy};
