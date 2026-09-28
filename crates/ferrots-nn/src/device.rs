use candle_core::Device;
use ferrots_core::{Device as DeviceSpec, DeviceError};

/// Resolve an explicit device. CUDA errors never select another backend.
pub fn resolve_device(spec: DeviceSpec) -> Result<Device, DeviceError> {
    match spec {
        DeviceSpec::Cpu => Ok(Device::Cpu),
        DeviceSpec::Cuda(index) => {
            #[cfg(feature = "cuda")]
            {
                if index > i32::MAX as usize {
                    return Err(DeviceError::CudaUnavailable {
                        index,
                        source: std::io::Error::other(
                            "CUDA index exceeds the driver ordinal range",
                        )
                        .into(),
                    });
                }
                Device::new_cuda(index).map_err(|source| DeviceError::CudaUnavailable {
                    index,
                    source: source.into(),
                })
            }
            #[cfg(not(feature = "cuda"))]
            Err(DeviceError::CudaUnavailable {
                index,
                source: std::io::Error::other("binary was built without the cuda feature").into(),
            })
        }
    }
}
