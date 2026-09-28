use std::{fmt, str::FromStr};

/// An explicit execution device. CUDA requests must never fall back to CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("invalid device {value:?}; expected cpu or cuda:N with a non-negative integer N")]
    InvalidSpec { value: String },
    #[error("cuda:{index} is unavailable: {source}")]
    CudaUnavailable {
        index: usize,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl FromStr for Device {
    type Err = DeviceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == "cpu" {
            return Ok(Self::Cpu);
        }
        if let Some(index) = value.strip_prefix("cuda:")
            && !index.is_empty()
            && index.bytes().all(|digit| digit.is_ascii_digit())
            && let Ok(index) = index.parse()
        {
            return Ok(Self::Cuda(index));
        }
        Err(DeviceError::InvalidSpec {
            value: value.to_owned(),
        })
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda(index) => write!(f, "cuda:{index}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_explicit_devices() {
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
        assert_eq!("cuda:0".parse::<Device>().unwrap(), Device::Cuda(0));
        assert_eq!("cuda:9".parse::<Device>().unwrap(), Device::Cuda(9));
        for device in [Device::Cpu, Device::Cuda(0), Device::Cuda(usize::MAX)] {
            assert_eq!(device.to_string().parse::<Device>().unwrap(), device);
        }
    }

    #[test]
    fn rejects_invalid_device_specs() {
        for value in [
            "", "CPU", "gpu", "cpu:0", "cuda", "cuda:", "cuda:-1", "cuda:+1", "cuda:1.0", "cuda:x",
            " cpu", "cuda:0 ", "cuda:0:1",
        ] {
            assert!(
                matches!(
                    value.parse::<Device>(),
                    Err(DeviceError::InvalidSpec { .. })
                ),
                "accepted invalid device: {value:?}"
            );
        }
        let overflow = format!("cuda:{}0", usize::MAX);
        assert!(overflow.parse::<Device>().is_err());
    }

    #[test]
    fn cuda_error_preserves_device_index_and_source() {
        let error = DeviceError::CudaUnavailable {
            index: 9,
            source: std::io::Error::other("device not found").into(),
        };
        assert!(error.to_string().contains("cuda:9"));
        assert_eq!(
            std::error::Error::source(&error).unwrap().to_string(),
            "device not found"
        );
    }
}
