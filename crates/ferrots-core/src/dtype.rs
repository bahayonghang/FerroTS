use std::{fmt, str::FromStr};

/// Supported production precision. Numerical tests can use backend F64 tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypePolicy {
    F32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DTypeError {
    #[error("unsupported dtype {dtype:?}; FerroTS currently supports only f32")]
    Unsupported { dtype: String },
}

impl FromStr for DTypePolicy {
    type Err = DTypeError;

    fn from_str(dtype: &str) -> Result<Self, Self::Err> {
        match dtype {
            "f32" => Ok(Self::F32),
            _ => Err(DTypeError::Unsupported {
                dtype: dtype.to_owned(),
            }),
        }
    }
}

impl fmt::Display for DTypePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_f32() {
        assert_eq!("f32".parse::<DTypePolicy>().unwrap(), DTypePolicy::F32);
        assert_eq!(DTypePolicy::F32.to_string(), "f32");
        for dtype in ["f16", "bf16", "f64", "i32", "F32", ""] {
            assert_eq!(
                dtype.parse::<DTypePolicy>(),
                Err(DTypeError::Unsupported {
                    dtype: dtype.to_owned(),
                })
            );
        }
    }
}
