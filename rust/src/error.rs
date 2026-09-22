// LombokVector — Error types
// License: Apache-2.0

use core::fmt;

/// Errors returned by LombokVector operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorError {
    /// Input slices have different lengths.
    DimensionMismatch { expected: usize, actual: usize },
    /// Input slice is empty (zero dimensions).
    EmptyVector,
    /// Vector has zero magnitude (cannot normalize).
    ZeroMagnitude,
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "dimension mismatch: expected {expected}, got {actual}"
                )
            }
            VectorError::EmptyVector => write!(f, "empty vector (zero dimensions)"),
            VectorError::ZeroMagnitude => write!(f, "zero magnitude vector (cannot normalize)"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VectorError {}
