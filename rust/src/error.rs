//! Error type (SPEC section 4).

use core::fmt;

/// Errors returned by LombokVector operations. [`VectorError::code`] gives the
/// code shared with the other language ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorError {
    /// Input slices have different lengths (`DIMENSION_MISMATCH`).
    DimensionMismatch {
        /// Length of the first (or query) vector.
        expected: usize,
        /// Length of the vector that differs.
        actual: usize,
    },
    /// An input vector has no elements (`EMPTY_VECTOR`).
    EmptyVector,
    /// A vector has magnitude zero where a direction is needed (`ZERO_MAGNITUDE`).
    ZeroMagnitude,
    /// A result or intermediate value is infinite or NaN: an input was not
    /// finite or the computation overflowed (`NON_FINITE`).
    NonFinite,
}

impl VectorError {
    /// The code shared by every port, for example `"DIMENSION_MISMATCH"`.
    pub fn code(&self) -> &'static str {
        match self {
            VectorError::DimensionMismatch { .. } => "DIMENSION_MISMATCH",
            VectorError::EmptyVector => "EMPTY_VECTOR",
            VectorError::ZeroMagnitude => "ZERO_MAGNITUDE",
            VectorError::NonFinite => "NON_FINITE",
        }
    }
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "DIMENSION_MISMATCH: expected {expected} elements, got {actual}"
                )
            }
            VectorError::EmptyVector => f.write_str("EMPTY_VECTOR: vector has no elements"),
            VectorError::ZeroMagnitude => f.write_str("ZERO_MAGNITUDE: vector has magnitude zero"),
            VectorError::NonFinite => f.write_str("NON_FINITE: result is infinite or NaN"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VectorError {}
