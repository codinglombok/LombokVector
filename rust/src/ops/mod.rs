// LombokVector — Core vector operations (public API)
// License: Apache-2.0

use crate::error::VectorError;
use crate::simd::dispatch;

// ============================================================================
// Validation helpers
// ============================================================================

#[inline]
fn check_pair(a_len: usize, b_len: usize) -> Result<(), VectorError> {
    if a_len == 0 {
        return Err(VectorError::EmptyVector);
    }
    if a_len != b_len {
        return Err(VectorError::DimensionMismatch {
            expected: a_len,
            actual: b_len,
        });
    }
    Ok(())
}

#[inline]
fn check_single(len: usize) -> Result<(), VectorError> {
    if len == 0 {
        return Err(VectorError::EmptyVector);
    }
    Ok(())
}

// ============================================================================
// f32 operations
// ============================================================================

/// Cosine similarity between two f32 vectors.
///
/// Returns a value in [-1.0, 1.0]. 1.0 = identical direction.
///
/// # Errors
/// - `EmptyVector` if inputs are empty
/// - `DimensionMismatch` if lengths differ
/// - `ZeroMagnitude` if either vector has zero magnitude
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    check_pair(a.len(), b.len())?;
    let dot = dispatch::dot_f32(a, b);
    let norm_a = dispatch::sum_squares_f32(a).sqrt();
    let norm_b = dispatch::sum_squares_f32(b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(VectorError::ZeroMagnitude);
    }
    Ok(dot / (norm_a * norm_b))
}

/// Dot product of two f32 vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    check_pair(a.len(), b.len())?;
    Ok(dispatch::dot_f32(a, b))
}

/// Euclidean (L2) distance between two f32 vectors.
pub fn l2_distance(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    check_pair(a.len(), b.len())?;
    Ok(dispatch::l2_sq_f32(a, b).sqrt())
}

/// Inner product (same as dot product for real-valued vectors).
pub fn inner_product(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    dot_product(a, b)
}

/// L2 norm (magnitude) of an f32 vector.
pub fn l2_norm(a: &[f32]) -> Result<f32, VectorError> {
    check_single(a.len())?;
    Ok(dispatch::sum_squares_f32(a).sqrt())
}

/// Normalize an f32 vector to unit length (L2 norm = 1.0).
/// Returns a new Vec<f32>.
///
/// # Errors
/// - `ZeroMagnitude` if the vector has zero magnitude
#[cfg(feature = "std")]
pub fn normalize(a: &[f32]) -> Result<Vec<f32>, VectorError> {
    check_single(a.len())?;
    let norm = dispatch::sum_squares_f32(a).sqrt();
    if norm == 0.0 {
        return Err(VectorError::ZeroMagnitude);
    }
    let inv = 1.0 / norm;
    Ok(a.iter().map(|&x| x * inv).collect())
}

/// Normalize an f32 vector in-place.
pub fn normalize_inplace(a: &mut [f32]) -> Result<(), VectorError> {
    check_single(a.len())?;
    let norm = dispatch::sum_squares_f32(a).sqrt();
    if norm == 0.0 {
        return Err(VectorError::ZeroMagnitude);
    }
    let inv = 1.0 / norm;
    for x in a.iter_mut() {
        *x *= inv;
    }
    Ok(())
}

/// Element-wise addition. Returns new Vec.
#[cfg(feature = "std")]
pub fn vec_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, VectorError> {
    check_pair(a.len(), b.len())?;
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
}

/// Element-wise subtraction. Returns new Vec.
#[cfg(feature = "std")]
pub fn vec_sub(a: &[f32], b: &[f32]) -> Result<Vec<f32>, VectorError> {
    check_pair(a.len(), b.len())?;
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect())
}

/// Scalar multiplication. Returns new Vec.
#[cfg(feature = "std")]
pub fn vec_mul_scalar(a: &[f32], s: f32) -> Result<Vec<f32>, VectorError> {
    check_single(a.len())?;
    Ok(a.iter().map(|&x| x * s).collect())
}

// ============================================================================
// f64 operations
// ============================================================================

/// Cosine similarity (f64).
pub fn cosine_similarity_f64(a: &[f64], b: &[f64]) -> Result<f64, VectorError> {
    check_pair(a.len(), b.len())?;
    let dot = dispatch::dot_f64(a, b);
    let norm_a = dispatch::sum_squares_f64(a).sqrt();
    let norm_b = dispatch::sum_squares_f64(b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(VectorError::ZeroMagnitude);
    }
    Ok(dot / (norm_a * norm_b))
}

/// Dot product (f64).
pub fn dot_product_f64(a: &[f64], b: &[f64]) -> Result<f64, VectorError> {
    check_pair(a.len(), b.len())?;
    Ok(dispatch::dot_f64(a, b))
}

/// Euclidean (L2) distance (f64).
pub fn l2_distance_f64(a: &[f64], b: &[f64]) -> Result<f64, VectorError> {
    check_pair(a.len(), b.len())?;
    Ok(dispatch::l2_sq_f64(a, b).sqrt())
}

/// Inner product (f64).
pub fn inner_product_f64(a: &[f64], b: &[f64]) -> Result<f64, VectorError> {
    dot_product_f64(a, b)
}

/// L2 norm (f64).
pub fn l2_norm_f64(a: &[f64]) -> Result<f64, VectorError> {
    check_single(a.len())?;
    Ok(dispatch::sum_squares_f64(a).sqrt())
}

/// Normalize (f64). Returns new Vec.
#[cfg(feature = "std")]
pub fn normalize_f64(a: &[f64]) -> Result<Vec<f64>, VectorError> {
    check_single(a.len())?;
    let norm = dispatch::sum_squares_f64(a).sqrt();
    if norm == 0.0 {
        return Err(VectorError::ZeroMagnitude);
    }
    let inv = 1.0 / norm;
    Ok(a.iter().map(|&x| x * inv).collect())
}

/// Element-wise addition (f64).
#[cfg(feature = "std")]
pub fn vec_add_f64(a: &[f64], b: &[f64]) -> Result<Vec<f64>, VectorError> {
    check_pair(a.len(), b.len())?;
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
}

/// Element-wise subtraction (f64).
#[cfg(feature = "std")]
pub fn vec_sub_f64(a: &[f64], b: &[f64]) -> Result<Vec<f64>, VectorError> {
    check_pair(a.len(), b.len())?;
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect())
}

/// Scalar multiplication (f64).
#[cfg(feature = "std")]
pub fn vec_mul_scalar_f64(a: &[f64], s: f64) -> Result<Vec<f64>, VectorError> {
    check_single(a.len())?;
    Ok(a.iter().map(|&x| x * s).collect())
}
