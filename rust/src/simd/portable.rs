// LombokVector — Portable scalar SIMD fallback
// Works on ALL targets including no_std embedded
// License: Apache-2.0

/// Scalar dot product (f32).
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let len = a.len();
    // Manual unroll by 4 for better ILP
    let chunks = len / 4;
    let remainder = len % 4;
    let mut i = 0;
    for _ in 0..chunks {
        sum += a[i] * b[i]
            + a[i + 1] * b[i + 1]
            + a[i + 2] * b[i + 2]
            + a[i + 3] * b[i + 3];
        i += 4;
    }
    for j in 0..remainder {
        sum += a[i + j] * b[i + j];
    }
    sum
}

/// Scalar dot product (f64).
#[inline]
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let mut i = 0;
    for _ in 0..chunks {
        sum += a[i] * b[i]
            + a[i + 1] * b[i + 1]
            + a[i + 2] * b[i + 2]
            + a[i + 3] * b[i + 3];
        i += 4;
    }
    for j in 0..remainder {
        sum += a[i + j] * b[i + j];
    }
    sum
}

/// Scalar sum of squares (f32) — used for L2 norm.
#[inline]
pub fn sum_squares_f32(a: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let mut i = 0;
    for _ in 0..chunks {
        sum += a[i] * a[i]
            + a[i + 1] * a[i + 1]
            + a[i + 2] * a[i + 2]
            + a[i + 3] * a[i + 3];
        i += 4;
    }
    for j in 0..remainder {
        sum += a[i + j] * a[i + j];
    }
    sum
}

/// Scalar sum of squares (f64).
#[inline]
pub fn sum_squares_f64(a: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let mut i = 0;
    for _ in 0..chunks {
        sum += a[i] * a[i]
            + a[i + 1] * a[i + 1]
            + a[i + 2] * a[i + 2]
            + a[i + 3] * a[i + 3];
        i += 4;
    }
    for j in 0..remainder {
        sum += a[i + j] * a[i + j];
    }
    sum
}

/// Scalar L2 squared distance (f32).
#[inline]
pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let mut i = 0;
    for _ in 0..chunks {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    for j in 0..remainder {
        let d = a[i + j] - b[i + j];
        sum += d * d;
    }
    sum
}

/// Scalar L2 squared distance (f64).
#[inline]
pub fn l2_sq_f64(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let mut i = 0;
    for _ in 0..chunks {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    for j in 0..remainder {
        let d = a[i + j] - b[i + j];
        sum += d * d;
    }
    sum
}
