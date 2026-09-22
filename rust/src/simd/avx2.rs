// LombokVector — AVX2 SIMD backend (x86_64, 256-bit, 8×f32)
// License: Apache-2.0

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;

/// AVX2 dot product (f32). Processes 8 floats per iteration.
///
/// # Safety
/// Caller must ensure AVX2 is available (checked via `is_x86_feature_detected!`).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    // Horizontal sum of 8 floats in acc
    let mut sum = hsum_256(acc);

    // Handle remainder
    let base = chunks * 8;
    for j in 0..remainder {
        sum += a[base + j] * b[base + j];
    }

    sum
}

/// AVX2 sum of squares (f32).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sum_squares_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();
    let a_ptr = a.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        acc = _mm256_fmadd_ps(va, va, acc);
    }

    let mut sum = hsum_256(acc);
    let base = chunks * 8;
    for j in 0..remainder {
        sum += a[base + j] * a[base + j];
    }
    sum
}

/// AVX2 L2 squared distance (f32).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }

    let mut sum = hsum_256(acc);
    let base = chunks * 8;
    for j in 0..remainder {
        let d = a[base + j] - b[base + j];
        sum += d * d;
    }
    sum
}

/// Horizontal sum of __m256 (8×f32 → 1×f32).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_256(v: __m256) -> f32 {
    // [a0+a4, a1+a5, a2+a6, a3+a7]
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    // [s0+s2, s1+s3, ?, ?]
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}
