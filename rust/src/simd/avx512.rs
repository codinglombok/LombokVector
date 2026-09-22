// LombokVector — AVX-512 SIMD backend (x86_64, 512-bit, 16×f32)
// License: Apache-2.0
//
// NOTE: AVX-512 requires nightly Rust for core::arch intrinsics on stable.
// On stable Rust, this module provides a 2×AVX2 fallback that processes
// 16 floats per iteration using two __m256 registers.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;

/// AVX-512-style dot product using 2×AVX2 (16 floats per iteration).
///
/// # Safety
/// Caller must ensure AVX2+FMA is available.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;
    let remainder = len % 16;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        let va0 = _mm256_loadu_ps(a_ptr.add(offset));
        let vb0 = _mm256_loadu_ps(b_ptr.add(offset));
        let va1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(offset + 8));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }

    let combined = _mm256_add_ps(acc0, acc1);
    let mut sum = hsum_256(combined);

    let base = chunks * 16;
    for j in 0..remainder {
        sum += a[base + j] * b[base + j];
    }
    sum
}

/// AVX-512-style sum of squares using 2×AVX2.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sum_squares_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;
    let remainder = len % 16;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let a_ptr = a.as_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        let va0 = _mm256_loadu_ps(a_ptr.add(offset));
        let va1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
        acc0 = _mm256_fmadd_ps(va0, va0, acc0);
        acc1 = _mm256_fmadd_ps(va1, va1, acc1);
    }

    let combined = _mm256_add_ps(acc0, acc1);
    let mut sum = hsum_256(combined);
    let base = chunks * 16;
    for j in 0..remainder {
        sum += a[base + j] * a[base + j];
    }
    sum
}

/// AVX-512-style L2 squared distance using 2×AVX2.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;
    let remainder = len % 16;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        let va0 = _mm256_loadu_ps(a_ptr.add(offset));
        let vb0 = _mm256_loadu_ps(b_ptr.add(offset));
        let va1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(offset + 8));
        let d0 = _mm256_sub_ps(va0, vb0);
        let d1 = _mm256_sub_ps(va1, vb1);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }

    let combined = _mm256_add_ps(acc0, acc1);
    let mut sum = hsum_256(combined);
    let base = chunks * 16;
    for j in 0..remainder {
        let d = a[base + j] - b[base + j];
        sum += d * d;
    }
    sum
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_256(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}
