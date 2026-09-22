// LombokVector — ARM NEON SIMD backend (aarch64, 128-bit, 4×f32)
// License: Apache-2.0

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// NEON dot product (f32). Processes 4 floats per iteration.
///
/// # Safety
/// Caller must be on aarch64 (NEON is always available on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut sum = vaddvq_f32(acc);
    let base = chunks * 4;
    for j in 0..remainder {
        sum += a[base + j] * b[base + j];
    }
    sum
}

/// NEON sum of squares (f32).
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn sum_squares_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    let a_ptr = a.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        acc = vfmaq_f32(acc, va, va);
    }

    let mut sum = vaddvq_f32(acc);
    let base = chunks * 4;
    for j in 0..remainder {
        sum += a[base + j] * a[base + j];
    }
    sum
}

/// NEON L2 squared distance (f32).
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        let diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }

    let mut sum = vaddvq_f32(acc);
    let base = chunks * 4;
    for j in 0..remainder {
        let d = a[base + j] - b[base + j];
        sum += d * d;
    }
    sum
}
