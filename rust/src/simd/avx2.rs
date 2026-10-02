//! AVX2 kernels (x86, x86_64): one 256-bit register holds the eight lanes.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{tail_into, Kind};
use crate::kernels::{combine_f32, LANES};

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn run(a: &[f32], b: &[f32], kind: Kind) -> f32 {
    // SAFETY: the intrinsics need the target feature, which the callers guarantee
    // (runtime detection or compile-time cfg); Rust before 1.87 treats them as
    // unsafe even inside a #[target_feature] function.
    #[allow(unused_unsafe)]
    unsafe {
        let n = a.len() - a.len() % LANES;
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < n {
            // SAFETY: i + 8 <= n <= a.len() and b.len() >= a.len() (checked by callers);
            // loadu has no alignment requirement.
            let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
            let p = match kind {
                Kind::Dot => {
                    let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
                    _mm256_mul_ps(va, vb)
                }
                Kind::SumSq => _mm256_mul_ps(va, va),
                Kind::L2Sq => {
                    let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
                    let d = _mm256_sub_ps(va, vb);
                    _mm256_mul_ps(d, d)
                }
            };
            acc = _mm256_add_ps(acc, p);
            i += LANES;
        }
        let mut s = [0.0f32; LANES];
        // SAFETY: s has room for eight f32 values; storeu has no alignment requirement.
        unsafe { _mm256_storeu_ps(s.as_mut_ptr(), acc) };
        let rb = if matches!(kind, Kind::SumSq) {
            &a[n..]
        } else {
            &b[n..a.len()]
        };
        tail_into(&mut s, &a[n..], rb, kind);
        combine_f32(&s)
    }
}

/// # Safety
/// The CPU must support AVX2, and `b.len() >= a.len()`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { run(a, b, Kind::Dot) }
}

/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn sumsq_f32(a: &[f32]) -> f32 {
    unsafe { run(a, a, Kind::SumSq) }
}

/// # Safety
/// The CPU must support AVX2, and `b.len() >= a.len()`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn l2sq_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { run(a, b, Kind::L2Sq) }
}
