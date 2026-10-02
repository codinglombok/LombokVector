//! NEON kernels (aarch64): two 128-bit registers hold lanes 0-3 and 4-7.
//! `vmulq_f32` and `vaddq_f32` round separately (never `vfmaq_f32`).

use core::arch::aarch64::*;

use super::{tail_into, Kind};
use crate::kernels::{combine_f32, LANES};

#[inline]
#[target_feature(enable = "neon")]
unsafe fn run(a: &[f32], b: &[f32], kind: Kind) -> f32 {
    // SAFETY: the intrinsics need the target feature, which the callers guarantee
    // (runtime detection or compile-time cfg); Rust before 1.87 treats them as
    // unsafe even inside a #[target_feature] function.
    #[allow(unused_unsafe)]
    unsafe {
        let n = a.len() - a.len() % LANES;
        let mut lo = vdupq_n_f32(0.0);
        let mut hi = vdupq_n_f32(0.0);
        let mut i = 0;
        while i < n {
            // SAFETY: i + 8 <= n <= a.len() and b.len() >= a.len() (checked by callers).
            let (a0, a1) = unsafe {
                (
                    vld1q_f32(a.as_ptr().add(i)),
                    vld1q_f32(a.as_ptr().add(i + 4)),
                )
            };
            let (p0, p1) = match kind {
                Kind::Dot => {
                    let (b0, b1) = unsafe {
                        (
                            vld1q_f32(b.as_ptr().add(i)),
                            vld1q_f32(b.as_ptr().add(i + 4)),
                        )
                    };
                    (vmulq_f32(a0, b0), vmulq_f32(a1, b1))
                }
                Kind::SumSq => (vmulq_f32(a0, a0), vmulq_f32(a1, a1)),
                Kind::L2Sq => {
                    let (b0, b1) = unsafe {
                        (
                            vld1q_f32(b.as_ptr().add(i)),
                            vld1q_f32(b.as_ptr().add(i + 4)),
                        )
                    };
                    let (d0, d1) = (vsubq_f32(a0, b0), vsubq_f32(a1, b1));
                    (vmulq_f32(d0, d0), vmulq_f32(d1, d1))
                }
            };
            lo = vaddq_f32(lo, p0);
            hi = vaddq_f32(hi, p1);
            i += LANES;
        }
        let mut s = [0.0f32; LANES];
        // SAFETY: s has room for eight f32 values.
        unsafe {
            vst1q_f32(s.as_mut_ptr(), lo);
            vst1q_f32(s.as_mut_ptr().add(4), hi);
        }
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
/// `b.len() >= a.len()`. NEON is always present on aarch64.
pub(crate) unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { run(a, b, Kind::Dot) }
}

/// # Safety
/// NEON is always present on aarch64.
pub(crate) unsafe fn sumsq_f32(a: &[f32]) -> f32 {
    unsafe { run(a, a, Kind::SumSq) }
}

/// # Safety
/// `b.len() >= a.len()`. NEON is always present on aarch64.
pub(crate) unsafe fn l2sq_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { run(a, b, Kind::L2Sq) }
}
