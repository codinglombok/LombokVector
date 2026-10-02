//! WebAssembly SIMD kernels (simd128): two v128 registers hold lanes 0-3 and 4-7.

use core::arch::wasm32::*;

use super::{tail_into, Kind};
use crate::kernels::{combine_f32, LANES};

#[inline]
unsafe fn load(p: *const f32) -> v128 {
    // SAFETY: callers pass a pointer with at least four readable f32 values.
    unsafe { v128_load(p as *const v128) }
}

unsafe fn run(a: &[f32], b: &[f32], kind: Kind) -> f32 {
    // SAFETY: the intrinsics need the target feature, which the callers guarantee
    // (runtime detection or compile-time cfg); Rust before 1.87 treats them as
    // unsafe even inside a #[target_feature] function.
    #[allow(unused_unsafe)]
    unsafe {
        let n = a.len() - a.len() % LANES;
        let mut lo = f32x4_splat(0.0);
        let mut hi = f32x4_splat(0.0);
        let mut i = 0;
        while i < n {
            // SAFETY: i + 8 <= n <= a.len() and b.len() >= a.len() (checked by callers).
            let (a0, a1) = unsafe { (load(a.as_ptr().add(i)), load(a.as_ptr().add(i + 4))) };
            let (p0, p1) = match kind {
                Kind::Dot => {
                    let (b0, b1) =
                        unsafe { (load(b.as_ptr().add(i)), load(b.as_ptr().add(i + 4))) };
                    (f32x4_mul(a0, b0), f32x4_mul(a1, b1))
                }
                Kind::SumSq => (f32x4_mul(a0, a0), f32x4_mul(a1, a1)),
                Kind::L2Sq => {
                    let (b0, b1) =
                        unsafe { (load(b.as_ptr().add(i)), load(b.as_ptr().add(i + 4))) };
                    let (d0, d1) = (f32x4_sub(a0, b0), f32x4_sub(a1, b1));
                    (f32x4_mul(d0, d0), f32x4_mul(d1, d1))
                }
            };
            lo = f32x4_add(lo, p0);
            hi = f32x4_add(hi, p1);
            i += LANES;
        }
        let s0 = [
            f32x4_extract_lane::<0>(lo),
            f32x4_extract_lane::<1>(lo),
            f32x4_extract_lane::<2>(lo),
            f32x4_extract_lane::<3>(lo),
            f32x4_extract_lane::<0>(hi),
            f32x4_extract_lane::<1>(hi),
            f32x4_extract_lane::<2>(hi),
            f32x4_extract_lane::<3>(hi),
        ];
        let mut s = s0;
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
/// `b.len() >= a.len()`.
pub(crate) unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { run(a, b, Kind::Dot) }
}

/// # Safety
/// Always safe on targets with simd128; `unsafe` only for a uniform signature.
pub(crate) unsafe fn sumsq_f32(a: &[f32]) -> f32 {
    unsafe { run(a, a, Kind::SumSq) }
}

/// # Safety
/// `b.len() >= a.len()`.
pub(crate) unsafe fn l2sq_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { run(a, b, Kind::L2Sq) }
}
