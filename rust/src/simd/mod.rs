//! SIMD kernels for binary32. Each one keeps eight lane sums, multiplies and
//! adds with separate rounding (no fused multiply-add), adds the tail into
//! lanes 0..n%8, and combines the lanes with `kernels::combine_f32`, so its
//! result is bit-identical to the reference kernel (SPEC section 2.3).
//!
//! Which kernel is reachable depends on the target and features (for example
//! none with `portable`), so unused-code warnings are silenced here.
#![allow(dead_code)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod avx2;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod wasm;

/// Adds the terms of the tail (fewer than 8 elements) into the low lanes.
#[inline(always)]
pub(crate) fn tail_into(s: &mut [f32; 8], ra: &[f32], rb: &[f32], kind: Kind) {
    for k in 0..ra.len() {
        let p = match kind {
            Kind::Dot => ra[k] * rb[k],
            Kind::SumSq => ra[k] * ra[k],
            Kind::L2Sq => {
                let d = ra[k] - rb[k];
                d * d
            }
        };
        s[k] += p;
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // not every kind is used on every target
pub(crate) enum Kind {
    Dot,
    SumSq,
    L2Sq,
}
