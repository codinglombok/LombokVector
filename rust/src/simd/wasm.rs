// LombokVector — WebAssembly SIMD backend (128-bit, 4×f32)
// License: Apache-2.0

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// WASM SIMD dot product (f32). Processes 4 floats per iteration.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = f32x4_splat(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = v128_load(a_ptr.add(offset) as *const v128);
        let vb = v128_load(b_ptr.add(offset) as *const v128);
        let prod = f32x4_mul(va, vb);
        acc = f32x4_add(acc, prod);
    }

    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);

    let base = chunks * 4;
    for j in 0..remainder {
        sum += a[base + j] * b[base + j];
    }
    sum
}

/// WASM SIMD sum of squares (f32).
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn sum_squares_f32(a: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = f32x4_splat(0.0);
    let a_ptr = a.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = v128_load(a_ptr.add(offset) as *const v128);
        let prod = f32x4_mul(va, va);
        acc = f32x4_add(acc, prod);
    }

    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);

    let base = chunks * 4;
    for j in 0..remainder {
        sum += a[base + j] * a[base + j];
    }
    sum
}

/// WASM SIMD L2 squared distance (f32).
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = f32x4_splat(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = v128_load(a_ptr.add(offset) as *const v128);
        let vb = v128_load(b_ptr.add(offset) as *const v128);
        let diff = f32x4_sub(va, vb);
        let sq = f32x4_mul(diff, diff);
        acc = f32x4_add(acc, sq);
    }

    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);

    let base = chunks * 4;
    for j in 0..remainder {
        let d = a[base + j] - b[base + j];
        sum += d * d;
    }
    sum
}
