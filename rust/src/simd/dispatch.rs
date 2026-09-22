// LombokVector — SIMD dispatch (runtime detection on x86_64)
// License: Apache-2.0

use super::portable;

/// Returns the name of the active SIMD backend.
pub fn active_backend_name() -> &'static str {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "auto-detect"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "avx2";
        }
        return "portable";
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "portable")))]
    {
        return "neon";
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "portable")))]
    {
        return "wasm-simd";
    }

    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), feature = "auto-detect"),
        all(target_arch = "aarch64", not(feature = "portable")),
        all(target_arch = "wasm32", not(feature = "portable")),
    )))]
    {
        "portable"
    }
}

// ============================================================================
// Dispatch functions — route to the best available SIMD backend
// ============================================================================

/// Dispatched dot product (f32).
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "auto-detect"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { super::avx2::dot_f32(a, b) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "portable")))]
    {
        return unsafe { super::neon::dot_f32(a, b) };
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "portable")))]
    {
        return unsafe { super::wasm::dot_f32(a, b) };
    }

    // Fallback: portable scalar
    #[allow(unreachable_code)]
    portable::dot_f32(a, b)
}

/// Dispatched dot product (f64). Always portable (SIMD f64 has less benefit).
#[inline]
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    portable::dot_f64(a, b)
}

/// Dispatched sum of squares (f32).
#[inline]
pub fn sum_squares_f32(a: &[f32]) -> f32 {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "auto-detect"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { super::avx2::sum_squares_f32(a) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "portable")))]
    {
        return unsafe { super::neon::sum_squares_f32(a) };
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "portable")))]
    {
        return unsafe { super::wasm::sum_squares_f32(a) };
    }

    #[allow(unreachable_code)]
    portable::sum_squares_f32(a)
}

/// Dispatched sum of squares (f64).
#[inline]
pub fn sum_squares_f64(a: &[f64]) -> f64 {
    portable::sum_squares_f64(a)
}

/// Dispatched L2 squared distance (f32).
#[inline]
pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "auto-detect"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { super::avx2::l2_sq_f32(a, b) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "portable")))]
    {
        return unsafe { super::neon::l2_sq_f32(a, b) };
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "portable")))]
    {
        return unsafe { super::wasm::l2_sq_f32(a, b) };
    }

    #[allow(unreachable_code)]
    portable::l2_sq_f32(a, b)
}

/// Dispatched L2 squared distance (f64).
#[inline]
pub fn l2_sq_f64(a: &[f64], b: &[f64]) -> f64 {
    portable::l2_sq_f64(a, b)
}
