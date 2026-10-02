//! Chooses a kernel. Every choice returns the same bits (SPEC section 2.3);
//! only the speed differs.

use crate::kernels;

/// Name of the binary32 kernel used on this machine.
pub fn active_backend() -> &'static str {
    #[cfg(all(
        feature = "std",
        any(target_arch = "x86", target_arch = "x86_64"),
        not(feature = "portable")
    ))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    #[cfg(all(
        not(feature = "std"),
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2",
        not(feature = "portable")
    ))]
    {
        return "avx2";
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "portable")))]
    {
        return "neon";
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "portable")
    ))]
    {
        return "wasm-simd128";
    }
    #[allow(unreachable_code)]
    "portable"
}

macro_rules! dispatch_f32 {
    ($name:ident, ($($arg:ident),*)) => {
        #[inline]
        pub(crate) fn $name($($arg: &[f32]),*) -> f32 {
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"), not(feature = "portable")))]
            {
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: AVX2 was detected; callers checked equal lengths.
                    return unsafe { crate::simd::avx2::$name($($arg),*) };
                }
            }
            #[cfg(all(
                not(feature = "std"),
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "avx2",
                not(feature = "portable")
            ))]
            {
                // SAFETY: the crate is compiled with AVX2 enabled; callers checked equal lengths.
                return unsafe { crate::simd::avx2::$name($($arg),*) };
            }
            #[cfg(all(target_arch = "aarch64", not(feature = "portable")))]
            {
                // SAFETY: NEON is part of the aarch64 baseline; callers checked equal lengths.
                return unsafe { crate::simd::neon::$name($($arg),*) };
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "portable")))]
            {
                // SAFETY: simd128 is enabled at compile time; callers checked equal lengths.
                return unsafe { crate::simd::wasm::$name($($arg),*) };
            }
            #[allow(unreachable_code)]
            kernels::$name($($arg),*)
        }
    };
}

dispatch_f32!(dot_f32, (a, b));
dispatch_f32!(sumsq_f32, (a));
dispatch_f32!(l2sq_f32, (a, b));

pub(crate) use kernels::{dot_f64, l2sq_f64, sumsq_f64};
