//! C ABI (feature `ffi`), declared in `c-headers/lombokvector.h`.
//!
//! Build a C library with
//! `cargo rustc --release --features ffi --crate-type cdylib` (or `staticlib`).
//! Every function returns 0 on success or one of the `LOMBOKVECTOR_ERR_*` codes.

use core::ffi::c_char;
use core::slice;

use crate::error::VectorError;

/// Success.
pub const LOMBOKVECTOR_OK: i32 = 0;
/// `DIMENSION_MISMATCH` (not produced by these functions: both vectors share `len`).
pub const LOMBOKVECTOR_ERR_DIMENSION_MISMATCH: i32 = 1;
/// `EMPTY_VECTOR`: `len` is 0.
pub const LOMBOKVECTOR_ERR_EMPTY_VECTOR: i32 = 2;
/// `ZERO_MAGNITUDE`.
pub const LOMBOKVECTOR_ERR_ZERO_MAGNITUDE: i32 = 3;
/// `NON_FINITE`.
pub const LOMBOKVECTOR_ERR_NON_FINITE: i32 = 4;
/// A required pointer is null.
pub const LOMBOKVECTOR_ERR_NULL_POINTER: i32 = 5;

fn code(e: VectorError) -> i32 {
    match e {
        VectorError::DimensionMismatch { .. } => LOMBOKVECTOR_ERR_DIMENSION_MISMATCH,
        VectorError::EmptyVector => LOMBOKVECTOR_ERR_EMPTY_VECTOR,
        VectorError::ZeroMagnitude => LOMBOKVECTOR_ERR_ZERO_MAGNITUDE,
        VectorError::NonFinite => LOMBOKVECTOR_ERR_NON_FINITE,
    }
}

/// # Safety
/// `p` must be null or point to `len` readable values.
unsafe fn view<'a, T>(p: *const T, len: usize) -> Option<&'a [T]> {
    if p.is_null() {
        None
    } else if len == 0 {
        Some(&[])
    } else {
        // SAFETY: guaranteed by the caller.
        Some(unsafe { slice::from_raw_parts(p, len) })
    }
}

macro_rules! pair_fn {
    ($(#[$m:meta])* $name:ident, $t:ty, $f:path) => {
        $(#[$m])*
        ///
        /// # Safety
        /// `a` and `b` must point to `len` readable values and `out` to one writable value.
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const $t, b: *const $t, len: usize, out: *mut $t) -> i32 {
            // SAFETY: forwarded from the caller's contract.
            let (Some(a), Some(b)) = (unsafe { view(a, len) }, unsafe { view(b, len) }) else {
                return LOMBOKVECTOR_ERR_NULL_POINTER;
            };
            if out.is_null() {
                return LOMBOKVECTOR_ERR_NULL_POINTER;
            }
            match $f(a, b) {
                Ok(v) => {
                    // SAFETY: out is non-null and writable per the contract.
                    unsafe { out.write(v) };
                    LOMBOKVECTOR_OK
                }
                Err(e) => code(e),
            }
        }
    };
}

pair_fn!(/// Cosine similarity of two `float` vectors.
    lombokvector_cosine_f32, f32, crate::cosine_similarity);
pair_fn!(/// Cosine similarity of two `double` vectors.
    lombokvector_cosine_f64, f64, crate::cosine_similarity_f64);
pair_fn!(/// Dot product of two `float` vectors.
    lombokvector_dot_f32, f32, crate::dot_product);
pair_fn!(/// Dot product of two `double` vectors.
    lombokvector_dot_f64, f64, crate::dot_product_f64);
pair_fn!(/// Euclidean distance of two `float` vectors.
    lombokvector_l2_f32, f32, crate::l2_distance);
pair_fn!(/// Euclidean distance of two `double` vectors.
    lombokvector_l2_f64, f64, crate::l2_distance_f64);

/// Euclidean norm of a `float` vector.
///
/// # Safety
/// `a` must point to `len` readable values and `out` to one writable value.
#[no_mangle]
pub unsafe extern "C" fn lombokvector_l2_norm_f32(a: *const f32, len: usize, out: *mut f32) -> i32 {
    // SAFETY: forwarded from the caller's contract.
    let Some(a) = (unsafe { view(a, len) }) else {
        return LOMBOKVECTOR_ERR_NULL_POINTER;
    };
    if out.is_null() {
        return LOMBOKVECTOR_ERR_NULL_POINTER;
    }
    match crate::l2_norm(a) {
        Ok(v) => {
            // SAFETY: out is non-null and writable per the contract.
            unsafe { out.write(v) };
            LOMBOKVECTOR_OK
        }
        Err(e) => code(e),
    }
}

/// Scales a `float` vector to unit length in place; unchanged on error.
///
/// # Safety
/// `a` must point to `len` readable and writable values.
#[no_mangle]
pub unsafe extern "C" fn lombokvector_normalize_inplace_f32(a: *mut f32, len: usize) -> i32 {
    if a.is_null() {
        return LOMBOKVECTOR_ERR_NULL_POINTER;
    }
    let v: &mut [f32] = if len == 0 {
        &mut []
    } else {
        // SAFETY: guaranteed by the caller.
        unsafe { slice::from_raw_parts_mut(a, len) }
    };
    match crate::normalize_inplace(v) {
        Ok(()) => LOMBOKVECTOR_OK,
        Err(e) => code(e),
    }
}

/// Name of the active binary32 kernel as a static NUL-terminated string.
#[no_mangle]
pub extern "C" fn lombokvector_active_backend() -> *const c_char {
    let name: &'static [u8] = match crate::active_backend() {
        "avx2" => b"avx2\0",
        "neon" => b"neon\0",
        "wasm-simd128" => b"wasm-simd128\0",
        _ => b"portable\0",
    };
    name.as_ptr().cast::<c_char>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_through_the_c_abi() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let mut out = 0.0f32;
        unsafe {
            assert_eq!(
                lombokvector_dot_f32(a.as_ptr(), b.as_ptr(), 3, &mut out),
                LOMBOKVECTOR_OK
            );
            assert_eq!(out, 32.0);
            assert_eq!(
                lombokvector_l2_norm_f32([3.0f32, 4.0].as_ptr(), 2, &mut out),
                LOMBOKVECTOR_OK
            );
            assert_eq!(out, 5.0);
            assert_eq!(
                lombokvector_cosine_f32(a.as_ptr(), b.as_ptr(), 0, &mut out),
                LOMBOKVECTOR_ERR_EMPTY_VECTOR
            );
            assert_eq!(
                lombokvector_l2_f32(core::ptr::null(), b.as_ptr(), 3, &mut out),
                LOMBOKVECTOR_ERR_NULL_POINTER
            );
            assert_eq!(
                lombokvector_dot_f32(a.as_ptr(), b.as_ptr(), 3, core::ptr::null_mut()),
                LOMBOKVECTOR_ERR_NULL_POINTER
            );
            let z = [0.0f64; 2];
            let mut o64 = 0.0f64;
            assert_eq!(
                lombokvector_cosine_f64(z.as_ptr(), z.as_ptr(), 2, &mut o64),
                LOMBOKVECTOR_ERR_ZERO_MAGNITUDE
            );
            let big = [1e200f64, 1e200];
            assert_eq!(
                lombokvector_dot_f64(big.as_ptr(), big.as_ptr(), 2, &mut o64),
                LOMBOKVECTOR_ERR_NON_FINITE
            );
            assert_eq!(
                lombokvector_l2_f64(big.as_ptr(), z.as_ptr(), 2, &mut o64),
                LOMBOKVECTOR_ERR_NON_FINITE
            );
            let mut v = [3.0f32, 4.0];
            assert_eq!(
                lombokvector_normalize_inplace_f32(v.as_mut_ptr(), 2),
                LOMBOKVECTOR_OK
            );
            assert_eq!(v, [0.6, 0.8]);
            assert_eq!(
                lombokvector_normalize_inplace_f32(core::ptr::null_mut(), 2),
                LOMBOKVECTOR_ERR_NULL_POINTER
            );
            assert_eq!(
                lombokvector_normalize_inplace_f32(v.as_mut_ptr(), 0),
                LOMBOKVECTOR_ERR_EMPTY_VECTOR
            );
            let name = core::ffi::CStr::from_ptr(lombokvector_active_backend());
            assert_eq!(name.to_str().unwrap(), crate::active_backend());
        }
        assert_eq!(
            code(VectorError::DimensionMismatch {
                expected: 1,
                actual: 2
            }),
            1
        );
    }
}
