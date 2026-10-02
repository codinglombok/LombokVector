//! LombokVector: vector math whose results are bit-identical across Rust,
//! TypeScript, Python, Go and PHP.
//!
//! Every sum uses the fixed 8-lane order of SPEC section 2 (multiply, round,
//! add, round; no fused multiply-add). The SIMD kernels (AVX2 on x86/x86_64,
//! NEON on aarch64, simd128 on wasm32) follow the same order, so they are
//! faster without changing a single bit of the result.
//!
//! ```
//! use lombokvector::{cosine_similarity, dot_product, l2_distance, normalize};
//!
//! let a = [1.0_f32, 2.0, 3.0];
//! let b = [4.0_f32, 5.0, 6.0];
//! assert_eq!(dot_product(&a, &b).unwrap(), 32.0);
//! assert_eq!(l2_distance(&a, &b).unwrap(), 27.0_f32.sqrt());
//! let c = cosine_similarity(&a, &b).unwrap();
//! assert!((c - 0.974_631_8).abs() < 1e-6);
//! assert_eq!(normalize(&[3.0_f32, 4.0]).unwrap(), vec![0.6, 0.8]);
//! ```
//!
//! # Features
//!
//! - `std` (default): runtime AVX2 detection, `std::error::Error`. Implies `alloc`.
//! - `alloc`: functions that return `Vec`. Without it the crate is `no_std`
//!   without an allocator and offers the scalar results and `normalize_inplace`.
//! - `portable`: always use the scalar kernels (same results).
//! - `ffi`: C functions declared in `c-headers/lombokvector.h`.
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod dispatch;
mod error;
mod kernels;
mod ops;
mod simd;
mod sqrt;

#[cfg(feature = "alloc")]
mod batch;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use dispatch::active_backend;
pub use error::VectorError;
pub use ops::{
    cosine_similarity, cosine_similarity_f64, dot_product, dot_product_f64, inner_product,
    inner_product_f64, l2_distance, l2_distance_f64, l2_norm, l2_norm_f64, normalize_inplace,
    normalize_inplace_f64,
};

#[cfg(feature = "alloc")]
pub use ops::{
    normalize, normalize_f64, vec_add, vec_add_f64, vec_mul_scalar, vec_mul_scalar_f64, vec_sub,
    vec_sub_f64,
};

#[cfg(feature = "alloc")]
pub use batch::{
    batch_cosine, batch_cosine_f64, batch_dot, batch_dot_f64, batch_l2, batch_l2_f64,
    distance_matrix_cosine, distance_matrix_cosine_f64,
};

/// The scalar reference kernels, for checking that a SIMD kernel matches bit for bit.
#[doc(hidden)]
pub mod reference {
    pub use crate::kernels::{dot_f32, dot_f64, l2sq_f32, l2sq_f64, sumsq_f32, sumsq_f64};
}
