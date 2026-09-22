// LombokVector — SIMD-optimized vector math for RAG systems
// Part of LombokRAGFrameworks (Tier 4 Hub) — Lombok Ecosystem (@codinglombok)
// License: Apache-2.0

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::excessive_precision)]

//! # LombokVector
//!
//! Zero-dependency, `no_std`-compatible SIMD vector math library.
//! Provides cosine similarity, dot product, L2 distance, normalization,
//! and batch operations optimized for RAG embedding workloads.
//!
//! ## SIMD Backends
//!
//! - `avx2` — x86_64 AVX2 (256-bit, 8×f32)
//! - `avx512` — x86_64 AVX-512 (512-bit, 16×f32)
//! - `neon` — ARM NEON (128-bit, 4×f32)
//! - `wasm-simd` — WebAssembly SIMD (128-bit, 4×f32)
//! - `portable` — scalar fallback (any target)
//! - `auto-detect` — runtime CPUID detection on x86_64 (default)
//!
//! ## Example
//!
//! ```rust
//! use lombokvector::{cosine_similarity, dot_product, l2_distance, normalize};
//!
//! let a = &[1.0_f32, 2.0, 3.0];
//! let b = &[4.0_f32, 5.0, 6.0];
//!
//! let cos = cosine_similarity(a, b);
//! let dot = dot_product(a, b);
//! let l2 = l2_distance(a, b);
//! let norm = normalize(a);
//! ```

pub mod error;
pub mod ops;
pub mod simd;

#[cfg(feature = "std")]
pub mod batch;

pub use error::VectorError;
pub use ops::{
    cosine_similarity, cosine_similarity_f64, dot_product, dot_product_f64, inner_product,
    inner_product_f64, l2_distance, l2_distance_f64, l2_norm, l2_norm_f64, normalize,
    normalize_f64, vec_add, vec_add_f64, vec_mul_scalar, vec_mul_scalar_f64, vec_sub,
    vec_sub_f64,
};

#[cfg(feature = "std")]
pub use batch::{batch_cosine, batch_dot, batch_l2, distance_matrix_cosine};

/// Returns the active SIMD backend name at runtime.
pub fn active_backend() -> &'static str {
    simd::dispatch::active_backend_name()
}
