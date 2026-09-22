// LombokVector — SIMD module
// License: Apache-2.0

pub mod dispatch;
pub mod portable;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx512;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
