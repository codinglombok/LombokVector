# Changelog

All notable changes to LombokVector will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-XX-XX

### Added

- **Core operations**: cosine similarity, dot product, L2 distance, inner product, L2 norm, normalize
- **Arithmetic**: vector addition, subtraction, scalar multiplication
- **Batch operations**: batch cosine, batch dot, batch L2 (sorted results with indices)
- **Distance matrix**: N×N cosine distance matrix computation
- **SIMD backends**:
  - AVX2+FMA (8×f32 per iteration, x86_64)
  - AVX-512 via 2×AVX2 (16×f32 per iteration, x86_64)
  - ARM NEON (4×f32 per iteration, aarch64)
  - WASM SIMD128 (4×f32 per iteration, browser/WASM runtime)
  - Portable scalar fallback (4× loop unrolled, works everywhere)
- **Runtime CPUID detection** on x86_64 (auto-select best SIMD backend)
- **`no_std` support** for Rust core (zero heap allocations in hot path)
- **f32 and f64 precision** variants for all operations
- **Language ports**:
  - Rust (primary, `no_std` compatible)
  - TypeScript (pure, zero-dep, ESM)
  - Python (pure, zero-dep, 3.9+)
  - Go (pure, 1.21+)
  - PHP (pure, 8.1+, PSR-4)
- **C FFI headers** (cbindgen-compatible)
- **Shared test vectors** JSON (cross-language validation)
- **CI/CD**:
  - Multi-language CI (Rust 3 OS + no_std + WASM, TS Node 18/20/22, Python 3.9-3.13, Go 1.21-1.23, PHP 8.1-8.4)
  - Tag-triggered release to crates.io, npm, GPR, PyPI, Packagist, GitHub Release
  - Weekly security audit (cargo-audit, npm audit, CodeQL)
  - Criterion benchmark tracking on gh-pages

[0.1.0]: https://github.com/codinglombok/LombokVector/releases/tag/v0.1.0
