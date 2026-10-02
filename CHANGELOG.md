# Changelog

All notable changes to **LombokVector** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.2.0] — 2026-10-02

Brings the library in line with the Lombok Ecosystem v3.6 standards: one
normative specification, results that are bit-identical in every port, and
claims that match the code. Numeric results change in the last bits (0.x release).

### Added
- `docs/SPEC_LombokVector_v0.2.0.md`: the 8-lane summation order (separate rounding, no fused multiply-add) and the exact semantics of every operation and error.
- `vectors/lombokvector-vectors-v1.json`: 223 cases (113 binary64, 110 binary32) with expected IEEE 754 bit patterns from an independent Python reference, each checked against exact rational arithmetic; runners in all five ports.
- `NON_FINITE` error for non-finite input and overflow, in every port.
- Rust: `batch_*_f64`, `distance_matrix_cosine_f64`, `normalize_inplace_f64`, `VectorError::code`, correctly rounded software `sqrt` for `no_std`, test that the SIMD kernels match the scalar reference bit for bit.
- Rust `ffi` feature implementing `c-headers/lombokvector.h`, with a C smoke test in CI.
- Go: binary32 API (`*F32`), `errors.Is` support, `BatchDot`, `DistanceMatrixCosine`.
- Python: `ScoredIndex`, `py.typed`; TypeScript: `batchDot`, typed-array input.
- Ten standard documents under `docs/`, `scripts/lombok-doctor.sh`, CI for all ports on Linux, Windows and macOS (Apple Silicon runs the NEON kernel).

### Changed
- Every port sums in the 8-lane order of SPEC section 2; 0.1.0 used different orders per port (4-way unrolling, FMA in AVX2), so results differed between languages and backends.
- Cosine similarity is clamped to [-1, 1].
- Batch results use a stable sort: equal scores keep index order.
- `distance_matrix_cosine` checks that all vectors have the same length.
- Rust AVX2 kernel no longer requires FMA.
- Go module path is now `github.com/codinglombok/lombokvector/go`.
- TypeScript port moved from `ts/` to `typescript/`; PHP tests no longer need PHPUnit.

### Fixed
- PHP: `VectorError` redeclared `Exception::$code` as readonly, so the package could not be loaded at all.
- Rust: `cargo build --no-default-features` failed (no_std build broken); clippy and rustfmt failures.
- Rust: the benchmark called a function that does not exist.
- C header claimed to be generated from Rust, but the crate had no C functions.

### Removed
- AVX-512 module (never called by the dispatcher) and the `avx2`, `avx512`, `neon`, `wasm-simd`, `auto-detect` Cargo features (they had no effect).
- README claims that tied the library to one application domain.
- Weekly audit and benchmark workflows with unpinned actions (CI now runs `npm pack`, coverage and the standards check).

## [0.1.0] — 2026-09-22

Initial version in Rust, TypeScript, Python, Go and PHP (not published to any registry).
