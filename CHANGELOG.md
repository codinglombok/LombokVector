# Changelog

All notable changes to **LombokVector** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Container release channel: root `Dockerfile` (multi-stage; builds the C FFI
  shared library with `--features ffi` and ships it in `debian:bookworm-slim`
  with `lombokvector.h` and a self-verifying smoke test as the entrypoint),
  `.dockerignore`, `.github/workflows/release-docker.yml` (staged dispatch,
  `dry_run` default `true`, fail-closed tag guard), a `docker` job in
  `release.yml`, and a build-plus-smoke job in `ci.yml`. Images are published
  to `ghcr.io/codinglombok/lombokvector` (multi-arch amd64/arm64); Docker Hub
  is pushed too when `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets exist.
- `publishConfig` (`access: public`, `provenance: true`) in
  `typescript/package.json`, so a local `npm publish` behaves exactly like
  the CI publish.
- Root `composer.json` (moved from `php/`, paths adjusted: `psr-4` -> `php/src/`,
  `composer test` -> `php php/tests/run.php`) so Packagist — which only reads
  the root `composer.json` of the default branch — can index the package
  `codinglombok/lombokvector` after a one-time submission.
- Root `package.json` (private npm workspace pointing at `typescript/`) and
  root `package-lock.json` for one-command workspace installs; the publishable
  package and its lockfile stay in `typescript/` (CI flows unchanged).

## [0.1.0] — 2026-10-23

First version published to any registry (npm, crates.io, PyPI, GitHub
Releases; Go through the `go/v0.1.0` module tag). The published version
ladder starts here and continues in staged steps: `v0.2.0`, `v0.3.0`, ...
(see `docs/RELEASE_RUNBOOK.md`).

This release consolidates the two earlier internal drafts that were never
published to any registry — the 2026-09-22 draft and the 2026-10-02 draft —
into a single release, so no published version number was skipped. It brings
the library in line with the Lombok Ecosystem v3.6 standards: one normative
specification, results that are bit-identical in every port, and claims that
match the code.

### Added

- `docs/SPEC_LombokVector_v0.1.0.md`: the 8-lane summation order (separate
  rounding, no fused multiply-add) and the exact semantics of every operation
  and error.
- `vectors/lombokvector-vectors-v1.json`: 223 cases (113 binary64, 110
  binary32) with expected IEEE 754 bit patterns from an independent Python
  reference, each checked against exact rational arithmetic; runners in all
  five ports.
- `NON_FINITE` error for non-finite input and overflow, in every port.
- Rust: `batch_*_f64`, `distance_matrix_cosine_f64`, `normalize_inplace_f64`,
  `VectorError::code`, correctly rounded software `sqrt` for `no_std`, and a
  test that the SIMD kernels match the scalar reference bit for bit.
- Rust `ffi` feature implementing `c-headers/lombokvector.h`, with a C smoke
  test in CI.
- Go: binary32 API (`*F32`), `errors.Is` support, `BatchDot`,
  `DistanceMatrixCosine`.
- Python: `ScoredIndex`, `py.typed`; TypeScript: `batchDot`, typed-array
  input.
- Ten standard documents under `docs/`, `scripts/lombok-doctor.sh`, CI for
  all ports on Linux, Windows and macOS (Apple Silicon runs the NEON kernel).
- Staged release automation: `release.yml` gains a `go-tag` job (automated,
  idempotent `go/v*` module tag) and a `github-release` job (GitHub Release
  with a checksummed source bundle built by `git archive`); the crates.io,
  npm and PyPI jobs declare environments (`crates`, `npm`, `pypi`,
  `github-release`) so releases can be gated registry by registry.
- Per-registry dispatch workflows `release-npm.yml`, `release-crates.yml`,
  `release-pypi.yml` with `dry_run` (default `true`) for staged rehearsal and
  emergency single-registry publishing.
- `docs/RELEASE_RUNBOOK.md`: the staged release commands (version ladder,
  playbooks, verification, version bumps, rollback) and
  `scripts/bump_version.sh` for staged version increases.

### Changed

- Every port sums in the 8-lane order of SPEC section 2; the unpublished
  internal drafts used different orders per port (4-way unrolling, FMA in
  AVX2), so results differed between languages and backends.
- Cosine similarity is clamped to [-1, 1].
- Batch results use a stable sort: equal scores keep index order.
- `distance_matrix_cosine` checks that all vectors have the same length.
- Rust AVX2 kernel no longer requires FMA.
- Go module path is now `github.com/codinglombok/lombokvector/go`.
- TypeScript port moved from `ts/` to `typescript/`; PHP tests no longer need
  PHPUnit.

### Fixed

- PHP: `VectorError` redeclared `Exception::$code` as readonly, so the
  package could not be loaded at all.
- Rust: `cargo build --no-default-features` failed (no_std build broken);
  clippy and rustfmt failures.
- Rust: the benchmark called a function that does not exist.
- C header claimed to be generated from Rust, but the crate had no C
  functions.

### Removed

- AVX-512 module (never called by the dispatcher) and the `avx2`, `avx512`,
  `neon`, `wasm-simd`, `auto-detect` Cargo features (they had no effect).
- README claims that tied the library to one application domain.
- Weekly audit and benchmark workflows with unpinned actions (CI now runs
  `npm pack`, coverage and the standards check).
