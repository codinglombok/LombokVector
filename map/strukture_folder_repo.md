# LombokVector — Folder Structure Map

> Complete repository structure for LombokVector.
> This layout is the **TEMPLATE** for all 39 libraries in LombokRAGFrameworks.

---

## Repository Root

```
LombokVector/
│
├── README.md                           # Main README (install, API, benchmarks)
├── LICENSE                             # Apache-2.0
├── CHANGELOG.md                        # Release changelog (Keep a Changelog format)
├── architecture_repo.md                # Architecture documentation
├── masterplan_repo.md                  # Release & distribution plan
├── .gitignore                          # Multi-language gitignore
├── .editorconfig                       # Cross-language editor config
│
├── map/
│   └── strukture_folder_repo.md        # This file — folder structure map
│
├── vectors/
│   └── lombokvector-vectors-v1.json    # Shared test vectors (all ports validate against this)
│
├── .github/
│   └── workflows/
│       ├── ci.yml                      # Multi-language CI (Rust, TS, Python, Go, PHP)
│       ├── release.yml                 # Tag-triggered publish to all registries
│       ├── audit.yml                   # Weekly security audit + CodeQL
│       └── bench.yml                   # Criterion benchmarks → gh-pages
│
├── rust/                               # ── Rust (primary implementation) ──
│   ├── Cargo.toml                      # Package config, feature flags, deps
│   ├── rustfmt.toml                    # Formatter config (edition=2021)
│   ├── clippy.toml                     # Linter thresholds
│   ├── src/
│   │   ├── lib.rs                      # Entry point, re-exports, #![no_std] cfg
│   │   ├── error.rs                    # VectorError enum
│   │   ├── ops/
│   │   │   └── mod.rs                  # Public API: cosine, dot, l2, normalize, arithmetic
│   │   ├── batch.rs                    # Batch operations (std only)
│   │   └── simd/
│   │       ├── mod.rs                  # SIMD module, conditional compilation
│   │       ├── dispatch.rs             # Runtime CPUID dispatch (x86_64)
│   │       ├── portable.rs             # Scalar fallback (4× unrolled)
│   │       ├── avx2.rs                 # AVX2+FMA intrinsics (8×f32)
│   │       ├── avx512.rs              # AVX-512 via 2×AVX2 (16×f32)
│   │       ├── neon.rs                 # ARM NEON intrinsics (4×f32)
│   │       └── wasm.rs                 # WASM SIMD128 (4×f32)
│   ├── tests/
│   │   └── test_ops.rs                 # Integration tests (shared vectors)
│   └── benches/
│       └── bench_ops.rs                # Criterion benchmarks
│
├── ts/                                 # ── TypeScript (pure port) ──
│   ├── package.json                    # npm config (ESM, zero-dep)
│   ├── tsconfig.json                   # TypeScript compiler config
│   └── src/
│       ├── index.ts                    # Full implementation
│       └── test.ts                     # Node test runner tests
│
├── python/                             # ── Python (pure port) ──
│   ├── pyproject.toml                  # PyPI config (setuptools)
│   ├── test_lombokvector.py            # unittest test suite
│   └── lombokvector/
│       └── __init__.py                 # Full implementation (single file)
│
├── go/                                 # ── Go (pure port) ──
│   ├── go.mod                          # Go module config
│   ├── lombokvector.go                 # Full implementation
│   └── lombokvector_test.go            # Test suite
│
├── php/                                # ── PHP (pure port) ──
│   ├── composer.json                   # Packagist config (PSR-4)
│   ├── phpunit.xml                     # PHPUnit 11 config
│   ├── src/
│   │   └── LombokVector.php            # Full implementation
│   └── tests/
│       └── LombokVectorTest.php        # PHPUnit test suite
│
└── c-headers/                          # ── C FFI headers ──
    └── lombokvector.h                  # C header (cbindgen-compatible)
```

## File Count Summary

| Category | Files | Description |
|----------|-------|-------------|
| Documentation | 5 | README, LICENSE, CHANGELOG, architecture, masterplan |
| Map | 1 | Folder structure (this file) |
| Config | 4 | .gitignore, .editorconfig, rustfmt.toml, clippy.toml |
| CI/CD | 4 | ci, release, audit, bench workflows |
| Shared | 1 | Test vectors JSON |
| Rust | 11 | Cargo.toml + 10 source/test/bench files |
| TypeScript | 4 | package.json, tsconfig.json + 2 source files |
| Python | 3 | pyproject.toml + 2 source files |
| Go | 3 | go.mod + 2 source files |
| PHP | 4 | composer.json, phpunit.xml + 2 source files |
| C headers | 1 | FFI header |
| **Total** | **41** | |

## Directory Purpose Key

| Directory | Purpose | Language/Tool |
|-----------|---------|---------------|
| `rust/` | Primary implementation, SIMD kernels | Rust (no_std) |
| `rust/src/simd/` | Backend-specific SIMD intrinsics | Rust unsafe |
| `rust/src/ops/` | Public API functions | Rust safe |
| `ts/` | Pure TypeScript port (Deno/Bun/browser) | TypeScript (ESM) |
| `python/` | Pure Python port (no C extensions) | Python 3.9+ |
| `go/` | Pure Go port | Go 1.21+ |
| `php/` | Pure PHP port | PHP 8.1+ |
| `c-headers/` | C FFI interface (cbindgen output) | C99 |
| `vectors/` | Cross-language golden test data | JSON |
| `.github/workflows/` | CI/CD automation | GitHub Actions |
| `map/` | Repository documentation maps | Markdown |

## Template Pattern

This structure is replicated for ALL 39 libraries in LombokRAGFrameworks:

```
Lombok<Name>/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── architecture_repo.md
├── masterplan_repo.md
├── .gitignore
├── .editorconfig
├── map/
│   └── strukture_folder_repo.md
├── vectors/
│   └── lombok<name>-vectors-v1.json
├── .github/workflows/
│   ├── ci.yml
│   ├── release.yml
│   ├── audit.yml
│   └── bench.yml
├── rust/
├── ts/
├── python/
├── go/
├── php/
└── c-headers/
```
