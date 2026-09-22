# LombokVector — Architecture

> SIMD-optimized vector math for RAG systems.
> Part of [LombokRAGFrameworks](https://github.com/codinglombok) · Wave 1 (zero-dep foundation)

---

## 1. Overview

LombokVector provides the core vector operations required by every RAG pipeline: cosine similarity, dot product, L2 distance, inner product, normalization, and batch operations. It sits at the absolute bottom of the dependency graph — **zero external dependencies** across all language ports.

```
┌─────────────────────────────────────────────────┐
│              RAG Pipeline (consumer)            │
│  LombokHNSW · LombokQuantize · LombokColBERT   │
├─────────────────────────────────────────────────┤
│              LombokVector (this)                │
│  cosine · dot · l2 · normalize · batch          │
├─────────────────────────────────────────────────┤
│              Hardware                           │
│  AVX2+FMA · AVX-512 · ARM NEON · WASM SIMD     │
└─────────────────────────────────────────────────┘
```

## 2. Design Principles

| Principle | Implementation |
|-----------|----------------|
| Zero dependencies | No crate/npm/pip deps. `no_std` compatible Rust core. |
| Hot-path allocation-free | No heap allocations in compute kernels. Stack + slices only. |
| SIMD-first | Compile-time feature flags + runtime CPUID dispatch on x86_64. |
| Cross-language consistency | Shared test vectors JSON. All ports produce identical results (±1e-6 f32, ±1e-14 f64). |
| 4× loop unrolling | Every kernel in every language uses the same ILP pattern. |

## 3. SIMD Backend Architecture

### 3.1 Dispatch Model (Rust)

```
                    ┌─────────────────────┐
                    │   Public API call    │
                    │ cosine_similarity()  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Feature gate check  │
                    │  (compile time)      │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
   ┌────────▼────────┐ ┌──────▼───────┐ ┌────────▼────────┐
   │  auto-detect    │ │  explicit    │ │  portable       │
   │  (runtime       │ │  feature     │ │  (scalar        │
   │   CPUID)        │ │  flag        │ │   fallback)     │
   └────────┬────────┘ └──────┬───────┘ └────────┬────────┘
            │                  │                  │
   ┌────────▼────────┐        │                  │
   │  is_x86_        │        │                  │
   │  feature_       │        │                  │
   │  detected!      │        │                  │
   └────┬───────┬────┘        │                  │
        │       │              │                  │
   ┌────▼──┐ ┌──▼────┐   ┌───▼───┐         ┌───▼────┐
   │ AVX2  │ │AVX512 │   │ NEON  │         │Portable│
   │ +FMA  │ │2×AVX2 │   │ ARM   │         │ Scalar │
   └───────┘ └───────┘   └───────┘         └────────┘
```

### 3.2 Backend Details

| Backend | Width | Intrinsics | Platform |
|---------|-------|-----------|----------|
| **AVX2+FMA** | 8×f32 | `_mm256_fmadd_ps`, `_mm256_mul_ps` | x86_64 (Haswell+) |
| **AVX-512** | 16×f32 (2×AVX2) | Two `__m256` per iteration | x86_64 (Skylake-X+) |
| **NEON** | 4×f32 | `vfmaq_f32`, `vaddvq_f32` | ARM (aarch64) |
| **WASM SIMD** | 4×f32 | `f32x4_mul`, `v128_load` | Browser/WASM runtime |
| **Portable** | 1×f32 (4× unrolled) | None (scalar) | Everywhere, `no_std` |

### 3.3 Feature Flags (Rust)

```toml
[features]
default = ["std", "auto-detect"]
std = []                    # Enable std-dependent features (batch, distance matrix)
auto-detect = ["std"]       # Runtime CPUID detection on x86_64
avx2 = []                   # Force AVX2+FMA backend
avx512 = []                 # Force AVX-512 (2×AVX2) backend
neon = []                   # Force ARM NEON backend
wasm-simd = []              # Force WASM SIMD backend
portable = []               # Force scalar fallback
```

## 4. Module Structure (Rust)

```
rust/src/
├── lib.rs              # Entry, re-exports, #![no_std] cfg
├── error.rs            # VectorError enum
├── ops/
│   └── mod.rs          # Public API: cosine, dot, l2, normalize, arithmetic
├── batch.rs            # Batch operations (std only)
└── simd/
    ├── mod.rs          # SIMD module, conditional compilation
    ├── dispatch.rs     # Runtime CPUID dispatch
    ├── portable.rs     # Scalar fallback (4× unrolled)
    ├── avx2.rs         # AVX2+FMA intrinsics
    ├── avx512.rs       # 2×AVX2 processing
    ├── neon.rs         # ARM NEON intrinsics
    └── wasm.rs         # WASM SIMD128
```

## 5. Public API

All language ports expose identical API surface:

| Function | Signature (Rust) | Description |
|----------|-------------------|-------------|
| `cosine_similarity` | `(&[f32], &[f32]) → Result<f32>` | Cosine similarity [-1, 1] |
| `dot_product` | `(&[f32], &[f32]) → Result<f32>` | Dot product |
| `l2_distance` | `(&[f32], &[f32]) → Result<f32>` | Euclidean distance |
| `inner_product` | `(&[f32], &[f32]) → Result<f32>` | Alias for dot product |
| `l2_norm` | `(&[f32]) → Result<f32>` | Vector magnitude |
| `normalize` | `(&[f32]) → Result<Vec<f32>>` | Unit vector normalization |
| `vec_add` | `(&[f32], &[f32]) → Result<Vec<f32>>` | Element-wise addition |
| `vec_sub` | `(&[f32], &[f32]) → Result<Vec<f32>>` | Element-wise subtraction |
| `vec_mul_scalar` | `(&[f32], f32) → Vec<f32>` | Scalar multiplication |
| `batch_cosine` | `(&[f32], &[&[f32]]) → Vec<ScoredIndex>` | Batch cosine (sorted desc) |
| `batch_l2` | `(&[f32], &[&[f32]]) → Vec<ScoredIndex>` | Batch L2 (sorted asc) |
| `distance_matrix_cosine` | `(&[&[f32]]) → Vec<Vec<f32>>` | N×N distance matrix |

All `_f64` variants also available.

## 6. Error Handling

```rust
pub enum VectorError {
    DimensionMismatch { expected: usize, got: usize },
    EmptyVector,
    ZeroMagnitude,
}
```

Consistent error types across all ports:
- **Rust**: `Result<T, VectorError>`
- **TypeScript**: `VectorError extends Error` (thrown)
- **Python**: `VectorError(Exception)` (raised)
- **Go**: `error` return value
- **PHP**: `VectorError extends RuntimeException` (thrown)

## 7. Performance Architecture

### 7.1 Kernel Pattern (all languages)

```
┌─────────────────────────────────────┐
│  Input: a[0..n], b[0..n]           │
├─────────────────────────────────────┤
│  Phase 1: 4× unrolled main loop    │
│  chunks = n & ~3                   │
│  for i in (0..chunks).step_by(4):  │
│    s += a[i]*b[i] + a[i+1]*b[i+1]  │
│         + a[i+2]*b[i+2]            │
│         + a[i+3]*b[i+3]            │
├─────────────────────────────────────┤
│  Phase 2: Scalar tail              │
│  for i in chunks..n:               │
│    s += a[i] * b[i]                │
├─────────────────────────────────────┤
│  Return: s                         │
└─────────────────────────────────────┘
```

### 7.2 SIMD Kernel (AVX2 example)

```
┌─────────────────────────────────────┐
│  8×f32 per iteration via __m256    │
│  _mm256_loadu_ps → load 8 floats   │
│  _mm256_fmadd_ps → fused mul-add   │
│  hsum_256 → horizontal reduction   │
│  + scalar tail for remainder       │
└─────────────────────────────────────┘
```

### 7.3 Targets

| Benchmark | Target | Backend |
|-----------|--------|---------|
| 1M cosine (768d) | <5ms | AVX2 |
| Batch 1000 × 1M vectors | <5s | AVX2 |
| WASM 100K vectors | <50ms | WASM SIMD |

## 8. Test Architecture

```
vectors/lombokvector-vectors-v1.json (shared golden vectors)
        │
        ├── rust/tests/test_ops.rs
        ├── ts/src/test.ts
        ├── python/test_lombokvector.py
        ├── go/lombokvector_test.go
        └── php/tests/LombokVectorTest.php
```

All ports test against the same expected values. Tolerance:
- f32: ±1e-6
- f64: ±1e-14

## 9. CI/CD Pipeline

```
Push/PR → ci.yml
  ├── Rust: 3 OS × clippy + fmt + test (default + portable + no_std + WASM)
  ├── TypeScript: Node 18/20/22
  ├── Python: 3.9–3.13
  ├── Go: 1.21–1.23
  └── PHP: 8.1–8.4

Tag v* → release.yml
  ├── crates.io (Rust)
  ├── npm + GPR (TypeScript)
  ├── PyPI (Python)
  ├── Packagist validation (PHP, auto-publish via webhook)
  ├── proxy.golang.org (Go, auto-publish via tag)
  └── GitHub Release (all)

Weekly → audit.yml
  ├── cargo-audit
  ├── npm audit
  └── CodeQL (JS + Python)

Push main → bench.yml
  └── Criterion benchmarks → gh-pages
```

## 10. Downstream Consumers

Libraries that depend on LombokVector (Wave 2+):

| Library | Usage |
|---------|-------|
| LombokHNSW | Distance computation in HNSW graph traversal |
| LombokQuantize | Vector quantization with distance scoring |
| LombokRRF | Score vector fusion |
| LombokLeiden | Community detection similarity computation |
| LombokEmbedding | Embedding similarity + normalization |
| LombokColBERT | Late-interaction token-level similarity |
| LombokEval | RAG evaluation metrics |
| LombokAgenticAuto | Agent similarity search |
