# LombokVector

**SIMD-optimized vector math for RAG systems.**
Zero dependencies. `no_std` compatible. Multi-language.

Part of [LombokRAGFrameworks](https://github.com/codinglombok/LombokRAGFrameworks) — Lombok Ecosystem ([@codinglombok](https://github.com/codinglombok)).

## Features

- **Cosine similarity**, dot product, L2 distance, inner product, normalization
- **Batch operations** — query vs N candidates, N×M distance matrix
- **SIMD backends** — AVX2, AVX-512, ARM NEON, WASM-SIMD, portable scalar
- **Runtime auto-detection** on x86_64 (CPUID)
- **f32 + f64 precision** — f32 default (fast), f64 optional (accurate)
- **no_std** — zero heap allocations in hot path, works on embedded/RTOS
- **Zero dependencies** — stdlib only, no tokio/rayon/ndarray

## Install

```bash
# Rust
cargo add lombokvector

# TypeScript/Node.js
npm install lombokvector

# Python
pip install lombokvector

# PHP
composer require codinglombok/lombokvector

# Go
go get github.com/codinglombok/lombokvector
```

## Quick Start

### Rust

```rust
use lombokvector::{cosine_similarity, dot_product, l2_distance, normalize};

let a = &[1.0_f32, 2.0, 3.0];
let b = &[4.0_f32, 5.0, 6.0];

let cos = cosine_similarity(a, b).unwrap();   // 0.9746318
let dot = dot_product(a, b).unwrap();          // 32.0
let l2 = l2_distance(a, b).unwrap();           // 5.196152
let norm = normalize(a).unwrap();              // [0.267, 0.535, 0.802]

// Check SIMD backend
println!("Backend: {}", lombokvector::active_backend()); // "avx2" or "portable"
```

### TypeScript

```typescript
import { cosineSimilarity, dotProduct, l2Distance, normalize } from 'lombokvector';

const cos = cosineSimilarity([1, 2, 3], [4, 5, 6]); // 0.9746318
const dot = dotProduct([1, 2, 3], [4, 5, 6]);        // 32.0
const l2 = l2Distance([1, 2, 3], [4, 5, 6]);         // 5.196152
const norm = normalize([1, 2, 3]);                    // Float64Array [0.267, 0.535, 0.802]
```

### Python

```python
from lombokvector import cosine_similarity, dot_product, l2_distance, normalize

cos = cosine_similarity([1, 2, 3], [4, 5, 6])  # 0.9746318
dot = dot_product([1, 2, 3], [4, 5, 6])        # 32.0
l2 = l2_distance([1, 2, 3], [4, 5, 6])         # 5.196152
norm = normalize([1, 2, 3])                     # [0.267, 0.535, 0.802]
```

### PHP

```php
use CodingLombok\LombokVector\LombokVector;

$cos = LombokVector::cosineSimilarity([1, 2, 3], [4, 5, 6]); // 0.9746318
$dot = LombokVector::dotProduct([1, 2, 3], [4, 5, 6]);        // 32.0
$l2 = LombokVector::l2Distance([1, 2, 3], [4, 5, 6]);         // 5.196152
$norm = LombokVector::normalize([1, 2, 3]);                    // [0.267, 0.535, 0.802]
```

### Go

```go
import lv "github.com/codinglombok/lombokvector"

cos, _ := lv.CosineSimilarity([]float64{1, 2, 3}, []float64{4, 5, 6}) // 0.9746318
dot, _ := lv.DotProduct([]float64{1, 2, 3}, []float64{4, 5, 6})       // 32.0
l2, _ := lv.L2Distance([]float64{1, 2, 3}, []float64{4, 5, 6})        // 5.196152
norm, _ := lv.Normalize([]float64{1, 2, 3})                            // [0.267, 0.535, 0.802]
```

## API

| Function | Description | Returns |
|----------|-------------|---------|
| `cosine_similarity(a, b)` | Cosine similarity | `f32` in [-1, 1] |
| `dot_product(a, b)` | Dot product | `f32` |
| `l2_distance(a, b)` | Euclidean distance | `f32` ≥ 0 |
| `inner_product(a, b)` | Inner product (= dot product) | `f32` |
| `l2_norm(a)` | Vector magnitude | `f32` ≥ 0 |
| `normalize(a)` | Unit-length vector | `Vec<f32>` |
| `vec_add(a, b)` | Element-wise addition | `Vec<f32>` |
| `vec_sub(a, b)` | Element-wise subtraction | `Vec<f32>` |
| `vec_mul_scalar(a, s)` | Scalar multiplication | `Vec<f32>` |
| `batch_cosine(query, candidates)` | 1 query vs N vectors (cosine) | Sorted desc |
| `batch_l2(query, candidates)` | 1 query vs N vectors (L2) | Sorted asc |
| `distance_matrix_cosine(A, B)` | N×M cosine matrix | `Vec<Vec<f32>>` |

All f32 functions have `_f64` variants.

## SIMD Backends

| Backend | Arch | Width | Floats/iter | Feature Flag |
|---------|------|-------|-------------|-------------|
| AVX2 + FMA | x86_64 | 256-bit | 8×f32 | `avx2` |
| AVX-512 (2×AVX2) | x86_64 | 512-bit | 16×f32 | `avx512` |
| NEON | aarch64 | 128-bit | 4×f32 | `neon` |
| WASM SIMD | wasm32 | 128-bit | 4×f32 | `wasm-simd` |
| Portable | any | scalar | 1×f32 (4× unrolled) | `portable` |

Default: `auto-detect` — runtime CPUID on x86_64, compile-time on ARM/WASM.

## Comparison

| Library | Language | SIMD | no_std | Zero-dep | Batch | Precision |
|---------|----------|------|--------|----------|-------|-----------|
| **LombokVector** | **Rust + 4 ports** | **AVX2/512/NEON/WASM** | **Yes** | **Yes** | **Yes** | **f32 + f64** |
| faiss | C++/Python | AVX2/512 | No | No (BLAS) | Yes | f32 |
| numpy | Python (C) | AVX2 | No | No (LAPACK) | Yes | f32/f64 |
| simsimd | C/Python/Rust | AVX2/512/NEON/SVE | No | Yes | Yes | f16/f32 |
| ndarray | Rust | via BLAS | No | No | Yes | f32/f64 |

## Performance Targets

| Operation | Dimension | Target |
|-----------|-----------|--------|
| Cosine (1M vectors) | 768 | <5ms AVX2 |
| Batch 1K queries × 1M | 768 | <5s |
| WASM (100K vectors) | 768 | <50ms |

## Test Vectors

All ports share `vectors/lombokvector-vectors-v1.json`. Every language MUST produce identical results within tolerance (f32: ±1e-6, f64: ±1e-14).

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Part of Lombok Ecosystem

```
LombokRAGFrameworks (Tier 4 Hub)
  └── LombokVector (this library — Layer 0: Core)
  └── LombokHNSW (depends on LombokVector)
  └── LombokQuantize (depends on LombokVector)
  └── LombokColBERT (depends on LombokVector)
  └── LombokEval (depends on LombokVector)
  └── ...38 more libraries
```
