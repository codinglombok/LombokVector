# LombokVector

> Vector math that gives the same bits in Rust, TypeScript, Python, Go, PHP, and C: dot product, cosine similarity, Euclidean distance, normalization, batch ranking. Zero runtime dependencies; the Rust crate is `no_std` and uses AVX2, NEON, or WASM SIMD without changing results.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/codinglombok/LombokVector/actions/workflows/ci.yml/badge.svg)](https://github.com/codinglombok/LombokVector/actions/workflows/ci.yml)
[![Vectors](https://img.shields.io/badge/shared%20vectors-223%20bit--exact-success)](vectors/)
[![Lombok Ecosystem](https://img.shields.io/badge/Lombok-Ecosystem-2e7d5b?logo=github)](https://github.com/codinglombok)

Part of the [Lombok Ecosystem](https://github.com/codinglombok).

## Mengapa library ini? (Why this library?)

- **Bit-identical across languages.** Floating-point sums depend on their order. LombokVector fixes the order (8 lanes, separate rounding, no fused multiply-add; [SPEC section 2](docs/SPEC_LombokVector_v0.2.0.md#2-reduksi-8-lajur-inti-kontrak)), so a score computed in a Python batch job, a Go service, and a TypeScript browser app is exactly the same number. All ports check 223 shared cases bit for bit in CI.
- **SIMD that does not change answers.** The Rust AVX2 (x86), NEON (aarch64), and simd128 (wasm32) kernels follow the same order as the scalar code, and a test compares them on every build. Measured: AVX2 dot product of 768 binary32 values in about 56 ns versus 72 ns for the scalar kernel on one x86_64 machine (the scalar code is already auto-vectorized), so expect a modest speed-up rather than a large one.
- **Explicit failure modes.** Empty input, length mismatch, zero magnitude, and overflow or non-finite input (`NON_FINITE`) are errors with the same code in every language, instead of silent NaN.
- **Runs small.** Rust `no_std` without an allocator (embedded), with a correctly rounded software square root, plus a C API. Pure TypeScript, Python, Go, and PHP with no dependencies.

## Installation

| Language | Package | Status |
|---|---|---|
| Rust | `lombokvector` (crates.io) | not yet published |
| TypeScript / JavaScript | `lombokvector` (npm) | not yet published |
| Python | `lombokvector` (PyPI) | not yet published |
| Go | `github.com/codinglombok/lombokvector/go` | tag `go/v0.2.0` on release |
| PHP | `codinglombok/lombokvector` (Packagist) | needs a split repository first |
| C | `c-headers/lombokvector.h` + library built from `rust/` | build from source |

## Quick start

### Rust

```rust
use lombokvector::{batch_cosine, cosine_similarity, dot_product, normalize};

let a = [1.0_f32, 2.0, 3.0];
let b = [4.0_f32, 5.0, 6.0];
assert_eq!(dot_product(&a, &b)?, 32.0);
let c = cosine_similarity(&a, &b)?;              // f32; *_f64 variants for binary64
let ranked = batch_cosine(&a, &[&b, &[0.0, 0.0, 1.0]])?;   // [(index, score)], descending
println!("backend: {}", lombokvector::active_backend());   // "avx2", "neon", "wasm-simd128" or "portable"
```

### TypeScript

```ts
import { cosineSimilarity, batchCosine, VectorError } from 'lombokvector';

cosineSimilarity([1, 2, 3], [4, 5, 6]);           // 0.9746318461970762
batchCosine(query, docs);                           // [{ index, score }, ...] descending
try { cosineSimilarity([0, 0], [1, 1]); } catch (e) { (e as VectorError).code; } // 'ZERO_MAGNITUDE'
```

### Python

```python
from lombokvector import cosine_similarity, batch_l2, normalize

cosine_similarity([1, 2, 3], [4, 5, 6])  # 0.9746318461970762
batch_l2([0, 0], [[3, 4], [1, 0]])       # [ScoredIndex(index=1, score=1.0), ScoredIndex(index=0, score=5.0)]
```

### Go

```go
import lombokvector "github.com/codinglombok/lombokvector/go"

c, err := lombokvector.CosineSimilarity([]float64{1, 2, 3}, []float64{4, 5, 6})
c32, err := lombokvector.CosineSimilarityF32([]float32{1, 2, 3}, []float32{4, 5, 6})
if errors.Is(err, lombokvector.ErrZeroMagnitude) { /* ... */ }
```

### PHP

```php
use CodingLombok\LombokVector\LombokVector;

LombokVector::cosineSimilarity([1, 2, 3], [4, 5, 6]); // 0.9746318461970762
LombokVector::batchDot([1, 1], [[1, 0], [2, 2]]);     // [['index' => 1, 'score' => 4.0], ...]
```

### C

```c
#include "lombokvector.h"
float out;
if (lombokvector_cosine_f32(a, b, n, &out) == LOMBOKVECTOR_OK) { /* ... */ }
```

Build the library with `cargo rustc --release --features ffi --crate-type cdylib` in `rust/`; see [c-headers/examples/check.c](c-headers/examples/check.c).

## Ports

| | Rust | TypeScript | Python | Go | PHP | C |
|---|---|---|---|---|---|---|
| binary64 | YES | YES | YES | YES | YES | dot, l2, cosine |
| binary32 | YES | – | – | YES | – | dot, l2, cosine, norm, normalize |
| batch + matrix | YES | YES | YES | YES | YES | – |
| SIMD | AVX2, NEON, simd128 | – | – | – | – | via Rust |
| shared vectors | 223/223 | 113/113 | 113/113 | 223/223 | 113/113 | smoke test |

## Known limitations

No approximate-nearest-neighbour index, no quantization, no distances beyond L2/dot/cosine. Very large elements overflow the sum of squares (`NON_FINITE`) because there is no rescaling. Only Rust and Go offer binary32. Details: [docs/full_summary_project_LombokVector_v0.2.0.md](docs/full_summary_project_LombokVector_v0.2.0.md#2-batasan-yang-diketahui).

## Upgrading from 0.1.0

0.2.0 changes numeric results in the last bits (new summation order), clamps cosine to [-1, 1], adds the `NON_FINITE` error, fixes the PHP port and the Rust `no_std` build, implements the C API, and moves the Go module to `github.com/codinglombok/lombokvector/go`. See [CHANGELOG.md](CHANGELOG.md).

## Development

```bash
cd rust && cargo test --features ffi && cargo clippy --all-targets --features ffi -- -D warnings
cd typescript && npm ci && npm run coverage
cd python && python -m pytest
cd go && go test ./...
cd php && php tests/run.php
python3 vectors/build_vectors.py && bash scripts/lombok-doctor.sh LombokVector
```

## License

Apache-2.0. See [LICENSE](LICENSE).
