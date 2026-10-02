# LombokVector for Rust

Rust crate `lombokvector` (binary32 + binary64, AVX2/NEON/simd128 kernels, `no_std`, optional C API): dot product, cosine similarity, Euclidean distance, normalization, and batch ranking with results that are bit-identical to the other LombokVector ports. Zero runtime dependencies.

```
cargo add lombokvector
```

The summation order and every operation are defined in [docs/SPEC_LombokVector_v0.2.0.md](https://github.com/codinglombok/LombokVector/blob/main/docs/SPEC_LombokVector_v0.2.0.md); this port reproduces the shared vector file bit for bit in CI. Usage for all languages: [main README](https://github.com/codinglombok/LombokVector#readme) and [docs/API_LombokVector_v0.2.0.md](https://github.com/codinglombok/LombokVector/blob/main/docs/API_LombokVector_v0.2.0.md).

License: Apache-2.0.
