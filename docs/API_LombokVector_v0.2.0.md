# LombokVector — API v0.2.0

Perilaku normatif ada di SPEC. Dokumen ini memetakan operasi SPEC ke nama di setiap port.

## 1. Ringkasan lintas port (binary64)

| Operasi (SPEC) | Rust | TypeScript | Python | Go | PHP (`LombokVector::`) |
|---|---|---|---|---|---|
| dot (3.1) | `dot_product_f64` | `dotProduct` | `dot_product` | `DotProduct` | `dotProduct` |
| inner (3.1) | `inner_product_f64` | `innerProduct` | `inner_product` | `InnerProduct` | `innerProduct` |
| norm (3.2) | `l2_norm_f64` | `l2Norm` | `l2_norm` | `L2Norm` | `l2Norm` |
| l2 (3.3) | `l2_distance_f64` | `l2Distance` | `l2_distance` | `L2Distance` | `l2Distance` |
| cosine (3.4) | `cosine_similarity_f64` | `cosineSimilarity` | `cosine_similarity` | `CosineSimilarity` | `cosineSimilarity` |
| normalize (3.5) | `normalize_f64`, `normalize_inplace_f64` | `normalize` | `normalize` | `Normalize` | `normalize` |
| add / sub / scale (3.6) | `vec_add_f64`, `vec_sub_f64`, `vec_mul_scalar_f64` | `vecAdd`, `vecSub`, `vecMulScalar` | `vec_add`, `vec_sub`, `vec_mul_scalar` | `VecAdd`, `VecSub`, `VecMulScalar` | `vecAdd`, `vecSub`, `vecMulScalar` |
| batch (3.7) | `batch_cosine_f64`, `batch_dot_f64`, `batch_l2_f64` | `batchCosine`, `batchDot`, `batchL2` | `batch_cosine`, `batch_dot`, `batch_l2` | `BatchCosine`, `BatchDot`, `BatchL2` | `batchCosine`, `batchDot`, `batchL2` |
| matrix (3.8) | `distance_matrix_cosine_f64` | `distanceMatrixCosine` | `distance_matrix_cosine` | `DistanceMatrixCosine` | `distanceMatrixCosine` |
| hasil batch | `Vec<(usize, f64)>` | `{ index, score }[]` | `list[ScoredIndex]` | `[]ScoredIndex[float64]` | `list<array{index, score}>` |
| error | `VectorError` (`.code()`) | `VectorError` (`.code`) | `VectorError` (`.code`, juga `ValueError`) | `*Error` (`.Code`, sentinel `Err*`) | `VectorError` (`->errorCode`, juga `InvalidArgumentException`) |

## 2. Rust (`lombokvector`)

| Item | Keterangan |
|---|---|
| binary32 | nama tanpa akhiran: `dot_product`, `inner_product`, `l2_norm`, `l2_distance`, `cosine_similarity`, `normalize`, `normalize_inplace`, `vec_add`, `vec_sub`, `vec_mul_scalar`, `batch_cosine`, `batch_dot`, `batch_l2`, `distance_matrix_cosine` |
| binary64 | akhiran `_f64` |
| tipe masukan | `&[f32]` / `&[f64]`; batch: `&[&[T]]` |
| tanpa alokator | dot, inner, norm, l2, cosine, `normalize_inplace(_f64)` |
| fitur `alloc` | fungsi yang mengembalikan `Vec` |
| `active_backend()` | `"avx2"`, `"neon"`, `"wasm-simd128"`, `"portable"` |
| `VectorError` | `DimensionMismatch { expected, actual }`, `EmptyVector`, `ZeroMagnitude`, `NonFinite`; `code()`, `Display`, `std::error::Error` (fitur `std`) |
| fitur `ffi` | modul `ffi` dengan fungsi C (§6) |

## 3. TypeScript (`lombokvector`)

Masukan `ArrayLike<number>` (`number[]`, `Float64Array`, `Float32Array`, ...). Fungsi vektor mengembalikan `Float64Array`. `distanceMatrixCosine` mengembalikan `number[][]`. `VectorError.code`: `'DIMENSION_MISMATCH' | 'EMPTY_VECTOR' | 'ZERO_MAGNITUDE' | 'NON_FINITE'`.

## 4. Python (`lombokvector`)

Masukan sequence apa pun berisi int/float. Hasil berupa `float` atau `list[float]`. `ScoredIndex` adalah `NamedTuple(index, score)`. `__version__ = "0.2.0"`.

## 5. Go (`github.com/codinglombok/lombokvector/go`, package `lombokvector`)

| Item | Keterangan |
|---|---|
| binary64 | `DotProduct`, `InnerProduct`, `L2Norm`, `L2Distance`, `CosineSimilarity`, `Normalize`, `VecAdd`, `VecSub`, `VecMulScalar`, `BatchCosine`, `BatchDot`, `BatchL2`, `DistanceMatrixCosine` |
| binary32 | akhiran `F32` untuk semua fungsi di atas kecuali `InnerProduct` |
| `ScoredIndex[T]` | `{ Index int; Score T }` |
| error | `type Error struct{ Code, Msg string }`; `ErrDimensionMismatch`, `ErrEmptyVector`, `ErrZeroMagnitude`, `ErrNonFinite`; `errors.Is` membandingkan `Code` |

## 6. C (`c-headers/lombokvector.h`)

| Fungsi | Keterangan |
|---|---|
| `lombokvector_{dot,l2,cosine}_{f32,f64}(a, b, len, out)` | hasil di `*out`, kembalian kode §4 SPEC (0 = sukses, 5 = pointer null) |
| `lombokvector_l2_norm_f32(a, len, out)` | |
| `lombokvector_normalize_inplace_f32(a, len)` | tidak mengubah `a` bila error |
| `lombokvector_active_backend()` | string statis |

## 7. Kompatibilitas dengan 0.1.0

Nama fungsi dipertahankan di semua port. Yang berubah:

- bit terakhir hasil;
- error `NON_FINITE` baru;
- Rust: hasil batch tetap `(usize, T)`; fitur Cargo `avx2`/`avx512`/`neon`/`wasm-simd`/`auto-detect` dihapus;
- Go: `ScoredIndex` menjadi generik `ScoredIndex[float64]`, dan path modul memakai `/go`;
- PHP: properti error `errorCode` (0.1 tidak dapat dimuat);
- TypeScript: direktori `typescript/`.

*Lisensi dokumen: Apache-2.0 · © codinglombok*
