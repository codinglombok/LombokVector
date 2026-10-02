# LombokVector — Guide How to Use v0.2.0

## 1. Pemasangan

| Bahasa | Perintah | Syarat |
|---|---|---|
| Rust | `cargo add lombokvector` (embedded: `default-features = false`) | Rust 1.70+ |
| TypeScript/JS | `npm install lombokvector` | Node.js 20+, peramban modern, Deno, Bun |
| Python | `pip install lombokvector` | Python 3.9+ |
| Go | `go get github.com/codinglombok/lombokvector/go` | Go 1.22+ |
| PHP | `composer require codinglombok/lombokvector` | PHP 8.1+ |
| C | `cargo rustc --release --features ffi --crate-type cdylib` di `rust/`, lalu `#include "lombokvector.h"` | kompiler C99 |

## 2. Operasi dasar

```ts
dotProduct([1, 2, 3], [4, 5, 6]);        // 32
l2Distance([1, 2, 3], [4, 5, 6]);        // 5.196152422706632
cosineSimilarity([1, 2, 3], [4, 5, 6]);  // 0.9746318461970762, selalu di [-1, 1]
normalize([3, 4]);                       // Float64Array [0.6000000000000001, 0.8] = a[i] * (1 / norm)
```

## 3. Peringkat

```python
from lombokvector import batch_cosine, distance_matrix_cosine
hits = batch_cosine(query, documents)        # [ScoredIndex(index, score)], menurun
top5 = hits[:5]
m = distance_matrix_cosine(group_a, group_b) # m[i][j]
```

- Skor sama: urutan indeks dipertahankan.
- Kandidat dengan norma 0 mendapat skor 0 pada `batch_cosine`.
- `batch_l2` mengurutkan menaik (paling dekat lebih dulu).

## 4. Error

| Kode | Kapan |
|---|---|
| `EMPTY_VECTOR` | vektor tanpa elemen |
| `DIMENSION_MISMATCH` | panjang berbeda |
| `ZERO_MAGNITUDE` | cosine/normalize pada vektor nol |
| `NON_FINITE` | masukan NaN/tak hingga, atau hasil meluap |

```go
if _, err := lombokvector.CosineSimilarity(a, b); errors.Is(err, lombokvector.ErrZeroMagnitude) { /* ... */ }
```

## 5. Rust: presisi, `no_std`, SIMD

- Fungsi tanpa akhiran memakai `f32`; akhiran `_f64` memakai `f64`.
- `default-features = false` memberi `no_std` tanpa alokator: dot, cosine, l2, norm, `normalize_inplace`. Fitur `alloc` menambah fungsi yang mengembalikan `Vec`.
- `active_backend()` menyebut kernel yang dipakai (`avx2`, `neon`, `wasm-simd128`, `portable`). Hasilnya sama di semua kernel.
- WASM: kompilasi dengan `RUSTFLAGS="-C target-feature=+simd128"` untuk kernel simd128.

## 6. Skenario pemakaian

1. **Pencarian semantik di beberapa layanan.** Embedding diperingkat di server Go dan di klien TypeScript. Skor identik, sehingga urutan hasil di layar sama dengan log server.
2. **Deduplikasi dokumen.** Pasangan dengan cosine ≥ ambang dianggap duplikat. Ambang berperilaku sama di skrip Python offline dan di layanan Rust, tanpa kasus tepi yang berbeda karena bit terakhir.
3. **Sensor di mikrokontroler.** Vektor fitur getaran dinormalisasi dan dibandingkan dengan pola acuan di Cortex-M (`no_std`, tanpa alokator), lalu hasilnya dicocokkan dengan perhitungan di server.
4. **Rekomendasi produk di aplikasi PHP.** `batchCosine` atas vektor produk, dengan hasil yang dapat diverifikasi ulang di pipeline Python.
5. **Uji regresi numerik.** Pola bit dari SPEC/vector dipakai sebagai nilai acuan saat memindahkan kode antar-bahasa.

## 7. Batasan

Lihat `full_summary_project_` §2: percepatan SIMD sedang, tanpa penskalaan ulang untuk elemen sangat besar, binary32 hanya di Rust/Go/C.

*Lisensi dokumen: Apache-2.0 · © codinglombok*
