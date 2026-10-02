# LombokVector — SPEC v0.2.0

This document is the normative cross-language contract. Every language port MUST produce byte-identical output for all specified inputs. Deviations from this specification are bugs.

| Atribut | Nilai |
|---|---|
| Versi SPEC | 0.2.0 (berlaku untuk paket `lombokvector` 0.2.x di crates.io, npm, PyPI, Packagist, dan modul Go) |
| Standar acuan | IEEE 754-2019 (binary32, binary64, pembulatan ke terdekat-genap, `squareRoot` dibulatkan benar); ECMAScript 2024 (Number = binary64, `Array.prototype.sort` stabil); spesifikasi bahasa Go 1.22 (konversi eksplisit mencegah fusi FMA); tinjauan 2026-10-02 |
| Vector | `vectors/lombokvector-vectors-v1.json` — 223 kasus (113 binary64 untuk semua port, 110 binary32 untuk port dengan API binary32) — SHA-256 `38e3c88a89529d190bb00b87c8281563280edb3b2ecdeb476269c9c1e2f01496` |
| Pemeriksa referensi | implementasi Python independen di `vectors/build_vectors.py` (binary32 diemulasikan eksak), dan setiap hasil dicek terhadap hitungan rasional eksak (`fractions.Fraction`) |
| Port | Rust (binary32 + binary64, kernel SIMD AVX2/NEON/simd128), Go (binary32 + binary64), TypeScript, Python, PHP (binary64) |
| Tanggal tinjauan | 2026-10-02 |

Kata MUST, MUST NOT, SHOULD, MAY mengikuti RFC 2119.

## 0. Konvensi

1. Semua aritmetika memakai IEEE 754 pada presisi operasi (binary64, atau binary32 untuk API binary32), dengan pembulatan ke terdekat-genap. Setiap operasi dasar (`+`, `-`, `*`, `/`, `sqrt`) dibulatkan sendiri.
2. Fused multiply-add MUST NOT dipakai. Port yang compiler-nya boleh memfusikan operasi (Go) MUST mencegahnya; Go melakukannya dengan konversi eksplisit `T(x*y)`.
3. `sqrt` MUST dibulatkan benar. Build Rust `no_std` memakai implementasi perangkat lunak yang diuji sama bit dengan instruksi hardware.
4. Hasil di vector ditulis sebagai pola bit IEEE 754: `0x` + 16 digit hex (binary64) atau 8 digit (binary32). Runner membandingkan pola bit secara eksak; `-0.0` dan `+0.0` berbeda.
5. Masukan bilangan bulat (PHP, Python) MUST dikonversi ke float sebelum aritmetika, supaya `-2 * 0` menghasilkan `-0.0`.

## 1. Ruang lingkup

Operasi: dot product, inner product, norma Euclid, jarak Euclid, cosine similarity, normalisasi, penjumlahan/pengurangan elemen, perkalian skalar, peringkat batch (cosine, dot, L2), dan matriks cosine. Di luar lingkup 0.2.0: indeks pencarian (ANN), kuantisasi, jarak lain (Manhattan, Hamming), dan API binary32 di TS/Python/PHP.

## 2. Reduksi 8 lajur (inti kontrak)

### 2.1 Definisi

Untuk suku `t[0..n-1]` dalam presisi P:

```
s[0..7] = +0.0
untuk i = 0 .. n-1:  s[i mod 8] = s[i mod 8] + t[i]          (dibulatkan ke P)
hasil = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))      (setiap + dibulatkan ke P)
```

Suku untuk setiap kernel:

| Kernel | Suku `t[i]` |
|---|---|
| `dot(a, b)` | `a[i] * b[i]` |
| `sumsq(a)` | `a[i] * a[i]` |
| `l2sq(a, b)` | `d * d` dengan `d = a[i] - b[i]` |

### 2.2 Mengapa urutan ini

Penjumlahan floating point tidak asosiatif, sehingga urutan menentukan bit hasil. Contohnya, `dot([1e16, 1, -1e16, 1], [1, 1, 1, 1])` menghasilkan `0.0` dengan urutan ini, sedangkan penjumlahan kiri-ke-kanan menghasilkan `1.0` (vector `dot-f64-005`). Urutan 8 lajur dipilih karena sama dengan bentuk alami kernel SIMD 256-bit (8 × binary32) dan 2 × 128-bit, dan galatnya lebih kecil daripada penjumlahan berurutan.

### 2.3 Kernel SIMD

Kernel SIMD MAY dipakai hanya bila menghasilkan bit yang sama dengan §2.1:

- delapan akumulator lajur;
- perkalian dan penjumlahan yang dibulatkan terpisah (AVX2 `mul_ps` + `add_ps`, NEON `vmulq` + `vaddq`, simd128 `f32x4_mul` + `f32x4_add`);
- sisa elemen ditambahkan ke lajur `0 .. n mod 8 - 1`;
- penggabungan lajur sesuai urutan §2.1.

Port Rust menguji kesetaraan ini pada setiap build (`tests/simd_matches_reference.rs`). CI menjalankannya di x86_64 (AVX2) dan Apple Silicon (NEON).

## 3. Operasi

Pemeriksaan dilakukan berurutan; error pertama yang cocok yang dilaporkan. `finite(x)` berarti: bila `x` tak hingga atau NaN → `NON_FINITE`.

### 3.1 `dot(a, b)` / `inner(a, b)`

`a` kosong → `EMPTY_VECTOR`; panjang berbeda → `DIMENSION_MISMATCH`; hasil `finite(dot(a, b))`.

### 3.2 `norm(a)`

`a` kosong → `EMPTY_VECTOR`; hasil `finite(sqrt(sumsq(a)))`.

### 3.3 `l2(a, b)`

Pemeriksaan seperti §3.1; hasil `finite(sqrt(l2sq(a, b)))`.

### 3.4 `cosine(a, b)`

1. Pemeriksaan seperti §3.1.
2. `d = finite(dot(a, b))`, `na = finite(sqrt(sumsq(a)))`, `nb = finite(sqrt(sumsq(b)))`, dihitung dalam urutan ini.
3. `na == 0` atau `nb == 0` → `ZERO_MAGNITUDE`.
4. `c = finite(d / (na * nb))`, lalu di-clamp: `c < -1` → `-1`, `c > 1` → `1`.

### 3.5 `normalize(a)`

1. `a` kosong → `EMPTY_VECTOR`.
2. `n = finite(sqrt(sumsq(a)))`; `n == 0` → `ZERO_MAGNITUDE`.
3. `inv = finite(1 / n)`; hasil `finite(a[i] * inv)` untuk setiap i.

Normalisasi memakai perkalian dengan `1 / n`, bukan pembagian dengan `n` (misalnya `normalize([3, 4])[0]` binary64 = `0.6000000000000001`). Tidak ada penskalaan ulang untuk elemen sangat besar: bila `sumsq` meluap, hasilnya `NON_FINITE`. Versi in-place (Rust, C) tidak mengubah masukan bila terjadi error.

### 3.6 `add`, `sub`, `scale`

`add`/`sub`: pemeriksaan seperti §3.1, lalu `finite(a[i] ± b[i])`. `scale(a, s)`: `a` kosong → `EMPTY_VECTOR`, lalu `finite(a[i] * s)`.

### 3.7 Batch: `batch_cosine`, `batch_dot`, `batch_l2`

1. `query` kosong → `EMPTY_VECTOR`.
2. `batch_cosine`: `qn = finite(sqrt(sumsq(query)))`; `qn == 0` → `ZERO_MAGNITUDE`.
3. Untuk setiap kandidat secara berurutan:
   - panjang berbeda → `DIMENSION_MISMATCH`;
   - skor dihitung seperti §3.1/§3.3/§3.4, dengan `qn` dipakai sebagai `na`;
   - kandidat dengan norma 0 diberi skor `+0.0` pada `batch_cosine`, bukan error.
4. Hasilnya daftar `(index, score)` yang diurutkan:
   - `batch_cosine` dan `batch_dot`: menurun; `batch_l2`: menaik;
   - skor sama mempertahankan urutan indeks (sort stabil), dan `-0.0` dianggap sama dengan `+0.0`.
5. Daftar kandidat kosong → daftar kosong.

### 3.8 `matrix_cosine(A, B)`

1. `A` atau `B` kosong → `EMPTY_VECTOR`; `A[0]` kosong → `EMPTY_VECTOR`.
2. Setiap vektor di `A` lalu `B` harus sepanjang `A[0]`, bila tidak → `DIMENSION_MISMATCH`.
3. Norma semua vektor `A` lalu `B` dihitung dengan `finite`.
4. `out[i][j] = finite(dot(A[i], B[j]))`, diubah menjadi `+0.0` bila salah satu norma 0, selain itu cosine §3.4 langkah 4.

## 4. Error

| Kode | Arti | Rust | TS | Python | Go | PHP | C |
|---|---|---|---|---|---|---|---|
| `DIMENSION_MISMATCH` | panjang berbeda | `VectorError::DimensionMismatch` | `VectorError` | `VectorError` (juga `ValueError`) | `ErrDimensionMismatch` | `VectorError` | 1 |
| `EMPTY_VECTOR` | vektor tanpa elemen | `EmptyVector` | idem | idem | `ErrEmptyVector` | idem | 2 |
| `ZERO_MAGNITUDE` | norma 0 tetapi arah diperlukan | `ZeroMagnitude` | idem | idem | `ErrZeroMagnitude` | idem | 3 |
| `NON_FINITE` | masukan tak hingga/NaN, atau luapan | `NonFinite` | idem | idem | `ErrNonFinite` | idem | 4 |

Teks pesan diawali kode lalu `": "`. Program MUST memeriksa kode. C menambahkan kode 5 (`NULL_POINTER`).

## 5. Port binary32

Rust (`cosine_similarity`, `dot_product`, ... tanpa akhiran) dan Go (akhiran `F32`) menjalankan §2-§3 dalam binary32: setiap operasi dibulatkan ke binary32, `sqrt` dibulatkan benar ke binary32. Masukan vector binary32 ditulis sebagai desimal eksak dari nilai binary32, sehingga konversi `binary64 → binary32` tidak membulatkan.

## 6. Keamanan (normatif)

1. Kode `unsafe` Rust hanya ada di `simd/` (intrinsik) dan `ffi.rs` (pointer dari C). Setiap blok berkomentar `SAFETY`, dan `#![deny(unsafe_op_in_unsafe_fn)]` berlaku. Panjang selalu diperiksa sebelum kernel SIMD dipanggil.
2. Fungsi C memeriksa pointer null dan menolak `len == 0`.
3. Waktu dan memori linear terhadap ukuran masukan; batch dan matriks O(jumlah × dimensi). Tidak ada alokasi di jalur skalar Rust.
4. Tidak ada dependensi runtime di semua port.

## 7. Perubahan dari 0.1.0

0.2.0 mengubah hasil numerik semua port (rilis 0.x):

- urutan penjumlahan 8 lajur menggantikan pembukaan loop 4× dan FMA;
- cosine di-clamp;
- error `NON_FINITE` baru untuk masukan non-finite dan luapan;
- `distance_matrix_cosine` kini memeriksa panjang;
- batch mengurutkan dengan stabil;
- Rust `no_std` diperbaiki (0.1 gagal dikompilasi), fitur Cargo tanpa efek dihapus, AVX-512 (tidak pernah dipanggil) dihapus, FFI C diimplementasikan;
- PHP diperbaiki (0.1 gagal di-parse);
- path modul Go menjadi `github.com/codinglombok/lombokvector/go`.

*Lisensi dokumen: Apache-2.0 · © codinglombok*
