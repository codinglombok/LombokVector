# LombokVector — Full Summary Project v0.2.0

| Item | Nilai |
|---|---|
| Deskripsi | Matematika vektor (dot, cosine, L2, normalisasi, peringkat batch) dengan hasil bit-identik di Rust, TS, Python, Go, PHP, dan C; tanpa dependensi runtime |
| Cluster · tingkat | 02 Matematika & Algoritma · L0 |
| Referensi | SPEC §2 + implementasi Python di `vectors/build_vectors.py`; kelima port setara |
| Port | Rust (binary32/64, SIMD, `no_std`, FFI C), Go (binary32/64), TypeScript, Python, PHP (binary64) |
| Vector | 223 kasus (113 binary64, 110 binary32) · SHA-256 `38e3c88a...f01496` |
| Test | Rust: unit + vector + kesetaraan SIMD (1.560 vektor acak per kernel) + API + FFI · TS 119 · Python 118 · Go unit + vector · PHP 30 cek API + 114 cek vector · uji asap C |
| Coverage | Rust 92,3% baris (fitur `ffi`) · TS 100% baris / 97,5% cabang · Python 99% · Go 96,3% · PHP 100% baris |
| Kinerja (diukur) | dot 768 binary32: AVX2 56 ns, kernel skalar 72 ns; cosine 768: 163 ns vs 215 ns (criterion, satu mesin x86_64 lingkungan pengembangan, bukan jaminan) |
| Registry | crates.io, npm, PyPI `lombokvector`; Go `github.com/codinglombok/lombokvector/go`; Packagist `codinglombok/lombokvector` (semua belum terbit) |
| Lisensi | Apache-2.0 |

## 1. Tabel gap vs pembanding (jujur)

| Kemampuan | LombokVector 0.2.0 | NumPy | simsimd | faiss |
|---|---|---|---|---|
| Hasil bit-identik lintas bahasa (SPEC + vector) | YA | TIDAK (bergantung BLAS) | TIDAK | TIDAK |
| SIMD x86/ARM/WASM | YA (AVX2, NEON, simd128; percepatan sedang) | YA (BLAS) | YA (banyak varian, jauh lebih cepat) | YA |
| AVX-512, SVE | TIDAK | tergantung BLAS | YA | YA |
| `no_std` tanpa alokator | YA | TIDAK | parsial (C) | TIDAK |
| Indeks pencarian ANN | TIDAK | TIDAK | TIDAK | YA |
| Kuantisasi (int8, biner) | TIDAK | TIDAK | YA | YA |
| Error eksplisit (kosong, beda panjang, non-finite) | YA | TIDAK (NaN/broadcast) | parsial | parsial |
| Tanpa dependensi | YA | TIDAK | YA | TIDAK |

Posisi unik yang dibuktikan test: skor yang dihitung di bahasa berbeda dan di CPU berbeda (AVX2, NEON, skalar) sama sampai bit terakhir, sehingga peringkat, deduplikasi, dan cache hasil konsisten di seluruh sistem.

## 2. Batasan yang Diketahui

1. Percepatan SIMD sedang (sekitar 1,3× untuk dot 768 dimensi pada satu mesin x86_64), karena kode skalar 8 lajur sudah divektorisasi otomatis dan urutan yang ditetapkan melarang FMA serta lebar 16 lajur.
2. Tidak ada AVX-512, SVE, atau kernel SIMD binary64; binary64 memakai kernel skalar.
3. Tidak ada penskalaan ulang: elemen di atas sekitar 1e154 (binary64) atau 1e19 (binary32) membuat jumlah kuadrat meluap dan menghasilkan `NON_FINITE`.
4. TypeScript, Python, dan PHP hanya binary64.
5. Python dan PHP murni interpreter; untuk jutaan vektor gunakan port Rust/Go.
6. Paket PHP berada di subdirektori; Packagist butuh repositori split.
7. Kernel NEON hanya diuji eksekusi di runner Apple Silicon CI; kernel simd128 hanya diuji kompilasi.

## 3. Prinsip Universal (ringkas, untuk publik)

| Prinsip | Status | Bukti |
|---|---|---|
| U1 Mandiri | YA | README tanpa klaim kepemilikan; skenario netral di guide_ §6 |
| U2 Modern | YA | SPEC: IEEE 754-2019, ES2024, Go 1.22, tanggal tinjauan |
| U3 Multi-platform | YA | CI ubuntu/windows/macos (arm64) untuk semua port; target no_std, aarch64, wasm32 |
| U4 Multi-bahasa | YA | 5 port + C; runner vector di semua port |
| U5 Rentang skala | YA | dari mikrokontroler (`no_std` tanpa alokator) hingga batch peringkat |
| U6 Lengkap & unik | SEBAGIAN | tabel gap di atas (tanpa ANN, kuantisasi) |
| U7 Aman & teruji | YA | SPEC §6; `unsafe` terisolasi + `SAFETY`; coverage ≥ 92% semua port; kesetaraan SIMD; cek rasional eksak |
| U8 Ekosistem tanpa kopling | YA | 0 dependensi wajib |
| U9 Internasional | SEBAGIAN | Lang_: tingkat E |
| U10 Lisensi | YA | Apache-2.0 di root dan setiap port |
| U11 Siap registri | SEBAGIAN | crates/npm/PyPI/Go siap; Packagist butuh split repo |
| U12 Dokumentasi | YA | 10 dokumen publik + 2 internal |
| U13 Kerahasiaan & dokumen bersih | YA | doctor: 0 emoji, `.gitignore` ADR-024, dokumen internal lama dikeluarkan dari repo |

*Lisensi dokumen: Apache-2.0 · © codinglombok*
