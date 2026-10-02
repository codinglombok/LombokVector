# LombokVector — Development IDE v0.2.0

## 1. Lingkungan

| Alat | Versi |
|---|---|
| Rust | stable (MSRV 1.70, diperiksa CI); target opsional `thumbv7em-none-eabihf`, `aarch64-unknown-linux-gnu`, `wasm32-unknown-unknown`; `cargo-llvm-cov` |
| Node.js | 20 LTS atau lebih baru (CI: 20, 22, 24) |
| Python | 3.9+ (CI: 3.9, 3.11, 3.13); `pytest`, `coverage` |
| Go | 1.22+ (CI: 1.22, 1.24) |
| PHP | 8.1+ (CI: 8.1, 8.3); ekstensi `pcov` untuk coverage |
| C | kompiler C99 untuk `c-headers/examples/check.c` |
| Bash | untuk `scripts/lombok-doctor.sh` (Windows: Git Bash atau WSL) |

## 2. Perintah

| Direktori | Perintah | Fungsi |
|---|---|---|
| `rust/` | `cargo test --features ffi` | unit, vector, kesetaraan SIMD, API, FFI |
| `rust/` | `cargo test --features portable --test vectors` | vector dengan kernel skalar |
| `rust/` | `cargo clippy --all-targets --features ffi -- -D warnings` · `cargo fmt --check` | lint |
| `rust/` | `cargo check --no-default-features --target thumbv7em-none-eabihf` | build `no_std` tanpa alokator |
| `rust/` | `cargo llvm-cov --features ffi --fail-under-lines 90` | coverage |
| `rust/` | `cargo bench` | benchmark criterion |
| `typescript/` | `npm ci` · `npm run lint` · `npm run coverage` | tipe strict, test dengan ambang |
| `python/` | `python -m pytest` | test |
| `go/` | `go vet ./...` · `go test -cover ./...` | lint, test |
| `php/` | `php tests/run.php` · `php -d pcov.enabled=1 tests/run.php --coverage=90` | test, coverage |
| root | `python3 vectors/build_vectors.py` | bangun ulang vector (gagal bila referensi menyimpang dari hitungan eksak) |
| root | `bash scripts/lombok-doctor.sh LombokVector` | pemeriksaan standar Lombok v3.6 |

## 3. Alur mengubah perilaku

1. Ubah SPEC lebih dulu.
2. Ubah referensi di `vectors/build_vectors.py` dan tambah kasus.
3. Jalankan builder, lalu perbarui `vectors/SHA256SUMS` dan hash di SPEC.
4. Ubah semua port, termasuk kernel SIMD Rust, sampai semua runner dan `simd_matches_reference` hijau.
5. Catat di `CHANGELOG.md`.

Kernel SIMD baru (misalnya AVX-512) hanya boleh ditambahkan bila tetap memakai delapan akumulator lajur dengan urutan §2.1; gunakan `tests/simd_matches_reference.rs` sebagai bukti.

## 4. Arah pengembangan

- API binary32 di TypeScript (`Math.fround`) dan Python (emulasi).
- Kernel SIMD binary64 dengan urutan yang sama.
- Penskalaan ulang opsional untuk elemen sangat besar sebagai operasi terpisah (tanpa mengubah hasil operasi yang ada).

*Lisensi dokumen: Apache-2.0 · © codinglombok*
