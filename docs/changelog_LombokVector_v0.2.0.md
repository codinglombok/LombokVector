# LombokVector — Changelog v0.2.0

Ringkasan; rincian lengkap di [CHANGELOG.md](../CHANGELOG.md). Entri terbaru di depan.

| Versi | Tanggal | Jenis | Ringkasan |
|---|---|---|---|
| 0.2.0 | 2026-10-02 | perubahan hasil numerik (0.x) + perbaikan | Urutan penjumlahan 8 lajur tanpa FMA di semua port dan backend SIMD, sehingga hasil bit-identik; SPEC + 223 kasus vector (pola bit IEEE 754) dijalankan kelima port; error `NON_FINITE`; cosine di-clamp; batch stabil; API C diimplementasikan (fitur `ffi`); PHP diperbaiki (0.1 gagal di-parse); build `no_std` diperbaiki dengan `sqrt` perangkat lunak; path modul Go `/go`; AVX-512 dan fitur Cargo tanpa efek dihapus; 10 dokumen standar; klaim README dikoreksi |
| 0.1.0 | 2026-09-22 | awal | Lima port dengan urutan penjumlahan berbeda-beda (tidak terbit di registry) |

*Lisensi dokumen: Apache-2.0 · © codinglombok*
