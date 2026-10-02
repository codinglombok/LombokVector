# LombokVector — Bahasa & i18n v0.2.0

| Atribut | Nilai |
|---|---|
| Versi | 0.2.0 |
| Tingkat i18n (masterplan §13) | **E**: kode error dan dokumentasi; perhitungan bebas locale |
| Katalog pesan | belum ada berkas `locales/`; ID pesan dicadangkan di §2 |
| Fallback | teks bahasa Inggris tertanam di kode |
| Cakupan katalog saat ini | en + id (2/20) di tabel §2; Nusantara 0/6 |

## 1. Prinsip

1. Setiap error membawa kode stabil (SPEC §4). Program MUST memeriksa kode, bukan teks.
2. Library tidak membaca locale. Angka diterima sebagai nilai numerik, bukan teks, sehingga pemisah desimal lokal tidak berperan.
3. Hasil numerik tidak bergantung pada platform, locale, atau mode floating point bawaan (pembulatan ke terdekat-genap diasumsikan, sesuai bawaan semua runtime yang didukung).

## 2. Katalog ID pesan (dicadangkan)

| ID | Kode | en | id |
|---|---|---|---|
| `lombokvector.dimension_mismatch` | `DIMENSION_MISMATCH` | Expected {$expected} elements, got {$actual}. | Diharapkan {$expected} elemen, didapat {$actual}. |
| `lombokvector.empty_vector` | `EMPTY_VECTOR` | Vector has no elements. | Vektor tidak memiliki elemen. |
| `lombokvector.zero_magnitude` | `ZERO_MAGNITUDE` | Vector has magnitude zero. | Panjang vektor nol. |
| `lombokvector.non_finite` | `NON_FINITE` | Result is infinite or NaN. | Hasil tak hingga atau bukan angka. |

## 3. RTL

Library tidak menghasilkan UI; RTL tidak berlaku.

## 4. Rencana

- Berkas `locales/en` dan `locales/id` dimuat lewat LombokLocale bila terpasang (dependensi opsional, bukan wajib).

*Lisensi dokumen: Apache-2.0 · © codinglombok*
