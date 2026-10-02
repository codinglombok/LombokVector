# LombokVector — Map v0.2.0

## 1. Posisi di ekosistem

```
Cluster 02 Matematika & Algoritma · tingkat L0 (tanpa dependensi Lombok wajib)

L0  LombokVector
     dependensi wajib    : (tidak ada)
     dependensi opsional : (tidak ada)
     dependensi dev      : serde_json, criterion (Rust); typescript, @types/node (TS); pytest, coverage (Python, CI)
```

## 2. Contoh pemakai di ekosistem

Library ini mandiri dan dapat dipakai siapa pun. Library dan aplikasi Lombok yang dapat memakainya (arah dependensi selalu pemakai ke library):

| Pemakai | Pemakaian |
|---|---|
| LombokSimHash (library) | jarak antar-vektor fitur, sebagai pembanding hasil |
| LombokRAGFrameworks (aplikasi) | peringkat dokumen berdasarkan cosine similarity embedding |
| LombokMiner (aplikasi) | jarak antar-titik untuk pengelompokan |
| LombokCharts (library) | normalisasi deret sebelum digambar |

## 3. Peta fitur x port

| Fitur | Rust | TypeScript | Python | Go | PHP | C |
|---|---|---|---|---|---|---|
| dot / norm / l2 / cosine (SPEC 3.1-3.4) | YA | YA | YA | YA | YA | YA (sebagian) |
| normalize, add, sub, scale (SPEC 3.5-3.6) | YA | YA | YA | YA | YA | normalize in-place binary32 |
| batch + matriks (SPEC 3.7-3.8) | YA | YA | YA | YA | YA | TIDAK |
| binary32 (SPEC 5) | YA | TIDAK | TIDAK | YA | TIDAK | YA |
| kernel SIMD (SPEC 2.3) | AVX2, NEON, simd128 | TIDAK | TIDAK | TIDAK | TIDAK | lewat Rust |
| vector | 223/223 | 113/113 | 113/113 | 223/223 | 113/113 | uji asap |

*Lisensi dokumen: Apache-2.0 · © codinglombok*
