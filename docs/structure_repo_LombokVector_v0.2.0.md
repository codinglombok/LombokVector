# LombokVector — Structure Repo v0.2.0

```
LombokVector/
├── README.md · CHANGELOG.md · LICENSE (Apache-2.0) · .editorconfig · .gitattributes
├── .github/workflows/  ci.yml (5 port, target silang, uji asap C, standards) · release.yml (terbit pada tag v*)
├── docs/               10 dokumen publik; masterplan_ dan architecture_ adalah dokumen internal (ADR-024), tidak di-commit
├── vectors/            lombokvector-vectors-v1.json · SHA256SUMS · build_vectors.py (referensi independen + cek rasional eksak)
├── scripts/            lombok-doctor.sh (pemeriksaan standar v3.6)
├── rust/               Cargo.toml · src/{lib,kernels,dispatch,ops,batch,error,sqrt,ffi}.rs · src/simd/{avx2,neon,wasm}.rs
│                       tests/{vectors,simd_matches_reference,api}.rs · benches/bench_ops.rs
├── typescript/         package.json · src/index.ts · tests/{api,vectors}.test.ts · scripts/coverage.mjs
├── python/             pyproject.toml · lombokvector/__init__.py · tests/{test_api,test_vectors}.py
├── go/                 go.mod · lombokvector.go (kernel generik) · api.go · {lombokvector,vectors}_test.go
├── php/                composer.json · src/{LombokVector,VectorError}.php · tests/{run,api,vectors,bootstrap}.php
└── c-headers/          lombokvector.h (API C) · examples/check.c (uji asap)
```

Aturan: perubahan perilaku mengubah SPEC dan vector lebih dulu, lalu semua port; setiap port punya README, LICENSE, dan manifest sendiri; hasil build tidak di-commit.

*Lisensi dokumen: Apache-2.0 · © codinglombok*
