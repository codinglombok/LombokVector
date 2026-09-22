# LombokVector — Masterplan

> Release & distribution plan for LombokVector across all registries.
> Part of [LombokRAGFrameworks](https://github.com/codinglombok) · Apache-2.0

---

## 1. Mission

Provide the fastest, zero-dependency vector math library for RAG systems — available in every major language, installable from every major registry, and consistent in behavior across all ports.

## 2. Registry Distribution

### 2.1 Target Registries

| Language | Registry | Package Name | Install Command |
|----------|----------|-------------|-----------------|
| Rust | crates.io | `lombokvector` | `cargo add lombokvector` |
| TypeScript | npm | `lombokvector` | `npm install lombokvector` |
| TypeScript | GitHub Packages | `@codinglombok/lombokvector` | `npm install @codinglombok/lombokvector` |
| Python | PyPI | `lombokvector` | `pip install lombokvector` |
| Go | proxy.golang.org | `github.com/codinglombok/lombokvector` | `go get github.com/codinglombok/lombokvector` |
| PHP | Packagist | `codinglombok/lombokvector` | `composer require codinglombok/lombokvector` |

### 2.2 Registry Config Files

| Registry | Config File | Location |
|----------|------------|----------|
| crates.io | `Cargo.toml` | `rust/Cargo.toml` |
| npm | `package.json` | `ts/package.json` |
| PyPI | `pyproject.toml` | `python/pyproject.toml` |
| Go | `go.mod` | `go/go.mod` |
| Packagist | `composer.json` | `php/composer.json` |

### 2.3 Required Secrets (GitHub)

| Secret | Registry | Purpose |
|--------|----------|---------|
| `CRATES_IO_TOKEN` | crates.io | `cargo publish` authentication |
| `NPM_TOKEN` | npm | `npm publish` authentication |
| `PYPI_TOKEN` | PyPI | PyPI Trusted Publisher or token |
| `GITHUB_TOKEN` | GPR | Auto-provided by GitHub Actions |
| Packagist webhook | Packagist | Auto-trigger on GitHub push (configured in Packagist dashboard) |

## 3. Version Strategy

### 3.1 Semantic Versioning

All ports share the same version number. A single `git tag v0.1.0` triggers publish to all registries simultaneously via `release.yml`.

```
v0.1.0 → Initial release
v0.1.x → Patch releases (bug fixes, perf improvements)
v0.2.0 → New operations or API additions
v1.0.0 → Stable API (no breaking changes after this)
```

### 3.2 Version Locations

Version must be updated in ALL files before tagging:

1. `rust/Cargo.toml` → `version = "x.y.z"`
2. `ts/package.json` → `"version": "x.y.z"`
3. `python/pyproject.toml` → `version = "x.y.z"`
4. `php/composer.json` → `"version": "x.y.z"`
5. `CHANGELOG.md` → New entry

Go versioning is tag-based (no file to update).

## 4. Quality Gates

### 4.1 Pre-Release Checklist

- [ ] All CI checks pass (ci.yml green on main)
- [ ] Benchmarks show no regression >15% (bench.yml)
- [ ] Security audit clean (audit.yml)
- [ ] CHANGELOG.md updated
- [ ] Version bumped in all config files
- [ ] Test vectors JSON unchanged (or all ports updated)

### 4.2 Post-Release Verification

- [ ] `cargo add lombokvector@x.y.z` installs
- [ ] `npm install lombokvector@x.y.z` installs
- [ ] `pip install lombokvector==x.y.z` installs
- [ ] `go get github.com/codinglombok/lombokvector@vx.y.z` installs
- [ ] `composer require codinglombok/lombokvector:x.y.z` installs
- [ ] GitHub Release created with install table

## 5. Roadmap

### v0.1.0 — Initial Release
- [x] Core operations: cosine, dot, l2, normalize
- [x] SIMD backends: AVX2, AVX-512 (2×AVX2), NEON, WASM SIMD, portable
- [x] Runtime CPUID auto-detection
- [x] Batch operations (cosine, dot, l2)
- [x] Distance matrix
- [x] Arithmetic (add, sub, mul_scalar)
- [x] 5 language ports: Rust, TypeScript, Python, Go, PHP
- [x] C FFI headers
- [x] Shared test vectors JSON
- [x] CI/CD for all languages
- [x] Multi-registry publish pipeline

### v0.2.0 — Performance & Features (planned)
- [ ] NAPI-RS bindings (Rust→Node.js native addon)
- [ ] PyO3 bindings (Rust→Python native extension)
- [ ] Product quantization helpers
- [ ] Sparse vector support
- [ ] f16/bf16 support (half precision)

### v0.3.0 — Advanced (planned)
- [ ] CUDA/Metal GPU kernels
- [ ] Streaming batch (iterator-based)
- [ ] Memory-mapped vector support
- [ ] Distance matrix parallelization (rayon)

### v1.0.0 — Stable API
- [ ] API freeze
- [ ] Full benchmark suite with comparison tables
- [ ] 100% doc coverage

## 6. Ecosystem Position

```
LombokRAGFrameworks Dependency Graph (Wave 1-4)
================================================

WAVE 1 (zero-dep):
  ┌─────────────────┐
  │  LombokVector ◄──── THIS LIBRARY
  ├─────────────────┤
  │  LombokSimHash  │
  │  LombokCompress │
  │  LombokSerde    │
  │  LombokAsync    │
  │  LombokCLIParse │
  │  LombokJSON     │
  │  LombokXML      │
  │  LombokHTML     │
  │  LombokValidator│
  └─────────────────┘
          │
          ▼
WAVE 2 (depends on Wave 1):
  ┌─────────────────┐
  │  LombokHNSW     │──── uses LombokVector
  │  LombokQuantize │──── uses LombokVector
  │  LombokRRF      │──── uses LombokVector
  │  LombokBM25     │
  │  LombokTokenizer│
  │  LombokHTTP     │
  │  LombokTLS      │
  └─────────────────┘
          │
          ▼
WAVE 3-4 (RAG pipeline):
  ┌─────────────────┐
  │  LombokColBERT  │──── uses LombokVector
  │  LombokEmbedding│──── uses LombokVector
  │  LombokEval     │──── uses LombokVector
  │  LombokChunker  │
  │  LombokReranker │
  │  LombokPipeline │
  │  LombokAgent    │
  │  LombokAgenticAuto│── uses LombokVector
  └─────────────────┘
```

## 7. Template Role

LombokVector is the **TEMPLATE** for all 39 libraries in LombokRAGFrameworks. Every convention established here is replicated:

| Convention | Detail |
|------------|--------|
| Folder structure | `rust/`, `ts/`, `python/`, `go/`, `php/`, `c-headers/`, `vectors/` |
| CI/CD | Same workflow structure (ci, release, audit, bench) |
| Registry targets | Same 6 registries |
| Test vectors | Shared JSON, cross-language validation |
| Documentation | `architecture_repo.md`, `masterplan_repo.md`, `map/`, `CHANGELOG.md` |
| License | Apache-2.0 header in every file |
| Naming | `lombok<name>` lowercase across all registries |
| Code style | `.editorconfig`, `rustfmt.toml`, `clippy.toml` per-language linters |
