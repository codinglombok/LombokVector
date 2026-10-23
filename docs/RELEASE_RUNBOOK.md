# LombokVector — Release Runbook (staged versions)

Staged (step-by-step) release procedure for every registry and GitHub
Releases. This runbook supersedes the publishing flow in
`docs/how_to_dist_LombokVector_v0.1.0.md`; that document stays the reference
for building local artifacts.

Ecosystem standard: every Lombok Ecosystem repository always carries workflows
for (a) releasing to its registries and (b) creating the GitHub Release. This
repository implements the standard with:

| Workflow | Trigger | Purpose |
|---|---|---|
| `.github/workflows/release.yml` | tag `v*` push | full release: version check, crates.io, npm, PyPI, container image (GHCR + optional Docker Hub), `go/v*` tag (automated), GitHub Release with checksummed bundle |
| `.github/workflows/release-npm.yml` | manual (`workflow_dispatch`) | staged npm publish, `dry_run` default `true` |
| `.github/workflows/release-crates.yml` | manual | staged crates.io publish, `dry_run` default `true` |
| `.github/workflows/release-pypi.yml` | manual | staged PyPI publish, `dry_run` default `true` |
| `.github/workflows/release-docker.yml` | manual | staged container push, `dry_run` default `true` |

## 1. Version ladder policy

- Published ladder: **v0.1.0 -> v0.2.0 -> v0.3.0 -> ...** (Semantic
  Versioning, 0.x while the API may still change).
- `v0.1.0` is the first version published to any registry. It consolidates
  the two earlier unpublished internal drafts (2026-09-22 and 2026-10-02), so
  no published version number was skipped.
- Tag formats: `vMAJOR.MINOR.PATCH` (full release) and
  `go/vMAJOR.MINOR.PATCH` (Go module tag; pushed automatically by the
  `go-tag` job of `release.yml`).
- A tag is never moved, reused, or deleted. crates.io and PyPI versions are
  immutable: fixes always ship as a new version.
- One version equals one commit on `main` where the tag, every manifest, the
  top `CHANGELOG.md` entry, and the `docs/*_v<version>.md` filenames all
  agree (enforced by `check-version` and `scripts/lombok-doctor.sh`).

## 2. One-time setup

1. Secrets (repo Settings -> Secrets and variables -> Actions):
   - `NPM_TOKEN` — npm automation token for the package `lombokvector`.
   - `CARGO_REGISTRY_TOKEN` — crates.io token for the crate `lombokvector`.
   - PyPI needs no token: trusted publishing (OIDC) with environment `pypi`;
     register the pending publisher on pypi.org for this repo, workflow file
     `release.yml` (and `release-pypi.yml`), environment `pypi`.
2. Environments (Settings -> Environments). The workflows reference `crates`,
   `npm`, `pypi`, `docker`, and `github-release`; GitHub creates them on first
   run. To make releases staged, add required reviewers to each environment —
   every publish then waits for your approval, and you approve them one
   registry at a time (this is the staged gate). Without reviewers they pass
   through automatically.
3. Container image: GHCR needs no secret (`GITHUB_TOKEN` with
   `packages: write` is enough). Docker Hub is optional: set
   `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets and the workflows push
   there too; without them the Docker Hub steps are skipped automatically.
   The image is `ghcr.io/codinglombok/lombokvector` — the C FFI shared
   library, its header, and a self-verifying smoke test as the entrypoint.
4. Packagist (PHP): one-time manual submission at
   <https://packagist.org/packages/submit> — paste
   `https://github.com/codinglombok/LombokVector` and submit. This only works
   after `main` is pushed: Packagist reads the root `composer.json` of the
   default branch (a `composer.json` inside `php/` is invisible to it, which
   is why the manifest lives at the repo root and points at `php/src`).
   Versions come from the `v*` tags; after the first submission Packagist
   auto-syncs on every push (no workflow, no secret).

## 3. Stage 1 — release v0.1.0 (first public release)

```bash
# 0. pre-flight (local)
git switch main && git pull
bash scripts/lombok-doctor.sh LombokVector
cd typescript && npm ci && npm test && cd ..
cd python && python -m pytest -q && cd ..

# 1. dry-run rehearsal of each registry (optional but recommended)
gh workflow run release-npm.yml    --ref main -f tag=v0.1.0 -f dry_run=true
gh workflow run release-crates.yml --ref main -f tag=v0.1.0 -f dry_run=true
gh workflow run release-pypi.yml   --ref main -f tag=v0.1.0 -f dry_run=true
gh workflow run release-docker.yml --ref main -f tag=v0.1.0 -f dry_run=true
gh run watch   # inspect each run; all four must be green

# 2. cut the release tag (annotated) and push it
git tag -a v0.1.0 -m "LombokVector 0.1.0 — first public release (bit-identical vector math)"
git push origin v0.1.0

# 3. watch the pipeline
gh run watch   # check-version -> crates + npm + pypi + docker -> go-tag -> github-release
```

4. Staged approvals (if required reviewers are configured): approve
   `npm` first, verify on npmjs.com, then `crates`, verify on crates.io, then
   `pypi`, verify on pypi.org, then `docker`, verify on GHCR. `github-release`
   last, so the release page only appears with the packages already live.
5. What the tag run does automatically:
   - `check-version` — tag = every manifest version, doctor passes.
   - `go-tag` — pushes `go/v0.1.0` (skipped if it already exists).
   - `github-release` — builds `LombokVector_0.1.0_bundle.zip` from tracked
     files (`git archive`; internal documents excluded, ADR-024), writes
     `SHA256SUMS`, and creates the GitHub Release with generated notes.
   - `crates` / `npm` / `pypi` — re-run tests, then publish.
   - `docker` — builds the image for linux/amd64 and linux/arm64 and pushes
     `ghcr.io/codinglombok/lombokvector:{0.1.0,v0.1.0,latest,sha-<commit>}`
     (plus the same set on Docker Hub when its secrets exist).

## 4. Stage 2 and later — v0.2.0, v0.3.0, ... (staged version increases)

Repeat this block for every next version on the ladder:

1. SPEC-first protocol: if the contract changes, edit
   `docs/SPEC_LombokVector_v<new>.md` first, then `vectors/build_vectors.py`,
   rebuild the vectors, refresh `vectors/SHA256SUMS`, and port the change to
   every runner until all ports are bit-identical again.
2. Bump everything in one step:

   ```bash
   scripts/bump_version.sh 0.2.0
   ```

   The script rewrites every manifest, `package-lock.json`,
   `python/lombokvector/__init__.py`, the test assertion, renames the
   `docs/*_v<old>.md` files, and updates the version references.
3. Write the new top section in `CHANGELOG.md` (must be `[0.2.0]` — the
   doctor checks this) and the matching row in
   `docs/changelog_LombokVector_v0.2.0.md`.
4. Run the doctor and the port test suites; push to `main`; wait for green
   CI.
5. Tag and publish exactly as in Stage 1, with the new version:

   ```bash
   git tag -a v0.2.0 -m "LombokVector 0.2.0 — <one-line summary>"
   git push origin v0.2.0
   ```

Never reuse or move a tag; a broken release is fixed by the next version,
never by re-tagging.

## 5. Post-release verification (any version)

```bash
npm view lombokvector@0.1.0 version
cargo search lombokvector --limit 1
pip index versions lombokvector
go list -m github.com/codinglombok/lombokvector/go@v0.1.0
docker run --rm ghcr.io/codinglombok/lombokvector:0.1.0   # prints "ok, backend ..."
docker buildx imagetools inspect ghcr.io/codinglombok/lombokvector:0.1.0
gh release view v0.1.0
gh release download v0.1.0 -D /tmp/rel && sha256sum -c /tmp/rel/SHA256SUMS
```

Then update the README "Installation" table (drop "not yet published") and
record the release in `CHANGELOG.md`.

## 6. Playbook B — per-registry publishing (rehearsal and emergencies)

The four `Release *` workflows can publish one registry at a time:

```bash
gh workflow run release-npm.yml --ref main -f tag=v0.1.0 -f dry_run=false
```

Rules: real publishes require `dry_run=false` AND a matching `tag` input
(fail-closed otherwise); each workflow still runs the port's full test suite;
duplicate publishes of the same version fail by design. If you use Playbook B
for all three registries and then push `v0.1.0`, the registry jobs of
`release.yml` will fail with "version already exists" — that is the expected
fail-closed behavior; the release is complete once `check-version`, `go-tag`,
and `github-release` succeed. Prefer Playbook A for a fully green run.

## 7. Rollback

```bash
cargo yank --version 0.1.0            # crates.io
npm deprecate lombokvector@0.1.0 "use 0.1.1 instead"
# PyPI: yank from the release page (web UI)
gh release delete v0.1.0 --yes        # GitHub Release page only
# Container: image tags are mutable — push a corrected image with the same
# version tag if the defect is packaging-only; a code defect ships as 0.1.1.
```

Tags are never deleted or moved (the Go module cache pins them); ship a fix
version instead.

*Lisensi dokumen: Apache-2.0 · © codinglombok*
