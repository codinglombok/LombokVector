#!/usr/bin/env bash
# lombok doctor (v3.7, varian D-ENV) - pemeriksaan mesin standar Lombok Ecosystem.
#
# Implementasi aturan DEPLOYMENT ENVIRONMENT dari Masterplan v3.6 -> v3.7 (R1-R4):
#   D-ENV-1  setiap job registry-publish WAJIB mendeklarasikan `environment:`
#            di level job (R1); job guard/config/deteksi DILARANG punya
#            environment (R3).
#   D-ENV-2  nama environment harus kanonik (R2):
#            npm|pypi|crates|packagist|docker|github-release|maven|nuget|cpan|pgxn
#   D-ENV-3  repo ber-port PHP wajib punya composer.json di root
#            (Packagist hanya membaca composer.json root dari branch default).
#
# Klasifikasi job publish: id/nama job memuat publish|mirror|gpr|packagist
# (case-insensitive). Job automation seperti release-please sengaja DI LUAR
# lingkup R1: ia jalan pada setiap push ke main dan hanya membuat tag/release,
# bukan unggah registry - environment di situ justru membanjiri tab
# Deployments dengan record no-op.
#
# Pemakaian: scripts/lombok-doctor.sh <RepoName>
set -u
REPO="${1:?usage: $0 <RepoName>}"
fail=0
err() { echo "FAIL: $*"; fail=1; }
ok() { echo "ok:   $*"; }

CANON_RE='^(npm|pypi|crates|packagist|docker|github-release|maven|nuget|cpan|pgxn)$'
PUBLISH_RE='publish|mirror|gpr|packagist'

# ---------------------------------------------------------------- D-ENV-1 + D-ENV-2
if [ ! -d .github/workflows ]; then
  err "D-ENV-1: .github/workflows tidak ditemukan"
else
  shopt -s nullglob
  found=0
  for f in .github/workflows/*.yml .github/workflows/*.yaml; do
    [ -f "$f" ] || continue
    # job-level key = indent 2 persis di bawah `jobs:`; isi job = indent >= 4.
    # Keluaran per job: <job-id>\001<environment>\001<name>
    # (delimiter \001 = SOH, bukan tab: tab adalah IFS whitespace sehingga
    # field kosong akan tergabung dan field bergeser - bug yang sudah pernah
    # terjadi. \001 tidak pernah muncul di teks YAML.)
    records="$(awk '
      /^[[:space:]]*#/ { next }
      /^jobs:/ { inj=1; next }
      inj && /^[^ #]/ { inj=0 }
      inj && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ {
        if (job != "") print job "\001" envv "\001" jname
        job=$0; sub(/^  /,"",job); sub(/:[[:space:]]*$/,"",job)
        envv=""; jname=""
        next
      }
      inj && job != "" {
        if (jname == "" && /^    name:/) { jname=$0; sub(/^ *name:[[:space:]]*/,"",jname) }
        if (/^    environment:/) { envv=$0; sub(/^ *environment:[[:space:]]*/,"",envv) }
      }
      END { if (job != "") print job "\001" envv "\001" jname }
    ' "$f")"
    [ -z "$records" ] && continue
    while IFS=$'\001' read -r job envv jname; do
      [ -n "$job" ] || continue
      envv="$(printf '%s' "$envv" | tr -d '"')"
      label="${jname:-$job}"
      if printf '%s %s' "$job" "$jname" | grep -qiE "$PUBLISH_RE"; then
        found=$((found+1))
        if [ -z "$envv" ]; then
          err "D-ENV-1: $f :: $job ('$label') adalah job registry-publish tanpa environment: (R1)"
        elif printf '%s' "$envv" | grep -qE "$CANON_RE"; then
          ok "D-ENV: $f :: $job -> environment: $envv"
        else
          err "D-ENV-2: $f :: $job environment '$envv' bukan nama kanonik (R2)"
        fi
      else
        if [ -n "$envv" ]; then
          err "D-ENV-1/R3: $f :: $job ('$label') bukan job publish tetapi punya environment: $envv"
        fi
      fi
    done <<< "$records"
  done
  if [ "$found" -gt 0 ]; then
    echo "job registry-publish terdeteksi: $found"
  else
    err "D-ENV-1: tidak ada job registry-publish terdeteksi (klasifikasi salah?)"
  fi
fi

# ---------------------------------------------------------------- D-ENV-3
if [ -d ports/php ] || [ -d php/src ]; then
  if [ -f composer.json ]; then
    ok "D-ENV-3: port PHP terdeteksi dan composer.json root ada"
  else
    err "D-ENV-3: repo ber-port PHP tetapi composer.json root tidak ada"
  fi
else
  echo "skip: D-ENV-3 (bukan repo ber-port PHP)"
fi

if [ "$fail" -eq 0 ]; then
  echo "OK: lombok doctor (D-ENV v3.7) lulus utk $REPO"
else
  echo "lombok doctor (D-ENV v3.7) GAGAL utk $REPO"
  exit 1
fi
