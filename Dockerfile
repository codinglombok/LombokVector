# LombokVector container image: the C FFI shared library, its header, and a
# self-verifying smoke test as the entrypoint. The library computes vector
# math with bit-identical IEEE 754 results in every language port; this image
# ships the native C ABI surface (c-headers/lombokvector.h, feature "ffi").
#
# Build:  docker build -t lombokvector:local .
# Run:    docker run --rm lombokvector:local
#         (prints "ok, backend <name>" and exits 0 on success)
#
# The build stage uses the exact recipe proven by the ci.yml c-api job:
#   cargo rustc --release --features ffi --crate-type cdylib
# The runtime stage is debian:bookworm-slim, glibc-compatible with the
# builder, and contains no toolchain at all.

# --- build stage -----------------------------------------------------------
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY rust/ rust/
COPY c-headers/ c-headers/
# The crate has zero runtime dependencies, so this build needs no network
# access beyond the toolchain image itself.
RUN cargo rustc --release --features ffi --crate-type cdylib \
      --manifest-path rust/Cargo.toml \
 && cc -O2 -Wall -Wextra -Werror c-headers/examples/check.c \
      -Ic-headers -Lrust/target/release -llombokvector \
      -o /tmp/lombokvector-check

# --- runtime stage ---------------------------------------------------------
FROM debian:bookworm-slim
ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="LombokVector" \
      org.opencontainers.image.description="Vector math with bit-identical results across languages: dot product, cosine similarity, L2 distance, normalization, batch ranking (C FFI runtime image)" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/codinglombok/LombokVector" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=build /src/rust/target/release/liblombokvector.so /usr/local/lib/
COPY --from=build /src/c-headers/lombokvector.h /usr/local/include/
COPY --from=build /tmp/lombokvector-check /usr/local/bin/lombokvector-check

# ldconfig registers the shared library system-wide; LD_LIBRARY_PATH is kept
# as a belt-and-braces fallback for containers that drop the ld cache.
RUN ldconfig \
 && useradd --system --no-create-home --shell /usr/sbin/nologin lombok

USER lombok
ENV LD_LIBRARY_PATH=/usr/local/lib

# Self-verification: the smoke test computes dot/cosine/normalize through the
# shared library and prints the active SIMD backend.
ENTRYPOINT ["/usr/local/bin/lombokvector-check"]
