/**
 * LombokVector - C API.
 *
 * Implemented by the Rust crate `lombokvector` with the `ffi` feature
 * (rust/src/ffi.rs). Build the library with:
 *
 *     cd rust
 *     cargo rustc --release --features ffi --crate-type cdylib     (shared)
 *     cargo rustc --release --features ffi --crate-type staticlib  (static)
 *
 * Results are bit-identical to the other LombokVector ports
 * (docs/SPEC_LombokVector_v0.2.0.md).
 *
 * License: Apache-2.0
 */

#ifndef LOMBOKVECTOR_H
#define LOMBOKVECTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Return codes: 0 = success, otherwise an error. */
typedef enum {
    LOMBOKVECTOR_OK = 0,
    LOMBOKVECTOR_ERR_DIMENSION_MISMATCH = 1, /* not produced: both inputs share `len` */
    LOMBOKVECTOR_ERR_EMPTY_VECTOR = 2,       /* len == 0 */
    LOMBOKVECTOR_ERR_ZERO_MAGNITUDE = 3,     /* a vector has magnitude zero */
    LOMBOKVECTOR_ERR_NON_FINITE = 4,         /* input not finite or result overflowed */
    LOMBOKVECTOR_ERR_NULL_POINTER = 5        /* a required pointer is NULL */
} lombokvector_error_t;

/* Every function reads `len` elements from `a` (and `b`) and writes one value to `out`. */

int32_t lombokvector_cosine_f32(const float *a, const float *b, size_t len, float *out);
int32_t lombokvector_cosine_f64(const double *a, const double *b, size_t len, double *out);
int32_t lombokvector_dot_f32(const float *a, const float *b, size_t len, float *out);
int32_t lombokvector_dot_f64(const double *a, const double *b, size_t len, double *out);
int32_t lombokvector_l2_f32(const float *a, const float *b, size_t len, float *out);
int32_t lombokvector_l2_f64(const double *a, const double *b, size_t len, double *out);
int32_t lombokvector_l2_norm_f32(const float *a, size_t len, float *out);

/** Scales `a` to unit length in place; `a` is unchanged on error. */
int32_t lombokvector_normalize_inplace_f32(float *a, size_t len);

/** Name of the active binary32 kernel ("avx2", "neon", "wasm-simd128" or "portable"); static storage. */
const char *lombokvector_active_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* LOMBOKVECTOR_H */
