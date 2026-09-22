/**
 * LombokVector — C FFI header
 * Generated from Rust crate `lombokvector` via cbindgen.
 * Part of LombokRAGFrameworks (@codinglombok).
 * License: Apache-2.0
 */

#ifndef LOMBOKVECTOR_H
#define LOMBOKVECTOR_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Error codes returned by LombokVector functions.
 * 0 = success, non-zero = error.
 */
typedef enum {
    LOMBOKVECTOR_OK = 0,
    LOMBOKVECTOR_ERR_DIMENSION_MISMATCH = 1,
    LOMBOKVECTOR_ERR_EMPTY_VECTOR = 2,
    LOMBOKVECTOR_ERR_ZERO_MAGNITUDE = 3,
} lombokvector_error_t;

/**
 * Cosine similarity between two f32 vectors.
 * Result written to `out`. Returns error code.
 */
lombokvector_error_t lombokvector_cosine_f32(
    const float *a, const float *b, size_t len, float *out);

/**
 * Cosine similarity between two f64 vectors.
 */
lombokvector_error_t lombokvector_cosine_f64(
    const double *a, const double *b, size_t len, double *out);

/**
 * Dot product of two f32 vectors.
 */
lombokvector_error_t lombokvector_dot_f32(
    const float *a, const float *b, size_t len, float *out);

/**
 * Dot product of two f64 vectors.
 */
lombokvector_error_t lombokvector_dot_f64(
    const double *a, const double *b, size_t len, double *out);

/**
 * Euclidean (L2) distance between two f32 vectors.
 */
lombokvector_error_t lombokvector_l2_f32(
    const float *a, const float *b, size_t len, float *out);

/**
 * Euclidean (L2) distance between two f64 vectors.
 */
lombokvector_error_t lombokvector_l2_f64(
    const double *a, const double *b, size_t len, double *out);

/**
 * L2 norm (magnitude) of an f32 vector.
 */
lombokvector_error_t lombokvector_l2_norm_f32(
    const float *a, size_t len, float *out);

/**
 * Normalize an f32 vector in-place to unit length.
 */
lombokvector_error_t lombokvector_normalize_inplace_f32(
    float *a, size_t len);

/**
 * Returns the name of the active SIMD backend as a null-terminated string.
 * The returned pointer is valid for the lifetime of the process.
 */
const char *lombokvector_active_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* LOMBOKVECTOR_H */
