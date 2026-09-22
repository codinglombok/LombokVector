// LombokVector — Integration tests (matches shared test vectors)
// License: Apache-2.0

use lombokvector::*;

const TOL_F32: f32 = 1e-6;
const TOL_F64: f64 = 1e-14;

fn approx_f32(a: f32, b: f32) -> bool {
    (a - b).abs() < TOL_F32
}

fn approx_f64(a: f64, b: f64) -> bool {
    (a - b).abs() < TOL_F64
}

// ============================================================================
// Cosine similarity
// ============================================================================

#[test]
fn test_cosine_basic() {
    let a = &[1.0_f32, 2.0, 3.0];
    let b = &[4.0_f32, 5.0, 6.0];
    let result = cosine_similarity(a, b).unwrap();
    assert!(approx_f32(result, 0.9746318), "got {result}");
}

#[test]
fn test_cosine_identical() {
    let a = &[1.0_f32, 0.0, 0.0];
    let result = cosine_similarity(a, a).unwrap();
    assert!(approx_f32(result, 1.0), "got {result}");
}

#[test]
fn test_cosine_orthogonal() {
    let a = &[1.0_f32, 0.0, 0.0];
    let b = &[0.0_f32, 1.0, 0.0];
    let result = cosine_similarity(a, b).unwrap();
    assert!(approx_f32(result, 0.0), "got {result}");
}

#[test]
fn test_cosine_opposite() {
    let a = &[1.0_f32, 2.0, 3.0];
    let b = &[-1.0_f32, -2.0, -3.0];
    let result = cosine_similarity(a, b).unwrap();
    assert!(approx_f32(result, -1.0), "got {result}");
}

#[test]
fn test_cosine_f64_basic() {
    let a = &[1.0_f64, 2.0, 3.0];
    let b = &[4.0_f64, 5.0, 6.0];
    let result = cosine_similarity_f64(a, b).unwrap();
    assert!(approx_f64(result, 0.9746318461970762), "got {result}");
}

#[test]
fn test_cosine_f64_negative_mixed() {
    let a = &[-0.5_f64, 0.3, -0.8, 0.1];
    let b = &[0.2_f64, -0.7, 0.4, 0.9];
    let result = cosine_similarity_f64(a, b).unwrap();
    assert!(approx_f64(result, -0.4431293675255979), "got {result}");
}

#[test]
fn test_cosine_768d() {
    let a: Vec<f64> = (0..768).map(|i| (i as f64 * 0.1).sin()).collect();
    let b: Vec<f64> = (0..768).map(|i| (i as f64 * 0.1).cos()).collect();
    let result = cosine_similarity_f64(&a, &b).unwrap();
    assert!(
        (result - 0.012394344943011405).abs() < 1e-10,
        "768d cosine got {result}"
    );
}

// ============================================================================
// Dot product
// ============================================================================

#[test]
fn test_dot_basic() {
    let result = dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
    assert!(approx_f32(result, 32.0), "got {result}");
}

#[test]
fn test_dot_zeros() {
    let result = dot_product(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]).unwrap();
    assert!(approx_f32(result, 0.0), "got {result}");
}

#[test]
fn test_dot_negative() {
    let result = dot_product(&[1.0, -2.0, 3.0], &[-4.0, 5.0, -6.0]).unwrap();
    assert!(approx_f32(result, -32.0), "got {result}");
}

#[test]
fn test_dot_single() {
    let result = dot_product(&[7.0], &[3.0]).unwrap();
    assert!(approx_f32(result, 21.0), "got {result}");
}

#[test]
fn test_dot_f64() {
    let result = dot_product_f64(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
    assert!(approx_f64(result, 32.0), "got {result}");
}

// ============================================================================
// L2 distance
// ============================================================================

#[test]
fn test_l2_basic() {
    let result = l2_distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
    assert!(approx_f32(result, 5.196152), "got {result}");
}

#[test]
fn test_l2_identical() {
    let result = l2_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();
    assert!(approx_f32(result, 0.0), "got {result}");
}

#[test]
fn test_l2_unit_axes() {
    let result = l2_distance(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).unwrap();
    assert!(approx_f32(result, 1.4142135), "got {result}");
}

#[test]
fn test_l2_f64() {
    let result = l2_distance_f64(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
    assert!(approx_f64(result, 5.196152422706632), "got {result}");
}

// ============================================================================
// Normalize
// ============================================================================

#[test]
fn test_normalize_basic() {
    let result = normalize(&[3.0, 4.0]).unwrap();
    assert!(approx_f32(result[0], 0.6), "got {:?}", result);
    assert!(approx_f32(result[1], 0.8), "got {:?}", result);
}

#[test]
fn test_normalize_3d() {
    let result = normalize_f64(&[1.0, 2.0, 3.0]).unwrap();
    assert!(approx_f64(result[0], 0.2672612419124244));
    assert!(approx_f64(result[1], 0.5345224838248488));
    assert!(approx_f64(result[2], 0.8017837257372732));
}

#[test]
fn test_normalize_already_unit() {
    let result = normalize_f64(&[1.0, 0.0, 0.0]).unwrap();
    assert!(approx_f64(result[0], 1.0));
    assert!(approx_f64(result[1], 0.0));
    assert!(approx_f64(result[2], 0.0));
}

// ============================================================================
// L2 norm
// ============================================================================

#[test]
fn test_l2_norm_basic() {
    let result = l2_norm(&[3.0, 4.0]).unwrap();
    assert!(approx_f32(result, 5.0), "got {result}");
}

#[test]
fn test_l2_norm_f64() {
    let result = l2_norm_f64(&[1.0, 2.0, 3.0]).unwrap();
    assert!(approx_f64(result, 3.7416573867739413), "got {result}");
}

// ============================================================================
// Arithmetic
// ============================================================================

#[test]
fn test_vec_add() {
    let result = vec_add(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_vec_sub() {
    let result = vec_sub(&[4.0, 5.0, 6.0], &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(result, vec![3.0, 3.0, 3.0]);
}

#[test]
fn test_vec_mul_scalar() {
    let result = vec_mul_scalar(&[1.0, 2.0, 3.0], 2.5).unwrap();
    assert!(approx_f32(result[0], 2.5));
    assert!(approx_f32(result[1], 5.0));
    assert!(approx_f32(result[2], 7.5));
}

// ============================================================================
// Error cases
// ============================================================================

#[test]
fn test_dimension_mismatch() {
    let result = cosine_similarity(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
    assert!(matches!(result, Err(VectorError::DimensionMismatch { .. })));
}

#[test]
fn test_empty_vector() {
    let result = cosine_similarity(&[], &[]);
    assert!(matches!(result, Err(VectorError::EmptyVector)));
}

#[test]
fn test_zero_magnitude() {
    let result = cosine_similarity(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
    assert!(matches!(result, Err(VectorError::ZeroMagnitude)));
}

#[test]
fn test_normalize_zero() {
    let result = normalize(&[0.0, 0.0]);
    assert!(matches!(result, Err(VectorError::ZeroMagnitude)));
}

// ============================================================================
// Batch operations
// ============================================================================

#[test]
fn test_batch_cosine() {
    let query = &[1.0_f32, 0.0, 0.0];
    let c0: &[f32] = &[1.0, 0.0, 0.0]; // cos = 1.0
    let c1: &[f32] = &[0.0, 1.0, 0.0]; // cos = 0.0
    let c2: &[f32] = &[0.707, 0.707, 0.0]; // cos ≈ 0.707
    let candidates: Vec<&[f32]> = vec![c0, c1, c2];
    let results = batch_cosine(query, &candidates).unwrap();
    // Sorted descending: index 0 (1.0), index 2 (~0.707), index 1 (0.0)
    assert_eq!(results[0].0, 0);
    assert_eq!(results[2].0, 1);
}

#[test]
fn test_batch_l2() {
    let query = &[0.0_f32, 0.0, 0.0];
    let c0: &[f32] = &[1.0, 0.0, 0.0]; // dist = 1.0
    let c1: &[f32] = &[3.0, 4.0, 0.0]; // dist = 5.0
    let candidates: Vec<&[f32]> = vec![c0, c1];
    let results = batch_l2(query, &candidates).unwrap();
    // Sorted ascending: index 0 (1.0), index 1 (5.0)
    assert_eq!(results[0].0, 0);
    assert!(approx_f32(results[0].1, 1.0));
    assert_eq!(results[1].0, 1);
    assert!(approx_f32(results[1].1, 5.0));
}

#[test]
fn test_distance_matrix() {
    let a0: &[f32] = &[1.0, 0.0];
    let a1: &[f32] = &[0.0, 1.0];
    let b0: &[f32] = &[1.0, 0.0];
    let b1: &[f32] = &[0.707, 0.707];
    let va: Vec<&[f32]> = vec![a0, a1];
    let vb: Vec<&[f32]> = vec![b0, b1];
    let matrix = distance_matrix_cosine(&va, &vb).unwrap();
    assert!(approx_f32(matrix[0][0], 1.0)); // a0·b0 = 1.0
    assert!(approx_f32(matrix[1][0], 0.0)); // a1·b0 = 0.0
}

// ============================================================================
// Backend detection
// ============================================================================

#[test]
fn test_active_backend() {
    let backend = lombokvector::active_backend();
    assert!(
        ["portable", "avx2", "avx512", "neon", "wasm-simd"].contains(&backend),
        "unexpected backend: {backend}"
    );
    println!("Active SIMD backend: {backend}");
}
