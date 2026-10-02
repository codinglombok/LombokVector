//! Behaviour that the vectors do not cover: error details, in-place
//! normalisation, Display text, f32/f64 symmetry.

use lombokvector::*;

#[test]
fn error_codes_and_display() {
    let e = dot_product(&[1.0], &[1.0, 2.0]).unwrap_err();
    assert_eq!(
        e,
        VectorError::DimensionMismatch {
            expected: 1,
            actual: 2
        }
    );
    assert_eq!(e.code(), "DIMENSION_MISMATCH");
    assert_eq!(
        e.to_string(),
        "DIMENSION_MISMATCH: expected 1 elements, got 2"
    );
    assert_eq!(
        VectorError::EmptyVector.to_string(),
        "EMPTY_VECTOR: vector has no elements"
    );
    assert_eq!(
        VectorError::ZeroMagnitude.to_string(),
        "ZERO_MAGNITUDE: vector has magnitude zero"
    );
    assert_eq!(
        VectorError::NonFinite.to_string(),
        "NON_FINITE: result is infinite or NaN"
    );
    assert_eq!(VectorError::NonFinite.code(), "NON_FINITE");
    let boxed: Box<dyn std::error::Error> = Box::new(VectorError::EmptyVector);
    assert!(boxed.to_string().starts_with("EMPTY_VECTOR"));
}

#[test]
fn non_finite_inputs_are_rejected() {
    assert_eq!(
        dot_product(&[f32::NAN], &[1.0]),
        Err(VectorError::NonFinite)
    );
    assert_eq!(l2_norm_f64(&[f64::INFINITY]), Err(VectorError::NonFinite));
    assert_eq!(
        cosine_similarity(&[1.0, f32::NAN], &[1.0, 1.0]),
        Err(VectorError::NonFinite)
    );
    assert_eq!(
        l2_distance(&[f32::MAX], &[-f32::MAX]),
        Err(VectorError::NonFinite)
    );
    assert_eq!(
        vec_sub_f64(&[f64::MAX], &[-f64::MAX]),
        Err(VectorError::NonFinite)
    );
    assert_eq!(
        vec_mul_scalar(&[2.0], f32::MAX),
        Err(VectorError::NonFinite)
    );
}

#[test]
fn normalize_inplace_keeps_input_on_error() {
    let mut v = [0.0f32, 0.0];
    assert_eq!(normalize_inplace(&mut v), Err(VectorError::ZeroMagnitude));
    assert_eq!(v, [0.0, 0.0]);
    let mut e: [f64; 0] = [];
    assert_eq!(normalize_inplace_f64(&mut e), Err(VectorError::EmptyVector));
    let mut w = [3.0f64, 4.0];
    normalize_inplace_f64(&mut w).unwrap();
    // a[i] * (1 / norm), not a[i] / norm (SPEC 3.5)
    assert_eq!(w, [3.0 * 0.2, 4.0 * 0.2]);
    assert_eq!(w[0], 0.6000000000000001);
    // the square of a subnormal underflows to 0
    let mut tiny = [f64::from_bits(1)];
    assert_eq!(
        normalize_inplace_f64(&mut tiny),
        Err(VectorError::ZeroMagnitude)
    );
    assert_eq!(tiny[0].to_bits(), 1);
    // the sum of squares overflows for very large elements (no rescaling, SPEC 3.5)
    let mut big = [f32::MAX, 1.0];
    assert_eq!(normalize_inplace(&mut big), Err(VectorError::NonFinite));
    assert_eq!(big, [f32::MAX, 1.0]);
}

#[test]
fn inner_product_is_dot_product() {
    let (a, b) = ([1.0f32, -2.0, 0.5], [3.0f32, 0.25, 8.0]);
    assert_eq!(inner_product(&a, &b), dot_product(&a, &b));
    let (c, d) = ([1.0f64, -2.0], [3.0f64, 0.25]);
    assert_eq!(inner_product_f64(&c, &d), dot_product_f64(&c, &d));
}

#[test]
fn cosine_is_clamped() {
    let v = [0.1f64, 0.2, 0.3];
    let c = cosine_similarity_f64(&v, &v).unwrap();
    assert!(c <= 1.0);
    let w = [-0.1f64, -0.2, -0.3];
    assert!(cosine_similarity_f64(&v, &w).unwrap() >= -1.0);
}

#[test]
fn batch_ranking_and_errors() {
    let q = [1.0f32, 1.0];
    let cands: [&[f32]; 4] = [&[1.0, 0.0], &[0.0, 1.0], &[2.0, 2.0], &[0.0, 0.0]];
    let r = batch_cosine(&q, &cands).unwrap();
    assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![2, 0, 1, 3]);
    assert_eq!(r[3].1, 0.0);
    let d = batch_dot(&q, &cands).unwrap();
    assert_eq!(d.iter().map(|x| x.0).collect::<Vec<_>>(), vec![2, 0, 1, 3]);
    let l = batch_l2_f64(&[0.0, 0.0], &[&[3.0, 4.0], &[1.0, 0.0], &[0.0, 1.0]]).unwrap();
    assert_eq!(l, vec![(1, 1.0), (2, 1.0), (0, 5.0)]);
    assert_eq!(batch_dot_f64(&[], &[]), Err(VectorError::EmptyVector));
    assert_eq!(
        batch_l2(&q, &[&[1.0, 2.0, 3.0]]),
        Err(VectorError::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    );
    assert_eq!(
        batch_cosine_f64(&[0.0, 0.0], &[]),
        Err(VectorError::ZeroMagnitude)
    );
    assert_eq!(
        batch_dot(&[f32::MAX], &[&[f32::MAX]]),
        Err(VectorError::NonFinite)
    );
}

#[test]
fn distance_matrix() {
    let a: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 0.0]];
    let b: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
    assert_eq!(
        distance_matrix_cosine_f64(&a, &b).unwrap(),
        vec![vec![1.0, 0.0], vec![0.0, 0.0]]
    );
    let e: [&[f32]; 1] = [&[]];
    assert_eq!(
        distance_matrix_cosine(&e, &e),
        Err(VectorError::EmptyVector)
    );
    assert_eq!(
        distance_matrix_cosine(&[], &[&[1.0f32][..]]),
        Err(VectorError::EmptyVector)
    );
    assert_eq!(
        distance_matrix_cosine(&[&[1.0f32, 0.0][..]], &[&[1.0f32][..]]),
        Err(VectorError::DimensionMismatch {
            expected: 2,
            actual: 1
        })
    );
}
