// LombokVector — Batch operations (requires std)
// License: Apache-2.0

use crate::error::VectorError;
use crate::simd::dispatch;

/// Compute cosine similarity of a query against N candidate vectors.
/// Returns Vec of (index, similarity) sorted by descending similarity.
pub fn batch_cosine(query: &[f32], candidates: &[&[f32]]) -> Result<Vec<(usize, f32)>, VectorError> {
    if query.is_empty() {
        return Err(VectorError::EmptyVector);
    }
    let q_norm = dispatch::sum_squares_f32(query).sqrt();
    if q_norm == 0.0 {
        return Err(VectorError::ZeroMagnitude);
    }

    let mut results: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());

    for (i, &cand) in candidates.iter().enumerate() {
        if cand.len() != query.len() {
            return Err(VectorError::DimensionMismatch {
                expected: query.len(),
                actual: cand.len(),
            });
        }
        let dot = dispatch::dot_f32(query, cand);
        let c_norm = dispatch::sum_squares_f32(cand).sqrt();
        if c_norm == 0.0 {
            results.push((i, 0.0));
        } else {
            results.push((i, dot / (q_norm * c_norm)));
        }
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    Ok(results)
}

/// Compute dot product of a query against N candidate vectors.
pub fn batch_dot(query: &[f32], candidates: &[&[f32]]) -> Result<Vec<(usize, f32)>, VectorError> {
    if query.is_empty() {
        return Err(VectorError::EmptyVector);
    }

    let mut results: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());

    for (i, &cand) in candidates.iter().enumerate() {
        if cand.len() != query.len() {
            return Err(VectorError::DimensionMismatch {
                expected: query.len(),
                actual: cand.len(),
            });
        }
        results.push((i, dispatch::dot_f32(query, cand)));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    Ok(results)
}

/// Compute L2 distance of a query against N candidate vectors.
/// Returns Vec of (index, distance) sorted by ascending distance.
pub fn batch_l2(query: &[f32], candidates: &[&[f32]]) -> Result<Vec<(usize, f32)>, VectorError> {
    if query.is_empty() {
        return Err(VectorError::EmptyVector);
    }

    let mut results: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());

    for (i, &cand) in candidates.iter().enumerate() {
        if cand.len() != query.len() {
            return Err(VectorError::DimensionMismatch {
                expected: query.len(),
                actual: cand.len(),
            });
        }
        results.push((i, dispatch::l2_sq_f32(query, cand).sqrt()));
    }

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    Ok(results)
}

/// Compute N×M cosine similarity distance matrix.
/// `matrix[i][j]` = cosine_similarity(vectors_a[i], vectors_b[j])
pub fn distance_matrix_cosine(
    vectors_a: &[&[f32]],
    vectors_b: &[&[f32]],
) -> Result<Vec<Vec<f32>>, VectorError> {
    if vectors_a.is_empty() || vectors_b.is_empty() {
        return Err(VectorError::EmptyVector);
    }

    let dim = vectors_a[0].len();
    if dim == 0 {
        return Err(VectorError::EmptyVector);
    }

    // Pre-compute norms
    let norms_a: Vec<f32> = vectors_a
        .iter()
        .map(|v| dispatch::sum_squares_f32(v).sqrt())
        .collect();
    let norms_b: Vec<f32> = vectors_b
        .iter()
        .map(|v| dispatch::sum_squares_f32(v).sqrt())
        .collect();

    let mut matrix = Vec::with_capacity(vectors_a.len());

    for (i, &va) in vectors_a.iter().enumerate() {
        if va.len() != dim {
            return Err(VectorError::DimensionMismatch {
                expected: dim,
                actual: va.len(),
            });
        }
        let mut row = Vec::with_capacity(vectors_b.len());
        for (j, &vb) in vectors_b.iter().enumerate() {
            if vb.len() != dim {
                return Err(VectorError::DimensionMismatch {
                    expected: dim,
                    actual: vb.len(),
                });
            }
            let dot = dispatch::dot_f32(va, vb);
            let denom = norms_a[i] * norms_b[j];
            if denom == 0.0 {
                row.push(0.0);
            } else {
                row.push(dot / denom);
            }
        }
        matrix.push(row);
    }

    Ok(matrix)
}
