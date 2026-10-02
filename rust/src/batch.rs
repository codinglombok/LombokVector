//! One query against many candidates, and all-pairs matrices (SPEC section 3.7-3.8).

use alloc::vec::Vec;

use crate::dispatch;
use crate::error::VectorError;
use crate::ops::{check_single, cos_value_f32, cos_value_f64, finite};
use crate::sqrt::Sqrt;

/// Sorts `(index, score)` pairs; the sort is stable, so equal scores keep index order.
fn rank<T: PartialOrd + Copy>(mut v: Vec<(usize, T)>, descending: bool) -> Vec<(usize, T)> {
    // scores are finite (checked by the callers), so partial_cmp never fails
    v.sort_by(|x, y| {
        let o = x.1.partial_cmp(&y.1).unwrap_or(core::cmp::Ordering::Equal);
        if descending {
            o.reverse()
        } else {
            o
        }
    });
    v
}

fn check_candidate(query: usize, cand: usize) -> Result<(), VectorError> {
    if cand != query {
        return Err(VectorError::DimensionMismatch {
            expected: query,
            actual: cand,
        });
    }
    Ok(())
}

macro_rules! batch {
    (
        $t:ty, $dotk:ident, $sumsqk:ident, $l2sqk:ident, $cosv:ident,
        $bcos:ident, $bdot:ident, $bl2:ident, $mat:ident
    ) => {
        /// Cosine similarity of `query` with each candidate, sorted by descending
        /// score (ties keep index order). A zero-magnitude candidate scores 0.
        ///
        /// # Errors
        /// `EmptyVector` (query), `ZeroMagnitude` (query), `DimensionMismatch`, `NonFinite`.
        pub fn $bcos(query: &[$t], candidates: &[&[$t]]) -> Result<Vec<(usize, $t)>, VectorError> {
            check_single(query.len())?;
            let qn = finite!(dispatch::$sumsqk(query).sqrt_())?;
            if qn == 0.0 {
                return Err(VectorError::ZeroMagnitude);
            }
            let mut out = Vec::with_capacity(candidates.len());
            for (i, c) in candidates.iter().enumerate() {
                check_candidate(query.len(), c.len())?;
                let d = finite!(dispatch::$dotk(query, c))?;
                let cn = finite!(dispatch::$sumsqk(c).sqrt_())?;
                out.push((i, if cn == 0.0 { 0.0 } else { $cosv(d, qn, cn)? }));
            }
            Ok(rank(out, true))
        }

        /// Dot product of `query` with each candidate, sorted by descending score.
        ///
        /// # Errors
        /// `EmptyVector`, `DimensionMismatch`, `NonFinite`.
        pub fn $bdot(query: &[$t], candidates: &[&[$t]]) -> Result<Vec<(usize, $t)>, VectorError> {
            check_single(query.len())?;
            let mut out = Vec::with_capacity(candidates.len());
            for (i, c) in candidates.iter().enumerate() {
                check_candidate(query.len(), c.len())?;
                out.push((i, finite!(dispatch::$dotk(query, c))?));
            }
            Ok(rank(out, true))
        }

        /// Euclidean distance from `query` to each candidate, sorted ascending.
        ///
        /// # Errors
        /// `EmptyVector`, `DimensionMismatch`, `NonFinite`.
        pub fn $bl2(query: &[$t], candidates: &[&[$t]]) -> Result<Vec<(usize, $t)>, VectorError> {
            check_single(query.len())?;
            let mut out = Vec::with_capacity(candidates.len());
            for (i, c) in candidates.iter().enumerate() {
                check_candidate(query.len(), c.len())?;
                out.push((i, finite!(dispatch::$l2sqk(query, c).sqrt_())?));
            }
            Ok(rank(out, false))
        }

        /// `out[i][j]` = cosine similarity of `a[i]` and `b[j]`; 0 when either has
        /// magnitude zero. Every vector must have the length of `a[0]`.
        ///
        /// # Errors
        /// `EmptyVector` (empty set or `a[0]` empty), `DimensionMismatch`, `NonFinite`.
        pub fn $mat(a: &[&[$t]], b: &[&[$t]]) -> Result<Vec<Vec<$t>>, VectorError> {
            if a.is_empty() || b.is_empty() {
                return Err(VectorError::EmptyVector);
            }
            let dim = a[0].len();
            check_single(dim)?;
            for v in a.iter().chain(b) {
                check_candidate(dim, v.len())?;
            }
            let norms = |set: &[&[$t]]| -> Result<Vec<$t>, VectorError> {
                set.iter()
                    .map(|v| finite!(dispatch::$sumsqk(v).sqrt_()))
                    .collect()
            };
            let (na, nb) = (norms(a)?, norms(b)?);
            let mut out = Vec::with_capacity(a.len());
            for (i, va) in a.iter().enumerate() {
                let mut row = Vec::with_capacity(b.len());
                for (j, vb) in b.iter().enumerate() {
                    let d = finite!(dispatch::$dotk(va, vb))?;
                    row.push(if na[i] == 0.0 || nb[j] == 0.0 {
                        0.0
                    } else {
                        $cosv(d, na[i], nb[j])?
                    });
                }
                out.push(row);
            }
            Ok(out)
        }
    };
}

batch!(
    f32,
    dot_f32,
    sumsq_f32,
    l2sq_f32,
    cos_value_f32,
    batch_cosine,
    batch_dot,
    batch_l2,
    distance_matrix_cosine
);
batch!(
    f64,
    dot_f64,
    sumsq_f64,
    l2sq_f64,
    cos_value_f64,
    batch_cosine_f64,
    batch_dot_f64,
    batch_l2_f64,
    distance_matrix_cosine_f64
);
