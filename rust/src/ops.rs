//! Operations on single vectors and pairs (SPEC section 3).

use crate::dispatch;
use crate::error::VectorError;
use crate::sqrt::Sqrt;

#[inline]
pub(crate) fn check_pair(a: usize, b: usize) -> Result<(), VectorError> {
    if a == 0 {
        return Err(VectorError::EmptyVector);
    }
    if a != b {
        return Err(VectorError::DimensionMismatch {
            expected: a,
            actual: b,
        });
    }
    Ok(())
}

#[inline]
pub(crate) fn check_single(a: usize) -> Result<(), VectorError> {
    if a == 0 {
        return Err(VectorError::EmptyVector);
    }
    Ok(())
}

macro_rules! finite {
    ($x:expr) => {{
        let x = $x;
        if x.is_finite() {
            Ok(x)
        } else {
            Err(VectorError::NonFinite)
        }
    }};
}
#[allow(unused_imports)] // used by batch.rs when `alloc` is enabled
pub(crate) use finite;

macro_rules! ops {
    (
        $t:ty, $dotk:ident, $sumsqk:ident, $l2sqk:ident, $cosv:ident,
        dot = $dot:ident, inner = $inner:ident, cosine = $cos:ident, l2 = $l2:ident, norm = $norm:ident,
        normalize_inplace = $ninp:ident, normalize = $nrm:ident, add = $add:ident, sub = $sub:ident,
        scale = $scale:ident
    ) => {
        /// Cosine from a dot product and two norms: `clamp(dot / (na * nb), -1, 1)`.
        #[inline]
        pub(crate) fn $cosv(dot: $t, na: $t, nb: $t) -> Result<$t, VectorError> {
            let c = finite!(dot / (na * nb))?;
            Ok(if c < -1.0 {
                -1.0
            } else if c > 1.0 {
                1.0
            } else {
                c
            })
        }

        #[doc = concat!("Dot product of two `", stringify!($t), "` vectors (SPEC 3.1).")]
        ///
        /// # Errors
        /// `EmptyVector`, `DimensionMismatch`, `NonFinite`.
        pub fn $dot(a: &[$t], b: &[$t]) -> Result<$t, VectorError> {
            check_pair(a.len(), b.len())?;
            finite!(dispatch::$dotk(a, b))
        }

        /// Same as the dot product for real vectors.
        ///
        /// # Errors
        /// `EmptyVector`, `DimensionMismatch`, `NonFinite`.
        pub fn $inner(a: &[$t], b: &[$t]) -> Result<$t, VectorError> {
            $dot(a, b)
        }

        /// Cosine similarity, clamped to [-1, 1] (SPEC 3.4).
        ///
        /// # Errors
        /// `EmptyVector`, `DimensionMismatch`, `NonFinite`, and `ZeroMagnitude`
        /// when either vector has norm 0.
        pub fn $cos(a: &[$t], b: &[$t]) -> Result<$t, VectorError> {
            check_pair(a.len(), b.len())?;
            let d = finite!(dispatch::$dotk(a, b))?;
            let na = finite!(dispatch::$sumsqk(a).sqrt_())?;
            let nb = finite!(dispatch::$sumsqk(b).sqrt_())?;
            if na == 0.0 || nb == 0.0 {
                return Err(VectorError::ZeroMagnitude);
            }
            $cosv(d, na, nb)
        }

        /// Euclidean distance (SPEC 3.3).
        ///
        /// # Errors
        /// `EmptyVector`, `DimensionMismatch`, `NonFinite`.
        pub fn $l2(a: &[$t], b: &[$t]) -> Result<$t, VectorError> {
            check_pair(a.len(), b.len())?;
            finite!(dispatch::$l2sqk(a, b).sqrt_())
        }

        /// Euclidean norm (SPEC 3.2).
        ///
        /// # Errors
        /// `EmptyVector`, `NonFinite`.
        pub fn $norm(a: &[$t]) -> Result<$t, VectorError> {
            check_single(a.len())?;
            finite!(dispatch::$sumsqk(a).sqrt_())
        }

        /// Scales `a` to unit length in place: `a[i] * (1 / norm)` (SPEC 3.5).
        /// On error `a` is unchanged.
        ///
        /// # Errors
        /// `EmptyVector`, `ZeroMagnitude`, `NonFinite`.
        pub fn $ninp(a: &mut [$t]) -> Result<(), VectorError> {
            check_single(a.len())?;
            let n = finite!(dispatch::$sumsqk(a).sqrt_())?;
            if n == 0.0 {
                return Err(VectorError::ZeroMagnitude);
            }
            let inv = finite!(1.0 / n)?;
            for &x in a.iter() {
                finite!(x * inv)?;
            }
            for x in a.iter_mut() {
                *x *= inv;
            }
            Ok(())
        }

        /// Returns `a` scaled to unit length (SPEC 3.5).
        ///
        /// # Errors
        /// `EmptyVector`, `ZeroMagnitude`, `NonFinite`.
        #[cfg(feature = "alloc")]
        pub fn $nrm(a: &[$t]) -> Result<alloc::vec::Vec<$t>, VectorError> {
            let mut v = a.to_vec();
            $ninp(&mut v)?;
            Ok(v)
        }

        /// Element-wise sum (SPEC 3.6).
        ///
        /// # Errors
        /// `EmptyVector`, `DimensionMismatch`, `NonFinite`.
        #[cfg(feature = "alloc")]
        pub fn $add(a: &[$t], b: &[$t]) -> Result<alloc::vec::Vec<$t>, VectorError> {
            check_pair(a.len(), b.len())?;
            a.iter().zip(b).map(|(&x, &y)| finite!(x + y)).collect()
        }

        /// Element-wise difference (SPEC 3.6).
        ///
        /// # Errors
        /// `EmptyVector`, `DimensionMismatch`, `NonFinite`.
        #[cfg(feature = "alloc")]
        pub fn $sub(a: &[$t], b: &[$t]) -> Result<alloc::vec::Vec<$t>, VectorError> {
            check_pair(a.len(), b.len())?;
            a.iter().zip(b).map(|(&x, &y)| finite!(x - y)).collect()
        }

        /// Every element multiplied by `s` (SPEC 3.6).
        ///
        /// # Errors
        /// `EmptyVector`, `NonFinite`.
        #[cfg(feature = "alloc")]
        pub fn $scale(a: &[$t], s: $t) -> Result<alloc::vec::Vec<$t>, VectorError> {
            check_single(a.len())?;
            a.iter().map(|&x| finite!(x * s)).collect()
        }
    };
}

ops!(
    f32,
    dot_f32,
    sumsq_f32,
    l2sq_f32,
    cos_value_f32,
    dot = dot_product,
    inner = inner_product,
    cosine = cosine_similarity,
    l2 = l2_distance,
    norm = l2_norm,
    normalize_inplace = normalize_inplace,
    normalize = normalize,
    add = vec_add,
    sub = vec_sub,
    scale = vec_mul_scalar
);
ops!(
    f64,
    dot_f64,
    sumsq_f64,
    l2sq_f64,
    cos_value_f64,
    dot = dot_product_f64,
    inner = inner_product_f64,
    cosine = cosine_similarity_f64,
    l2 = l2_distance_f64,
    norm = l2_norm_f64,
    normalize_inplace = normalize_inplace_f64,
    normalize = normalize_f64,
    add = vec_add_f64,
    sub = vec_sub_f64,
    scale = vec_mul_scalar_f64
);
