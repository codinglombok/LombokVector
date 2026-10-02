//! The reference reduction of SPEC section 2, in binary32 and binary64.
//!
//! Every term goes into lane `i % 8`; each lane adds its terms in index order
//! (multiply, round, add, round; never a fused multiply-add). The lanes are
//! then combined as `((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))`.
//! The SIMD kernels in `simd/` perform exactly these operations.

pub(crate) const LANES: usize = 8;

macro_rules! reference_kernels {
    ($t:ty, $dot:ident, $sumsq:ident, $l2sq:ident, $combine:ident) => {
        /// Combines the eight lane sums in the fixed order of SPEC section 2.
        #[inline]
        pub(crate) fn $combine(s: &[$t; LANES]) -> $t {
            ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]))
        }

        /// Reference dot product.
        pub fn $dot(a: &[$t], b: &[$t]) -> $t {
            let mut s = [0.0 as $t; LANES];
            let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
            let (ra, rb) = (ca.remainder(), cb.remainder());
            for (x, y) in ca.zip(cb) {
                for k in 0..LANES {
                    let p = x[k] * y[k];
                    s[k] += p;
                }
            }
            for k in 0..ra.len() {
                let p = ra[k] * rb[k];
                s[k] += p;
            }
            $combine(&s)
        }

        /// Reference sum of squares.
        pub fn $sumsq(a: &[$t]) -> $t {
            let mut s = [0.0 as $t; LANES];
            let ca = a.chunks_exact(LANES);
            let ra = ca.remainder();
            for x in ca {
                for k in 0..LANES {
                    let p = x[k] * x[k];
                    s[k] += p;
                }
            }
            for k in 0..ra.len() {
                let p = ra[k] * ra[k];
                s[k] += p;
            }
            $combine(&s)
        }

        /// Reference sum of squared differences.
        pub fn $l2sq(a: &[$t], b: &[$t]) -> $t {
            let mut s = [0.0 as $t; LANES];
            let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
            let (ra, rb) = (ca.remainder(), cb.remainder());
            for (x, y) in ca.zip(cb) {
                for k in 0..LANES {
                    let d = x[k] - y[k];
                    let p = d * d;
                    s[k] += p;
                }
            }
            for k in 0..ra.len() {
                let d = ra[k] - rb[k];
                let p = d * d;
                s[k] += p;
            }
            $combine(&s)
        }
    };
}

reference_kernels!(f32, dot_f32, sumsq_f32, l2sq_f32, combine_f32);
reference_kernels!(f64, dot_f64, sumsq_f64, l2sq_f64, combine_f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_order_is_visible() {
        // lanes: (1e16 + 1) + (-1e16 + 1) = 1e16 + -1e16 = 0, because each
        // `1` is absorbed in its pair; plain left-to-right summation gives 1.
        let a = [1e16, 1.0, -1e16, 1.0];
        assert_eq!(dot_f64(&a, &[1.0; 4]), 0.0);
        assert_eq!(a.iter().fold(0.0, |s, &x| s + x), 1.0);
        // ten products of 0.1 * 0.1: lanes give 0.10000000000000002,
        // left-to-right gives 0.10000000000000003
        assert_eq!(
            dot_f64(&[0.1; 10], &[0.1; 10]).to_bits(),
            0x3fb999999999999b
        );
    }

    #[test]
    fn tail_goes_to_low_lanes() {
        let a: [f32; 17] = core::array::from_fn(|i| (i + 1) as f32);
        assert_eq!(dot_f32(&a, &[1.0; 17]), 153.0);
        assert_eq!(sumsq_f32(&[3.0, 4.0]), 25.0);
        assert_eq!(l2sq_f64(&[1.0, 2.0], &[4.0, 6.0]), 25.0);
    }
}
