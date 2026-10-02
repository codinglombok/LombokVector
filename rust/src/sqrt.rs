//! Correctly rounded square root for `no_std` builds (IEEE 754 requires
//! `sqrt` to be correctly rounded, so this returns the same bits as the
//! hardware instruction that `std` uses).
//!
//! Method: digit-by-digit integer square root of the scaled significand with
//! a sticky bit, then round to nearest, ties to even.

/// Square root of a non-negative finite or infinite value; NaN otherwise.
pub(crate) trait Sqrt {
    fn sqrt_(self) -> Self;
}

#[cfg(feature = "std")]
impl Sqrt for f32 {
    #[inline]
    fn sqrt_(self) -> f32 {
        self.sqrt()
    }
}

#[cfg(feature = "std")]
impl Sqrt for f64 {
    #[inline]
    fn sqrt_(self) -> f64 {
        self.sqrt()
    }
}

#[cfg(not(feature = "std"))]
impl Sqrt for f32 {
    #[inline]
    fn sqrt_(self) -> f32 {
        f32::from_bits(soft_sqrt(self.to_bits() as u128, 23, 8) as u32)
    }
}

#[cfg(not(feature = "std"))]
impl Sqrt for f64 {
    #[inline]
    fn sqrt_(self) -> f64 {
        f64::from_bits(soft_sqrt(self.to_bits() as u128, 52, 11) as u64)
    }
}

/// Integer square root: floor(sqrt(n)) and whether n was a perfect square.
#[allow(dead_code)]
fn isqrt(n: u128) -> (u128, bool) {
    if n == 0 {
        return (0, true);
    }
    let mut x = n;
    let mut res: u128 = 0;
    let mut bit: u128 = 1 << ((127 - n.leading_zeros()) & !1);
    while bit != 0 {
        if x >= res + bit {
            x -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    (res, x == 0)
}

/// Bit-level square root for a binary format with `mant` fraction bits and
/// `exp` exponent bits; `bits` holds the encoding in the low bits.
#[allow(dead_code)]
pub(crate) fn soft_sqrt(bits: u128, mant: u32, exp: u32) -> u128 {
    let sign = bits >> (mant + exp);
    let e_mask = (1u128 << exp) - 1;
    let m_mask = (1u128 << mant) - 1;
    let e = (bits >> mant) & e_mask;
    let m = bits & m_mask;
    let quiet_nan = (e_mask << mant) | (1 << (mant - 1));
    if e == e_mask {
        // NaN stays NaN; +inf -> +inf; -inf -> NaN
        return if m != 0 || sign == 0 {
            if m != 0 {
                bits | (1 << (mant - 1))
            } else {
                bits
            }
        } else {
            quiet_nan
        };
    }
    if e == 0 && m == 0 {
        return bits; // +0 or -0
    }
    if sign == 1 {
        return quiet_nan;
    }
    let bias = (1i64 << (exp - 1)) - 1;
    // significand as an integer and unbiased exponent so that value = sig * 2^ex
    let (mut sig, mut ex) = if e == 0 {
        (m, 1 - bias - mant as i64)
    } else {
        (m | (1 << mant), e as i64 - bias - mant as i64)
    };
    // normalise subnormals so sig has mant+1 bits
    while sig < (1 << mant) {
        sig <<= 1;
        ex -= 1;
    }
    // make the exponent even
    if ex & 1 != 0 {
        sig <<= 1;
        ex -= 1;
    }
    // scale so the root has mant+2 bits (one guard bit): sig * 2^(2*(mant+2) - bits(sig)+...)
    let shift = 2 * (mant + 2) - (128 - sig.leading_zeros()) + 1;
    let shift = shift & !1; // keep the exponent even
    let scaled = sig << shift;
    let (mut r, exact) = isqrt(scaled);
    let mut rex = (ex - shift as i64) / 2;
    // r has mant+2 or mant+3 bits; reduce to mant+2 bits keeping a sticky bit
    let mut sticky = !exact;
    while r >= (1 << (mant + 2)) {
        sticky |= r & 1 == 1;
        r >>= 1;
        rex += 1;
    }
    // r = 1.xxx (mant fraction bits) followed by one guard bit
    let guard = r & 1;
    let mut q = r >> 1;
    rex += 1;
    if guard == 1 && (sticky || q & 1 == 1) {
        q += 1;
        if q == (1 << (mant + 1)) {
            q >>= 1;
            rex += 1;
        }
    }
    let biased = rex + mant as i64 + bias;
    // square roots of finite positive values are always normal and in range
    ((biased as u128) << mant) | (q & m_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check64(x: f64) {
        let got = f64::from_bits(soft_sqrt(x.to_bits() as u128, 52, 11) as u64);
        let want = x.sqrt();
        assert!(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            "sqrt({x:e}) = {got:e}, want {want:e}"
        );
    }

    fn check32(x: f32) {
        let got = f32::from_bits(soft_sqrt(x.to_bits() as u128, 23, 8) as u32);
        let want = x.sqrt();
        assert!(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            "sqrt({x:e}) = {got:e}, want {want:e}"
        );
    }

    #[test]
    fn matches_hardware_sqrt() {
        for x in [
            0.0,
            -0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            0.5,
            1e-310,
            5e-324,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::INFINITY,
            -1.0,
            f64::NEG_INFINITY,
            f64::NAN,
            27.0,
            1e16,
            0.1,
        ] {
            check64(x);
        }
        for x in [
            0.0f32,
            -0.0,
            1.0,
            2.0,
            1e-45,
            1e-40,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            -2.0,
            f32::NAN,
            0.1,
        ] {
            check32(x);
        }
        // pseudo-random bit patterns across the whole positive range
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            check64(f64::from_bits(s >> 1));
            check32(f32::from_bits((s >> 33) as u32));
        }
    }
}
