//! The dispatched kernels (AVX2, NEON, simd128 or portable) must return the
//! same bits as the scalar reference for every input (SPEC section 2.3).

use lombokvector as lv;
use lv::reference as r;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        // mixed magnitudes, both signs
        let m = (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        let e = (self.next() % 20) as i32 - 10;
        m * 2f32.powi(e)
    }
}

#[test]
fn dispatched_kernels_are_bit_identical_to_the_reference() {
    let mut g = Rng(0x2545_f491_4f6c_dd1d);
    let backend = lv::active_backend();
    for len in (1..=70).chain([127, 128, 129, 255, 384, 768, 1000, 1536]) {
        for _ in 0..20 {
            let a: Vec<f32> = (0..len).map(|_| g.f32()).collect();
            let b: Vec<f32> = (0..len).map(|_| g.f32()).collect();
            assert_eq!(
                lv::dot_product(&a, &b).unwrap().to_bits(),
                r::dot_f32(&a, &b).to_bits(),
                "dot, len {len}, {backend}"
            );
            assert_eq!(
                lv::l2_norm(&a).unwrap().to_bits(),
                r::sumsq_f32(&a).sqrt().to_bits(),
                "norm, len {len}, {backend}"
            );
            assert_eq!(
                lv::l2_distance(&a, &b).unwrap().to_bits(),
                r::l2sq_f32(&a, &b).sqrt().to_bits(),
                "l2, len {len}, {backend}"
            );
        }
    }
}

#[test]
fn backend_name_is_known() {
    assert!(["avx2", "neon", "wasm-simd128", "portable"].contains(&lv::active_backend()));
}
