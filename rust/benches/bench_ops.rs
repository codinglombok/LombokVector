use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lombokvector::{cosine_similarity, dot_product, l2_distance, batch_cosine_similarity};

fn generate_vectors(dim: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.001).sin())
                .collect()
        })
        .collect()
}

fn bench_cosine(c: &mut Criterion) {
    let a: Vec<f32> = (0..768).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..768).map(|i| (i as f32 * 0.1).cos()).collect();

    c.bench_function("cosine_768d", |bencher| {
        bencher.iter(|| cosine_similarity(black_box(&a), black_box(&b)))
    });
}

fn bench_dot(c: &mut Criterion) {
    let a: Vec<f32> = (0..768).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..768).map(|i| (i as f32 * 0.1).cos()).collect();

    c.bench_function("dot_768d", |bencher| {
        bencher.iter(|| dot_product(black_box(&a), black_box(&b)))
    });
}

fn bench_l2(c: &mut Criterion) {
    let a: Vec<f32> = (0..768).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..768).map(|i| (i as f32 * 0.1).cos()).collect();

    c.bench_function("l2_768d", |bencher| {
        bencher.iter(|| l2_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_batch_cosine(c: &mut Criterion) {
    let query: Vec<f32> = (0..768).map(|i| (i as f32 * 0.1).sin()).collect();
    let candidates = generate_vectors(768, 1000);
    let refs: Vec<&[f32]> = candidates.iter().map(|v| v.as_slice()).collect();

    c.bench_function("batch_cosine_1000x768d", |bencher| {
        bencher.iter(|| batch_cosine_similarity(black_box(&query), black_box(&refs)))
    });
}

criterion_group!(benches, bench_cosine, bench_dot, bench_l2, bench_batch_cosine);
criterion_main!(benches);
