/// MSM Benchmark: arkworks Pippenger MSM for BN254 G1 and G2
/// Head-to-head comparison with Zolt's MSM implementation.
///
/// Tests same sizes as Dory opening proof (2..4096 points).
/// Outputs machine-readable [MSM-BENCH] lines for comparison.

use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup, scalar_mul::variable_base::VariableBaseMSM};
use ark_ff::UniformRand;
use ark_std::rand::SeedableRng;
use std::time::Instant;

const WARMUP: usize = 2;
const MIN_ITERS: usize = 5;
const MIN_TIME_NS: u128 = 500_000_000; // 500ms minimum per size

fn generate_g1_bases(n: usize) -> Vec<G1Affine> {
    // Deterministic: repeated doubling from generator
    let gen = G1Affine::generator();
    let mut proj = G1Projective::from(gen);
    let mut bases = Vec::with_capacity(n);
    for _ in 0..n {
        bases.push(proj.into_affine());
        proj = proj.double();
    }
    bases
}

fn generate_g2_bases(n: usize) -> Vec<G2Affine> {
    let gen = G2Affine::generator();
    let mut proj = G2Projective::from(gen);
    let mut bases = Vec::with_capacity(n);
    for _ in 0..n {
        bases.push(proj.into_affine());
        proj = proj.double();
    }
    bases
}

fn generate_fr_scalars(n: usize) -> Vec<Fr> {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0xdeadbeef);
    (0..n).map(|_| Fr::rand(&mut rng)).collect()
}

fn generate_i128_scalars(n: usize) -> Vec<i128> {
    // Match Zolt's deterministic pseudo-random i128s
    let mut seed: u64 = 0xcafebabe;
    (0..n).map(|_| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let hi = (seed as u128) << 64;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let lo = seed as u128;
        (hi | lo) as i128
    }).collect()
}

fn bench_msm_g1_fr(bases: &[G1Affine], scalars: &[Fr]) -> f64 {
    // Warmup
    for _ in 0..WARMUP {
        let _ = G1Projective::msm(bases, scalars).unwrap();
    }

    // Timed runs
    let mut total_ns: u128 = 0;
    let mut iters: usize = 0;
    while iters < MIN_ITERS || total_ns < MIN_TIME_NS {
        let start = Instant::now();
        let _ = G1Projective::msm(bases, scalars).unwrap();
        total_ns += start.elapsed().as_nanos();
        iters += 1;
    }
    (total_ns as f64) / (iters as f64) / 1_000_000.0
}

fn bench_msm_g1_i128(bases: &[G1Affine], scalars: &[i128]) -> f64 {
    use ark_ec::scalar_mul::variable_base::msm_i128;

    // Warmup
    for _ in 0..WARMUP {
        let _ = msm_i128::<G1Projective>(bases, scalars, true);
    }

    // Timed runs
    let mut total_ns: u128 = 0;
    let mut iters: usize = 0;
    while iters < MIN_ITERS || total_ns < MIN_TIME_NS {
        let start = Instant::now();
        let _ = msm_i128::<G1Projective>(bases, scalars, true);
        total_ns += start.elapsed().as_nanos();
        iters += 1;
    }
    (total_ns as f64) / (iters as f64) / 1_000_000.0
}

fn bench_msm_g2_fr(bases: &[G2Affine], scalars: &[Fr]) -> f64 {
    // Warmup
    for _ in 0..WARMUP {
        let _ = G2Projective::msm(bases, scalars).unwrap();
    }

    // Timed runs
    let mut total_ns: u128 = 0;
    let mut iters: usize = 0;
    while iters < MIN_ITERS || total_ns < MIN_TIME_NS {
        let start = Instant::now();
        let _ = G2Projective::msm(bases, scalars).unwrap();
        total_ns += start.elapsed().as_nanos();
        iters += 1;
    }
    (total_ns as f64) / (iters as f64) / 1_000_000.0
}

fn main() {
    // Force Rayon to use all CPUs
    let num_threads = rayon::current_num_threads();

    let sizes: Vec<usize> = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
    let max_size = 262144;

    eprintln!("=== Arkworks MSM Benchmark (BN254) ===");
    eprintln!("Rayon threads: {}\n", num_threads);

    // Pre-generate
    let g1_bases = generate_g1_bases(max_size);
    let g2_bases = generate_g2_bases(max_size);
    let fr_scalars = generate_fr_scalars(max_size);
    let i128_scalars = generate_i128_scalars(max_size);

    eprintln!("{:>6} │ {:>10} {:>10} {:>10}",
        "N", "G1-Fr", "G1-i128", "G2-Fr");
    eprintln!("───────┼─────────────────────────────────");

    for &n in &sizes {
        let b1 = &g1_bases[..n];
        let sf = &fr_scalars[..n];
        let si = &i128_scalars[..n];

        let g1_fr_ms = bench_msm_g1_fr(b1, sf);
        let g1_i128_ms = bench_msm_g1_i128(b1, si);
        // G2 only up to 8192 (very slow beyond)
        let g2_fr_ms = if n <= 8192 { bench_msm_g2_fr(&g2_bases[..n], sf) } else { 0.0 };

        eprintln!("{:>6} │ {:>9.3}  {:>9.3}  {:>9.3}",
            n, g1_fr_ms, g1_i128_ms, g2_fr_ms);

        // Machine-readable
        eprintln!("[MSM-BENCH] n={} g1_fr={:.3} g1_i128={:.3} g2_fr={:.3}",
            n, g1_fr_ms, g1_i128_ms, g2_fr_ms);
    }
}
