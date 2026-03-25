//! Rayon micro-benchmark: parallel reduce over field element pairs.
//! Direct comparison with the Zig ThreadPool equivalent (main.zig).
//!
//! Usage: cargo run --release

use ark_bn254::Fr;
use ark_ff::{Field, AdditiveGroup};
use rayon::prelude::*;
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 50;

fn bench_parallel_reduce(a: &[Fr], b: &[Fr], half: usize) -> f64 {
    // Warmup
    for _ in 0..WARMUP {
        let _ = (0..half)
            .into_par_iter()
            .map(|i| (a[2 * i] * b[2 * i], a[2 * i + 1] * b[2 * i + 1]))
            .reduce(|| (Fr::ZERO, Fr::ZERO), |acc, x| (acc.0 + x.0, acc.1 + x.1));
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        let r = (0..half)
            .into_par_iter()
            .map(|i| (a[2 * i] * b[2 * i], a[2 * i + 1] * b[2 * i + 1]))
            .reduce(|| (Fr::ZERO, Fr::ZERO), |acc, x| (acc.0 + x.0, acc.1 + x.1));
        std::hint::black_box(r);
    }
    start.elapsed().as_secs_f64() / ITERS as f64 * 1000.0
}

fn bench_sequential(a: &[Fr], b: &[Fr], half: usize) -> f64 {
    // Warmup
    for _ in 0..WARMUP {
        let mut s0 = Fr::ZERO;
        let mut s1 = Fr::ZERO;
        for i in 0..half {
            s0 += a[2 * i] * b[2 * i];
            s1 += a[2 * i + 1] * b[2 * i + 1];
        }
        std::hint::black_box((s0, s1));
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        let mut s0 = Fr::ZERO;
        let mut s1 = Fr::ZERO;
        for i in 0..half {
            s0 += a[2 * i] * b[2 * i];
            s1 += a[2 * i + 1] * b[2 * i + 1];
        }
        std::hint::black_box((s0, s1));
    }
    start.elapsed().as_secs_f64() / ITERS as f64 * 1000.0
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("Rayon micro-benchmark (Rust)");
    println!("Threads: {threads}");
    println!("Workload: parallel reduce Σ a[i]*b[i] with ark-bn254 Fr\n");

    let sizes = [1024, 4096, 16384, 65536, 262144, 524288];

    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "N (pairs)", "Sequential", "Parallel", "Speedup"
    );
    println!("{:->10} {:->12} {:->12} {:->10}", "", "", "", "");

    for &n in &sizes {
        let half = n;
        let len = half * 2;

        let a: Vec<Fr> = (0..len)
            .map(|i| Fr::from((i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1)))
            .collect();
        let b: Vec<Fr> = (0..len)
            .map(|i| Fr::from((i as u64).wrapping_mul(0x517CC1B727220A95).wrapping_add(1)))
            .collect();

        let seq_ms = bench_sequential(&a, &b, half);
        let par_ms = bench_parallel_reduce(&a, &b, half);
        let speedup = seq_ms / par_ms;

        println!(
            "{n:>10} {seq_ms:>10.3} ms {par_ms:>10.3} ms {speedup:>9.2}x"
        );
    }
}
