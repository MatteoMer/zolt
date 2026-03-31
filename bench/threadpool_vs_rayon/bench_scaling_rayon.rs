//! Scaling micro-benchmarks (Rayon side): parallelFor, repeated dispatch, multi-array bind
//!
//! Matches bench_scaling.zig workload-for-workload.
//!
//! Usage: cargo run --release --bin rayon-scaling-bench

use ark_bn254::Fr;
use ark_ff::AdditiveGroup;
use rayon::prelude::*;
use std::time::Instant;

const WARMUP: usize = 5;
const ITERS: usize = 50;
const RUNS: usize = 5;

// ====================================================================
// Benchmark 1: par_iter_mut — in-place u64 write (light work per elem)
// ====================================================================

fn for_light_par(data: &mut [u64]) -> f64 {
    for _ in 0..WARMUP {
        data.par_iter_mut().for_each(|v| *v = v.wrapping_mul(v.wrapping_add(1)));
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        data.par_iter_mut().for_each(|v| *v = v.wrapping_mul(v.wrapping_add(1)));
    }
    start.elapsed().as_secs_f64() / ITERS as f64 * 1000.0
}

fn for_light_seq(data: &mut [u64]) -> f64 {
    for _ in 0..WARMUP {
        data.iter_mut().for_each(|v| *v = v.wrapping_mul(v.wrapping_add(1)));
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        data.iter_mut().for_each(|v| *v = v.wrapping_mul(v.wrapping_add(1)));
    }
    start.elapsed().as_secs_f64() / ITERS as f64 * 1000.0
}

// ====================================================================
// Benchmark 2: par_iter_mut — BN254 field bind (heavy work per elem)
//   out[i] = a[i] * scalar
// ====================================================================

fn for_heavy_par(a: &[Fr], out: &mut [Fr], scalar: Fr) -> f64 {
    for _ in 0..WARMUP {
        out.par_iter_mut().enumerate().for_each(|(i, o)| *o = a[i] * scalar);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        out.par_iter_mut().enumerate().for_each(|(i, o)| *o = a[i] * scalar);
    }
    start.elapsed().as_secs_f64() / ITERS as f64 * 1000.0
}

fn for_heavy_seq(a: &[Fr], out: &mut [Fr], scalar: Fr) -> f64 {
    for _ in 0..WARMUP {
        out.iter_mut().enumerate().for_each(|(i, o)| *o = a[i] * scalar);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        out.iter_mut().enumerate().for_each(|(i, o)| *o = a[i] * scalar);
    }
    start.elapsed().as_secs_f64() / ITERS as f64 * 1000.0
}

// ====================================================================
// Benchmark 3: Repeated dispatch — call par_iter reduce DISPATCH_COUNT
// times at a fixed size. Measures dispatch + wake overhead.
// ====================================================================

const DISPATCH_COUNT: usize = 200;

fn repeated_dispatch_par(data: &[u64]) -> f64 {
    // warmup
    for _ in 0..WARMUP {
        for _ in 0..DISPATCH_COUNT {
            std::hint::black_box(
                data.par_iter()
                    .copied()
                    .map(|v| v.wrapping_mul(v.wrapping_add(1)))
                    .reduce(|| 0u64, |a, b| a.wrapping_add(b)),
            );
        }
    }
    let start = Instant::now();
    for _ in 0..RUNS {
        for _ in 0..DISPATCH_COUNT {
            std::hint::black_box(
                data.par_iter()
                    .copied()
                    .map(|v| v.wrapping_mul(v.wrapping_add(1)))
                    .reduce(|| 0u64, |a, b| a.wrapping_add(b)),
            );
        }
    }
    start.elapsed().as_secs_f64() / (RUNS * DISPATCH_COUNT) as f64 * 1000.0
}

fn repeated_dispatch_seq(data: &[u64]) -> f64 {
    // warmup
    for _ in 0..WARMUP {
        for _ in 0..DISPATCH_COUNT {
            std::hint::black_box(
                data.iter()
                    .copied()
                    .map(|v| v.wrapping_mul(v.wrapping_add(1)))
                    .fold(0u64, |a, b| a.wrapping_add(b)),
            );
        }
    }
    let start = Instant::now();
    for _ in 0..RUNS {
        for _ in 0..DISPATCH_COUNT {
            std::hint::black_box(
                data.iter()
                    .copied()
                    .map(|v| v.wrapping_mul(v.wrapping_add(1)))
                    .fold(0u64, |a, b| a.wrapping_add(b)),
            );
        }
    }
    start.elapsed().as_secs_f64() / (RUNS * DISPATCH_COUNT) as f64 * 1000.0
}

// ====================================================================
// Benchmark 4: Multi-array bind — bind NUM_ARRAYS arrays of size T
// using rayon::scope + spawn — the stage 2-6 pattern.
// ====================================================================

const NUM_ARRAYS: usize = 8;

fn multi_array_bind_par(arrays: &mut [Vec<Fr>], scalar: Fr) -> f64 {
    for _ in 0..WARMUP {
        rayon::scope(|s| {
            for arr in arrays.iter_mut() {
                s.spawn(|_| {
                    arr.iter_mut().for_each(|v| *v *= scalar);
                });
            }
        });
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        rayon::scope(|s| {
            for arr in arrays.iter_mut() {
                s.spawn(|_| {
                    arr.iter_mut().for_each(|v| *v *= scalar);
                });
            }
        });
    }
    start.elapsed().as_secs_f64() / ITERS as f64 * 1000.0
}

fn multi_array_bind_seq(arrays: &mut [Vec<Fr>], scalar: Fr) -> f64 {
    for _ in 0..WARMUP {
        for arr in arrays.iter_mut() {
            arr.iter_mut().for_each(|v| *v *= scalar);
        }
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        for arr in arrays.iter_mut() {
            arr.iter_mut().for_each(|v| *v *= scalar);
        }
    }
    start.elapsed().as_secs_f64() / ITERS as f64 * 1000.0
}

// ====================================================================
// Benchmark 5: Nested parallel — rayon::scope + par_iter_mut inside
// ====================================================================

fn nested_parallel_par(arrays: &mut [Vec<Fr>], scalar: Fr) -> f64 {
    for _ in 0..WARMUP {
        rayon::scope(|s| {
            for arr in arrays.iter_mut() {
                s.spawn(|_| {
                    arr.par_iter_mut().for_each(|v| *v *= scalar);
                });
            }
        });
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        rayon::scope(|s| {
            for arr in arrays.iter_mut() {
                s.spawn(|_| {
                    arr.par_iter_mut().for_each(|v| *v *= scalar);
                });
            }
        });
    }
    start.elapsed().as_secs_f64() / ITERS as f64 * 1000.0
}

// ====================================================================
// Main
// ====================================================================

fn main() {
    let threads = rayon::current_num_threads();
    println!("Scaling micro-benchmark (Rayon)");
    println!("Threads: {threads}");

    let sizes: &[usize] = &[1024, 4096, 16384, 65536, 262144, 524288, 1048576];
    let max_n: usize = 1048576;

    // Allocate u64 data
    let mut u64_data: Vec<u64> = (0..max_n)
        .map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
        .collect();

    // Allocate field data
    let fa: Vec<Fr> = (0..max_n)
        .map(|i| Fr::from((i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1)))
        .collect();
    let mut fout: Vec<Fr> = vec![Fr::ZERO; max_n];
    let scalar = Fr::from(0xDEADBEEFCAFEBABEu64);

    // --- parallelFor light (u64 in-place write) ---
    println!("\n--- parallelFor: u64 in-place write ---");
    println!(
        "Config: {WARMUP} warmup, {ITERS} iters, {RUNS} runs (min-of-runs)\n"
    );
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "N", "Sequential", "Parallel", "Speedup"
    );
    println!("{:->10} {:->12} {:->12} {:->10}", "", "", "", "");
    for &n in sizes {
        let (mut bs, mut bp) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..RUNS {
            let s = for_light_seq(&mut u64_data[..n]);
            let p = for_light_par(&mut u64_data[..n]);
            bs = bs.min(s);
            bp = bp.min(p);
        }
        println!("{n:>10} {bs:>10.3} ms {bp:>10.3} ms {:>9.2}x", bs / bp);
    }

    // --- parallelFor heavy (BN254 field bind) ---
    println!("\n--- parallelFor: BN254 field bind (out[i] = a[i] * scalar) ---");
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "N", "Sequential", "Parallel", "Speedup"
    );
    println!("{:->10} {:->12} {:->12} {:->10}", "", "", "", "");
    for &n in sizes {
        let (mut bs, mut bp) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..RUNS {
            let s = for_heavy_seq(&fa[..n], &mut fout[..n], scalar);
            let p = for_heavy_par(&fa[..n], &mut fout[..n], scalar);
            bs = bs.min(s);
            bp = bp.min(p);
        }
        println!("{n:>10} {bs:>10.3} ms {bp:>10.3} ms {:>9.2}x", bs / bp);
    }

    // --- Repeated dispatch ---
    println!(
        "\n--- Repeated dispatch: {DISPATCH_COUNT} calls of par_iter reduce (per-call avg) ---"
    );
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "N", "Sequential", "Parallel", "Speedup"
    );
    println!("{:->10} {:->12} {:->12} {:->10}", "", "", "", "");
    let dispatch_sizes: &[usize] = &[256, 1024, 4096, 16384, 65536, 262144];
    for &n in dispatch_sizes {
        let s = repeated_dispatch_seq(&u64_data[..n]);
        let p = repeated_dispatch_par(&u64_data[..n]);
        println!("{n:>10} {s:>10.3} ms {p:>10.3} ms {:>9.2}x", s / p);
    }

    // --- Multi-array bind ---
    println!("\n--- Multi-array bind: {NUM_ARRAYS} arrays, rayon::scope ---");
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "T (each)", "Sequential", "Parallel", "Speedup"
    );
    println!("{:->10} {:->12} {:->12} {:->10}", "", "", "", "");
    let arr_sizes: &[usize] = &[1024, 4096, 16384, 65536, 131072];
    for &t in arr_sizes {
        let mut arrays: Vec<Vec<Fr>> = (0..NUM_ARRAYS)
            .map(|k| {
                (0..t)
                    .map(|j| {
                        Fr::from(
                            ((k * t + j) as u64)
                                .wrapping_mul(0x9E3779B97F4A7C15)
                                .wrapping_add(1),
                        )
                    })
                    .collect()
            })
            .collect();

        let (mut bs, mut bp) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..RUNS {
            let s = multi_array_bind_seq(&mut arrays, scalar);
            let p = multi_array_bind_par(&mut arrays, scalar);
            bs = bs.min(s);
            bp = bp.min(p);
        }
        println!("{t:>10} {bs:>10.3} ms {bp:>10.3} ms {:>9.2}x", bs / bp);
    }

    // --- Nested parallel ---
    println!("\n--- Nested parallel: scope({NUM_ARRAYS} arrays) x par_iter_mut(T) ---");
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "T (each)", "Sequential", "Nested par", "Speedup"
    );
    println!("{:->10} {:->12} {:->12} {:->10}", "", "", "", "");
    for &t in arr_sizes {
        let mut arrays: Vec<Vec<Fr>> = (0..NUM_ARRAYS)
            .map(|k| {
                (0..t)
                    .map(|j| {
                        Fr::from(
                            ((k * t + j) as u64)
                                .wrapping_mul(0x9E3779B97F4A7C15)
                                .wrapping_add(1),
                        )
                    })
                    .collect()
            })
            .collect();

        let (mut bs, mut bp) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..RUNS {
            let s = multi_array_bind_seq(&mut arrays, scalar);
            let p = nested_parallel_par(&mut arrays, scalar);
            bs = bs.min(s);
            bp = bp.min(p);
        }
        println!("{t:>10} {bs:>10.3} ms {bp:>10.3} ms {:>9.2}x", bs / bp);
    }
}
