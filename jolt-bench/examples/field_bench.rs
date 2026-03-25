use ark_bn254::Fr;
use ark_ff::Field;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let n: usize = 128;
    let iters: usize = 2_000_000;

    let a_orig: Vec<Fr> = (0..n).map(|i| Fr::from((i + 1) as u64)).collect();
    let b_orig: Vec<Fr> = (0..n).map(|i| Fr::from((i + 1000) as u64)).collect();

    println!("=== Jolt (arkworks) BN254 Fr field benchmarks ===");

    for _ in 0..3 {
        let mut a = a_orig.clone();
        let b = &b_orig;

        // mul throughput
        {
            let t = Instant::now();
            for _ in 0..iters {
                for i in 0..n { a[i] *= b[i]; }
            }
            let ns = t.elapsed().as_nanos() as f64 / (iters as f64 * n as f64);
            println!("Fr mul throughput:      {:.1}ns", ns);
            black_box(&a);
        }

        // square throughput
        {
            let t = Instant::now();
            for _ in 0..iters {
                for i in 0..n { a[i] = a[i].square(); }
            }
            let ns = t.elapsed().as_nanos() as f64 / (iters as f64 * n as f64);
            println!("Fr square throughput:   {:.1}ns", ns);
            black_box(&a);
        }

        // add throughput
        {
            let t = Instant::now();
            for _ in 0..iters {
                for i in 0..n { a[i] += b[i]; }
            }
            let ns = t.elapsed().as_nanos() as f64 / (iters as f64 * n as f64);
            println!("Fr add throughput:      {:.1}ns", ns);
            black_box(&a);
        }

        // sub throughput
        {
            let t = Instant::now();
            for _ in 0..iters {
                for i in 0..n { a[i] -= b[i]; }
            }
            let ns = t.elapsed().as_nanos() as f64 / (iters as f64 * n as f64);
            println!("Fr sub throughput:      {:.1}ns", ns);
            black_box(&a);
        }

        // sop (sum of products) throughput
        {
            let t = Instant::now();
            let mut r = Fr::from(0u64);
            for i in 0..iters {
                let idx = i % n;
                r = Fr::sum_of_products(&[a[idx], b[idx]], &[b[idx], a[idx]]);
            }
            let ns = t.elapsed().as_nanos() as f64 / iters as f64;
            if r == Fr::from(0u64) { println!("never"); }
            println!("Fr sop throughput:     {:.1}ns", ns);
        }

        // mul chain latency
        {
            let mut acc = a[0];
            let t = Instant::now();
            for i in 0..iters {
                acc *= b[i % n];
            }
            let ns = t.elapsed().as_nanos() as f64 / iters as f64;
            if acc == Fr::from(0u64) { println!("never"); }
            println!("Fr mul chain latency:  {:.1}ns", ns);
        }

        // square chain latency
        {
            let mut acc = a[0];
            let t = Instant::now();
            for _ in 0..iters {
                acc = acc.square();
            }
            let ns = t.elapsed().as_nanos() as f64 / iters as f64;
            if acc == Fr::from(0u64) { println!("never"); }
            println!("Fr square chain:       {:.1}ns", ns);
        }

        // add chain latency
        {
            let mut acc = a[0];
            let t = Instant::now();
            for i in 0..iters {
                acc += b[i % n];
            }
            let ns = t.elapsed().as_nanos() as f64 / iters as f64;
            if acc == Fr::from(0u64) { println!("never"); }
            println!("Fr add chain latency:  {:.1}ns", ns);
        }

        // sub chain latency
        {
            let mut acc = a[0];
            let t = Instant::now();
            for i in 0..iters {
                acc -= b[i % n];
            }
            let ns = t.elapsed().as_nanos() as f64 / iters as f64;
            if acc == Fr::from(0u64) { println!("never"); }
            println!("Fr sub chain latency:  {:.1}ns", ns);
        }

        println!();
    }
}
