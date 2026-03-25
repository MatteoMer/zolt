use ark_bn254::Fr;
use ark_ff::Field;
use std::time::Instant;

fn main() {
    let n = 128;
    let iters = 2_000_000;
    
    // Initialize arrays with non-trivial values
    let mut a: Vec<Fr> = (0..n).map(|i| Fr::from((i + 1) as u64)).collect();
    let mut b: Vec<Fr> = (0..n).map(|i| Fr::from((i + 100) as u64)).collect();
    
    println!("=== Jolt BN254 Fr field benchmarks ===");
    
    for _ in 0..3 {
        // Throughput: independent muls
        let t = Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                a[i] *= b[i];
            }
        }
        let elapsed = t.elapsed().as_nanos() as f64;
        let per_op = elapsed / (iters as f64 * n as f64);
        println!("Fr mul throughput: {:.1}ns", per_op);
        
        // Latency: chained muls
        let t = Instant::now();
        let mut acc = a[0];
        for _ in 0..(iters * n) {
            acc *= b[0];
        }
        let elapsed = t.elapsed().as_nanos() as f64;
        let per_op = elapsed / (iters as f64 * n as f64);
        // Prevent optimization
        if acc == Fr::from(0u64) { println!("never"); }
        println!("Fr mul chain latency: {:.1}ns", per_op);
        
        // Add latency
        let t = Instant::now();
        let mut acc2 = a[0];
        for _ in 0..(iters * n) {
            acc2 += b[0];
        }
        let elapsed = t.elapsed().as_nanos() as f64;
        let per_op = elapsed / (iters as f64 * n as f64);
        if acc2 == Fr::from(0u64) { println!("never"); }
        println!("Fr add chain latency: {:.1}ns", per_op);
        
        // Sub latency
        let t = Instant::now();
        let mut acc3 = a[0];
        for _ in 0..(iters * n) {
            acc3 -= b[0];
        }
        let elapsed = t.elapsed().as_nanos() as f64;
        let per_op = elapsed / (iters as f64 * n as f64);
        if acc3 == Fr::from(0u64) { println!("never"); }
        println!("Fr sub chain latency: {:.1}ns", per_op);
        
        println!();
    }
}
