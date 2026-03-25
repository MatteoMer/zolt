use ark_bn254::Fr;
use ark_ff::Field;
use std::time::Instant;

fn main() {
    let n = 128;
    let iters = 5_000_000;
    
    let b_val = Fr::from(42u64);
    
    for _ in 0..3 {
        // Add throughput (independent)
        let mut a: Vec<Fr> = (0..n).map(|i| Fr::from((i + 1) as u64)).collect();
        let t = Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                a[i] += b_val;
            }
        }
        let elapsed = t.elapsed().as_nanos() as f64;
        println!("Fr add throughput: {:.1}ns", elapsed / (iters as f64 * n as f64));
        if a[0] == Fr::from(0u64) { println!("never"); }
        
        // Sub throughput (independent)
        let mut a: Vec<Fr> = (0..n).map(|i| Fr::from((i + 1) as u64)).collect();
        let t = Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                a[i] -= b_val;
            }
        }
        let elapsed = t.elapsed().as_nanos() as f64;
        println!("Fr sub throughput: {:.1}ns", elapsed / (iters as f64 * n as f64));
        if a[0] == Fr::from(0u64) { println!("never"); }
        println!();
    }
}
