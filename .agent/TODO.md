# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Formula Mismatch

## Session 116 Summary

### Fixed: File Path Issue
The Jolt test was reading `/tmp/zolt_proof_dory.bin` (old file from Feb 1) instead of `logs/zolt_proof_dory.bin` (current file from Feb 2). Copied the correct file to /tmp.

After this fix, the table_flag claims now deserialize correctly:
- LookupTableFlag(0) = non-zero ✓
- LookupTableFlag(1) = non-zero ✓
- LookupTableFlag(9) = non-zero ✓ (for NotEqual table)
- All others = zero ✓

### Current Issue: Stage 5 Formula Mismatch

Stage 5 sumcheck verification still fails:
```
output_claim:   [9d, 3d, dd, d4...]
expected_claim: [1b, 43, f0, ba...]
```

The root cause is that **Zolt's sumcheck polynomial computation differs from Jolt's formula**.

### Formula Analysis

Jolt's `expected_output_claim` formula (read_raf_checking.rs:1285-1321):
```rust
val_claim = Σ (table_MLE[i](r_address_prime) * table_flag[i])  // <-- KEY DIFFERENCE
raf_claim = (1 - raf_flag) * (left_op + gamma*right_op) + raf_flag * gamma * identity
expected = eq_eval * ra_claim * (val_claim + gamma * raf_claim)
```

Zolt's computation (stage5_prover.zig:811):
```zig
lookups_combined_vals[j] = lookup_output.add(gamma_raf.mul(left_op)).add(gamma_raf2.mul(right_op));
// Where lookup_output = step.rd_value (actual instruction output)
```

### Key Difference

1. **Jolt**: `val_claim = Σ table_MLE[i](r_address) * table_flag[i]`
   - Each `table_MLE[i]` is the multilinear extension of lookup table i
   - Evaluated at the address point `r_address_prime`
   - This requires implementing table MLE evaluation using prefix-suffix decomposition

2. **Zolt**: Uses `lookup_output = step.rd_value`
   - This is the actual output value from the execution trace
   - NOT the table MLE evaluated at the opening point

### What Needs to be Fixed

Zolt needs to implement the lookup table MLE evaluation for all 42 tables:
- Table 0: RangeCheck (used by Fibonacci)
- Table 9: NotEqual (used by Fibonacci branches)
- Tables 1-8, 10-41: Other instruction lookup tables

The `evaluate_mle` function for each table takes `r_address` (128 bits) and returns the table MLE evaluated at that point.

For example, in Jolt (andn.rs:18-32):
```rust
fn evaluate_mle<F, C>(&self, r: &[C]) -> F {
    let mut result = F::zero();
    for i in 0..XLEN {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        result += F::from_u64(1u64 << (XLEN - 1 - i)) * x_i * (F::one() - y_i);
    }
    result
}
```

### Implementation Options

1. **Full Prefix-Suffix Decomposition**: Implement Jolt's optimization with prefix checkpoints and suffix polynomials. Complex but efficient.

2. **Direct MLE Evaluation**: For each of the 42 tables, implement `evaluate_mle(r_address)` and multiply by the corresponding `table_flag`.

Option 2 is simpler to implement but may be slower for large traces.

### Files to Modify

- `src/zkvm/spartan/stage5_prover.zig`: Implement table MLE evaluations
- `src/zkvm/lookup_table/`: Add evaluate_mle for each table type

### Next Steps

1. Understand all 42 lookup table formulas from Jolt
2. Implement evaluate_mle for each table in Zolt
3. Modify Stage 5 prover to use `Σ table_MLE[i](r_address) * table_flag[i]` formula
4. Test verification

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy to /tmp for Jolt test
cp logs/zolt_*.bin /tmp/

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
