# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - RamRaClaimReduction PhaseCycle Fix

## Session 123 Summary

### Progress Made
1. **Implemented P*Q Decomposition for PhaseCycle**
   - Added P_raf, P_rw, P_val arrays for prefix eq evaluations
   - Added Q_raf, Q_rw, Q_val arrays for suffix-weighted sums
   - Added H_prime array for PhaseCycle2
   - Implemented PhaseCycle1 polynomial computation using P*Q products
   - Implemented PhaseCycle2 polynomial computation using H'*eq_hi products
   - Implemented proper binding for both phases

2. **Build and Test Results**
   - Code compiles successfully
   - Stage 5 sumcheck verification still failing
   - The output_claim doesn't match expected_claim

### Current Analysis

The verifier computes expected_output_claim as:
```
eq_combined * ra_claim_reduced

where:
eq_combined = eq_addr_1 * eq_cycle_A + γ² * eq_addr_2 * eq_cycle_B
eq_cycle_A = eq(r_cycle_raf, r_cycle_reduced) + γ * eq(r_cycle_val, r_cycle_reduced)
eq_cycle_B = eq(r_cycle_rw, r_cycle_reduced) + γ * eq(r_cycle_val, r_cycle_reduced)
```

Key insight: The verifier uses `eq(r_cycle_*, r_cycle_reduced)` where r_cycle_reduced
are the sumcheck challenges from the 8 cycle rounds.

My P*Q decomposition:
- P_x[c_lo] = eq(r_cycle_x_lo, c_lo)
- Q_x[c_lo] = Σ_{c_hi} H[c_lo,c_hi] * eq(r_cycle_x_hi, c_hi)

The polynomial contribution is: Σ_j coeff * P_x[j] * Q_x[j]

After binding prefix bits (first prefix_n_vars rounds), P reduces to:
P[0] = eq(r_cycle_x_lo, prefix_challenges)

After binding suffix bits (last suffix_n_vars rounds), we should get:
eq(r_cycle_x, r_cycle_reduced) = eq(r_cycle_x_lo, prefix_challenges) * eq(r_cycle_x_hi, suffix_challenges)

But Q already contains H[c] and eq_hi, so there might be a mismatch.

### Possible Issues

1. **Order of eq polynomial indices**: Need to verify that BIG_ENDIAN vs LowToHigh ordering is correct
2. **Suffix vs Prefix split**: Jolt splits r_cycle into [high bits | low bits], need to match this
3. **H[c] vs F_values**: H[c] = F_values[address[c]] is correct, but need to verify address indexing

### Next Steps

1. Add more debug output to trace P*Q values during cycle rounds
2. Compare intermediate values with Jolt's expected values
3. Consider if we need to track ra_claim_reduced separately

## Test Commands
```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy to /tmp for Jolt test
cp logs/zolt_*.bin /tmp/

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Files Modified This Session
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig`
  - Added P*Q decomposition arrays (P_raf, P_rw, P_val, Q_raf, Q_rw, Q_val)
  - Added H_prime, eq_hi arrays
  - Added cycle_challenges tracking
  - Updated PhaseCycle polynomial computation for PhaseCycle1 and PhaseCycle2
  - Updated binding code for both phases

## SESSION_ENDING

Progress saved. Implemented P*Q decomposition but verification still failing. Need to debug the exact values being computed vs expected.
