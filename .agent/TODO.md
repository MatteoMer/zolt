# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Mismatch

## Session 131 Final Summary

### Completed Analysis

1. **eq_prefix decomposition is mathematically correct**:
   - eq(2j, r) = eq_prefix(j) * (1 - r[-1])
   - eq(2j+1, r) = eq_prefix(j) * r[-1]
   - eq_prefix = eq(2j, r) / (1 - r[-1])

2. **Jolt's EqPolynomial::evals convention matches Zolt**:
   - bit (n-1-j) of index k ↔ r[j]
   - No change needed to computeEqAtIndex

3. **r_round values match between Zolt and Jolt**:
   - Zolt: r_reduction[n_cycle_vars - 1 - lookups_round]
   - Jolt: eq_poly.get_current_w() which returns w[current_index - 1]
   - Both produce the same sequence of values

4. **Binding logic is correct**:
   - lookups_eq_evals[j] after k bindings includes accumulated eq scalars
   - This automatically matches Jolt's current_scalar behavior

5. **finishMlesProductSumFromEvals matches Jolt**:
   - Same formula for computing eval_at_0 from claim
   - Same interpolation and eq multiplication

### Current Implementation

The cycle round polynomial computation in stage5_prover.zig:
1. Computes eq_prefix = eq_0 / (1 - r_round)
2. Sets pairs[0] = (eq_prefix * val[2j], eq_prefix * val[2j+1])
3. Sets pairs[1..9] = ra_chunk pairs
4. Evaluates product at [1, 2, ..., 8, ∞]
5. Calls finishMlesProductSumFromEvals to recover polynomial
6. Combines with instances 0, 1 using batch coefficients

### Verification Still Fails

```
output_claim:   [84, 83, e6, 0a, 81, 4f, 33, 12, ...]
expected_claim: [c6, 19, df, ae, 44, 5b, ac, 2e, ...]
```

Individual instance claims match but batched sumcheck fails.

### Remaining Possibilities

1. **Transcript synchronization**: Maybe challenges differ between Zolt and Jolt?
2. **Combined polynomial encoding**: The batched polynomial might use a different format
3. **Instance 0/1 contribution**: The RegistersValEvaluation or RamRaClaimReduction polynomials might be wrong
4. **Numerical precision**: Some edge case in field arithmetic?

### Next Steps

1. Add detailed debug output at each cycle round:
   - Print eq_prefix, sum_evals, full_coeffs
   - Compare with Jolt's values (needs Jolt debugging enabled)

2. Verify transcript matches:
   - Print all challenges from Stage 5
   - Ensure Zolt and Jolt use same challenge sequence

3. Check Instance 0 and 1:
   - Their contribution might be wrong during cycle rounds
   - Verify combined_poly format

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cp /tmp/zolt_*.bin /home/vivado/projects/jolt/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
