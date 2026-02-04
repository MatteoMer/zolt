# Zolt-Jolt Compatibility Implementation

## Status: Session 70 - Stage 1 Sumcheck Deep Investigation (Continued)

## Previous Issue: R1CS vs Memory Trace Inconsistency

Stage 5 sumcheck was failing due to fundamental inconsistency between R1CS trace and memory trace.
This has been FIXED by injecting a full synthetic trace cycle for termination writes.

## Current Issue: Stage 1 Sumcheck Verification Failure

### Symptoms

Stage 1 (Spartan Outer) sumcheck verification fails with:
```
output_claim:   [8f, 49, 4e, 9d, ca, 22, 84, a0, ...]
expected_claim: [f5, 77, db, ef, 9d, 38, 62, 52, ...]
```

### KEY FINDING (This Session)

**The sumcheck polynomial chain is CORRECT!**

1. Prover's `final_claim * batching_coeff` = [8f, 49, 4e, 9d, ...]
2. Verifier's `output_claim` = [8f, 49, 4e, 9d, ...] (MATCHES!)

This confirms:
- Initial claims MATCH
- Round polynomial coefficients MATCH
- Challenges MATCH
- Claim tracking through sumcheck is CORRECT

**The issue is in the expected_output_claim oracle computation!**

The verifier computes:
```
expected_claim = inner_sum_prod * tau_high_bound_r0 * tau_bound_r_tail_reversed * batching_coeff
```

Where:
- `inner_sum_prod = Az_final * Bz_final` (R1CS constraint evaluation)
- `tau_high_bound_r0 = [0b, d4, 56, 27, ...]` = Prover's `lagrange_tau_r0` (MATCHES!)
- `tau_bound_r_tail_reversed = eq(tau_low, challenges_reversed)`

### Eq Factor Analysis

Prover's eq_factor:
- Initial: `lagrange_tau_r0 = [0b, d4, 56, 27, ...]`
- After all bindings: `[f7, be, 45, d2, ...]`

Verifier's tau factors:
- `tau_high_bound_r0 = [0b, d4, 56, 27, ...]` ✓ MATCHES
- `tau_bound_r_tail_reversed = [99, 31, d1, 65, ...]`

The binding order in Zolt matches Jolt:
- Zolt binds tau[n-1] with first challenge, tau[n-2] with second, ..., tau[0] with last
- Jolt reverses challenges and computes eq(tau_low, challenges_reversed)
- These produce the SAME result (multiplication is commutative)

### Root Cause Hypothesis

The issue is likely in `inner_sum_prod`:
- The R1CS input evaluations in the opening claims may not satisfy the R1CS constraints
- Or the constraint evaluation formula differs between prover and verifier

The prover's polynomial sums:
```
Σ Az(x) * Bz(x) * eq(tau, x)
```

The verifier expects this sum to equal:
```
inner_sum_prod * eq_factor
```

If these don't match, it means:
```
prover's Az(r) * Bz(r) ≠ verifier's inner_sum_prod
```

### Next Steps

1. Add debug to print Zolt's Az_final and Bz_final at the constraint point
2. Compare with Jolt's inner_sum_prod components (az_g0, bz_g0, az_g1, bz_g1)
3. Verify the R1CS input evaluations are being computed correctly
4. Check if the witness MLE evaluations at r_cycle match between prover and verifier

### Verified Items

- [x] Initial claims MATCH
- [x] Round polynomial coefficients MATCH
- [x] Challenges MATCH
- [x] Batching coefficients MATCH
- [x] lagrange_tau_r0 MATCHES tau_high_bound_r0
- [x] Eq polynomial binding order is consistent
- [ ] Inner sum product computation

## Test Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Files Modified This Session

- `src/zkvm/proof_converter.zig`: Added debug for lagrange_tau_r0 and final_claim
- `.agent/TODO.md`: Updated with investigation findings

## Debug Output Summary

### Zolt Prover Output (Stage 1)
```
lagrange_tau_r0 = [0b, d4, 56, 27, ...]
prover final_claim = [be, 81, 99, 16, ...]
prover final_claim * batching_coeff = [8f, 49, 4e, 9d, ...]
prover eq_factor = [f7, be, 45, d2, ...]
```

### Jolt Verifier Output (Stage 1)
```
inner_sum_prod = [6e, d1, 32, 0b, ...]
tau_high_bound_r0 = [0b, d4, 56, 27, ...]
tau_bound_r_tail_reversed = [99, 31, d1, 65, ...]
expected result (no batching) = [31, 4a, 3a, 6f, ...]
expected_claim (with batching) = [f5, 77, db, ef, ...]
output_claim = [8f, 49, 4e, 9d, ...]  (from sumcheck chain)
```

### Analysis

The sumcheck output_claim is correct and matches prover's claim chain.
The expected_claim differs because inner_sum_prod * eq_factors differs from what the prover computed.

This means the constraint polynomial Az*Bz*eq being summed by the prover
produces a different final evaluation than what the verifier recomputes.

Likely causes:
1. Different witness values being used
2. Different constraint grouping or indexing
3. Different Lagrange basis weights
4. Different r_stream blending formula
