# Zolt-Jolt Compatibility TODO

## Current Status: Session 40 - January 2, 2026

**712 tests pass. Stage 1 sumcheck output_claim still mismatches expected_output_claim.**

---

## Summary of Investigation

### Session 40 Progress

1. **Discovered batching coefficient issue**: The round polynomials were not being scaled by the batching coefficient. Fixed in `proof_converter.zig` - now:
   - Raw evaluations are computed by the prover (unscaled)
   - Scaled evaluations = raw * batching_coeff are written to proof
   - Scaled coefficients are hashed to transcript
   - Unscaled claim is used for prover's internal state

2. **Verified transcript consistency**: The challenges now match between prover and verifier (same r_i values in sumcheck debug output).

3. **Remaining issue**: The final output_claim still doesn't match expected_output_claim:
   - output_claim: 11745972059365673324717055336378505103382790433770080606002230314528714321637
   - expected: 13147110630967021857497758076978613720325907259294229523986769287815268967658

### Analysis of expected_output_claim

From Jolt's verifier (outer.rs):
```rust
expected_output_claim = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod * batching_coeff
```

Where:
- `tau_high_bound_r0 = L(tau_high, r0_uniskip)` - Lagrange kernel at UniSkip challenge
- `tau_bound_r_tail_reversed = eq(tau_low, [r10, r9, ..., r1, r0_sumcheck])` - eq poly at reversed challenges
- `inner_sum_prod = Az(rx_constr) * Bz(rx_constr)` - R1CS matrix products
- `rx_constr = [r0_sumcheck, r0_uniskip]` - constraint row point

### Key Questions

1. **Eq polynomial direction**: Is `r_tail_reversed = [r10, ..., r0]` being computed correctly?
   - In Jolt: `sumcheck_challenges.iter().rev().copied().collect()`
   - sumcheck_challenges has 11 elements (rounds 0-10)
   - reversed = [r10, r9, ..., r1, r0]

2. **Split_eq binding order**: Zolt's split_eq uses LowToHigh binding. After binding:
   - Round 0 challenge r0 is bound to lowest bit
   - Round 10 challenge r10 is bound to highest bit
   - Final eq evaluation should match eq(tau_low, [r0, r1, ..., r10]) in low-to-high bit order
   - Which equals eq(tau_low_reversed, [r10, ..., r0]) in big-endian order

3. **Is tau_low being constructed correctly?**
   - tau = full challenge vector (length num_rows_bits = num_cycle_vars + 2)
   - tau_high = tau[tau.len - 1]
   - tau_low = tau[0..tau.len - 1]

4. **Is the Lagrange kernel being applied correctly?**
   - At initialization, split_eq gets scaling_factor = L(tau_high, r0_uniskip)
   - This should be multiplied into all eq evaluations

### Possible Issues

1. **Variable ordering in eq polynomial**: The eq(tau_low, r_tail_reversed) might have wrong bit ordering

2. **Off-by-one in tau split**: tau_low might be wrong length or content

3. **Lagrange scaling factor**: The L(tau_high, r0) might not be correctly incorporated

---

## Next Steps

1. **Add debug output to Zolt prover**:
   - Print tau, tau_high, tau_low at initialization
   - Print the split_eq's current_scalar after initialization
   - Print the final values of E_out, E_in tables

2. **Add debug output to track eq evaluation**:
   - After final round, compute eq(tau_low, r) directly and compare
   - Verify L(tau_high, r0) value matches Jolt

3. **Trace the sumcheck claim evolution**:
   - Print claim after each round
   - Compare with what verifier expects

---

## Test Commands

```bash
# All tests pass
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification (fails at Stage 1)
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
