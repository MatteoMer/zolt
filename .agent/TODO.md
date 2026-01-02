# Zolt-Jolt Compatibility TODO

## Current Status: Session 39 - January 2, 2026

**712 tests pass. Stage 1 sumcheck output_claim mismatch persists.**

---

## Summary of Investigation

### What Works ✅
1. EqPolynomial - partition of unity holds, sum equals 1
2. Individual Az and Bz MLE evaluations match between prover and verifier
3. All 712 unit tests pass
4. Proof generation completes successfully

### What Doesn't Work ❌
1. Stage 1 sumcheck output_claim doesn't match expected_output_claim
   - output_claim: 21656329869382715893372831461077086717482664293827627865217976029788055707943
   - expected: 4977070801800327014657227951104439579081780871540314422928627443513195286072

### Investigation in Session 39

**Finding 1: Round Zero vs General Materialization**

The jolt-rust-expert agent identified that Jolt has TWO materialization paths:
- `fused_materialise_polynomials_round_zero`: Simple indexing with `full_idx = grid_size * i + j`
- `fused_materialise_polynomials_general_with_multiquadratic`: Complex indexing with r_grid

Zolt was using the general path for all rounds. I updated `materializeLinearPhasePolynomials` to use the round zero logic, but the proof output is IDENTICAL before and after the change.

**Possible explanations:**
1. The materialization change doesn't affect the values written to az/bz at round zero
2. The issue is elsewhere in the code path
3. The formula is correct but some constants differ

**Finding 2: Verification Formula**

The verifier computes expected_output_claim as:
```
tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
```

Where inner_sum_prod = Az(rx_constr, r_cycle) * Bz(rx_constr, r_cycle).

The rx_constr = [r_stream (sumcheck_challenges[0]), r0].

---

## Next Steps

1. **Add debug output to Zolt's computeRemainingRoundPoly**
   - Print t_zero and t_infinity values
   - Print the round polynomial evaluations
   - Compare these with what Jolt would produce

2. **Debug buildTPrimePoly**
   - Print the t_prime_poly.evaluations array
   - Verify it matches Jolt's round_zero t_prime construction

3. **Check if the issue is in split_eq or multiquadratic**
   - The E_out/E_in tables might differ
   - The expandGrid logic might differ

4. **Verify transcript consistency**
   - The challenges must be generated identically

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

---

## Key Files

### Zolt
- `src/zkvm/spartan/streaming_outer.zig` - materializeLinearPhasePolynomials, buildTPrimePoly
- `src/poly/split_eq.zig` - getWindowEqTables, computeCubicRoundPoly
- `src/poly/multiquadratic.zig` - MultiquadraticPolynomial

### Jolt (Reference)
- `jolt-core/src/zkvm/spartan/outer.rs` - fused_materialise_polynomials_round_zero
- `jolt-core/src/poly/split_eq_poly.rs` - E_out_in_for_window
