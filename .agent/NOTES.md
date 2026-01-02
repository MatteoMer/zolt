# Zolt-Jolt Compatibility Notes

## Current Status (Session 39 - January 2, 2026)

### Summary

The Stage 1 sumcheck output_claim doesn't match the expected_output_claim. All 712 unit tests pass, and the transcript challenges are identical between prover and verifier.

### Session 39 Investigation

1. **Round zero materialization**: Updated `materializeLinearPhasePolynomials` to use the simple `full_idx = grid_size * i + j` indexing (matching Jolt's `fused_materialise_polynomials_round_zero`). However, this didn't change the proof output because when `num_r_bits = 0` (at round zero), both old and new formulas compute identical indices.

2. **Transcript consistency**: The challenges are correctly shared between prover and verifier. The same `r_stream` value appears in both the round polynomial and the verification formula.

3. **E_out/E_in tables**: These are correctly initialized with `[1]` at index 0, matching Jolt's invariant.

4. **Multiquadratic expansion**: The `expandGrid` function correctly computes `[f0, f1, f1-f0]` for window_size=1, matching Jolt's `expand_linear_dim1`.

### Verification Formula (from Jolt)

```
expected_output_claim = L(tau_high, r0) * eq(tau_low, r_reversed) * Az(rx_constr) * Bz(rx_constr)
```

Where:
- `L(tau_high, r0)` = Lagrange kernel
- `eq(tau_low, r_reversed)` = EqPolynomial MLE evaluation
- `rx_constr = [r_stream, r0]` = constraint row randomness
- `Az`, `Bz` = constraint polynomial evaluations

### Current Values

```
output_claim:          21656329869382715893372831461077086717482664293827627865217976029788055707943
expected_output_claim: 4977070801800327014657227951104439579081780871540314422928627443513195286072
```

### Next Steps

1. Add debug output to trace t_prime_poly construction:
   - Print t_prime_poly.evaluations[0], [1], [2] after construction
   - Compare with what Jolt produces

2. Verify E_out and E_in table values at round zero:
   - Print E_out.len, E_in.len, first few values
   - Compare with Jolt's getWindowEqTables output

3. Check if the issue is in how current_scalar is used in computeCubicRoundPoly

---

## Previous Status (Session 32 - January 2, 2026)

### CRITICAL DISCOVERY: EqPolynomial is CORRECT!

**Verified:**
1. ✅ EqPolynomial partition of unity test PASSES - sum equals 1
2. ✅ Individual Az and Bz MLE evaluations MATCH between prover and verifier
3. ✅ EqPolynomial with r=[5555, 6666] sums to exactly ONE

**Debug output confirms:**
```
Sum:
  ac96341c4ffffffb 36fc76959f60cd29 666ea36f7879462e 0e0a77c19a07df2f
One:
  ac96341c4ffffffb 36fc76959f60cd29 666ea36f7879462e 0e0a77c19a07df2f
Sum == One? true
```

### The Inner Sum Product Mismatch Explained

The test "inner_sum_prod: prover vs verifier computation" shows:
- `prover_sum = Σ_t eq(r_cycle, t) * Az(t) * Bz(t)` (MLE of product)
- `verifier_inner_sum_prod = Az_MLE(r_cycle) * Bz_MLE(r_cycle)` (product of MLEs)

These are mathematically DIFFERENT quantities:
```
MLE(f*g)(r) ≠ MLE(f)(r) * MLE(g)(r)
```

HOWEVER, the sumcheck protocol produces the product of MLEs, NOT the MLE of products:
- After binding all variables to r, the claim becomes: `eq(τ, r) * Az(r) * Bz(r)`
- Here Az(r) and Bz(r) are single-point evaluations, which equal their MLE values

### Root Cause Hypothesis

The issue is in how the streaming outer prover's round polynomials are constructed. The sumcheck should produce `Az_MLE * Bz_MLE` at the final point, but something in the computation is causing it to compute `MLE(Az*Bz)` instead.

Possible issues:
1. The `buildTPrimePoly` function stores `Σ eq_weight * Az * Bz` per grid point
2. This might be computing a different quantity than what Jolt's prover does
3. The projection via `E_active` and `computeCubicRoundPoly` may have subtle bugs

---

## Architecture Notes

### Sumcheck Structure

Stage 1 has:
- 1 UniSkip round (produces r0)
- 1 + num_cycle_vars remaining rounds

For trace_length = 1024:
- num_cycle_vars = 10
- num_rows_bits = 12
- tau.len = 12
- tau_low.len = 11
- Remaining rounds = 11
- r_tail_reversed = [r_10, r_9, ..., r_1, r_stream]

### Big-Endian Convention

From Jolt's eq_poly.rs:
```
evals(r)[i] = eq(r, b₀…b_{n-1})
where i has MSB b₀ and LSB b_{n-1}
```

### Code Paths

**Zolt prover flow:**
1. `StreamingOuterProver.initWithScaling` - sets up split_eq with tau and scaling_factor
2. `bindFirstRoundChallenge(r0)` - sets r_stream, current_round=1
3. `computeRemainingRoundPoly()` calls:
   - `materializeLinearPhasePolynomials()` - fills az_poly, bz_poly, t_prime_poly
   - `computeTEvals()` - gets (t_zero, t_infinity) from t_prime_poly
   - `split_eq.computeCubicRoundPoly(t_zero, t_infinity, previous_claim)` - builds round polynomial
4. After each round: `bindRemainingRoundChallenge(r)` binds split_eq, t_prime_poly, az_poly, bz_poly

**Jolt prover flow:**
1. `OuterSharedState::new` - sets up split_eq_poly, r0
2. `OuterLinearStage::initialize` calls:
   - `fused_materialise_polynomials_round_zero` or `compute_evaluation_grid_from_polynomials_parallel`
   - Builds az, bz, t_prime_poly
3. `next_round` calls:
   - `compute_t_evals` - gets (t_zero, t_infinity)
   - `compute_cubic_round_poly` - builds round polynomial
4. After each round: `ingest_challenge` binds all polynomials
