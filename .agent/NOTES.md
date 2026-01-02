# Zolt-Jolt Compatibility Notes

## Current Status (Session 32 - January 2, 2026)

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

### Verification Values

From test output:
```
output_claim:          21656329869382715893372831461077086717482664293827627865217976029788055707943
expected_output_claim: 4977070801800327014657227951104439579081780871540314422928627443513195286072
```

### 712+ Tests Pass

All tests pass including:
- EqPolynomial partition of unity
- Az/Bz MLE computation
- Gruen cubic polynomial construction
- Multiquadratic polynomial operations

### Next Steps

1. Add detailed tracing to each sumcheck round
2. Compare t_prime_poly values between Zolt and Jolt
3. Verify the round polynomial aggregation produces the correct final claim

---

## Previous Status (Session 31 - January 2, 2026)

### Recent Progress

1. **Fixed transcript to use compressed coefficients** - The prover was appending evaluation points `[s(0), s(1), s(2), s(3)]` instead of compressed coefficients `[c0, c2, c3]`. Fixed in `streaming_outer.zig`.

2. **Verified mathematical correctness** - For linear Az and Bz functions, the identity `MLE(Az)(r) * MLE(Bz)(r) = Az(z_MLE(r)) * Bz(z_MLE(r))` holds due to linearity.

3. **Identified potential structural issue** - The Az/Bz computation in `materializeLinearPhasePolynomials` may not match Jolt's structure.

### Key Observations

1. **Individual Az and Bz MLEs match** - The test shows `Az MLE match: true, Bz MLE match: true`
2. **But the products differ** - This suggests a subtle structural difference in how the values are combined

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
