# Zolt-Jolt Compatibility TODO

## Current Status: Session 38 - January 2, 2026

**Progress: Verified EqPolynomial is correct, identified mathematical issue in sumcheck**

---

## Session 38 Findings

### CRITICAL: EqPolynomial is CORRECT!

1. ✅ Partition of unity test passes - sum of eq evaluations equals 1
2. ✅ Test with r=[5555, 6666] confirms sum == F.one()
3. ✅ Individual Az and Bz MLE values match between prover and verifier

### The Mathematical Issue Explained

The test computes two different quantities:
- `prover_sum = Σ_t eq(r_cycle, t) * Az(t) * Bz(t)` = MLE(Az*Bz)
- `verifier_inner_sum_prod = Az_MLE * Bz_MLE` = product of MLEs

These are NOT equal in general: `MLE(f*g) ≠ MLE(f) * MLE(g)`

HOWEVER, the sumcheck protocol should produce `MLE(f) * MLE(g)` after all bindings, because the final claim is a single-point evaluation.

### Root Cause

The issue is in how the round polynomials are constructed. The sumcheck should:
1. In each round, compute the univariate polynomial over the remaining sum
2. After binding all variables, arrive at `eq(τ, r) * Az(r) * Bz(r)`
3. This equals `eq(τ, r) * MLE_Az(r) * MLE_Bz(r)`

But something in Zolt's prover is causing it to compute the wrong intermediate values.

---

## Current Issue: Sumcheck Output Claim Mismatch

```
output_claim:          21656329869382715893372831461077086717482664293827627865217976029788055707943
expected_output_claim: 4977070801800327014657227951104439579081780871540314422928627443513195286072
```

All 11 rounds pass individually (p(0) + p(1) = claim), but the final claim doesn't match expected.

---

## Remaining Investigation Areas

### 1. Round Polynomial Construction
The `computeRemainingRoundPoly` uses:
- `t_zero` and `t_infinity` from t_prime projection
- `computeCubicRoundPoly` from split_eq

Need to verify:
- Is t_prime storing the right values?
- Is the projection formula correct?
- Is the cubic construction matching Jolt's?

### 2. t_prime_poly Building
`buildTPrimePoly` accumulates:
```
t_prime[idx] = Σ E_out * E_in * Az * Bz
```

Need to compare index-by-index with Jolt's equivalent.

### 3. Variable Binding Order
Both use LowToHigh, but verify:
- split_eq binds correctly
- t_prime_poly binds correctly
- az_poly/bz_poly bind correctly

---

## Test Commands

```bash
# Run Zolt tests (all 712+ pass)
zig build test --summary all

# Generate proof
rm -f /tmp/zolt_proof_dory.bin
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Files Modified This Session

1. `src/poly/mod.zig` - Added partition of unity tests for EqPolynomial
2. `.agent/NOTES.md` - Updated with findings
3. `.agent/TODO.md` - Updated status

---

## Next Steps

1. Add detailed round-by-round debug output to streaming outer prover
2. Compare t_prime values at each step between Zolt and Jolt
3. Verify the mathematical relationship between prover computation and expected value
