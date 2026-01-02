# Zolt-Jolt Compatibility TODO

## Current Status: Session 37 - January 2, 2026

**Progress: Deep analysis of sumcheck mismatch, mathematical equivalence verified**

---

## Session 37 Changes

### 1. Fixed Transcript (DONE)
- Prover now appends compressed coefficients `[c0, c2, c3]` instead of evaluations
- Matches Jolt's `UniPoly::compress()` format

### 2. Mathematical Analysis (DONE)
- Verified: `MLE(Az)(r) = Az(z_MLE(r))` due to linearity
- Verified: `MLE(Az * Bz) = MLE(Az) * MLE(Bz)` for this special case
- Both Jolt and Zolt compute `Σ eq * (Az * Bz)` in the prover

### 3. Key Finding: Individual MLEs Match!
- `prover_az_mle == verifier_az_final` ✓
- `prover_bz_mle == verifier_bz_final` ✓
- But `prover_sum (Az*Bz MLE) ≠ verifier_inner_sum_prod (Az*Bz from z_MLE)`
- The difference is ~0.22%, not a simple factor

---

## Current Issue: Sumcheck Output Claim Mismatch (UNCHANGED)

```
output_claim:          21656329869382715893372831461077086717482664293827627865217976029788055707943
expected_output_claim: 4977070801800327014657227951104439579081780871540314422928627443513195286072
```

### Key Observations

1. **Az/Bz MLE values match perfectly** between prover and verifier
   - `Az MLE match: true, Bz MLE match: true`

2. **The output_claim hasn't changed** after the split_eq fixes
   - This suggests the issue is NOT in E_out/E_in table access
   - The pop is an optimization, not a correctness fix

3. **All 11 individual rounds pass** (p(0) + p(1) = claim)
   - The polynomials are internally consistent
   - But the final claim doesn't match expected

4. **The mismatch is large** - different by ~4x
   - Not a field arithmetic issue (would be close but off by a small amount)

---

## Remaining Hypotheses

### 1. t_prime Polynomial Construction
The `buildTPrimePoly` function accumulates:
```
t_prime[idx] = Σ E_out[x_out] * E_in[x_in] * Az[i][j] * Bz[i][j]
```

But this includes the eq factor from E_out * E_in, which is SEPARATE from current_scalar.
The final output should be:
```
output_claim = current_scalar * (remaining_eq_factor) * Az_final * Bz_final
```

Maybe the remaining_eq_factor is being computed incorrectly?

### 2. current_scalar Initialization
The split_eq is initialized with `lagrange_tau_r0` as the scaling factor.
This becomes the initial `current_scalar`.

But after all rounds are bound:
```
current_scalar = lagrange_tau_r0 * Π eq(tau[10-i], r_i)
```

The verifier expects:
```
eq_factor = lagrange_tau_r0 * eq(tau_low, r_tail_reversed)
```

These should be the same, but maybe there's an ordering issue?

### 3. Round Polynomial Evaluation Point
When computing the next round's claim:
```
claim = poly(r) where poly = [s(0), s(1), s(2), s(3)]
```

Are we evaluating at the right point?

---

## Next Steps for Future Session

### 1. Add Debug Output to Compare Values
- Print t_zero, t_infinity for each round
- Print current_scalar before/after each bind
- Compare with Jolt's values (add similar prints to Jolt)

### 2. Verify eq Factor Directly
After all rounds, compute:
```
prover_eq = prover.split_eq.current_scalar
verifier_eq = lagrange_tau_r0 * eq(tau_low, r_tail_reversed)
```
These should match!

### 3. Check t_prime Final Value
After all bindings, the t_prime polynomial should have a single value.
This should equal `Az_final * Bz_final`.

---

## Test Commands

```bash
# Run Zolt tests (all 710 pass)
zig build test --summary all

# Generate proof
rm -f /tmp/zolt_proof_dory.bin
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
