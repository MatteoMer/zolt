# Zolt-Jolt Compatibility TODO

## Current Status: Session 36 Final - January 2, 2026

**Progress: Binding order fixed, split_eq pop implemented, sumcheck output mismatch persists**

---

## Changes Made This Session

### 1. Fixed Binding Order in `bindRemainingRoundChallenge()` (DONE)
- Changed order: split_eq first, then t_prime, then az/bz
- Matches Jolt's `ingest_challenge()` exactly

### 2. Fixed split_eq `bind()` to Pop Tables (DONE)
- Now pops from E_in_vec or E_out_vec after binding (like Jolt)
- For LowToHigh: first pop from E_in_vec (when current_index > m), then E_out_vec

### 3. Fixed `getWindowEqTables()` Bounds (DONE)
- Uses actual vector lengths instead of original num_x_out/num_x_in

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
