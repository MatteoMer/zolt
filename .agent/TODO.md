# Zolt-Jolt Compatibility TODO

## Current Status: Session 36 Continued - January 2, 2026

**Progress: Binding order fixed (split_eq first), sumcheck output_claim mismatch persists**

---

## Completed This Session

### 1. Fixed Az/Bz Binding on ALL Rounds (DONE)
- Changed `bindRemainingRoundChallenge()` to bind az/bz on ALL rounds, not just rounds > 1
- This matches Jolt's `ingest_challenge()` which binds az/bz on every round
- Reference: `jolt-core/src/zkvm/spartan/outer.rs:1452-1455`

### 2. Fixed Proof Converter to Use computeRemainingRoundPoly for ALL Rounds (DONE)
- Previously used `computeRemainingRoundPolyMultiquadratic()` for rounds 1+
- Now uses `computeRemainingRoundPoly()` for all rounds

### 3. Fixed Binding Order to Match Jolt Exactly (DONE)
- Changed order in `bindRemainingRoundChallenge()`:
  - OLD: az/bz → split_eq → t_prime
  - NEW: split_eq → t_prime → az/bz (matches Jolt's ingest_challenge)

---

## Current Issue: Sumcheck Output Claim Mismatch

```
output_claim:          21656329869382715893372831461077086717482664293827627865217976029788055707943
expected_output_claim: 4977070801800327014657227951104439579081780871540314422928627443513195286072
```

### Analysis This Session

1. **Eq Factor Accumulation**: VERIFIED CORRECT
   - Prover computes `Π_{i=0}^{10} eq(tau[10-i], r_i)` via bind()
   - Verifier computes `eq(tau_low, r_tail_reversed) = Π eq(tau[i], r_{10-i})`
   - Both are mathematically equivalent products

2. **Az/Bz MLE Evaluation**: VERIFIED MATCHING
   - Test shows `Az MLE match: true, Bz MLE match: true`
   - Individual Az/Bz values at the final point are correct

3. **Constraint Mapping**: VERIFIED CORRECT
   - Jolt's `a` = Zolt's `condition`
   - Jolt's `b` = Zolt's `left - right`
   - `FIRST_GROUP_INDICES` and `SECOND_GROUP_INDICES` match Jolt

4. **Polynomial Indexing**: VERIFIED CORRECT
   - `az[grid_size * i + j]` = Az for cycle i, group j
   - Matches Jolt's `fused_materialise_polynomials_round_zero`

---

## The Puzzle

Individual components all match between Zolt and Jolt:
- ✅ Eq factor accumulation formula
- ✅ Az/Bz MLE final values
- ✅ Constraint definitions
- ✅ Polynomial indexing

Yet the sumcheck output_claim doesn't match expected_output_claim. This suggests the issue is in how these components are **combined** during the sumcheck rounds.

### Most Likely Culprit: t_prime_poly Accumulation

The `buildTPrimePoly` function builds:
```
t_prime[idx] = Σ_{x_out, x_in} E_out[x_out] * E_in[x_in] * Az[i][j] * Bz[i][j]
```

After binding all variables:
- E_out = [1], E_in = [1]
- t_prime should contain a single value = Az_final * Bz_final
- current_scalar should contain the full eq factor

But if E_out/E_in aren't being updated correctly during binding, the accumulated values could be wrong.

---

## Next Steps

### 1. Add Debug to buildTPrimePoly
Print after each call:
- `E_out.len`, `E_in.len`
- First few values of E_out, E_in
- Size and first few values of resulting t_prime

### 2. Compare t_prime Round-by-Round
For each round, compare Zolt's t_prime to Jolt's by adding debug prints to both sides.

### 3. Verify E_out/E_in After Binding
After each bind, verify that `E_out_vec` and `E_in_vec` are being trimmed correctly in split_eq.

---

## Test Commands

```bash
# Run Zolt tests (all 710 pass)
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
