# Zolt-Jolt Compatibility TODO

## Current Status: Session 36 - January 2, 2026

**Progress: Fixed Az/Bz binding issue, but output_claim still doesn't match expected_output_claim**

---

## Completed This Session

### 1. Fixed Az/Bz Binding on ALL Rounds
- Changed `bindRemainingRoundChallenge()` to bind az/bz on ALL rounds, not just rounds > 1
- This matches Jolt's `ingest_challenge()` which binds az/bz on every round
- Reference: `jolt-core/src/zkvm/spartan/outer.rs:1452-1455`

### 2. Fixed Proof Converter to Use computeRemainingRoundPoly for ALL Rounds
- Previously used `computeRemainingRoundPolyMultiquadratic()` for rounds 1+
- Now uses `computeRemainingRoundPoly()` for all rounds
- This ensures proper Az/Bz materialization, binding, and t_prime rebuilding

---

## Current Issue: Sumcheck Output Claim Mismatch (Updated)

After the fix, the round polynomials changed but there's still a mismatch:

```
output_claim:          21656329869382715893372831461077086717482664293827627865217976029788055707943
expected_output_claim: 4977070801800327014657227951104439579081780871540314422928627443513195286072
```

### Key Observations
1. Round 0 polynomial is UNCHANGED (correct)
2. Round 1+ polynomials are DIFFERENT (Az/Bz binding now affects them)
3. The challenges (r_i) are different because the polynomials changed
4. The R1CS input evaluations changed accordingly

### Possible Remaining Issues
1. **t_prime_poly rebuilding** - May not match Jolt's `compute_evaluation_grid_from_polynomials_parallel`
2. **E_active computation** - May not match Jolt's `E_active_for_window`
3. **projectToFirstVariable** - Index mapping may differ from Jolt

---

## Next Steps

### Debug Strategy
1. Add debug output to compare t_prime_poly values with Jolt
2. Compare (t_zero, t_infinity) at each round between Zolt and Jolt
3. Verify E_out and E_in tables match between implementations
4. Add debug prints to Jolt's prover for comparison

### Key Functions to Verify
- [ ] `buildTPrimePoly()` - ternary grid expansion
- [ ] `computeTEvals()` - E_active projection
- [ ] `projectToFirstVariable()` - index mapping
- [ ] `getEActiveForWindow()` - active window bits

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/zkvm/spartan/streaming_outer.zig` | Bind az/bz on all rounds, not just > 1 |
| `src/zkvm/proof_converter.zig` | Use computeRemainingRoundPoly for all rounds |

---

## Test Commands

```bash
# Run Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification tests
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Architecture Notes (from previous sessions)

### Jolt's LinearOnlySchedule Flow
```
Round 0 (switch_over): Linear::initialize() materializes Az/Bz, builds t_prime_poly
Round 1+: next_window() rebuilds t_prime_poly from bound Az/Bz
All rounds: ingest_challenge() binds split_eq, t_prime_poly, az, bz
```

### Zolt's Equivalent (after fix)
```
Round 1 (= Jolt's Round 0):
  → materializeLinearPhasePolynomials()
  → buildTPrimePoly()
  → computeRemainingRoundPoly() uses t_prime_poly
  → bindRemainingRoundChallenge() binds split_eq, t_prime, az, bz

Round 2+:
  → rebuildTPrimePoly() when t_prime_poly.num_vars == 0
  → computeRemainingRoundPoly() uses t_prime_poly
  → bindRemainingRoundChallenge() binds split_eq, t_prime, az, bz
```
