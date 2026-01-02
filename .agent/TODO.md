# Zolt-Jolt Compatibility TODO

## Current Status: Session 36 - January 2, 2026

**Progress: Fixed Az/Bz binding, sumcheck output_claim mismatch persists**

---

## Completed This Session

### 1. Fixed Az/Bz Binding on ALL Rounds (DONE)
- Changed `bindRemainingRoundChallenge()` to bind az/bz on ALL rounds, not just rounds > 1
- This matches Jolt's `ingest_challenge()` which binds az/bz on every round
- Reference: `jolt-core/src/zkvm/spartan/outer.rs:1452-1455`

### 2. Fixed Proof Converter to Use computeRemainingRoundPoly for ALL Rounds (DONE)
- Previously used `computeRemainingRoundPolyMultiquadratic()` for rounds 1+
- Now uses `computeRemainingRoundPoly()` for all rounds
- This ensures proper Az/Bz materialization, binding, and t_prime rebuilding

---

## Current Issue: Sumcheck Output Claim Mismatch

```
output_claim:          21656329869382715893372831461077086717482664293827627865217976029788055707943
expected_output_claim: 4977070801800327014657227951104439579081780871540314422928627443513195286072
```

### Extensive Analysis Performed
1. ✅ `projectToFirstVariable` index mapping - MATCHES Jolt
2. ✅ `expandGrid` ternary expansion - MATCHES Jolt
3. ✅ `ternaryToBinaryIndex` and `isBooleanTernaryIndex` - CORRECT
4. ✅ Binding order (LowToHigh) - MATCHES Jolt
5. ✅ E_out/E_in table sizes after binding - CORRECT
6. ✅ `computeCubicRoundPoly` Gruen formula - MATCHES Jolt
7. ✅ `lagrangeKernel` computation - MATCHES Jolt
8. ✅ split_eq `bind` formula for eq factor - MATCHES Jolt

### Key Finding
- Round 0 polynomial is UNCHANGED (correct, uses initial t_prime)
- Round 1+ polynomials DIFFER (use rebuilt t_prime from bound az/bz)
- All individual formulas checked match Jolt
- Issue must be in how values flow through the system

---

## Remaining Suspects

### 1. E_out/E_in Index Ordering in buildTPrimePoly
The iteration order over `(x_out, x_in)` pairs might differ from Jolt's parallel iteration order.

### 2. Polynomial Binding Order Effects
While individual bindings match, there might be a cumulative effect from different ordering:
- Jolt: split_eq, t_prime, THEN az/bz
- Zolt: az/bz, THEN split_eq, t_prime

### 3. Window Eq Tables After Binding
The `getWindowEqTables` might return different values after binding because it uses `current_index` which is updated during bind.

### 4. Current Scalar Accumulation
The `current_scalar` multiplies eq factors for bound variables. If the binding sequence differs, this could accumulate differently.

---

## Next Steps (For Next Session)

### Debug Strategy
1. Add Jolt-side debug output showing t_prime_poly values at each round
2. Print (t_zero, t_infinity) from BOTH Zolt and Jolt side-by-side
3. Trace through binding order effects on current_scalar

### Specific Checks Needed
1. After round 1 bind, compare:
   - az_poly.evaluations[0..boundLen]
   - bz_poly.evaluations[0..boundLen]
   - split_eq.current_scalar
   - E_out and E_in table contents

2. For round 2 rebuild, compare:
   - Input to buildTPrimePoly (E_out, E_in, az, bz)
   - Output t_prime_poly.evaluations

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

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/zkvm/spartan/streaming_outer.zig` | Bind az/bz on all rounds |
| `src/zkvm/proof_converter.zig` | Use computeRemainingRoundPoly for all rounds |

---

## Architecture Summary

### Jolt's Stage 1 Flow (Outer Remaining Sumcheck)
```
1. UniSkip: degree-27 poly on domain {-4..5}
2. After UniSkip: split_eq gets lagrange_tau_r0 scaling
3. For each round:
   a. If round > 0: next_window() rebuilds t_prime from bound az/bz
   b. compute_message() gets (t0, t_inf) from t_prime
   c. gruen_poly_deg_3() computes round polynomial
   d. ingest_challenge() binds: split_eq, t_prime, az, bz
4. Final output_claim = p(r) after all rounds
5. Expected = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
```

### Zolt's Equivalent Implementation
```
1. UniSkip: same polynomial construction
2. After UniSkip: split_eq.initWithScaling(lagrange_tau_r0)
3. For each round:
   a. If t_prime.num_vars == 0: rebuildTPrimePoly() from bound az/bz
   b. computeTEvals() gets (t0, t_inf) from t_prime
   c. computeCubicRoundPoly() computes round polynomial
   d. bindRemainingRoundChallenge() binds: az, bz, split_eq, t_prime
4. Final output_claim = current_claim after all rounds
5. Jolt verifier checks same expected formula
```
