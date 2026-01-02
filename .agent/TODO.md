# Zolt-Jolt Compatibility TODO

## Current Status: Session 35 - January 2, 2026

**All 710 Zolt tests pass, but Jolt verification fails with output claim mismatch**

---

## Completed This Session

### 1. MultiquadraticPolynomial.bind()
- Added quadratic interpolation: `f(r) = f(0)*(1-r) + f(1)*r + f(∞)*r(r-1)`
- Added `isBound()` and `finalSumcheckClaim()` helpers
- Reference: `jolt-core/src/poly/multiquadratic_poly.rs:bind_first_variable`

### 2. GruenSplitEqPolynomial.getEActiveForWindow()
- Computes eq table over active window bits (window_size - 1 bits)
- Returns `[1]` for window_size=1 (no active bits)
- Reference: `jolt-core/src/poly/split_eq_poly.rs:E_active_for_window`

### 3. t_prime_poly Integration in StreamingOuterProver
- Added `t_prime_poly: ?MultiquadraticPolynomial(F)` field
- `buildTPrimePoly()` - Creates from bound Az/Bz using multiquadratic expansion
- `rebuildTPrimePoly()` - nextWindow equivalent for rebuilding
- `computeTEvals()` - Projects t_prime using E_active weights
- Binds t_prime_poly after each round (critical for Jolt compatibility)

### 4. LinearOnlySchedule Fix
- Round 1 (first remaining round) now initializes linear stage
- This matches Jolt's round 0 being the switch-over point
- All rounds use linear phase (no streaming rounds)
- Removed separate streaming round handling

---

## Current Issue: Sumcheck Output Claim Mismatch

```
output_claim:          8206907536993754864973510285637683658139731930814938521485939885759521476392
expected_output_claim: 5887936957248500858334092112703331331673171118046881060635640978343116912473
```

### Root Cause Hypothesis

The sumcheck round polynomials are being computed incorrectly. Likely issues:

1. **buildTPrimePoly** - The base-3 index mapping or grid expansion may be wrong
2. **computeTEvals** - The E_active projection may not match Jolt's indexing
3. **materializeLinearPhasePolynomials** - Index parity for group selection may be off

### Verifier's Expected Claim Formula
```rust
let rx_constr = &[r_stream, r0];  // r_stream = sumcheck_challenges[0]
let inner_sum_prod = key.evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);

let tau_high_bound_r0 = LagrangePolynomial::lagrange_kernel(&tau_high, &r0);
let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);

expected = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
```

---

## Next Steps

### Debug Strategy
1. Add debug prints for t_prime_poly values at each round
2. Compare (t_zero, t_infinity) with Jolt's compute_t_evals output
3. Compare round polynomial [s(0), s(1), s(2), s(3)] with Jolt
4. Verify base-3 indexing in buildTPrimePoly matches Jolt

### Specific Checks
- [ ] Verify `expandGrid` produces correct infinity values
- [ ] Verify E_out/E_in factorization matches Jolt's getWindowEqTables
- [ ] Verify projectToFirstVariable matches Jolt's project_to_first_variable
- [ ] Add Jolt debug prints to compare intermediate values

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/poly/multiquadratic.zig` | Added `bind()`, `isBound()`, `finalSumcheckClaim()` |
| `src/poly/split_eq.zig` | Added `getEActiveForWindow()` |
| `src/zkvm/spartan/streaming_outer.zig` | Major restructure for LinearOnlySchedule, added t_prime_poly |

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

## Architecture Summary

### Jolt's LinearOnlySchedule Flow
```
Round 0: switch_over point
  → OuterLinearStage::initialize()
  → fused_materialise_polynomials_round_zero()
  → Builds Az/Bz polynomials AND t_prime_poly
  → compute_message() uses t_prime_poly

Round 1+: linear phase continues
  → next_window() rebuilds t_prime_poly from bound Az/Bz
  → compute_message() uses t_prime_poly
  → ingest_challenge() binds split_eq, t_prime, az, bz
```

### Zolt's Equivalent
```
Round 1 (= Jolt's Round 0):
  → materializeLinearPhasePolynomials()
  → buildTPrimePoly()
  → computeRemainingRoundPoly() uses t_prime_poly

Round 2+:
  → rebuildTPrimePoly() if t_prime_poly.num_vars == 0
  → computeRemainingRoundPoly() uses t_prime_poly
  → bindRemainingRoundChallenge() binds split_eq, t_prime, az, bz
```
