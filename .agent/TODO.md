# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Output Claim Mismatch (Session 20)

### Summary
All Stage 1 sumcheck rounds pass (p(0)+p(1) = claim), but the **expected output claim** doesn't match the **output claim from sumcheck walk**.

### Key Fixes Applied
1. R1CSInputIndex order matches Jolt's ALL_R1CS_INPUTS ✓
2. UniSkip polynomial uses tau_low for eq (not full tau) ✓

### Latest Test Results
After tau_low fix:
- **output_claim**: 7120341815860535077792666425421583012196152296139946730075156877231654137396
- **expected_output_claim**: 2000541294615117218219795634222435854478303422072963760833200542270573423153
- **Ratio**: ~3.56 (was ~1.338 before fix)

### Analysis of Jolt's Structure

The Lagrange kernel L(τ_high, r0) appears in **BOTH**:
1. UniSkip polynomial: `s1(Y) = L(τ_high, Y) * t1(Y)` → `s1(r0) = L(τ_high, r0) * t1(r0)`
2. Remaining rounds: `current_scalar = L(r0, τ_high)` (same value, symmetric kernel)

This is correct because:
- The UniSkip claim `s1(r0)` already includes the Lagrange factor
- The remaining rounds include it in `current_scalar`
- The expected output also includes `tau_high_bound_r0`

### Remaining Investigation Areas

1. **Streaming Round (current_round == 1)**
   - Uses multiquadratic: `t(0) = Az_g0*Bz_g0`, `t(∞) = dAz*dBz`
   - This should be correct for the quadratic Az*Bz polynomial

2. **Cycle Rounds (current_round > 1)**
   - Uses r_grid for bound streaming challenges
   - Complex index structure with constraint group selection
   - May have issues with how selector interacts with r_grid

3. **Az/Bz Computation**
   - Prover computes via constraint evaluators + Lagrange weights
   - Verifier computes via opening claims (R1CS input MLEs) + linear combinations
   - These should produce the same values at the challenge point

4. **EQ Polynomial Binding**
   - Prover: binds tau[n-1-i] with r_i (LowToHigh order)
   - Verifier: computes eq(tau_low, r_tail_reversed)
   - These should be equivalent (same products, different order)

### Expected Output Claim Formula
```
expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
         = L(τ_high, r0) * eq(τ_low, r_tail_reversed) * Az_final * Bz_final

Az_final = az_g0 + r_stream * (az_g1 - az_g0)
Bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)
inner_sum_prod = Az_final * Bz_final
```

## Completed

### Phase 1-5: Core Infrastructure ✓
1. Transcript Compatibility - Blake2b
2. Proof Structure - 7-stage
3. Serialization - Arkworks format
4. Commitment - Dory with Jolt SRS
5. Verifier Preprocessing Export

### Stage 1 Fixes ✓
- Big-endian EqPolynomial.evals()
- R1CSInputIndex order matches Jolt
- UniSkip uses tau_low for eq polynomial
- All 656 Zolt tests pass

## Test Commands

```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
