# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck Verification (Session 32)

### Issue Summary
The Stage 1 sumcheck output_claim doesn't match expected_output_claim.

```
output_claim:          18149181199645709635565994144274301613989920934825717026812937381996718340431
expected_output_claim:  9784440804643023978376654613918487285551699375196948804144755605390806131527
ratio: 9880136244111364639762211720308610935310266258272323252035902739067030045854 (not a simple number)
```

### Recent Fix Applied
- Added transcript operations before generating remaining round polynomials:
  1. Append uni_skip_claim to transcript
  2. Consume batching coefficient challenge
- This matches Jolt's BatchedSumcheck::verify flow

### Verified Correct
- [x] Blake2b transcript (byte-for-byte match)
- [x] UniSkip polynomial coefficients
- [x] Compressed poly format [c0, c2, c3]
- [x] interpolateDegree3 and evalsToCompressed
- [x] computeCubicRoundPoly formula matches Jolt's gruen_poly_deg_3
- [x] Streaming round uses SUM-OF-PRODUCTS structure
- [x] Cycle rounds use selector = full_idx & 1
- [x] r_grid update logic matches Jolt's HalfSplitSchedule
- [x] E_out/E_in table construction (big-endian indexing)
- [x] split_eq.bind() formula matches Jolt
- [x] Transcript flow (input_claim, batching_coeff, then round polys)
- [x] All 656 Zolt tests pass

### Root Cause Analysis
The transcript flow is now correct, but the round polynomial VALUES themselves are wrong.
This means the issue is in how t_zero/t_infinity are being computed in:
- computeRemainingRoundPoly (streaming round)
- computeRemainingRoundPolyMultiquadratic (cycle rounds)

### Possible Issues

1. **Az/Bz Computation**
   - The constraint evaluation might differ from Jolt
   - Lagrange weight application might be wrong

2. **E_out/E_in Values**
   - The eq table values might be computed differently
   - Big-endian indexing might not match exactly

3. **r_grid Accumulation**
   - The r_grid weights might be computed at wrong points
   - The streaming round might not properly populate r_grid

4. **Multiquadratic Expansion**
   - The prod_0 and prod_inf computation might differ
   - The slope calculation might have sign issues

### Next Steps (Priority Order)
1. Add debug output to print t_zero/t_infinity at round 0 (streaming)
2. Compare r_grid values with expected
3. Verify computeCycleAzBzForGroup produces correct results
4. Check if E_out/E_in match Jolt's values at each round

## Test Commands
```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

## Completed Milestones
- [x] Blake2b transcript implementation
- [x] Field serialization (Arkworks format)
- [x] UniSkip polynomial generation
- [x] Stage 1 remaining rounds sumcheck
- [x] R1CS constraint definitions
- [x] Split eq polynomial factorization
- [x] Lagrange kernel computation
- [x] Opening claims with MLE evaluation
- [x] Streaming round sum-of-products structure
- [x] Cycle rounds multiquadratic method
- [x] Transcript flow matching Jolt
- [x] All 656 Zolt tests pass
