# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck Verification (Session 33)

### Issue Summary
The Stage 1 sumcheck output_claim doesn't match expected_output_claim.

```
output_claim:          18149181199645709635565994144274301613989920934825717026812937381996718340431
expected_output_claim:  9784440804643023978376654613918487285551699375196948804144755605390806131527
```

### Key Investigation Findings

1. **All individual components appear correct:**
   - ✅ Blake2b transcript (byte-for-byte match)
   - ✅ UniSkip polynomial coefficients
   - ✅ Compressed poly format [c0, c2, c3]
   - ✅ computeCubicRoundPoly matches Jolt's gruen_poly_deg_3
   - ✅ Constraint structure matches Jolt exactly (19 constraints, correct groups)
   - ✅ R1CSInputIndex ordering matches Jolt's ALL_R1CS_INPUTS
   - ✅ E_out/E_in factorization structure matches Jolt
   - ✅ All 656 Zolt tests pass

2. **Formula Analysis:**
   - expected_output_claim = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
   - inner_sum_prod = Az(rx_constr) * Bz(rx_constr) where rx_constr = [r_stream, r0]
   - The prover should produce output_claim = L(tau_high,r0) * eq(tau_low, r) * Az(r) * Bz(r)

3. **Potential Issues to Investigate:**
   - The streaming round computes t_zero and t_infinity for the first remaining round
   - These feed into computeCubicRoundPoly (gruen_poly_deg_3)
   - The index structure for iterating over cycles might be off

### Next Steps
1. Add detailed debug output comparing t_zero/t_infinity between Zolt and Jolt
2. Create a minimal test case with known correct values
3. Trace the exact index structure in streaming round vs Jolt's fused_materialise_polynomials_round_zero
4. Compare eq polynomial values at specific cycle indices

### Verified Correct
- [x] Blake2b transcript implementation
- [x] Field serialization (Arkworks format)
- [x] UniSkip polynomial generation
- [x] R1CS constraint definitions (19 constraints, 2 groups)
- [x] Split eq polynomial factorization (E_out/E_in tables)
- [x] Lagrange kernel L(tau_high, r0)
- [x] MLE evaluation for opening claims
- [x] Gruen cubic polynomial formula
- [x] All 656 Zolt tests pass

### Analysis Notes
The sumcheck verification passes all rounds (p(0) + p(1) = claim), meaning:
- The sumcheck polynomial is internally consistent
- But the final claim doesn't match what the verifier computes from opening claims

This suggests either:
1. The R1CS input MLE evaluations don't match
2. The t_zero/t_infinity computation differs from Jolt
3. Some subtle index/ordering difference in how we iterate over cycles

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
