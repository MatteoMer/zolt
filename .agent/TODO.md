# Zolt-Jolt Compatibility Implementation

## Status: Session 40 - Stage 5 Sumcheck Verification Debugging

## Major Progress!

**Stage 4 passes!** Stage 5 (InstructionReadRaf) fails with sumcheck verification error.

## Current Failure: Stage 5 (InstructionReadRaf)

### Key Findings This Session

**Confirmed Matching Values:**
1. ✓ Initial batched claim
2. ✓ Batching coefficients (batch0, batch1, batch2)
3. ✓ All 136 sumcheck challenges
4. ✓ Round polynomial coefficients (verified rounds 0-3, 128-129)
5. ✓ Intermediate claims after each round (output_claim matches at round 135)
6. ✓ Instance 0 opening claims (inc_claim, wa_claim)
7. ✓ Instance 1 opening claims (ram_ra_claim)
8. ✓ Instance 2 opening claims (ra_chunks, table_flags, raf_flag)
9. ✓ Instance 2 components (eq_eval_r_reduction, ra_product, val_claim, raf_claim)
10. ✓ All individual instance expected_output_claim values

**The Mystery:**
- The sumcheck polynomial evaluations are ALL CORRECT
- All opening claims match between Zolt and Jolt
- BUT: output_claim ≠ expected_claim

**Output claim (what sumcheck produces):**
`[af, 51, 7b, 30, ff, 29, 91, 26, ...]`

**Expected claim (sum of instance expected_outputs * coeffs):**
`[f0, c1, c7, e7, 7e, fd, c3, 3b, ...]`

**Individual instance claim*coeff contributions (from Jolt):**
- Instance 0: `[36, 6f, 00, 34, 01, 26, 50, 93, ...]`
- Instance 1: `[48, d9, da, ee, a3, 74, 88, ae, ...]`
- Instance 2: `[73, 79, ec, b4, 6d, 58, cd, 3d, ...]`

The sum should equal expected_claim. This is verified by Jolt.

### Hypothesis

The sumcheck output_claim equals batch0*poly0(r0) + batch1*poly1(r1) + batch2*poly2(r2).

The expected_claim equals batch0*exp0 + batch1*exp1 + batch2*exp2.

For these to match, we need poly_i(r_i) = exp_i for each instance.

Since all opening claims match AND all expected_output formulas seem correct, there might be:
1. A subtle bug in the polynomial evaluation during cycle rounds
2. A mismatch in how the polynomial is defined vs what the verifier expects
3. A byte order issue in the sum computation

### Next Steps

1. **Trace polynomial evaluation more carefully:**
   - Verify that poly_i(r_i) actually equals exp_i for each instance
   - Print these values side by side at the end of the sumcheck

2. **Verify the batched sum:**
   - Check if batch0*exp0 + batch1*exp1 + batch2*exp2 = expected_claim
   - Check if batch0*poly0 + batch1*poly1 + batch2*poly2 = output_claim

3. **Check for off-by-one or endianness issues:**
   - The challenges might be indexed differently
   - The opening points might have different orderings

### Working Commands

```bash
# Build optimized (~13 sec proof generation)
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Session Summary

- Verified all individual components match between Zolt and Jolt
- The sumcheck polynomial coefficients and evaluations are correct
- The opening claims are correct
- The expected_output_claim formulas give correct results
- BUT the final sums don't match
- Need to trace the actual polynomial values vs expected values at the evaluation point

SESSION_ENDING - Deep investigation of Stage 5. All individual components match but the final sums don't. Need to trace polynomial evaluation vs expected output for each instance.
