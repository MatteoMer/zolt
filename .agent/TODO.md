# Zolt-Jolt Compatibility - Stage 2 Sumcheck Investigation

## Current Status: Stage 2 Verification Fails

### What Works
- [x] Stage 1 (SpartanOuter) verification passes
- [x] Tau sampling matches Jolt
- [x] Polynomial coefficients (c0, c2, c3) match Jolt at each round
- [x] Challenges match between Zolt and Jolt
- [x] Soundness constraint `s(0)+s(1) = current_claim` holds for rounds 0-20 (but fails 21-25)

### Current Problem
- Stage 2 sumcheck verification fails with mismatched output_claim
- `output_claim = 11948928263400051798463901278432764058724926493141863520413443728531572654384`
- `expected_output_claim = 14998460073388315545242452814285195471990034347995786920854240537701021643062`

### Root Cause Analysis

1. **Polynomial Evaluation Inconsistency**:
   - Zolt computes polynomial evaluations [s0, s1, s2, s3] from instance provers
   - These evaluations satisfy `s0 + s1 = current_claim` (soundness) for early rounds
   - But when Jolt verifies, it uses `eval_from_hint` formula:
     - `c1 = hint - 2*c0 - c2 - c3`
     - `P(r) = c0 + c1*r + c2*r^2 + c3*r^3`
   - This gives a DIFFERENT result than Lagrange interpolation from [s0, s1, s2, s3]
   - The divergence starts at round 16/17 when Instance 0 (ProductVirtual) becomes active

2. **Why Coefficients Match But Evaluations Differ**:
   - Zolt's `evalsToCompressed` correctly computes c0, c2, c3 from interpolation
   - But Zolt's s1 is wrong (different from what Jolt's c1 recovery expects)
   - The s2, s3 values compensate to produce correct c0, c2, c3
   - However, evaluation at challenge r diverges because Lagrange interpolation
     uses the wrong s1, s2, s3 values

3. **Specific Finding at Round 16**:
   - Coefficients match: c0, c2, c3 all identical
   - Challenge matches
   - Zolt's `next_claim` from Lagrange: 18890382963589086332434320318952386057822041579309690361978685610844422969181
   - Jolt's `next_claim` from eval_from_hint: 18434812717882284909799595479264485648355085528645025427055036769841404564943
   - These differ even though the compressed coefficients match!

4. **Investigation Needed**:
   - [ ] Trace how ProductVirtualRemainderProver computes [s0, s1, s2, s3]
   - [ ] Compare with Jolt's ProductVirtualRemainder polynomial computation
   - [ ] Verify that the left_poly, right_poly, and eq_poly are correct
   - [ ] The polynomial degree is cubic (degree-3) - this is correct

### Key Insight
The fundamental issue is that Zolt computes polynomial evaluations [s0, s1, s2, s3]
that produce CORRECT compressed coefficients [c0, c2, c3] but INCORRECT Lagrange evaluation.

This happens because:
1. interpolateDegree3 uses all 4 points (s0, s1, s2, s3) to recover c0, c1, c2, c3
2. evalsToCompressed returns [c0, c2, c3] (dropping c1)
3. Jolt's verifier recovers c1 from hint as `c1 = hint - 2*c0 - c2 - c3`
4. If s1 is wrong but s2, s3 compensate in interpolation, c1 matches
5. BUT evaluating P(r) via Lagrange uses wrong s1, s2, s3

### Files to Investigate
- `src/zkvm/proof_converter.zig` - generateStage2BatchedSumcheckProof
- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainderProver
- Jolt's `jolt-core/src/zkvm/spartan/inner.rs` - ProductVirtualRemainder equivalent

### Fix Options
1. Make Zolt compute s1, s2, s3 that are consistent with Jolt's expected values
2. Or change Zolt to use `eval_from_hint` for claim tracking (but this causes soundness violations when combined_evals are wrong)

## Verification Commands

```bash
# Build and test Zolt
zig build test --summary all

# Generate Jolt-compatible proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Verify with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```

## Code Locations
- ProductVirtual prover: `/Users/matteo/projects/zolt/src/zkvm/spartan/product_remainder.zig`
- Jolt ProductVirtual: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/product.rs`
- Split Eq polynomial: `/Users/matteo/projects/zolt/src/poly/split_eq.zig`
- Proof converter: `/Users/matteo/projects/zolt/src/zkvm/proof_converter.zig`
