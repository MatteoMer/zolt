# Zolt-Jolt Compatibility - Status Update

## Session 15 Summary

### Current Status
- ✅ All 712 internal tests pass (`zig build test`)
- ✅ Stage 1 passes Jolt verification
- ❌ Stage 2 fails Jolt verification

### Stage 2 Root Cause Analysis

The Stage 2 batched sumcheck fails due to **multiple issues**:

#### Issue 1: r0 Mismatch
The Stage 2 r0 (UniSkip challenge) differs between Zolt and Jolt:
- Zolt r0: `8768758914789955585787902790032491769856779696899125603611137465800193155946`
- Jolt r0: `16176819525807790011525369806787798080841445107270164702191186390206256879646`

r0 comes from the opening_accumulator after UniSkip verification. The transcript must match for r0 to match.

#### Issue 2: Sumcheck Claim Violations
Errors starting at round 23:
```
[ZOLT CLAIM ERROR] round 23: s(0)+s(1) != old_claim!
[ZOLT CLAIM ERROR] round 24: s(0)+s(1) != old_claim!
[ZOLT CLAIM ERROR] round 25: s(0)+s(1) != old_claim!
```

This happens because instances 1, 2, 4 use zero polynomial fallback instead of proper provers.

#### Issue 3: Missing Sumcheck Provers
Stage 2 has 5 batched sumcheck instances:
- Instance 0: ProductVirtualRemainder ✅ (implemented)
- Instance 1: RamRafEvaluation ❌ (zero fallback)
- Instance 2: RamReadWriteChecking ❌ (zero fallback)
- Instance 3: OutputSumcheck ✅ (implemented)
- Instance 4: InstructionLookupsClaimReduction ❌ (zero fallback)

Input claims for instances 1, 2, 4 are NON-ZERO but their polynomials contribute zero.

### What Works
- ✅ fused_left matches Jolt: `3680814111042145831100417079225278919431426777627349458700618452903652360804`
- ✅ fused_right matches Jolt: `5628401284835057616148875782341094898402011560234054472864896388346845354264`
- ✅ tau_high matches Jolt: `1724079782492782403949918631195347939403999634829548103697761600182229454970`
- ✅ All 26 sumcheck challenges (rounds 16-25 for ProductVirtualRemainder) match
- ✅ Stage 1 complete verification

### Next Steps (Priority Order)

1. **Debug r0 Derivation**
   - Compare Stage 2 UniSkip first round polynomial between Zolt and Jolt
   - Check transcript state before/after UniSkip
   - Fix any coefficient serialization issues

2. **Implement Missing Provers** (after r0 is fixed)
   - RamRafEvaluation prover (16 rounds)
   - RamReadWriteChecking prover (26 rounds, 3-phase)
   - InstructionLookupsClaimReduction prover (10 rounds)

3. **Ensure Sumcheck Soundness**
   - s(0) + s(1) must equal old_claim at every round
   - All 5 instances must contribute correct polynomial evaluations

### Files to Investigate
- `/Users/matteo/projects/zolt/src/zkvm/proof_converter.zig` - Stage 2 proof generation
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/product.rs` - ProductVirtualUniSkip
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/ram/raf_evaluation.rs` - RafEvaluation
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/ram/read_write_checking.rs` - ReadWriteChecking
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs` - InstructionClaimReduction

## Previous Sessions

### Session 14
- Fixed Stage 2 UniSkip extended evaluations
- Verified fused_left/fused_right/split_eq match
- Instance 0 final claim matches

### Session 13
- Identified r0 mismatch as root cause
- Transcript states match but r0 differs

### Session 11-12
- Fixed Stage 1 output-sumcheck
- Stage 1 started passing

### Session 10
- Implemented Stage 1 streaming outer prover
