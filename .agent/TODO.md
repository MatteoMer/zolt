# Zolt-Jolt Compatibility - Status Update

## Current Status: Session 14

### Stage 2 Cross-Verification - IN PROGRESS

The UniSkip extended evaluations are fixed, but the ProductVirtualRemainder sumcheck is still producing the wrong output claim.

**Analysis from Jolt cross-verification:**
- `output_claim = 18399905727734613027016857362443321745605316746004735455047505419992328320300`
- `expected_output_claim = 17135332148321379181445270571371034736283030019078327201597244975562090816163`
- Difference = ~1.26e75 (significant mismatch)

The batched sumcheck has 5 instances:
- Instance 0 (ProductVirtualRemainder): expected contribution = 17135332...
- Instances 1-4: contribution = 0 (claims are zero)

So the ProductVirtualRemainder prover is producing an incorrect final claim.

**Key Jolt formulas:**
- `expected_output_claim = L(τ_high, r0) · Eq(τ_low, r_tail^rev) · fused_left · fused_right`
- τ is [r_cycle (BIG_ENDIAN), τ_high]
- r_tail is the Stage 2 sumcheck challenges (reversed before use)

**Investigation progress:**
1. ✅ fused_left and fused_right values MATCH between Zolt and Jolt
2. ✅ L*Eq (split_eq.current_scalar) MATCHES
3. ✅ ProductVirtualRemainder instance 0 final claim MATCHES: 17831747...
4. ✅ Sumcheck challenges (r_tail) MATCH between Zolt and Jolt
5. ❌ Batched sumcheck output differs by ~1.26e75

**Root cause identified:**
The difference comes from instances 1, 2, 4 which should contribute zero to the sum:
- In Jolt: instances 1, 2, 4 have expected_output_claim = 0
- In Zolt: these instances have non-zero input_claims from Stage 1

The issue is that Stage 2's non-ProductVirtualRemainder instances (RamRafEvaluation, RamReadWriteChecking, InstructionLookupsClaimReduction) are contributing non-zero values when they should contribute zero.

**Key insight from Jolt analysis:**
- Instances 1, 2, 4 DO have non-zero input claims from Stage 1
- But their expected_output_claim = 0 because the sumcheck polynomial is designed to reduce to zero
- These instances have SPECIFIC polynomial structures that must be sumchecked

**The problem:**
Zolt doesn't implement real provers for instances 1, 2, 4. It's falling through to a "constant polynomial" fallback that doesn't correctly reduce the input claims to zero output claims.

**Fix options:**
1. Implement proper provers for RamRafEvaluation, RamReadWriteChecking, InstructionLookupsClaimReduction
2. Or understand why these should evaluate to zero and handle specially

**Next steps:**
1. Implement RafEvaluationProver for instance 1
2. Implement RamReadWriteCheckingProver for instance 2
3. Implement InstructionClaimReductionProver for instance 4

### Stage 2 UniSkip Extended Evaluations - FIXED!

**Key Fix**: The `createUniSkipProofStage2WithClaims` function was using zeros for extended evaluations. Now it computes the actual fused products at extended points {-3, 3, -4, 4} from the trace data.

### What Was Done
1. Added precomputed Lagrange coefficients for Product Virtual extended targets
2. Added `computeProductVirtualExtendedEvals` function to compute extended evals from cycle witnesses
3. Added `extractProductFactors` helper to extract the 8 product factors from R1CS witnesses
4. Updated `createUniSkipProofStage2WithClaims` to use actual extended evaluations

### Test Results
- All 712 tests pass
- example-pipeline passes with all stages verified

### Previous Status: Session 13

### Stage 2 Failure - MULTIPLE ROOT CAUSES IDENTIFIED!

- **Stage 1: PASSING ✅** - Sumcheck output_claim matches expected
- **Stage 2: FAILING ❌** - output_claim != expected_output_claim

```
output_claim:          15813746968267243297450630513407769417288591023625754132394390395019523654383
expected_output_claim: 21370344117342657988577911810586668133317596586881852281711504041258248730449
```

### ROOT CAUSE 1: Stage 2 UniSkip Extended Evaluations

**CRITICAL BUG FOUND**: `createUniSkipProofStage2WithClaims` uses zeros for extended evaluations:
```zig
const extended_evals: [DEGREE]F = [_]F{F.zero()} ** DEGREE;
```

The extended evaluations at points {-4, -3, 3, 4} should be the actual fused products computed from the trace. Using zeros produces an incorrect UniSkip polynomial which causes:
1. Wrong polynomial coefficients appended to transcript
2. Different transcript state
3. Different r0 challenge derived
4. Mismatch between Zolt's prover r0 and Jolt's verifier r0

**Evidence**:
- Transcript states match at rounds 54 and 176
- But r0 values differ:
  - Zolt r0: `5629772851639812945906736172593031815056148939881883788449064297659372967906`
  - Jolt r0: `16176819525807790011525369806787798080841445107270164702191186390206256879646`

**Fix**: Need trace access to compute extended evaluations for ProductVirtual UniSkip.

### ROOT CAUSE 2: Constant Polynomials for Active Instances

For instances 1, 2, 4 (RamRafEvaluation, RamReadWriteChecking, InstructionLookupsClaimReduction):
- Zolt generates constant polynomials instead of actual sumcheck round polynomials
- These instances have non-zero input claims, so they need actual provers
- Expected output for these is 0 (since ra=0, val=0), but constant polynomials don't reduce to 0

### Key Findings This Session

1. **Transcript states match up to round 176** - the divergence happens during Stage 2 UniSkip
2. **fused_left and fused_right values match** - the witness evaluations are correct
3. **The r0 is derived from UniSkip polynomial** - not stored in opening claims
4. **Opening claims only store values, not points** - but Jolt needs opening points too

### What Works
1. ✅ Stage 1 passes completely
2. ✅ Initial batched claim matches
3. ✅ All 26 round polynomial coefficients match (c0, c2, c3)
4. ✅ All 26 round challenges match
5. ✅ fused_left and fused_right factor claims match

### What Needs Fixing (in priority order)
1. **Stage 2 UniSkip extended evaluations** - need trace access
2. **Constant polynomial approach** - need actual provers for instances 1, 2, 4

### Fix Strategy

**Step 1: Fix UniSkip Extended Evaluations**
- Add trace access to proof converter
- Compute actual extended evaluations for ProductVirtual UniSkip
- This should make r0 match

**Step 2: Fix Instance Provers (after r0 is fixed)**
- Implement actual provers for RAF, RWC, and Instruction claim reduction
- Or verify that these instances should have zero input claims for fibonacci

## Previous Sessions

### Session 12
- Found that polynomial coefficients and challenges match
- Identified that s(0)+s(1) != claim at round 0

### Session 11
- Fixed output-sumcheck r_address_prime reversal
- Stage 1 started passing

### Session 10
- Implemented Stage 1 streaming outer prover
