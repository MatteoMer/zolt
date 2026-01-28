# Zolt-Jolt Compatibility: Status

## Status: Stage 2 Sumcheck Debug ⏳

## Session Summary (2026-01-28)

### Verified Working
1. **Proof Deserialization** ✅ - Jolt can deserialize Zolt proofs
2. **Stage 1 Verification** ✅ - Passes
3. **Stage 2 Transcript Sync** ✅ - Challenges match exactly
4. **Stage 2 Initial Claims** ✅ - All 5 instances have matching input claims
5. **Stage 2 Batching Coefficients** ✅ - Match exactly
6. **Stage 2 Round Polynomials** ✅ - c0, c2, c3 match for all rounds
7. **Stage 2 Final Output Claim** ✅ - `[50, 8d, 70, 43, ...]` matches

### The Bug: Expected Output Claim Mismatch

After 24 rounds of Stage 2 sumcheck:
- **Zolt/Jolt output_claim** (from sumcheck): `[50, 8d, 70, 43, ...]`
- **Jolt expected_claim** (from instance evaluations): `[38, d1, cc, 37, ...]`

The expected claim is computed as:
```
expected = sum(instance[i].output_claim(r_sumcheck) * batching_coeff[i])
```

This means the verifier is computing different polynomial evaluations than what the prover computed. The issue is in how Zolt's Stage 2 prover evaluates the instance polynomials.

### Next Steps

1. Debug which instance(s) have incorrect output claims
2. Compare Zolt vs Jolt evaluation at the final sumcheck point
3. Fix the polynomial evaluation to match Jolt's expectations

### Files to Check
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 2 prover
- Stage 2 instances:
  - ProductVirtualRemainder
  - RamRafEvaluation
  - RamReadWriteChecking
  - OutputSumcheck
  - InstructionLookupsClaimReduction

### Technical Details

Stage 2 batches 5 sumcheck instances:
- Instance 0: ProductVirtualRemainder (8 rounds)
- Instance 1: RamRafEvaluation (16 rounds)
- Instance 2: RamReadWriteChecking (24 rounds - max)
- Instance 3: OutputSumcheck (16 rounds)
- Instance 4: InstructionLookupsClaimReduction (8 rounds)

Each instance computes `output_claim(r) = eq(opening_point, r[offset:]) * polynomial_value`
