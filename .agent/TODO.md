# Zolt-Jolt Compatibility: Stage 2 Debug

## Status: Factor Evaluation Mismatch Investigation ⏳

## Session Summary (2026-01-28 evening)

### Verified Working
1. **Proof Deserialization** ✅ - Jolt can deserialize Zolt proofs
2. **Stage 1 Verification** ✅ - Passes
3. **Stage 2 Transcript Sync** ✅ - Challenges match exactly through all 24 rounds
4. **Stage 2 Initial Claims** ✅ - All 5 instance input claims match
5. **Stage 2 Batching Coefficients** ✅ - Match exactly
6. **Stage 2 Round Polynomials** ✅ - c0, c2, c3 match for all 24 rounds
7. **Stage 2 Sumcheck Output Claim** ✅ - `[50, 8d, 70, 43, ...]` matches

### KEY BUG IDENTIFIED: Factor Evaluations Don't Match

The factor polynomial evaluations stored in `opening_claims` by Zolt don't match what Jolt reads when computing `expected_output_claim`.

**Jolt reads from opening_claims** (ProductVirtualRemainder Instance 0):
- `l_inst (LeftInstructionInput): [fd, 52, a8, 83, 5d, 65, a5, 6f, ...]`
- `r_inst (RightInstructionInput): [21, 81, 2b, 04, 91, 40, 14, 5e, ...]`
- etc.

**Zolt stores** (cache_openings[0] LeftInstructionInput):
- First 8 BE bytes: `43 85 16 42 5f 95 17 fd` - **DIFFERENT!**

### Why Sumcheck Passes But Verification Fails

1. Zolt generates Stage 2 round polynomials
2. All 24 rounds satisfy `p(0) + p(1) = claim` (internally consistent)
3. Challenges are synchronized (transcript matches)
4. Final `output_claim` matches what verifier computes from sumcheck
5. **BUT**: Verifier then computes `expected_claim` using factor evals from `opening_claims`
6. Since factor evals are wrong, `expected_claim ≠ output_claim`

### Root Cause Hypothesis

The factor evaluations are computed as MLE evaluations:
```
factor[i] = Σ_t eq(r_cycle, t) * witness[t].factor[i]
```

Where `r_cycle = reverse(challenges[16..24])` (last 8 of 24 Stage 2 challenges)

Possible issues:
1. **r_cycle extraction** - wrong challenge slice or wrong reversal order
2. **eq polynomial** - wrong evaluation (though eq_sum=1 suggests it's OK)
3. **witness values** - extracting wrong fields from R1CS inputs
4. **MLE formula** - using different indexing than Jolt expects

### Next Debugging Steps

1. Print exact r_cycle from Jolt verifier and compare with Zolt
2. Print single witness value and trace through MLE computation
3. Check Jolt's factor evaluation formula in cache_openings
4. Verify Zolt extracts correct R1CS fields for each factor

### Technical Context

Stage 2 has 5 batched instances with different round counts:
- Instance 0: ProductVirtualRemainder (8 rounds, offset=16)
- Instance 1: RamRafEvaluation (16 rounds, offset=8)
- Instance 2: RamReadWriteChecking (24 rounds, offset=0) - MAX
- Instance 3: OutputSumcheck (16 rounds, offset=8)
- Instance 4: InstructionLookupsClaimReduction (8 rounds, offset=16)

For Instance 0, the factor claims should be MLE evaluations at normalized(challenges[16..24]).

### Files

- `/home/vivado/projects/jolt/jolt-core/src/zkvm/spartan/product.rs` - Jolt's expected_output_claim
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Zolt's factor computation
- `/home/vivado/projects/zolt/src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainder prover
