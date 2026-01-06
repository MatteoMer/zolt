# Zolt-Jolt Compatibility Notes

## Current Status (Session 65 - January 7, 2026)

### Summary

**Stage 1 PASSES, Stage 2 UniSkip PASSES, Stage 2 Sumcheck WIRED UP**

Stage 2 batched sumcheck is now wired up with:
- ProductVirtualRemainderProver for real product constraint proofs
- Zero claims for other 4 instances (RamRaf, RamRW, Output, InstructionClaim)

### What Was Done This Session

1. **Created ProductVirtualRemainderProver** (`src/zkvm/spartan/product_remainder.zig`)
   - Fuses 5 product constraints into left/right polynomials using Lagrange weights
   - Computes cubic round polynomials [c0, c2, c3]
   - Extracts 8 unique factor polynomial values per cycle
   - Matches Jolt's ProductVirtualRemainderProver structure

2. **Created BatchedSumcheckProver** (`src/zkvm/batched_sumcheck.zig`)
   - Infrastructure for combining 5 instances with different round counts
   - Handles batching protocol: scale claims, sample coefficients, combine polynomials
   - Proper transcript operations matching Jolt exactly

3. **Wired generateStage2BatchedSumcheckProof** in proof_converter.zig
   - Replaces generateZeroSumcheckProof call
   - Passes r0_stage2, uni_skip_claim, tau, cycle_witnesses
   - ProductVirtualRemainder uses real prover
   - Other 4 instances contribute zero (valid for simple programs)

### Stage 2 Transcript Flow

Verified to match Jolt exactly:
```
1. Append input_claims[0..5] to transcript
2. Sample batching_coeffs[0..5] using challenge_vector
3. For each round:
   a. Compute combined polynomial from all instances
   b. transcript.append_message("UniPoly_begin")
   c. transcript.append_scalar(c0, c2, c3)
   d. transcript.append_message("UniPoly_end")
   e. Sample round_challenge
4. Cache opening claims
```

### Test Status

- All 712+ tests pass
- Stage 2 batched sumcheck generates non-zero polynomials
- Transcript operations are in correct order

### Known Issues / Next Steps

1. **ProductVirtualRemainder polynomial generation** - The current implementation may need debugging
   - Need to verify the compressed format is correct
   - Need to verify left/right fusion is matching Jolt

2. **Opening claims** - Currently set to zero for instances 2-5
   - This is valid for programs without RAM/lookups
   - Need to compute real claims for full programs

3. **Testing with Jolt verifier** - Need end-to-end test
   - Generate proof with `zolt prove`
   - Verify with Jolt's verifier

### Files Modified

| File | Changes |
|------|---------|
| `src/zkvm/spartan/product_remainder.zig` | NEW - ProductVirtualRemainderProver |
| `src/zkvm/batched_sumcheck.zig` | NEW - Batched sumcheck infrastructure |
| `src/zkvm/spartan/mod.zig` | Added exports |
| `src/zkvm/mod.zig` | Added batched_sumcheck |
| `src/zkvm/proof_converter.zig` | Added generateStage2BatchedSumcheckProof, wired up |

---

## Previous Sessions

See earlier notes for Stage 1 fixes including:
- Montgomery form serialization
- Batching coefficient handling
- UniSkip polynomial construction
- Opening claims witness matching
