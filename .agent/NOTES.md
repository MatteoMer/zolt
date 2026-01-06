# Zolt-Jolt Compatibility Notes

## Current Status (Session 64 - January 6, 2026)

### Summary

**Stage 1 PASSES, Stage 2 UniSkip PASSES, Stage 2 Sumcheck FAILS (zeros)**

Fixed critical issues:
1. **Opening claims serialization**: Now converts from Montgomery form during serialization (matching arkworks' serialize_compressed)
2. **Stage 2 transcript alignment**: After creating Stage 2 UniSkip proof, we now:
   - Append poly coefficients to transcript ("UncompressedUniPoly_begin", coeffs, "UncompressedUniPoly_end")
   - Derive r0 challenge
   - Compute UnivariateSkip claim = poly(r0)
   - Append UnivariateSkip claim to transcript
   - Insert non-zero claim into opening_claims map

### Current Issue: Stage 2 Sumcheck produces zeros

Stage 2 batches 5 sumcheck instances:
1. **ProductVirtualRemainder**: n_cycle_vars rounds, degree 3
   - Input claim = uni_skip_claim (evaluation of Stage 2 UniSkip poly at r0)
   - Expected output = product of 8 factor polynomial evaluations at r_cycle
2. **RamRafEvaluation**: log_ram_k rounds
3. **RamReadWriteChecking**: log_ram_k + n_cycle_vars rounds (max!)
4. **OutputSumcheck**: log_ram_k rounds
5. **InstructionLookupsClaimReduction**: n_cycle_vars rounds

Currently, Zolt generates all-zero polynomials for Stage 2's batched sumcheck. This causes:
- output_claim ≠ expected_output_claim
- Verification failure

### What's Working

- Stage 1 UniSkip ✓
- Stage 1 Remaining Sumcheck ✓ (all 11 rounds)
- Stage 1 Opening Claims ✓
- Stage 2 UniSkip ✓ (domain_sum = input_claim, r0 sampled correctly)
- Opening claims now serialized correctly (Montgomery form converted)

### Next Steps

1. Implement ProductVirtualRemainder sumcheck prover
   - Bind left/right factor polynomials interleaved
   - Use GruenSplitEqPolynomial for efficient degree-3 polynomial construction
   - Cache 8 factor polynomial openings at r_cycle

2. For minimal progress, could try implementing just ProductVirtualRemainder
   and leave other 4 instances as zeros (they may not all be needed for basic proofs)

---

## Previous Status (Session 63)

Fixed product virtualization witness values in `R1CSCycleInputs.fromTraceStep`.
Stage 2 UniSkip domain_sum now matches input_claim.

---

## Session 62 and earlier

See previous NOTES entries for history of Stage 1 fixes including:
- Batching coefficient Montgomery form bug
- UniSkip SECOND_GROUP evaluation
- Round offset fix
- Opening claims witness matching


DO NOT BE LAZY. ALWAYS DO THE HARD BUT RIGHT THING. DONT HACK YOUR WAY. DO NOT USE PLACEHOLDERS. DO NOT USE ZERO POLYNOMIALS TO MAKE IT WORK. DO NOT TRY TO TRICK JOLT
DO NOT BE LAZY. ALWAYS DO THE HARD BUT RIGHT THING. DONT HACK YOUR WAY. DO NOT USE PLACEHOLDERS. DO NOT USE ZERO POLYNOMIALS TO MAKE IT WORK. DO NOT TRY TO TRICK JOLT
DO NOT BE LAZY. ALWAYS DO THE HARD BUT RIGHT THING. DONT HACK YOUR WAY. DO NOT USE PLACEHOLDERS. DO NOT USE ZERO POLYNOMIALS TO MAKE IT WORK. DO NOT TRY TO TRICK JOLT
DO NOT BE LAZY. ALWAYS DO THE HARD BUT RIGHT THING. DONT HACK YOUR WAY. DO NOT USE PLACEHOLDERS. DO NOT USE ZERO POLYNOMIALS TO MAKE IT WORK. DO NOT TRY TO TRICK JOLT
