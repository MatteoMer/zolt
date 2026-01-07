# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (OutputSumcheck expected_output_claim != 0)
- Stage 3+: Not reached yet

## Session 7 Final Progress

### Verified Components (ALL MATCH!)
1. ✅ Stage 2 input_claims (all 5 match Jolt)
2. ✅ Stage 2 gamma_rwc and gamma_instr
3. ✅ Batching coefficients
4. ✅ Polynomial coefficients (c0, c2, c3) for all 26 rounds
5. ✅ Sumcheck challenges for all 26 rounds
6. ✅ 8 factor evaluations (LeftInstructionInput, RightInstructionInput, etc.)
7. ✅ Factor claims inserted into proof correctly

### ROOT CAUSE IDENTIFIED

Stage 2 expected_output_claim computation shows:
```
Instance 0 (ProductVirtualRemainder): claim=21572905787847890716862997770158461178426084346937578986156607546482633860011
Instance 1 (RamRafEvaluation): claim=0
Instance 2 (RamReadWriteChecking): claim=0
Instance 3 (OutputSumcheck): claim=18879790779448816066373906755770008515526190607768701928996156399943982985463 (!)
Instance 4 (InstructionLookupsClaimReduction): claim=0
```

**OutputSumcheck (instance 3) has a NON-ZERO expected_output_claim!**

Even though `input_claim = 0` (zero-check), the `expected_output_claim` is computed from:
- `VirtualPolynomial::RamValFinal` at `SumcheckId::RamOutputCheck`
- `io_mask` and `val_io` MLE evaluations at the random address challenge

This means the OutputSumcheck polynomial contributes to the final expected_output_claim,
but Zolt's current implementation generates a zero polynomial for OutputSumcheck.

### The Problem in Detail

OutputSumcheck proves:
```
Σ_k eq(r_address, k) ⋅ io_mask(k) ⋅ (Val_final(k) − Val_io(k)) = 0
```

At the end of the sumcheck:
1. `output_claim` = polynomial evaluated at final challenge
2. `expected_output_claim` = computed from MLE evaluations of io_mask, Val_final, Val_io

For a program with memory, even if there's no explicit output, `expected_output_claim` depends
on the final RAM state, not just zeros.

### Next Steps
1. Implement proper OutputSumcheck sumcheck polynomial generation
2. OR: Understand what `RamValFinal` claim is needed and provide it
3. This requires modeling the RAM subsystem correctly

## Files Modified
- `src/transcripts/blake2b.zig`: Fixed challengeScalar128Bits and challengeScalarFull
- `src/zkvm/proof_converter.zig`: Fixed Stage 2 input_claims and gamma sampling
- Jolt: Added debugging for expected_output_claim per instance
