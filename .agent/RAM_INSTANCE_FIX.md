# RAM Instance Issue - Investigation Update

## Date: 2026-01-25

## Problem Summary

Cross-verification test shows Stage 4 FAIL with:
```
output_claim:          4025718365397880246610377225086562173672770992931618085272964253447434290014
expected_output_claim: 12140057478814186156378889252409437120722846392760694384171609527722202919821
```

The `output_claim` matches Stage 1's final claim, indicating a potential issue with how Stage 4's sumcheck is being processed by the verifier.

## Investigation Results

### ✅ Zolt IS Generating Stage 4 Correctly

Contrary to initial suspicion, the proof converter's Stage 4 code **IS executing properly**:

1. **Execution confirmed**: Debug logs show all 15 rounds being generated:
   - `[ZOLT STAGE4] Round 0` through `[ZOLT STAGE4] Round 14`
   - `[ZOLT STAGE4] Final batched_claim` is computed
   - All opening claims are generated

2. **Trace data IS passed**: Both `execution_trace` and `memory_trace` are correctly passed via `ProofConverterConfig` in all three code paths (lines 687, 962, 1176 in mod.zig)

3. **Batching IS correct**: The batched claim combines:
   - Register RW prover (non-zero contribution)
   - RAM val_eval (zero for Fibonacci - no RAM)
   - RAM val_final (zero for Fibonacci - no RAM)

### Evidence

1. **Missing debug output**: The log should show `[PROOF_CONV] ===== STARTING STAGE 4 REGISTER CHECKING =====` but it doesn't appear.

2. **Zero proof fallback**: The code at `proof_converter.zig:1590-1619` has guards:
   ```zig
   const trace = config.execution_trace orelse {
       std.debug.print("[STAGE4] No execution trace, using zero proof\n", .{});
       try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, ...);
       break :stage4_block;
   };
   const memory_trace = config.memory_trace orelse {
       std.debug.print("[STAGE4] No memory trace, using zero proof\n", .{});
       try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, ...);
       break :stage4_block;
   };
   ```

3. **Stage 4 prover output shows zeros**: The RAM val evaluation prover shows all-zero polynomials, which is the separate RAM prover, not the proof_converter's batched Stage 4.

## The Fix

The proof converter needs to receive the execution trace and memory trace from the emulator. These should be passed via `ProofConverterConfig` when calling `converter.convert()`.

### Required Changes

In `src/zkvm/mod.zig`, where the converter is called (around lines 520, 791, 1017), we need to:

1. Pass `execution_trace: &emulator.trace` to the `ProofConverterConfig`
2. Pass `memory_trace: &emulator.ram.trace` to the `ProofConverterConfig`

This will allow the proof_converter to:
- Generate the proper RegistersReadWriteChecking sumcheck (15 rounds for Fibonacci)
- Include batched RAM val_eval and val_final instances (both zero-contributing for no-RAM programs)
- Produce the correct batched Stage 4 claim that matches Jolt's expectations

## Implementation Plan

1. Locate all calls to `converter.convert()` or similar methods
2. Modify the `ProofConverterConfig` initialization to include:
   - `execution_trace = &emulator.trace`
   - `memory_trace = &emulator.ram.trace`
3. Verify that the Stage 4 debug output appears in logs
4. Run cross-verification test to confirm fix

## Expected Outcome

After the fix:
- Stage 4's batched sumcheck will combine all 3 instances properly
- For Fibonacci (no RAM): contribution = register_claim * batch0 + 0 * batch1 + 0 * batch2
- The output_claim will match expected_output_claim
- Cross-verification will PASS all stages
