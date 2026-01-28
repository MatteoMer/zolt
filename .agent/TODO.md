# Zolt-Jolt Compatibility: Status Update

## Status: DEBUGGING STAGE 4 BATCHED SUMCHECK MISMATCH

## Summary

Zolt can now:
1. Generate proofs for RISC-V programs (`./zig-out/bin/zolt prove`)
2. Verify proofs internally (`./zig-out/bin/zolt verify`) - ALL 6 STAGES PASS
3. Export proofs in Jolt-compatible format (`--jolt-format`)
4. Export preprocessing for Jolt verifier (`--export-preprocessing`)
5. Pass all 714 unit tests ✓
6. **Proof successfully deserializes in Jolt** ✓
7. **Preprocessing successfully deserializes in Jolt** ✓

## Cross-Verification Status

Verification fails at Stage 4: `batched_claim ≠ total_expected`

## Current Issue Analysis (Session 71)

### Key Finding: Instance 0 (RegistersRWC) is CORRECT

The RegistersRWC instance computes correctly:
- `regs_current_claim (poly_0 final)` = `expected_output (eq * combined)` ✓
- The eq polynomial (`merged_eq[0]`) matches `eq_val_be` computed via MLE ✓

### The Problem: Batched Claim Mismatch

After all 15 rounds of Stage 4:
- `batched_claim` = `{ 13, 174, 120, 9, 233, 120, 62, 18, ... }` (sumcheck output)
- `total_expected` = `{ 18, 61, 142, 143, 28, 54, 66, 104, ... }` (computed from openings)

These don't match because the weighted sum of instance outputs doesn't equal the evolved batched_claim.

### Stage 4 Batched Sumcheck Structure

The batched sumcheck has 3 instances with different round counts:
- Instance 0 (RegistersRWC): 15 rounds, offset=0, uses challenges[0..15]
- Instance 1 (ValEvaluation): 8 rounds, offset=7, uses challenges[7..15]
- Instance 2 (ValFinal): 8 rounds, offset=7, uses challenges[7..15]

### Root Cause Hypothesis

The expected output computation may not be using the correct challenge slice for Instances 1 and 2.

In Jolt's verifier:
- Instance 1's expected output uses `r_sumcheck[7..15]` (the LAST 8 challenges)
- Instance 2's expected output uses `r_sumcheck[7..15]` (same slice)

In Zolt's proof_converter, need to verify:
- `val_eval_openings` are computed at the correct bound point
- `val_final_openings` are computed at the correct bound point
- The expected output formula uses the correct LT/EQ evaluations

### Verification Check Details

```
batched_claim = { 13, 174, 120, 9, ... }  (sumcheck output after 15 rounds)
Instance 0: expected={ eq*combined }, coeff={...}, weighted={...}
Instance 1: expected={ inc*wa*lt }, coeff={...}, weighted={...}
Instance 2: expected={ inc*wa }, coeff={...}, weighted={...}
total_expected = weighted_0 + weighted_1 + weighted_2 = { 18, 61, 142, 143, ... }
Do they match? false
```

## Next Steps

1. **Verify Instance 1 and 2 binding order**:
   - Ensure `val_eval_prover` and `val_final_prover` bind challenges in the correct order
   - The first variable should bind to challenge[7], not challenge[0]

2. **Verify LT polynomial evaluation**:
   - `lt_eval` should be `LT(r_sumcheck[7..15], r_cycle_from_Stage2)`
   - Check if the LT polynomial is being evaluated at the correct point

3. **Debug the scaling/deferred contribution**:
   - Rounds 0-6: Instances 1&2 contribute `2^(remaining-8) * input_claim * coeff` as constant
   - Rounds 7-14: Instances 1&2 contribute actual polynomial evaluations
   - Verify the transition is happening correctly

## Files Generated

- `/tmp/fib_proof.bin` - Zolt native format proof
- `/tmp/zolt_proof_dory.bin` - Jolt-compatible format proof
- `/tmp/zolt_preprocessing.bin` - Preprocessing for Jolt verifier

## Test Commands

```bash
# Run full proof generation
./zig-out/bin/zolt prove examples/fibonacci.elf -o /tmp/fib_proof.bin \
    --jolt-format /tmp/zolt_proof_dory.bin \
    --export-preprocessing /tmp/zolt_preprocessing.bin 2>&1 | \
    grep -E "VERIFY CHECK|Do they match"
```
