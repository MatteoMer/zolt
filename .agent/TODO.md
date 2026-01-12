# Zolt-Jolt Compatibility TODO

## Current Progress

### Stages 1-3: PASS
### Stage 4: IN PROGRESS - Transcript Divergence Found

## Stage 4 Investigation (Updated 2026-01-12)

### VERIFIED CORRECT

1. **normalize_opening_point Works Correctly** - r_cycle is correctly reconstructed
2. **gamma matches between Zolt and Jolt** - Value: 250464584498748727615350532275610162451
3. **Round polynomial coefficients (c0, c2) match** - Verified byte-by-byte
4. **Instance 0 input_claim (RegistersReadWriteChecking) matches**

### ROOT CAUSE FOUND: Transcript Diverges Due to Input Claim Mismatch

**Symptom:**
- Batching coefficients differ between Zolt and Jolt
- Round 0 challenges differ despite same round polynomial coefficients
- output_claim (4025...) != expected_output_claim (12140...)

**Root Cause:**
The input claims for **Instance 1 (RamValEvaluation)** and **Instance 2 (ValFinalSumcheck)** differ between Zolt and Jolt.

**Jolt's Stage 4 Input Claims (from verifier debug):**
- Instance 0: 307483459531037776443095509592867762877490066210083905177366244070544605991
- Instance 1: 1063618430616024228610198021464576543949054712450992483279253898727921684198
- Instance 2: 21330192240049295511200146006695953640064028237054731883905956060590810557878

**How Input Claims are Computed:**
- `input_claim_val_eval = rwc_val_claim - val_init_eval`
- `input_claim_val_final = output_val_final_claim - output_val_init_claim`

If either component is computed differently, the transcript diverges immediately after appending input claims, causing all subsequent challenges and batching coefficients to differ.

### Stage 4 Transcript Protocol (verified correct order)

1. gamma = challenge_scalar()
2. (no advice appends for fibonacci)
3. append_scalar(input_claim_registers)    ← Instance 0 - MATCHES
4. append_scalar(input_claim_val_eval)     ← Instance 1 - MISMATCH?
5. append_scalar(input_claim_val_final)    ← Instance 2 - MISMATCH?
6. batch0 = challenge_vector[0]
7. batch1 = challenge_vector[1]
8. batch2 = challenge_vector[2]
9. For each round: append poly coeffs, derive challenge

### TODO: Next Steps

1. **Compare input_claim_val_eval components:**
   - Compare `rwc_val_claim` between Zolt and Jolt
   - Compare `val_init_eval` between Zolt and Jolt

2. **Compare input_claim_val_final components:**
   - Compare `output_val_final_claim` between Zolt and Jolt
   - Compare `output_val_init_claim` between Zolt and Jolt

3. **Trace r_address derivation:**
   - The val_init_eval depends on r_address from Stage 2
   - If r_address differs, val_init_eval will differ

### Files to Investigate

**Zolt:**
- `src/zkvm/proof_converter.zig` lines 1650-1660 (Stage 4 input claim computation)
- Stage 2 output: rwc_val_claim, output_val_final_claim, output_val_init_claim

**Jolt:**
- `jolt-core/src/zkvm/ram/val_evaluation.rs:160-166` (RamValEvaluation input_claim)
- `jolt-core/src/zkvm/ram/val_final.rs:135-141` (ValFinalSumcheck input_claim)

## Testing
```bash
bash scripts/build_verify.sh  # Output goes to logs/ directory
```
