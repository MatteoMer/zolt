# Zolt-Jolt Compatibility: Status Update

## Status: STAGE 4 INPUT CLAIM MISMATCH IDENTIFIED

## Session 74 Progress (2026-01-28)

### Key Finding: ValEvaluation Input Claim Mismatch

The Stage 4 verification fails because the ValEvaluation prover's initial claim doesn't match `input_claim_val_eval`:

```
input_claim_val_eval (from accumulator) = { 164, 183, 91, 114, 236, 92, 48, 36, ... }
val_eval_prover initial_claim = { 178, 177, 67, 89, 63, 118, 102, 181, ... }
Match? false
```

### Root Cause Analysis

The ValEvaluation sumcheck proves:
```
Val(r) - Val_init(r_address) = Σ_{j} inc(j) * wa(r_address, j) * LT(j, r_cycle)
```

Where:
- LHS = `rwc_val_claim - init_eval_for_val_eval` = `input_claim_val_eval` (from Stage 2)
- RHS = `Σ inc(j) * wa(j) * lt(j)` = prover's `initial_claim` (computed locally)

These SHOULD be equal but they're not. This means:
1. The prover is using different r_address/r_cycle than Stage 2
2. OR the inc/wa/lt polynomials are incorrectly computed
3. OR there's a mismatch in how the opening point is normalized

### Debugging Steps Needed

1. **Verify r_address matches**: Check if the r_address passed to ValEvaluation prover matches Stage 2's r_address
2. **Verify r_cycle matches**: Same for r_cycle
3. **Verify start_address**: The inc/wa polynomials use start_address - ensure it's consistent
4. **Check synthetic termination write handling**: Fibonacci has only one RAM write (termination). Verify it's handled consistently.

### Key Code Locations

- `proof_converter.zig:1748-1805` - r_cycle_be/r_cycle_le construction from Stage 2 challenges
- `proof_converter.zig:1942-1949` - ValEvaluation prover initialization
- `val_evaluation.zig:423-522` - ValEvaluationProver init and initial claim computation
- `read_write_checking.zig:1210-1290` - RWC getOpeningClaims (rwc_val_claim computation)

### Native Verification Status

All 6 stages pass in native verification. The issue is specifically in cross-verification where:
- Stage 4 input claims are computed from Stage 2 opening accumulator
- These must match the prover's internal polynomial sums

### Next Steps

1. Add debug to compare r_address/r_cycle between Stage 2 RWC and ValEvaluation prover
2. Trace through the synthetic termination write to see if it affects both sides equally
3. Verify start_address is consistent (0x7fff8000 for both?)

SESSION_ENDING - Identified input claim mismatch as root cause. Need to verify r_address/r_cycle consistency.
