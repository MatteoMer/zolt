# Zolt-Jolt Compatibility: Stage 4 Sumcheck Issue

## Status: In Progress (Session 70)

## Current Issue

The Stage 4 sumcheck produces a final claim (`batched_claim`) that doesn't match the expected output:
- `batched_claim` = `{ 56, 208, 132, ... }` (from sumcheck evolution)
- `expected_output` = `{ 165, 170, 243, ... }` (eq * combined)
- `total_expected` = `{ 40, 66, 17, ... }` (coeff * expected)

These don't match, causing verification to fail.

## Key Findings (Session 70)

### 1. Sumcheck Constraint IS Satisfied
- Each round correctly maintains `p(0) + p(1) = previous_claim`
- No constraint violations detected
- The Phase 2/3 fix for `from_evals_and_hint` pattern is working

### 2. Final Claim Mismatch
```
[ZOLT STAGE4 FINAL BIND] expected (eq * combined) = { 165, 170, 243, ... }
[ZOLT STAGE4 FINAL DEBUG] regs_current_claim = { 85, 93, 128, ... }
```

The prover's final sumcheck claim doesn't equal `eq_scalar * combined`.

### 3. Polynomial Values Are Correct
- `eq_scalar` (merged_eq[0]) matches `eq_val_be` computed from challenges
- `combined = ra*val + wa*(val+inc)` formula matches Jolt's
- Individual claims (val_claim, ra_claims, etc.) are computed correctly

### 4. Root Cause Hypothesis: Variable Binding Order
The sumcheck binds variables in a specific order:
- Phase 1: First log_T/2 cycle variables
- Phase 2: LOG_K address variables
- Phase 3: Remaining cycle variables

Jolt's `normalize_opening_point` reorganizes challenges:
- `r_cycle = reversed(phase3_cycle) ++ reversed(phase1)`
- `r_address = reversed(phase3_address) ++ reversed(phase2)`

The mismatch may be in how `regs_current_claim` evolves vs how the expected output is computed.

## Files Modified This Session

- `/home/vivado/projects/zolt/src/zkvm/ram/val_evaluation.zig`: Added `lt_eval` to `getFinalOpenings`
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig`: Improved debug output for 3-instance comparison
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage4_gruen_prover.zig`: Added sumcheck constraint verification

## Next Steps

1. **Investigate variable binding order**:
   - Compare Zolt's polynomial binding with Jolt's MLE evaluation order
   - Ensure eq polynomial initialization matches Jolt's GruenSplitEqPolynomial

2. **Debug specific round values**:
   - Print eq * combined at each round to trace divergence
   - Compare with Jolt's prover output (if buildable)

3. **Consider alternative approaches**:
   - May need to restructure polynomial binding to match Jolt's exact variable order
   - Or compute expected output using the same binding order as sumcheck

## Session Summary

- Phase 2/3 sumcheck fix verified (constraint satisfied at every round)
- Root cause identified: final claim doesn't match expected output
- Most likely cause: variable binding order mismatch
- Need to align Zolt's binding order with Jolt's expected MLE evaluation order
