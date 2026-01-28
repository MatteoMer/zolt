# Zolt-Jolt Compatibility: Stage 4 Final Claim Mismatch

## Status: In Progress (Session 70 - continued)

## Current Issue

The Stage 4 sumcheck produces a final claim that doesn't match the expected output:
- `regs_current_claim` = `{ 85, 93, 128, 139, ... }` (from sumcheck evolution)
- `expected_output` = `{ 165, 170, 243, 243, ... }` (eq * combined)
- These SHOULD be equal but they're not!

## Key Finding: Constraint Satisfied But Final Claim Wrong

The sumcheck constraint `p(0)+p(1)=claim` is satisfied at EVERY round:
- No "[STAGE4 SUMCHECK CONSTRAINT VIOLATION]" messages printed
- Phase 2/3 correctly use `from_evals_and_hint` pattern

But the polynomial being computed doesn't produce the expected final value.

## Analysis Done

1. **Formula Verified**: `combined = ra*val + wa*(val+inc)` matches Jolt
2. **Eq Polynomial**: `merged_eq[0]` after all bindings matches `eq_val_be` computed from challenges
3. **Phase Configuration**: Phase 1 (4 rounds) + Phase 2 (7 rounds) + Phase 3 (4 rounds) = 15 total
4. **Binding Operation**: Standard multilinear interpolation `new = lo*(1-c) + hi*c` is correct

## Remaining Investigation Areas

1. **Phase 1 Gruen Polynomial**: The `gruenPolyDeg3` function combines eq and body polynomials
   - Need to verify the cubic polynomial construction is correct
   - Particularly the recovery of the linear coefficient using the hint

2. **Phase 2/3 Eq Multiplication**: Check if `eval_0 = sum(eq[j] * combined)` includes eq correctly
   - Formula looks correct but could have indexing issues

3. **Variable Ordering**: The sumcheck binds variables in a specific order
   - Phase 1: First log_T/2 cycle vars (Gruen)
   - Phase 2: All LOG_K address vars
   - Phase 3: Remaining cycle vars
   - Expected output uses `normalize_opening_point` which reorders challenges

4. **Possible Issue**: The `expected_output` computation uses a different challenge ordering
   than what the sumcheck actually binds. Need to verify they produce the same eq*combined value.

## Files to Check

- `gruen_eq.zig:214` - `gruenPolyDeg3` function
- `stage4_gruen_prover.zig:561` - `phase1ComputeMessage`
- `stage4_gruen_prover.zig:764` - `phase2ComputeMessage`
- `stage4_gruen_prover.zig:853` - `phase3ComputeMessage`
- `proof_converter.zig:2269-2281` - `r_cycle_sumcheck_be` construction

## Next Steps

1. Add debug output comparing eq polynomial at each round
2. Verify `regs_current_claim` evolution matches expected at each round
3. Check if the issue is in Phase 1 (Gruen) or Phase 2/3 (dense eq)

## Session Summary

- Confirmed sumcheck constraint is satisfied at every round
- Identified that final claim doesn't match expected output
- Issue likely in polynomial coefficient computation or variable ordering
- Need to trace round-by-round to find where divergence occurs

SESSION_ENDING - saved progress to TODO.md for next session.
