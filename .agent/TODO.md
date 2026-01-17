# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ✅ PASS | RWC - removed synthetic termination write |
| 3 | ✅ PASS | RegistersClaimReduction |
| 4 | ❌ FAIL | Sumcheck output_claim != expected_output_claim |
| 5-7 | ⏳ Blocked | Waiting for Stage 4 |

## Session 40 Progress (2026-01-17)

### Completed
1. ✅ Fixed Stage 2 RWC mismatch
   - Root cause: Zolt was adding synthetic termination write to memory trace
   - Jolt only sets termination bit in val_final, NOT in execution trace
   - Fix: Removed recordTerminationWrite() calls from tracer
   - Commit: 5cec222

### Stage 4 Investigation Summary

**Verified ALL of these match between Zolt and Jolt:**
- Transcript state at all checkpoints ✓
- Challenge bytes and reversed bytes ✓
- Input claims for all 3 instances ✓
- Batching coefficients ✓
- Polynomial coefficients in proof file ✓

**Key discovery:**
The challenge is stored as F{ .limbs = .{ 0, 0, low, high } } which is Jolt's
MontU128Challenge format. This works for Stage 1 but NOT for Stage 4.

When I tried converting to proper Montgomery form F{ .limbs = .{ low, high, 0, 0 } }.toMontgomery(),
Stage 1 broke while Stage 4 was not fixed.

**Hypothesis:**
Stage 4's eq polynomial computation or binding may have a different expectation
for how the challenge value is represented/used. The [0, 0, low, high] format
works for direct polynomial evaluation but may fail when used in eq(r, r')
computations.

### Next Steps
1. Compare Stage 4's eq_cycle_evals values with Jolt
2. Trace through the polynomial binding step by step
3. Check if the r_cycle_be ordering is correct
4. Verify the final eq_claim computation matches Jolt's expected_output_claim formula

### Technical Notes
- Stage 1 uses: challengeScalar() -> [0, 0, low, high] format, WORKS
- Stage 4 uses: challengeScalar() -> [0, 0, low, high] format, FAILS
- The difference is in how Stage 4 computes eq(r_cycle_Stage3, r_cycle_Stage4)
- computeEqEvalsBE processes r_cycle_be to build eq(r, j) for all j

## Commit History
- 5cec222: fix: remove synthetic termination write from memory trace (Stage 2 fix)
- 51b5f1b: docs: update TODO with Stage 4 investigation progress
- ad6eb9d: docs: update notes with Stage 4 investigation findings
