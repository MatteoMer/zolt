# Zolt-Jolt Compatibility: Stage 4 Final Claim Fix

## Status: In Progress (Session 71)

## Recent Fix Applied

### Phase 3 Degree-3 Polynomial Fix (Committed: e40d5cf)

**Root Cause Found:** Phase 3 was always computing degree-2 polynomial, but Jolt has TWO cases:
1. **Cycles remaining** (current_T > 1): Degree 3, compute [p(0), p(2), p(3)]
2. **Cycles fully bound** (current_T == 1): Degree 2, compute [p(0), p(2)]

**Fix Applied:** Updated `phase3ComputeMessage` to check `cycles_remaining = self.current_T > 1` and compute degree-3 polynomial with evaluations at [0, 2, 3] when cycle variables remain.

## Current Testing

Need to test the Phase 3 fix with actual Jolt verification. The test suite is running into OOM issues before reaching Stage 4.

## Files Modified

- `src/zkvm/spartan/stage4_gruen_prover.zig` - Phase 3 fix for degree-3 polynomial
- `src/zkvm/spartan/gruen_eq.zig` - Changed division-by-zero handling to panic

## Key Findings from Jolt Analysis

### Phase Structure (RegistersRWC)
- **Phase 1** (first log_T/2 rounds): Gruen optimization, degree-3
- **Phase 2** (next LOG_K=7 rounds): Address vars, degree-2
- **Phase 3** (remaining rounds):
  - If cycles remain: degree-3
  - If cycles bound: degree-2

### Expected Output Claim Formula
```
expected_output = eq(r_cycle_from_sumcheck, r_cycle_from_stage3) * combined
```
where `combined = rd_wa*(inc+val) + gamma*rs1_ra*val + gamma^2*rs2_ra*val`

### Variable Binding Order
- All phases use LowToHigh binding
- Challenges come out in reverse order from big-endian representation
- normalize_opening_point reverses phases to reconstruct BE order

## Next Steps

1. Run full test with Jolt verifier to check Stage 4 fix
2. If still failing, investigate Phase 1 Gruen polynomial computation
3. Check if merged_eq[0] matches expected eq_val after all bindings

## Session Summary

- Identified Phase 3 degree issue through Jolt code analysis
- Fixed Phase 3 to compute degree-3 polynomial when cycles remain
- Updated edge case handling in gruen_eq.zig
- Committed fix to main branch
