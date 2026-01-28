# Zolt-Jolt Compatibility: Status Update

## Status: LT POLYNOMIAL FIXED - NEW ISSUE IN BATCHED CLAIM

## Session 73 Progress (2026-01-28)

### ✓ Fixed: LT Polynomial Binding Order

The LT polynomial mismatch has been resolved. Changed all polynomial binding from HighToLow (half-based indexing) to LowToHigh (adjacent-pair indexing):

- **Before**: `new[i] = (1-r)*old[i] + r*old[i+half]` (HighToLow)
- **After**: `new[i] = (1-r)*old[2*i] + r*old[2*i+1]` (LowToHigh)

Fixed files:
- `val_evaluation.zig` - ValEvaluationProver
- `val_final.zig` - ValFinalProver
- `raf_checking.zig` - RaPolynomial and RafSumcheckProver
- `jolt_r1cs.zig` - R1CS sumcheck
- `jolt_outer_prover.zig` - Outer prover
- `outer.zig` - Spartan outer prover

**Commit**: `859cd7e` - "fix: Use LowToHigh binding order for polynomial folding"

### Debug Output Shows LT Now Matches

```
[ZOLT LT DEBUG] Computing LT(r, r_cycle):
  lt_eval_computed (Jolt formula, BE) = { 41, 152, 72, ... }
  lt_eval_prover (from binding) = { 41, 152, 72, ... }
  Match? true

[ZOLT LT DEBUG] Computing LT using LE formulation:
  lt_eval_le (LE formulation) = { 41, 152, 72, ... }
  lt_eval_prover (from binding) = { 41, 152, 72, ... }
  Match LE? true
```

### Current Issue: Stage 4 Batched Claim Mismatch

Despite the LT polynomial fix, the Stage 4 verification still shows:
```
batched_claim (sumcheck output) = { 35, 3, 197, 27, ... }
total_expected = { 15, 44, 17, 109, ... }
Do they match? false
```

The individual instance computations:
- Instance 0 (RegistersRWC): eq * combined
- Instance 1 (ValEval): inc * wa * lt - **LT NOW CORRECT**
- Instance 2 (ValFinal): inc * wa

The weighted sum of these doesn't equal the sumcheck output claim.

### Possible Causes

1. **Claim update during binding**: The round-by-round claim update may not correctly track the batched claim through all rounds.

2. **Coefficient handling**: The batching coefficients look correct (non-zero in lower 128 bits), but need to verify they're applied correctly.

3. **Round polynomial construction**: After fixing binding order, need to verify `computeRoundPolynomial` uses consistent indexing across all instances.

4. **Phase transitions**: Stage 4 has multiple phases (cycle binding, address binding) - check phase transition handling.

### Next Steps

1. Add more detailed debug output to trace the batched claim evolution round-by-round
2. Compare the round polynomial values at each step between Zolt prover and what Jolt verifier expects
3. Verify the `regs_current_claim` matches `eq * combined` after all bindings
4. Check if the issue is in how the 3 instance contributions are batched

### Files for Investigation

- `src/zkvm/proof_converter.zig:2032-2537` - Stage 4 sumcheck loop
- `src/zkvm/spartan/stage4_gruen_prover.zig` - Gruen prover for RegistersRWC
- `src/zkvm/ram/val_evaluation.zig` - ValEvaluation prover (fixed)
- `src/zkvm/ram/val_final.zig` - ValFinal prover (fixed)

### Test Results

- 714 unit tests pass ✓
- Native verification passes all 6 stages ✓
- Jolt export produces proof file ✓
- **Cross-verification fails at Stage 4**: batched claim mismatch

SESSION_ENDING - LT polynomial fixed, need to debug batched claim computation.
