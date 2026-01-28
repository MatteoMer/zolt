# Zolt-Jolt Compatibility: Status Update

## Status: LT POLYNOMIAL FIXED - BATCHED CLAIM MISMATCH REMAINING

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

### Current Issue: Stage 4 Final Verification Mismatch

The sumcheck itself runs correctly (Round 0 `p(0)+p(1) == batched_claim` passes), but the final verification check fails:

```
batched_claim (sumcheck output) = { 35, 3, 197, 27, 84, 39, 252, 206, ... }
total_expected = { 15, 44, 17, 109, 110, 245, 207, 87, ... }
Do they match? false
```

Where `total_expected` = weighted sum of:
- Instance 0: eq(r_sumcheck, r_cycle) * combined
- Instance 1: inc * wa * lt (ValEvaluation)
- Instance 2: inc * wa (ValFinal)

### Key Observations

1. **Initial claim matches**: `[ZOLT STAGE4 CHECK] Round 0: match? true`
2. **LT values match**: Both Jolt formula and LE formulation produce same result
3. **Final claim mismatch**: `total_expected != batched_claim`

### Possible Root Causes

1. **eq polynomial computation**: The `eq(r_sumcheck, r_cycle)` may be using wrong endianness for one or both inputs

2. **combined polynomial**: The `combined` value from RegistersRWC claims might not match what the verifier expects

3. **Claim tracking**: The `regs_current_claim` tracking through rounds may diverge from the actual batched polynomial

### Key Files for Next Session

- `src/zkvm/proof_converter.zig:2380-2537` - eq_val_be and expected_output computation
- `src/zkvm/spartan/stage4_gruen_prover.zig` - RegistersRWC prover
- `src/poly/mod.zig` - EqPolynomial.mle function

### Next Steps

1. Add debug to compare Zolt's `eq(r_sumcheck, r_cycle)` with Jolt's computation
2. Verify `r_sumcheck` normalization matches Jolt's `normalize_opening_point`
3. Check if `combined` computation follows Jolt's formula exactly

### Test Results

- 714 unit tests pass ✓
- Native verification passes all 6 stages ✓
- Jolt export produces proof file ✓
- **Cross-verification fails at Stage 4**: total_expected != batched_claim

SESSION_ENDING - LT polynomial fixed. Need to investigate eq polynomial and combined computation for the Stage 4 mismatch.
