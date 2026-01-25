# BUG FOUND: Double-Batching in Stage 4

## Date: 2026-01-24

## Bug Description

**Location**: `src/zkvm/spartan/stage4_gruen_prover.zig`, lines 430-434

**Issue**: Round polynomials were being batched TWICE:
1. First batched when appending to transcript (lines 419-421)
2. Then batched AGAIN when storing in round_polys (lines 430-434)

**Original buggy code**:
```zig
var batched_poly = round_poly;
for (0..4) |i| {
    batched_poly.coeffs[i] = round_poly.coeffs[i].mul(self.batching_coeff);
}
round_polys[round] = batched_poly;
```

**Fix Applied**:
```zig
// Store UNBATCHED polynomial (batching is only for transcript/challenge)
round_polys[round] = round_poly;
```

## Root Cause Analysis

The confusion arose because:
1. Batching is needed for the transcript to compute Fiat-Shamir challenges
2. But the stored round_polys should be unbatched
3. The proof_converter calls `computeRoundEvals()` which generates fresh polynomials, so it doesn't use the stored round_polys
4. This meant the fix doesn't affect Jolt proof generation!

## Current Status

- ✅ Fix applied to remove double-batching in Stage4GruenProver
- ✅ Sumcheck relation verified: `p(0) + p(1) = current_claim` for register prover
- ❌ Cross-verification still FAILS at Stage 4

## Remaining Issue

The cross-verification test shows:
- Stages 1-3: PASS ✅
- Stage 4: FAIL ❌

Error:
```output_claim:          3110039466046447483900250223050551234127534443712699458605485358576992771996
expected_output_claim: 1714094508039949549364454035354307069625501160663719062284025310540203859155
```

The output_claim doesn't match expected_output_claim, which means the combined polynomial from the proof_converter is producing wrong values.

## Next Steps

Since the individual register prover polynomial is correct (sumcheck relation holds), the issue must be in:

1. **Batching in proof_converter**: Check how combined_evals are computed from:
   - regs_evals * batching_coeffs[0]
   - val_eval_evals * batching_coeffs[1] (should be 0 for programs without RAM)
   - val_final_evals * batching_coeffs[2] (should be 0 for programs without RAM)

2. **RAM instance contributions**: Verify that val_eval and val_final contribute ZERO when there are no RAM operations

3. **Batching coefficients**: Verify batching_coeffs[0], [1], [2] are correct

## Files Modified

- `src/zkvm/spartan/stage4_gruen_prover.zig`: Removed double-batching bug
- `src/zkvm/spartan/gruen_eq.zig`: Added debug logging for gruenPolyDeg3
- `src/zkvm/proof_converter.zig`: Added debug logging for Stage 4 round evaluations

## Deep Code Audit Results

Performed comprehensive line-by-line comparison of Zolt vs Jolt implementations:

✅ **All formulas match exactly:**
- Polynomial formulas (c_0, c_X2)
- val_poly semantics
- x_in/x_out indexing
- evalsCached table building
- gruenPolyDeg3 conversion
- Coefficient interpolation

The bug was NOT in the algorithms, but in how the results were stored/used.
