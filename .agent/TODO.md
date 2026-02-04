# Zolt-Jolt Compatibility Implementation

## Status: Session 65 - Stage 4 FIXED, Stage 5 failing

## Current Issue

Stage 5 (InstructionReadRaf) sumcheck verification fails with output_claim vs expected_claim mismatch.

## Session 65 Progress

### Stage 4 Fix (COMPLETED)

Fixed several critical issues in Stage 4 sumcheck:

1. **ValEvaluation hint mechanism**: Changed from `p(1) = claim - p(0)` to `p(0) = claim - p(1)` to match Jolt's convention. Jolt computes `eval_at_0 = previous_claim - eval_at_1` in val_evaluation.rs.

2. **ValFinal polynomial evaluation**:
   - Changed from Toom-Cook formula (expecting c3) to Lagrange interpolation through 3 points
   - ValFinal is degree-2 (quadratic), so uses [p(0), p(1), p(2)] not [p(0), p(1), p(2), c3]
   - Implemented `evaluateQuadraticAtChallengeFromEvals` using Lagrange basis polynomials

3. **ValFinal hint mechanism**: Keep actual p(2) unchanged when applying hint to p(1)
   - Jolt's `from_evals_and_hint` takes [p(0), p(2)] and computes p(1) = claim - p(0)
   - Then interpolates through all 3 points via Vandermonde
   - The actual p(2) is kept, only p(1) is modified by the hint

4. **ValFinal combined polynomial contribution**:
   - ValFinal is degree-2, so c3 = 0 (no cubic term)
   - Fixed to not add c2 (ValFinal's evals[3]) to combined_evals[3] (which expects c3)
   - Only add weighted [p(0), p(1), p(2), 0] to combined_evals

### Key Files Modified

1. `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig`:
   - Added `evaluateQuadraticAtChallengeFromEvals` function for degree-2 polynomial evaluation
   - Fixed ValEvaluation hint: `p(0)_hint = claim - p(1)_actual`
   - Fixed ValFinal hint: keep actual p(2), only modify p(1)
   - Fixed ValFinal combined_evals contribution: don't add c2 to c3 slot

2. `/home/vivado/projects/zolt/src/zkvm/ram/val_final.zig`:
   - Previous fix: compute actual sum after binding in `bindChallengeWithPoly`

3. `/home/vivado/projects/zolt/src/zkvm/ram/val_evaluation.zig`:
   - Previous fix: compute actual sum after binding in `bindChallengeWithPoly`

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Next Steps

1. Investigate Stage 5 (InstructionReadRaf) sumcheck failure
2. Stage 5 has 3 instances with complex polynomial structures
3. May have similar degree/hint issues to Stage 4

## Technical Notes

### Polynomial Degree Issues
- Stage 4 has mixed-degree instances: RegistersRWC (degree-3), ValEvaluation (degree-3), ValFinal (degree-2)
- When combining, must handle different leading coefficient slots correctly
- Degree-2 contributes 0 to c3 (cubic coefficient)
- Degree-3 contributes actual c3 value

### Hint Mechanism Variations
- Different sumcheck instances use different hint conventions:
  - Most use `p(1) = claim - p(0)` (hint on p(1), keep actual p(0))
  - ValEvaluation uses `p(0) = claim - p(1)` (hint on p(0), keep actual p(1))
  - Both satisfy sumcheck invariant `p(0) + p(1) = claim`

### Polynomial Evaluation
- Cubic (degree-3): Use Toom-Cook to coefficients, then Horner's method
- Quadratic (degree-2): Use Lagrange interpolation through 3 points
