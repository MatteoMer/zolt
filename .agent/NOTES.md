# Zolt-Jolt Compatibility Notes

## Current Status (December 28, 2024)

### Working Components
- ✅ Blake2b Transcript (matches Jolt)
- ✅ Dory Commitment Scheme (GT elements verified matching)
- ✅ Proof Structure (JoltProof with 7 stages)
- ✅ Serialization Format (opening claims, commitments)
- ✅ LagrangeHelper with shift_coeffs_i32 (matches Jolt)
- ✅ COEFFS_PER_J precomputed Lagrange weights
- ✅ SpartanOuterProver.computeUniskipFirstRoundPoly (fixed to use COEFFS_PER_J)
- ✅ All 618 unit tests pass

### Stage 1 Sumcheck Debugging

**Current Issue**: `Verification failed: Stage 1 - Sumcheck verification failed`

**Root Cause Analysis Complete**:

The issue was in how we compute extended evaluations for the UniSkip polynomial:
1. For satisfied R1CS constraints, Az(y) * Bz(y) = 0 at base domain points
2. Old approach: Interpolate from Az*Bz products → all zeros
3. New approach (implemented): Evaluate Az(y_j) and Bz(y_j) **separately** using COEFFS_PER_J, then multiply

**Fix Implemented**:
- Added `LagrangeHelper.shiftCoeffsI32()` to compute Lagrange interpolation weights
- Precomputed `COEFFS_PER_J[9][10]` for extended domain evaluation
- Fixed `computeUniskipFirstRoundPoly()` to use separate Az/Bz evaluation

**Test Verification**:
- Unit test confirms non-zero coefficients are produced for non-trivial Az/Bz
- All 618 tests pass

**Remaining Issue**:
- Production proof still has all-zero UniSkip coefficients
- Need to trace why the production path produces different results than the test
- Possible causes:
  1. cycle_witnesses array is empty or has zero values
  2. The constraint evaluators produce zeros for all constraints
  3. Some early return path is being taken

## Architecture Overview

### UniSkip Extended Evaluation

The key insight from Jolt's implementation:

```
t1(y_j) = Σ_x eq(τ,x) * Az(x, y_j) * Bz(x, y_j)

For base domain points:
- y ∈ {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
- These directly map to constraint indices 0-9
- For satisfied constraints, Az * Bz = 0 at these points

For extended domain points:
- y ∈ {-9, -8, -7, -6, -5} ∪ {6, 7, 8, 9}
- Use COEFFS_PER_J[j][i] to compute:
  Az(y_j) = Σ_i coeffs[j][i] * Az[i]
  Bz(y_j) = Σ_i coeffs[j][i] * Bz[i]
- Even if Az*Bz = 0 at base points, Az(y_j)*Bz(y_j) ≠ 0 at extended points
```

### Key Files
- `src/zkvm/r1cs/univariate_skip.zig` - LagrangeHelper, COEFFS_PER_J, UniPoly
- `src/zkvm/spartan/outer.zig` - SpartanOuterProver with fixed computeUniskipFirstRoundPoly
- `src/zkvm/r1cs/evaluators.zig` - Az/Bz evaluation structures
- `src/zkvm/proof_converter.zig` - Proof conversion with UniSkip generation

## Next Steps

1. **Debug production path**: Add runtime logging (not debug.print which is stripped in release) to trace the actual values in the production proof generation

2. **Verify constraint evaluator output**: Check if the AzFirstGroup/BzFirstGroup structs are populated with expected boolean/integer values

3. **Test with Jolt-generated witnesses**: Export a witness from Jolt and verify Zolt can process it correctly
