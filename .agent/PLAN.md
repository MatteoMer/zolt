# Zolt zkVM Implementation Plan

## Current Status (December 2024 - Iteration 14)

### Critical Fix: Base Field vs Scalar Field

**DISCOVERED BUG**: The pairing implementation was using the WRONG field!
- All Fp2, Fp6, Fp12 operations were using Fr (scalar field) instead of Fp (base field)
- These are DIFFERENT primes with different moduli
- Fr = 21888242871839275222246405745257275088548364400416034343698204186575808495617
- Fp = 21888242871839275222246405745257275088696311157297823662689037894645226208583

**SOLUTION IMPLEMENTED**:
1. Added `BN254BaseField` type (`Fp`) with correct Fp modulus and Montgomery constants
2. Created generic `MontgomeryField` function for parameterized field types
3. Updated all extension field types (Fp2, Fp6, Fp12) to use Fp
4. Added `G1PointFp` type for G1 points with Fp coordinates
5. Added `g1ToFp` function for proper Montgomery form conversion

### Line Evaluation Rewrite (gnark-crypto style)

Rewrote line evaluation to match gnark-crypto's affine approach:
- Line coefficients: R0 = λ, R1 = λ·x_Q - y_Q
- Sparse element: (1, 0, 0, c3, c4, 0) in Fp12
- evaluateLineSparse() computes c3 = R0 * xNegOverY, c4 = R1 * yInv

### Status

- **All 328 tests pass**
- Pairing bilinearity test still failing (disabled)

## Known Issues

### Pairing Bilinearity (Critical for Verification)
The pairing still doesn't satisfy e(aP, Q) = e(P, Q)^a.
This affects HyperKZG verification.

Remaining possible issues:
1. Final exponentiation hard part formula may need adjustment
2. Frobenius endomorphism on G2 may have coefficient issues
3. Twist isomorphism handling may need review
4. Frobenius coefficients may not match gnark-crypto exactly

## Files Modified (Iteration 14)

### src/field/mod.zig
- Added BN254_FP_MODULUS, BN254_FP_R, BN254_FP_R2, BN254_FP_INV
- Added `BN254BaseField` = `MontgomeryField(...)`
- Added generic `MontgomeryField` function with all field operations
- Added `double()` method to MontgomeryField

### src/field/pairing.zig
- Import Fp = BN254BaseField (base field)
- Updated Fp2 to use Fp instead of BN254Scalar
- Updated fp2ScalarMul to use Fp
- Updated fp2FromLimbs and fpFromLimbs to use Fp
- Updated G2Point.generator() to use Fp for coordinates
- Added G1PointFp struct for G1 points in base field
- Added g1ToFp() for proper Fr→Fp conversion
- Updated pairing() to convert G1 points before processing
- Updated evaluateLine and evaluateLineSparse to use Fp
- Changed LineCoeffs from (lambda, mu) to (r0, r1)
- Updated doublingStep and additionStep for R0/R1 format

### src/integration_tests.zig
- Added Fp import for extension field tests

## Architecture Summary

### Field Tower (CORRECTED)
- **Fp = BN254 base field** (254 bits) - for point coordinates and pairing
- Fr = BN254 scalar field (254 bits) - for scalars in MSM
- Fp2 = Fp[u]/(u² + 1)
- Fp6 = Fp2[v]/(v³ - ξ) where ξ = 9 + u
- Fp12 = Fp6[w]/(w² - v)

### Pairing Algorithm
1. Convert G1 point from Fr to Fp Montgomery form
2. Miller loop over 6x+2 (x = 4965661367192848881)
3. Two additional lines with Frobenius endomorphism
4. Final exponentiation: easy part + hard part

### Commitment Schemes
- HyperKZG: Uses BN254 pairing (verify stub until pairing works)
- Dory: Streaming implementation (commit works, open placeholder)

## Next Steps for Future Iterations

1. **Debug Pairing**:
   - Compare intermediate Miller loop values with gnark-crypto
   - Verify Frobenius coefficients match exactly
   - Check final exponentiation hard part formula
   - Consider using arkworks test vectors

2. **Performance Optimization**:
   - SIMD operations
   - Parallel sumcheck rounds
   - Batch inversions

3. **Production Readiness**:
   - Import Ethereum SRS
   - Full E2E testing
   - Benchmarking
