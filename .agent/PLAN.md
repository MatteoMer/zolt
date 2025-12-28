# Zolt zkVM Implementation Plan

## Current Status (December 2024 - Iteration 16)

### MAJOR FIX: Projective Point Doubling + HyperKZG Architecture

Two critical bugs fixed in this iteration:

1. **Projective Doubling Bug**: The Jacobian doubling formula was using `D` instead of `2*D` in the Y3 calculation, causing scalar multiplication to produce wrong results.

2. **HyperKZG Type Architecture**: G1 points were using the scalar field (Fr) for coordinates instead of the base field (Fp). Fixed by using `AffinePoint(Fp)` for G1 points.

### Verified Properties
1. **Projective == Affine double**: Now `ProjectivePoint.double().toAffine()` equals `AffinePoint.double()` ✅
2. **SRS Pairing Relationship**: e([τ]G1, G2) = e(G1, [τ]G2) ✅
3. **Scalar multiplication consistency**: `scalarMul(G1, 2)` equals `G1.double()` ✅

## Architecture Summary

### Field Tower
```
Fp  = BN254 base field (254 bits) - G1 point coordinates
Fr  = BN254 scalar field (254 bits) - scalars for multiplication

Fp2 = Fp[u] / (u² + 1)
Fp6 = Fp2[v] / (v³ - ξ)  where ξ = 9 + u
Fp12 = Fp6[w] / (w² - v)
```

### Correct Type Usage for HyperKZG
```
Polynomial evaluations: Fr (scalar field)
G1 point coordinates: Fp (base field)
G2 point coordinates: Fp2 (extension field)
Scalars for EC multiplication: Fr
MSM: MSM(Fr, Fp) - Fr scalars, Fp coordinates
```

### Key Types
- `HyperKZG.Point = AffinePoint(Fp)` - G1 point with Fp coords
- `HyperKZG.Fp = BN254BaseField` - base field for coords
- `G1PointFp` - pairing-compatible G1 point struct
- `G2Point` - G2 point with Fp2 coordinates

## Files Modified (Iteration 16)

### src/msm/mod.zig
- **Fixed ProjectivePoint.double()**:
  - Changed variable name from `D` to `half_D`
  - `D = 2 * half_D` (correct EFD formula)
  - `X3 = F - 2*D` uses `two_D = D.add(D)`
  - `Y3 = E*(D - X3) - 8*C` uses the correct `D` (= 2*half_D)

### src/poly/commitment/mod.zig
- Changed `Point = AffinePoint(F)` to `Point = AffinePoint(Fp)`
- Updated MSM calls from `MSM(F, F)` to `MSM(F, Fp)`
- Added `toG1PointFp()` helper for pairing conversion
- Added tests:
  - `hyperkzg projective vs affine double`
  - `hyperkzg srs has correct tau relationship`

### src/host/mod.zig
- Changed `G1Point = AffinePoint(F)` to `G1Point = AffinePoint(Fp)`
- This makes host preprocessing consistent with HyperKZG types

## Next Steps

### High Priority
1. **Implement proper HyperKZG verification**
   - Current `verifyWithPairing()` uses a simplified equation
   - Need Gemini-style reduction for multilinear openings
   - Reference: Gemini paper / jolt-core prover.rs

2. **Wire up JoltProver.prove()**
   - Remove panic at src/zkvm/mod.zig:149
   - Connect all proving stages

3. **Wire up JoltVerifier.verify()**
   - Remove panic at src/zkvm/mod.zig:174
   - Implement verification logic

### Medium Priority
- Implement host.execute()
- Implement Preprocessing.preprocess()
- Complete memory RAF checking

### Still Needed (from task guide)
- Lasso lookup arguments
- Instruction R1CS constraint generation
- Multi-stage sumcheck orchestration

## Test Status

All 341 tests pass:
- Field arithmetic: Fp, Fp2, Fp6, Fp12
- Curve arithmetic: G1, G2 points (now with correct projective doubling)
- Pairing: bilinearity, identity, non-degeneracy
- HyperKZG: SRS pairing relationship verified
- Sumcheck protocol
- RISC-V emulation
- ELF loading
- MSM operations
