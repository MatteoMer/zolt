# Zolt zkVM Implementation Plan

## Current Status (December 2024 - Iteration 15)

### MAJOR MILESTONE: BN254 Pairing Working! 🎉

The pairing is now fully functional with bilinearity verified.

**Critical Fix**: The Fp6 extension field was using the wrong non-residue!
- BUG: ξ was (1 + u) - WRONG
- FIX: ξ is (9 + u) - CORRECT

This single bug was causing the entire pairing to fail.

### Verified Pairing Properties
1. **Bilinearity in G1**: e([2]P, Q) = e(P, Q)²  ✅
2. **Bilinearity in G2**: e(P, [2]Q) = e(P, Q)²  ✅
3. **Identity**: e(P, O) = e(O, Q) = 1  ✅
4. **Non-degeneracy**: e(P, Q) ≠ 1 for non-identity P, Q  ✅

## Architecture Summary

### Field Tower (Correct)
```
Fp  = BN254 base field (254 bits) - point coordinates
Fr  = BN254 scalar field (254 bits) - scalar multiplication

Fp2 = Fp[u] / (u² + 1)
Fp6 = Fp2[v] / (v³ - ξ)  where ξ = 9 + u  ← CRITICAL
Fp12 = Fp6[w] / (w² - v)
```

### Types for Pairing
- `Fp = BN254BaseField` - base field element
- `G1PointFp` - G1 point with Fp coordinates (for pairing)
- `G1PointInFp` - G1 point via AffinePoint(Fp) (for EC ops)
- `G2Point` - G2 point with Fp2 coordinates
- `Fp12` - pairing target group element

### Key Functions
- `pairingFp(G1PointFp, G2Point) -> Fp12` - pairing with Fp coords
- `pairing(G1Point, G2Point) -> Fp12` - pairing with Fr->Fp conversion
- `millerLoop(G1PointFp, G2Point) -> Fp12` - ate loop
- `finalExponentiation(Fp12) -> Fp12` - (p^12-1)/r exponentiation

## Files Modified (Iteration 15)

### src/field/pairing.zig
1. **Fixed mulByXi()**:
   - Old: `(a-b) + (a+b)u` for ξ = 1 + u
   - New: `(9a-b) + (a+9b)u` for ξ = 9 + u

2. **Added G1PointInFp type** for proper EC ops in base field

3. **Added pairingFp()** function for direct Fp pairing

4. **Added pairing tests**:
   - Bilinearity in G1 and G2
   - Identity property
   - Non-degeneracy

## Next Steps

### Now Ready (Pairing Enables)
1. **Implement real HyperKZG verification**
   - Use pairingFp() for the pairing check
   - e(L, [1]₂) = e(R, [τ]₂)

2. **Complete Dory verification**
   - Pairing-based polynomial commitment verification

3. **Wire up JoltProver/JoltVerifier**
   - The core Jolt SNARK flow

### Still Needed
- Lasso lookup arguments (THE core technique)
- Instruction R1CS constraint generation
- Memory RAF checking
- Multi-stage sumcheck orchestration

## Test Status

All 339 tests pass:
- Field arithmetic: Fp, Fp2, Fp6, Fp12
- Curve arithmetic: G1, G2 points
- Pairing: bilinearity, identity, non-degeneracy
- Sumcheck protocol
- RISC-V emulation
- ELF loading
- MSM operations
