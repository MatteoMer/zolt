# Zolt zkVM Implementation Plan

## Current Status (December 2024 - Iteration 17)

### HyperKZG Verification Enhancement

Improved the HyperKZG verification to use proper batching and pairing checks:

1. **Batched quotient commitments**: W = sum_i gamma^i * Q_i
2. **Correction term**: sum_i gamma^i * r_i * Q_i (for evaluation points)
3. **Pairing equation**: e(C - v*G1 - correction, G2) == e(W, tau_G2)

### Architecture Summary

#### Field Tower
```
Fp  = BN254 base field (254 bits) - G1 point coordinates
Fr  = BN254 scalar field (254 bits) - scalars for multiplication

Fp2 = Fp[u] / (u² + 1)
Fp6 = Fp2[v] / (v³ - ξ)  where ξ = 9 + u
Fp12 = Fp6[w] / (w² - v)
```

#### HyperKZG Type Usage
```
Polynomial evaluations: Fr (scalar field)
G1 point coordinates: Fp (base field)
G2 point coordinates: Fp2 (extension field)
Scalars for EC multiplication: Fr
MSM: MSM(Fr, Fp) - Fr scalars, Fp coordinates
```

### Key Files Modified

#### src/poly/commitment/mod.zig
- Enhanced `verifyWithPairing()`:
  - Gamma-based batching of quotient commitments
  - Correction term computation from evaluation points
  - Proper pairing check equation
- Added `verifyAlgebraic()` for testing

## Next Steps

### Priority 1: Integration Tests
- Create end-to-end tests that prove and verify a simple program
- Verify that all components work together correctly

### Priority 2: Commitment Integration
- Wire HyperKZG commitments into JoltProof
- Implement batch opening for multiple polynomials

### Priority 3: Dory Completion
- Implement proper inner product argument for Dory.open()
- Add Dory verification

## Test Status

All 350 tests pass:
- Field arithmetic: Fp, Fp2, Fp6, Fp12
- Curve arithmetic: G1, G2 points
- Pairing: bilinearity, identity, non-degeneracy
- HyperKZG: commit, open, verify, SRS verification
- Sumcheck protocol
- RISC-V emulation
- ELF loading
- MSM operations
