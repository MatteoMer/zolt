# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 16)

### Critical Fix: Projective Point Doubling Bug
- [x] **FIXED BUG**: ProjectivePoint.double() had wrong D parameter in Y3 formula
  - Was using `D` instead of `2*D` (which is the correct formula from EFD)
  - This caused scalar multiplication to give wrong results
  - Now `scalarMul(G1, 2)` equals `G1.double()` correctly

### HyperKZG Architecture Fix
- [x] Changed HyperKZG to use Fp (base field) for G1 point coordinates
  - Point = AffinePoint(Fp) instead of AffinePoint(Fr)
  - This is the correct representation for elliptic curve points
  - Scalars (polynomial evals) still in Fr, but point coords in Fp

- [x] Updated MSM calls to use MSM(F, Fp) for Fr scalars with Fp coords
- [x] Fixed host/mod.zig to use the same G1Point type
- [x] Added G1PointFp conversion helper for pairing operations
- [x] Enabled and fixed HyperKZG SRS pairing relationship test

### Test Status
- [x] All tests pass (341 tests)
- [x] Projective vs Affine double consistency verified
- [x] SRS pairing relationship: e([τ]G1, G2) = e(G1, [τ]G2) verified

## Completed (Previous Sessions)

### Iteration 15: BN254 Pairing
- [x] Fixed Fp6 non-residue ξ = 9 + u (was incorrectly 1 + u)
- [x] Added G1PointInFp type for proper base field coordinates
- [x] Added pairingFp() and pairingCheckFp() functions
- [x] Verified pairing bilinearity: e([2]P, Q) = e(P, Q)²

### Iterations 1-14: Core Infrastructure
- [x] Lookup table infrastructure (14 tables)
- [x] Lasso prover/verifier
- [x] Instruction proving with flags
- [x] Memory checking with RAF and Val Evaluation
- [x] Multi-stage prover (6 stages)
- [x] BN254 G1/G2 generators
- [x] HyperKZG SRS generation
- [x] G2Point.scalarMul()
- [x] Sumcheck protocol
- [x] RISC-V emulator
- [x] ELF loader

## Working Components

### Fully Working
- **BN254 Pairing** - Bilinearity verified, used for SRS verification
- **Extension Fields** - Fp2, Fp6, Fp12 with correct ξ = 9 + u
- **Field Arithmetic** - Montgomery form, all ops
- **G1/G2 Point Arithmetic** - Addition, doubling, scalar mul (now fixed!)
- **Projective Points** - Jacobian doubling now correct
- **Frobenius Endomorphism** - All coefficients from ziskos
- **Sumcheck Protocol** - Complete prover/verifier
- **RISC-V Emulator** - Full RV64IMC execution
- **ELF Loader** - Complete ELF32/ELF64 parsing
- **MSM** - Multi-scalar multiplication with bucket method
- **HyperKZG** - commit() and basic verify() work, SRS verified

### Partially Working
- **HyperKZG verifyWithPairing** - Needs proper Gemini reduction
- **Dory** - commit() works, open() placeholder
- **Spartan** - Matrix ops work, verifier incomplete

## Next Steps (Future Iterations)

### High Priority
- [ ] Implement proper HyperKZG verification with Gemini reduction
- [ ] Wire up JoltProver.prove() (currently panics)
- [ ] Wire up JoltVerifier.verify() (currently panics)

### Medium Priority
- [ ] Implement host.execute() (currently panics)
- [ ] Implement Preprocessing.preprocess() (currently panics)
- [ ] Complete memory RAF checking

### Low Priority
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation
- [ ] Import production SRS from Ethereum ceremony
