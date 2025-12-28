# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 15)

### Critical Fix: BN254 Non-Residue (ξ)
- [x] **FIXED BUG**: ξ was (1 + u), should be (9 + u)
- [x] Updated `mulByXi()` formula: (9a - b) + (a + 9b)u
- [x] Pairing bilinearity test now passes!

### Pairing Improvements
- [x] Added `G1PointInFp` type for proper base field coordinates
- [x] Added `pairingFp()` function for direct Fp pairing
- [x] Added comprehensive pairing tests:
  - [x] Bilinearity in G1: e([2]P, Q) = e(P, Q)²
  - [x] Bilinearity in G2: e(P, [2]Q) = e(P, Q)²
  - [x] Identity property: e(P, O) = e(O, Q) = 1
  - [x] Non-degeneracy: e(P, Q) ≠ 1

### Test Status
- [x] All 339 tests pass
- [x] Pairing bilinearity verified

## Completed (Previous Sessions)

### Iteration 14: Base Field vs Scalar Field
- [x] Added `BN254BaseField` (Fp) type
- [x] Created generic `MontgomeryField` function
- [x] Updated Fp2, Fp6, Fp12 to use Fp
- [x] Added G1PointFp for G1 in base field
- [x] Rewrote line evaluation (gnark-crypto style)

### Iterations 1-13: Core Infrastructure
- [x] Lookup table infrastructure (14 tables)
- [x] Lasso prover/verifier
- [x] Instruction proving with flags
- [x] Memory checking with RAF and Val Evaluation
- [x] Multi-stage prover (6 stages)
- [x] BN254 G1/G2 generators
- [x] HyperKZG SRS generation
- [x] G2Point.scalarMul()

## Working Components

### Fully Working
- **BN254 Pairing** - Bilinearity verified ✅
- **Extension Fields** - Fp2, Fp6, Fp12 with correct ξ = 9 + u
- **Field Arithmetic** - Montgomery form, all ops
- **G1/G2 Point Arithmetic** - Addition, doubling, scalar mul
- **Frobenius Endomorphism** - All coefficients from ziskos
- **Sumcheck Protocol** - Complete prover/verifier
- **RISC-V Emulator** - Full RV64IMC execution
- **ELF Loader** - Complete ELF32/ELF64 parsing
- **Basic MSM** - Affine/projective arithmetic

### Partially Working
- **HyperKZG** - commit() works, verify() needs pairing (now ready!)
- **Dory** - commit() works, open() placeholder
- **Spartan** - Matrix ops work, verifier incomplete

## Next Steps (Future Iterations)

### High Priority (Ready Now That Pairing Works)
- [ ] Implement HyperKZG.verify() using pairingFp
- [ ] Add real verification in commitment schemes
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
