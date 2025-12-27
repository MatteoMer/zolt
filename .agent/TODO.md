# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 14)

### Critical Architecture Fix: Base Field vs Scalar Field
- [x] Discovered pairing was using wrong field (Fr instead of Fp)
- [x] Added `BN254BaseField` (Fp) type with correct modulus
- [x] Created generic `MontgomeryField` function for parameterized fields
- [x] Updated all Fp2, Fp6, Fp12 types to use base field Fp
- [x] Added `G1PointFp` type for G1 points in base field
- [x] Added `g1ToFp` function for proper Montgomery form conversion

### Line Evaluation Rewrite (gnark-crypto style)
- [x] Rewrote line coefficients to R0/R1 format
- [x] R0 = λ (slope), R1 = λ·x_Q - y_Q
- [x] Sparse element representation: (1, 0, 0, c3, c4, 0) in Fp12
- [x] Updated doublingStep and additionStep to match gnark-crypto

### Tests
- [x] All 328 tests pass
- [ ] Pairing bilinearity test still failing (disabled)

## Completed (Previous Sessions)

### Iteration 13: Frobenius and Line Evaluation
- [x] Added complete Frobenius coefficients from Zisk
- [x] Implemented frobenius2() and frobenius3() for Fp12
- [x] Updated final exponentiation hard part
- [x] Updated Miller loop iteration order

### Iterations 1-12: Core Infrastructure
- [x] Lookup table infrastructure (14 tables)
- [x] Lasso prover/verifier
- [x] Instruction proving with flags
- [x] Memory checking with RAF and Val Evaluation
- [x] Multi-stage prover (6 stages)
- [x] BN254 G1/G2 generators
- [x] HyperKZG SRS generation
- [x] G2Point.scalarMul() using double-and-add

## Known Issues

### Pairing Bilinearity Still Failing
The test `e([2]P, Q) = e(P, Q)^2` fails. Possible causes:
1. Final exponentiation formula may need adjustment
2. Frobenius endomorphism on G2 may have coefficient issues
3. Twist isomorphism handling may need review
4. The Frobenius coefficients may not match gnark-crypto exactly

## Next Steps (Future Iterations)

### High Priority (Pairing Fix)
- [ ] Compare intermediate Miller loop values against reference
- [ ] Verify Frobenius coefficients match gnark-crypto exactly
- [ ] Check final exponentiation hard part formula
- [ ] Enable and pass pairing bilinearity test

### Medium Priority (Jolt Core)
- [ ] Lookup Arguments / Lasso (THE core technique)
- [ ] Instruction proving with R1CS constraints
- [ ] Memory RAF checking

### Low Priority
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation
- [ ] Import production SRS from Ethereum ceremony
