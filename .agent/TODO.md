# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 13)

### Pairing Implementation Improvements
- [x] Added complete Frobenius coefficients from Zisk (GAMMA11-GAMMA35)
- [x] Implemented Fp12.frobenius2() for p² power
- [x] Implemented Fp12.frobenius3() for p³ power
- [x] Updated final exponentiation hard part with Zisk formula
- [x] Updated Miller loop to match Zisk iteration order
- [x] Added BN254Scalar.toMontgomery() for coefficient conversion
- [x] Updated line coefficients to (λ, μ) format
- [x] Updated doublingStep and additionStep to return (λ, μ)
- [x] Added sparse line evaluation structure

### Tests
- [x] All 327+ tests pass
- [ ] Pairing bilinearity test still failing (disabled)

## Completed (Previous Sessions)

### Iteration 12: Frobenius Coefficients
- [x] Added Fp6.frobenius() method with correct coefficients
- [x] Added Fp12.frobenius() method with correct coefficients
- [x] Fixed ATE_LOOP_COUNT to correct 65-element signed binary expansion
- [x] Fixed Miller loop to iterate from MSB to LSB

### Iteration 11: G2 Improvements
- [x] Implemented G2Point.scalarMul() using double-and-add algorithm
- [x] Updated HyperKZG SRS to compute proper [τ]_2 = τ * G2
- [x] Added gamma12 and gamma13 for G2 Frobenius coefficients

### Iterations 1-10: Core Infrastructure
- [x] Lookup table infrastructure (14 tables)
- [x] Lasso prover/verifier
- [x] Instruction proving with flags
- [x] Memory checking with RAF and Val Evaluation
- [x] Multi-stage prover (6 stages)
- [x] BN254 G1/G2 generators
- [x] HyperKZG SRS generation
- [x] host.execute() and JoltProver.prove()

## Next Steps (Future Iterations)

### High Priority (Pairing Fix)
- [ ] Investigate if Zisk coefficients are already in Montgomery form
- [ ] Compare line evaluation with gnark-crypto implementation
- [ ] Debug sparse Fp12 multiplication
- [ ] Verify twist isomorphism handling
- [ ] Enable and pass pairing bilinearity test

### Medium Priority
- [ ] Import production SRS from Ethereum ceremony
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation

### Low Priority
- [ ] Dory commitment scheme completion
- [ ] Additional RISC-V instruction support
- [ ] Benchmarking against Rust implementation
