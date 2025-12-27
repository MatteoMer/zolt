# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 12)

### Frobenius Coefficients Fix
- [x] Added Fp6.frobenius() method with correct coefficients
- [x] Added Fp12.frobenius() method with correct coefficients
- [x] Updated frobeniusFp12() helper to use new implementation
- [x] Computed correct byte arrays from arkworks decimal values:
  - FROBENIUS_COEFF_FP6_C1[1] = gamma12() = ξ^{(p-1)/3}
  - FROBENIUS_COEFF_FP6_C2[1] = ξ^{2(p-1)/3}
  - FROBENIUS_COEFF_FP12_C1[1] = ξ^{(p-1)/6}
- [x] Verified all 327+ tests still pass

### ATE Loop Count and Miller Loop Fix
- [x] Fixed ATE_LOOP_COUNT to correct 65-element signed binary expansion
- [x] Fixed Miller loop to iterate from MSB to LSB (was wrong direction)

### Pairing Bilinearity (Still Failing)
The pairing bilinearity test e([2]P, Q) = e(P, Q)^2 still fails.
With Frobenius coefficients and ATE loop now correct, the remaining issues are likely in:
1. Line evaluation in doubling/addition step
2. Final exponentiation hard part formula
3. π(Q) twist factor computation

## Completed (Previous Sessions - Iteration 11)

### Test Interference Investigation
- [x] Confirmed test interference issue with detailed investigation
- [x] Determined it's a Zig 0.15.2 compiler bug (not runtime memory corruption)
- [x] Updated documentation in .agent/NOTES.md

### G2 Scalar Multiplication
- [x] Implemented `G2Point.scalarMul()` using double-and-add algorithm
- [x] Added `G2Point.scalarMulU64()` convenience method
- [x] Updated HyperKZG SRS to compute proper [τ]_2 = τ * G2
- [x] Added test for G2 scalar multiplication consistency

### G2 Frobenius Coefficients
- [x] Added gamma12 = ξ^{(p-1)/3} for G2 x-coordinate
- [x] Added gamma13 = ξ^{(p-1)/2} for G2 y-coordinate
- [x] Updated frobeniusG2() to multiply by Frobenius coefficients

### Pairing Issue Analysis
- [x] Discovered pairing implementation uses placeholder/simplified code
- [x] Identified root cause: missing Frobenius coefficients and incomplete final exponentiation
- [x] Documented all issues in .agent/NOTES.md

## Completed (Previous Sessions - Iterations 1-10)

### Iteration 10: LassoProver and E2E Testing
- [x] Fixed LassoProver to use `params.log_T` directly
- [x] Discovered test interference issue when calling `JoltProver.prove()`

### Iteration 9: Preprocessing and Zig 0.15 Fixes
- [x] Implemented SharedPreprocessing(F) with bytecode size, padding, memory layout
- [x] Computed initial memory hash for public verification
- [x] Preprocessing.preprocess() generates ProvingKey and VerifyingKey
- [x] Many Zig 0.15 API compatibility fixes

### Iteration 8: R1CS-Spartan Integration
- [x] Created JoltR1CS type with witness generation
- [x] Implemented Az, Bz, Cz computation for Spartan
- [x] Created JoltSpartanInterface for sumcheck integration

### Iterations 1-7: Core Infrastructure
- [x] Lookup table infrastructure (14 tables)
- [x] Lasso prover/verifier with ExpandingTable and SplitEqPolynomial
- [x] Instruction proving with CircuitFlags and InstructionFlags
- [x] Memory checking with RAF and Val Evaluation
- [x] Multi-stage prover (6 stages)
- [x] BN254 G1/G2 generators with real coordinates
- [x] HyperKZG SRS generation
- [x] host.execute() and JoltProver.prove() implementation

## Summary (Iteration 11)

1. **Test Interference**: Confirmed the issue is a Zig compiler bug, not memory corruption.

2. **G2 Scalar Multiplication**: Implemented correctly. Verified via add/double consistency tests.

3. **HyperKZG SRS**: Now computes proper [τ]_2 = τ * G2.

4. **Frobenius Coefficients**: Added gamma12 and gamma13 for G2 Frobenius endomorphism.

5. **Pairing Analysis**: The pairing still doesn't satisfy bilinearity because:
   - Fp12 Frobenius is missing coefficients
   - Hard part of final exponentiation uses simplified formula
   - Possibly issues in line evaluation

All 327 tests pass.

## Next Steps (Future Iterations)

### High Priority (Pairing Fix)
- [ ] Add Fp12 Frobenius coefficients (γ_{1,1} through γ_{1,5})
- [ ] Implement proper frobeniusFp12() with coefficient multiplication
- [ ] Fix hard part of final exponentiation
- [ ] Verify Miller loop line evaluation is correct
- [ ] Enable and pass pairing bilinearity test

### Medium Priority
- [ ] Import production SRS from Ethereum ceremony
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation

### Low Priority
- [ ] Dory commitment scheme completion
- [ ] Additional RISC-V instruction support
- [ ] Benchmarking against Rust implementation
