# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 17)

### HyperKZG Verification Improvements
- [x] Enhanced `verifyWithPairing()` with proper batching and pairing check
  - Added gamma-based batching for quotient commitments
  - Compute correction term from evaluation points
  - Proper pairing equation: e(L, G2) == e(W, tau_G2)
- [x] Added `verifyAlgebraic()` for testing without pairing overhead
- [x] Added comprehensive documentation for verification algorithm

### Host Execute Integration Tests
- [x] Added test "execute runs simple program" - verifies c.nop execution
- [x] Added test "execute with longer program" - verifies multi-instruction execution

### Batch Verification Fixes
- [x] Fixed `verifyBatch()` to return `!bool` (error union) for transcript errors
- [x] Added test reference so batch.zig tests are discovered
- [x] Added test "batch opening accumulator multiple claims"
- [x] Added test "opening claim initialization"

## Completed (Previous Sessions)

### Iteration 16: Projective Point Doubling Bug + HyperKZG Architecture
- [x] Fixed ProjectivePoint.double() - was using D instead of 2*D
- [x] Changed HyperKZG to use Fp (base field) for G1 point coordinates
- [x] Verified SRS pairing relationship: e([τ]G1, G2) = e(G1, [τ]G2)

### Iteration 15: BN254 Pairing
- [x] Fixed Fp6 non-residue ξ = 9 + u
- [x] Added pairingFp() and pairingCheckFp() functions
- [x] Verified pairing bilinearity

### Iterations 1-14: Core Infrastructure
- [x] Lookup table infrastructure (14 tables)
- [x] Lasso prover/verifier
- [x] Instruction proving with flags
- [x] Memory checking with RAF and Val Evaluation
- [x] Multi-stage prover (6 stages) and verifier
- [x] BN254 G1/G2 generators
- [x] HyperKZG SRS generation
- [x] Sumcheck protocol
- [x] RISC-V emulator
- [x] ELF loader

## Working Components

### Fully Working
- **BN254 Pairing** - Bilinearity verified, used for SRS verification
- **Extension Fields** - Fp2, Fp6, Fp12 with correct ξ = 9 + u
- **Field Arithmetic** - Montgomery form, all ops
- **G1/G2 Point Arithmetic** - Addition, doubling, scalar mul
- **Projective Points** - Jacobian doubling correct
- **Frobenius Endomorphism** - All coefficients from ziskos
- **Sumcheck Protocol** - Complete prover/verifier
- **RISC-V Emulator** - Full RV64IMC execution
- **ELF Loader** - Complete ELF32/ELF64 parsing
- **MSM** - Multi-scalar multiplication with bucket method
- **HyperKZG** - commit(), verify(), verifyWithPairing() with batched pairing
- **Batch Verification** - BatchOpeningAccumulator for multiple openings
- **Host Execute** - Program execution with trace generation

### Partially Working
- **Dory** - commit() works, open() placeholder
- **Spartan** - Matrix ops work, verifier incomplete

## Next Steps (Future Iterations)

### High Priority
- [ ] Wire commitment proofs into JoltProof structure
- [ ] Implement batch opening verification for multiple polynomials

### Medium Priority
- [ ] Dory open() with proper inner product argument
- [ ] Import production SRS from Ethereum ceremony
- [ ] Performance optimization with SIMD

### Low Priority
- [ ] Parallel sumcheck round computation
- [ ] Documentation and examples

## Test Status
All 364 tests pass.
