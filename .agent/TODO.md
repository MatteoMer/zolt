# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 18)

### Commitment Type Infrastructure
- [x] Created `commitment_types.zig` with PolyCommitment wrapping G1 points
- [x] Added OpeningProof type for batch verification
- [x] Updated BytecodeProof to use PolyCommitment instead of field elements
- [x] Updated MemoryProof to use PolyCommitment with final_state_commitment
- [x] Updated RegisterProof to use PolyCommitment with final_state_commitment
- [x] Added init() and withCommitments() constructors for proofs

### Commitment Generation in Prover
- [x] Added ProvingKey struct with HyperKZG SRS
- [x] Added initWithKey() to JoltProver for commitment-enabled proving
- [x] Implemented commitBytecode() - commits bytecode polynomial
- [x] Implemented commitMemory() - commits memory trace polynomial
- [x] Implemented commitRegisters() - commits register trace polynomial
- [x] Prover generates real G1 commitments when ProvingKey is provided

## Completed (Previous Sessions)

### Iteration 17: HyperKZG Verification + Host Execute
- [x] Enhanced verifyWithPairing() with proper batching
- [x] Added verifyAlgebraic() for testing
- [x] Added host execute tests
- [x] Fixed batch verification return type

### Iterations 1-16: Core Infrastructure
- [x] Lookup table infrastructure (14 tables)
- [x] Lasso prover/verifier
- [x] Instruction proving with flags
- [x] Memory checking with RAF and Val Evaluation
- [x] Multi-stage prover (6 stages) and verifier
- [x] BN254 G1/G2 generators and pairing
- [x] HyperKZG SRS generation
- [x] Sumcheck protocol
- [x] RISC-V emulator
- [x] ELF loader
- [x] Fixed Projective point doubling
- [x] Fixed Fp6 non-residue ξ = 9 + u

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
- **PolyCommitment** - G1 point wrapper for proof commitments
- **ProvingKey** - SRS-based commitment generation
- **Prover Commitments** - Bytecode, memory, register polynomials committed

### Partially Working
- **Dory** - commit() works, open() placeholder
- **Spartan** - Matrix ops work, verifier incomplete

## Next Steps (Future Iterations)

### High Priority
- [ ] Implement batch opening proofs (prove multiple polynomial openings)
- [ ] Wire commitment verification into JoltVerifier
- [ ] Add VerifyingKey with SRS subset for verification

### Medium Priority
- [ ] Dory open() with proper inner product argument
- [ ] Import production SRS from Ethereum ceremony
- [ ] Performance optimization with SIMD

### Low Priority
- [ ] Parallel sumcheck round computation
- [ ] Documentation and examples

## Test Status
All 366 tests pass.
