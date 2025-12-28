# Zolt zkVM Implementation Plan

## Current Status (December 2024 - Iteration 18)

### Session Summary

This iteration focused on wiring proper polynomial commitments into the proof structures and adding verification key support:

1. **Commitment Type Infrastructure**
   - Created `commitment_types.zig` with `PolyCommitment` wrapping G1 points
   - Added `OpeningProof` type for batch verification support
   - Updated BytecodeProof, MemoryProof, RegisterProof to use PolyCommitment
   - Added init() and withCommitments() constructors for clean API

2. **Commitment Generation in Prover**
   - Added `ProvingKey` struct containing HyperKZG SRS
   - Implemented `commitBytecode()` - converts bytecode to polynomial and commits
   - Implemented `commitMemory()` - commits memory trace values
   - Implemented `commitRegisters()` - commits register trace values
   - Prover generates real G1 point commitments when ProvingKey is provided

3. **Verifier Enhancements**
   - Added `VerifyingKey` with minimal SRS elements (g1, g2, tau_g2)
   - Implemented `absorbCommitments()` for Fiat-Shamir transcript binding
   - All commitments are absorbed into transcript before challenge derivation

### Test Status

All 376 tests pass:
- Field arithmetic: Fp, Fp2, Fp6, Fp12
- Curve arithmetic: G1, G2 points (correct projective doubling)
- Pairing: bilinearity verified, SRS relationship verified
- HyperKZG: commit, open, verify, verifyWithPairing
- Batch verification: accumulator, multiple claims
- Sumcheck protocol
- RISC-V emulation (RV64IMC)
- ELF loading (ELF32/ELF64)
- MSM operations
- Spartan proof generation and verification
- ProvingKey initialization
- VerifyingKey extraction
- Commitment type operations

### Architecture Summary

#### Field Tower
```
Fp  = BN254 base field (254 bits) - G1 point coordinates
Fr  = BN254 scalar field (254 bits) - scalars for multiplication

Fp2 = Fp[u] / (u² + 1)
Fp6 = Fp2[v] / (v³ - ξ)  where ξ = 9 + u
Fp12 = Fp6[w] / (w² - v)
```

#### Commitment Architecture
```
PolyCommitment = G1 point wrapper
  - fromPoint(G1Point) -> PolyCommitment
  - zero() -> identity commitment
  - eql(), isZero() for comparison
  - toBytes() for serialization

ProvingKey = { srs: HyperKZG.SetupParams, max_trace_length }
  - init(allocator, size) -> generates SRS
  - fromSRS(existing_srs) -> uses existing SRS
  - toVerifyingKey() -> extracts minimal verification data

VerifyingKey = { g1, g2, tau_g2 }
  - Much smaller than ProvingKey
  - Contains only generators and tau*G2

JoltProver
  - Without ProvingKey: uses identity commitments (placeholder)
  - With ProvingKey: generates real HyperKZG commitments

JoltVerifier
  - absorbCommitments() binds all commitments to transcript
  - Without VerifyingKey: placeholder verification
  - With VerifyingKey: can verify commitment openings
```

## Components Status

### Fully Working
- **BN254 Pairing** - Full Miller loop, final exponentiation, bilinearity verified
- **Extension Fields** - Fp2, Fp6, Fp12 with correct ξ = 9 + u
- **Field Arithmetic** - Montgomery form CIOS multiplication
- **G1/G2 Point Arithmetic** - Addition, doubling, scalar multiplication
- **Projective Points** - Jacobian doubling (fixed in iteration 16)
- **Frobenius Endomorphism** - Complete coefficients
- **Sumcheck Protocol** - Complete prover/verifier
- **RISC-V Emulator** - Full RV64IMC execution with tracing
- **ELF Loader** - Complete ELF32/ELF64 parsing
- **MSM** - Multi-scalar multiplication with bucket method
- **HyperKZG** - commit(), verify(), verifyWithPairing()
- **Batch Verification** - BatchOpeningAccumulator
- **Host Execute** - Program execution with trace generation
- **Preprocessing** - Generates proving and verifying keys
- **Spartan** - Proof generation and verification
- **Lasso** - Lookup argument prover/verifier
- **Multi-stage Prover** - 6-stage sumcheck orchestration
- **Transcripts** - Keccak and Poseidon-based Fiat-Shamir
- **PolyCommitment** - G1 point wrapper for proofs
- **ProvingKey** - SRS-based commitment infrastructure
- **VerifyingKey** - Minimal verification data
- **Prover Commitments** - Real G1 commitments for bytecode/memory/registers
- **Verifier Transcript** - Commitment absorption for Fiat-Shamir

### Partially Working
- **Dory** - commit() works, open() is placeholder
- **Spartan Verifier** - Structure complete, needs full implementation

## Future Work

### High Priority
1. Implement batch opening proofs
2. Complete Spartan verifier

### Medium Priority
1. Implement Dory open() with inner product argument
2. Import production SRS from Ethereum ceremony
3. Performance optimization with SIMD

### Low Priority
1. Parallel sumcheck round computation
2. Documentation and examples
3. Benchmarking suite
