# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-12-27

Initial port of Jolt zkVM from Rust to Zig.

### Added

#### Core Field Arithmetic
- BN254 scalar field with full Montgomery form implementation
- CIOS (Coarsely Integrated Operand Scanning) multiplication algorithm
- Optimized squaring using Karatsuba-like technique (~25% fewer multiplications)
- Batch operations: add, multiply, inverse (Montgomery's trick), inner product
- Horner's method for polynomial evaluation
- Extension fields: Fp2, Fp6, Fp12 tower for BN254 pairings
- SIMD-optimized batch operations using Zig vectors (`@Vector`)

#### Polynomial Infrastructure
- Dense multilinear polynomials with boolean hypercube representation
- Equality polynomials for sumcheck protocol
- Univariate polynomials with coefficient representation

#### Commitment Schemes
- HyperKZG for multilinear polynomial commitment
- Dory transparent setup scheme
- Mock commitment scheme for testing
- G2 point operations for pairing verification

#### Subprotocols
- Complete sumcheck prover with round generation
- Sumcheck verifier for proof checking

#### RISC-V Virtual Machine
- Full RV64I base instruction set
- M extension (multiply/divide operations with edge cases)
- C extension (compressed 16-bit instructions with expansion)
- Instruction decoder with opcode classification
- Memory checking infrastructure (RAM state)
- Register file with read/write operations

#### Proving System
- R1CS constraint system with sparse matrix representation
- Spartan prover with sumcheck integration
- Spartan verifier
- Fiat-Shamir transcript using Keccak-f[1600] permutation

#### Multi-Scalar Multiplication
- Affine and projective point representations
- Point addition and doubling
- Pippenger's bucket method with optimal window selection
- Parallel MSM execution using std.Thread

#### Host Interface
- Complete ELF32/ELF64 parser
- Program loading and segment extraction
- Memory layout configuration

#### Tracer
- RISC-V emulator with full instruction execution
- Execution tracing with cycle counting
- Witness generation from execution trace

#### Utilities
- Error types and handling
- Math utilities (log2, ceiling division, etc.)
- Proof serialization/deserialization with versioning

#### Pairing Operations
- Miller loop with optimal ate pairing
- NAF (Non-Adjacent Form) representation for efficiency
- Final exponentiation with BN254 hard part
- Frobenius endomorphism on extension fields

#### Examples
- `field_arithmetic.zig`: Demonstrates field operations and batch processing
- `simple_proof.zig`: Shows polynomial commitment and Fiat-Shamir usage
- `risc_v_emulation.zig`: RISC-V instruction decoding examples

### Architecture Decisions

- **Memory Management**: Explicit allocators throughout (Zig idiom)
- **Generics**: Comptime generics instead of Rust traits
- **Error Handling**: Error unions (`T!E`) instead of `Result<T,E>`
- **Parallelism**: `std.Thread` instead of rayon
- **Serialization**: Custom binary format instead of serde

### Compatibility

- Requires Zig 0.15.0 or later
- Uses `std.ArrayListUnmanaged` pattern for collections
- All 193 tests passing

### Not Yet Implemented

- GPU acceleration hooks
- Performance benchmarks vs Rust (basic benchmarks exist in bench.zig)

## Comparison with Rust Jolt

| Component | Rust | Zig |
|-----------|------|-----|
| Field | `ark_bn254::Fr` | `BN254Scalar` |
| Polynomials | `DensePolynomial` | `DensePolynomial(F)` |
| Commitments | `HyperKZG`, `Dory` | `HyperKZG(F)`, `Dory(F)` |
| Spartan | `R1CSSatisfied` | `SpartanProver(F)` |
| Sumcheck | `SumcheckProof` | `Sumcheck(F).Proof` |
| Transcripts | `Keccak256Transcript` | `Transcript(F)` |
| RISC-V | RV64IMC | RV64IMC |
| Serialization | serde | `ProofSerializer(F)` |
| Witness | `JoltWitness` | `WitnessGenerator(F)` |
