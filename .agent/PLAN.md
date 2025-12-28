# Zolt zkVM Implementation Plan

## Current Status (December 2024 - Iteration 34)

### Session Summary - Sumcheck Degree Mismatch Fix

This iteration focused on fixing the sumcheck polynomial format mismatch between prover and verifier:

**Problem Identified:**
The Jolt protocol uses a compressed polynomial format for degree-2 sumchecks:
- Prover sends `[p(0), p(2)]` (evaluations at points 0 and 2)
- Verifier uses constraint `p(0) + p(1) = claim` to recover p(1)
- This saves 1 field element per round compared to sending all 3 evaluations

**Changes Made:**
1. Updated RAF prover (`raf_checking.zig`) to compute [p(0), p(2)]
2. Added `evaluateQuadraticAt3Points` helper for Lagrange interpolation
3. Updated Stage 2, 3, 5, 6 verifiers to use correct polynomial formats
4. Updated Stage 5, 6 provers to send [p(0), p(2)]
5. Fixed Stage 3 verifier to handle Lasso's coefficient form

**Polynomial Format Summary:**
- Stage 1 (Spartan): Degree 3, sends 4 coefficients
- Stage 2 (RAF): Degree 2, sends [p(0), p(2)]
- Stage 3 (Lasso): Degree 2, sends polynomial coefficients [c0, c1, c2]
- Stage 4 (Val): Degree 3, sends [p(0), p(1), p(2)]
- Stage 5 (Register): Degree 2, sends [p(0), p(2)]
- Stage 6 (Booleanity): Degree 2, sends [p(0), p(2)]

### Previous Session (Iteration 33) - Module Structure Improvements

Added claim_reductions and instruction_lookups module placeholders.

### Previous Session (Iteration 32) - Investigation & CLI Improvements

Fixed CLI error handling, investigated test interference issue.

### Previous Session (Iteration 31) - Strict Verification Mode

Added `VerifierConfig` with `strict_sumcheck` and `debug_output` options.

### Previous Session (Iteration 30) - Critical Fix

Fixed prover/verifier transcript synchronization for end-to-end verification.

## Architecture Summary

### Field Tower
```
Fp  = BN254 base field (254 bits) - G1 point coordinates
Fr  = BN254 scalar field (254 bits) - scalars for multiplication

Fp2 = Fp[u] / (u² + 1)
Fp6 = Fp2[v] / (v³ - ξ)  where ξ = 9 + u
Fp12 = Fp6[w] / (w² - v)
```

### Proof Structure
```
JoltProof:
  ├── bytecode_proof: Commitment to program bytecode
  ├── memory_proof: Memory access commitments
  ├── register_proof: Register file commitments
  ├── r1cs_proof: R1CS/Spartan proof
  └── stage_proofs: 6-stage sumcheck proofs
        ├── Stage 1: Outer Spartan (R1CS correctness)
        ├── Stage 2: RAM RAF evaluation
        ├── Stage 3: Lasso lookup (instruction lookups)
        ├── Stage 4: Value evaluation (memory consistency)
        ├── Stage 5: Register evaluation
        └── Stage 6: Booleanity (flag constraints)
```

### Commitment Schemes
```
HyperKZG (trusted setup)
  - commit(params, evals) -> Commitment
  - open(params, evals, point, value) -> Proof
  - verify(params, commitment, point, value, proof) -> bool
  - batchCommit, batchOpen, verifyBatchOpening

Dory (transparent setup, IPA-based)
  - setup(allocator, size) -> SetupParams
  - commit(params, evals) -> Commitment
  - open(params, evals, point, value, allocator) -> Proof
  - verify(params, commitment, point, value, proof) -> bool

SRS Utilities
  - generateMockSRS, loadFromRawBinary, serializeToRawBinary
  - loadFromPtau, loadFromPtauFile (snarkjs format)
```

### Lookup Tables (24 total)
```
Bitwise: And, Or, Xor, Andn
Comparison: Equal, NotEqual, UnsignedLessThan, SignedLessThan,
            UnsignedGreaterThanEqual, SignedGreaterThanEqual, UnsignedLessThanEqual
Arithmetic: RangeCheck, Sub, Movsign
Shifts: LeftShift, RightShift, RightShiftArithmetic, Pow2
Sign Extension: SignExtend8, SignExtend16, SignExtend32
Division: ValidDiv0, ValidUnsignedRemainder, ValidSignedRemainder
```

## Components Status

### Fully Working
- **BN254 Pairing** - Full Miller loop, final exponentiation, bilinearity verified
- **Extension Fields** - Fp2, Fp6, Fp12 with correct ξ = 9 + u
- **Field Arithmetic** - Montgomery form CIOS multiplication
- **Field Serialization** - Big-endian and little-endian I/O
- **G1/G2 Point Arithmetic** - Addition, doubling, scalar multiplication
- **Sumcheck Protocol** - Complete prover/verifier
- **RISC-V Emulator** - Full RV64IMC execution with tracing
- **ELF Loader** - Complete ELF32/ELF64 parsing
- **MSM** - Multi-scalar multiplication with bucket method
- **HyperKZG** - All operations including batch
- **Dory** - Full IPA-based commitment scheme
- **Host Execute** - Program execution with trace generation
- **Preprocessing** - Generates proving and verifying keys
- **Spartan** - Proof generation and verification
- **Lasso** - Lookup argument prover/verifier
- **Multi-stage Prover** - 6-stage sumcheck orchestration
- **All Lookup Tables** - 24 tables covering all RV64IM operations
- **Full Instruction Coverage** - 60+ instruction types
- **SRS Utilities** - PTAU file parsing, serialization
- **End-to-End Verification** - PASSES (lenient mode)

## Known Issues

### Test Interference (Iteration 32)
Adding new integration tests causes unrelated lasso/spartan tests to fail.
Need to investigate root cause before adding more e2e tests.

## Future Work

### High Priority
1. Investigate test interference issue
2. Enable strict_sumcheck mode by default

### Medium Priority
1. Performance optimization with SIMD
2. Parallel sumcheck round computation
3. Test with real Ethereum ceremony ptau files

### Low Priority
1. More comprehensive benchmarking
2. Add more example programs

## Performance Metrics (from benchmarks)
- Field addition: 4.0 ns/op
- Field multiplication: 55.5 ns/op
- Field inversion: 11.8 us/op
- MSM (256 points): 0.50 ms/op
- HyperKZG commit (1024): 1.5 ms/op

## Commit History (Iteration 34)
1. Fix sumcheck polynomial format mismatch between prover and verifier

## Commit History (Iteration 33)
1. Add claim_reductions and instruction_lookups module structure
2. Update README and tracking files for iteration 33

## Commit History (Iteration 32)
1. Update tracking files for iteration 32
2. Improve CLI error handling (no stack traces, clean exit codes)

## Commit History (Iteration 31)
1. Add configurable strict sumcheck verification mode
2. Improve sumcheck prover/verifier

## Commit History (Iteration 30)
1. Fix prover/verifier transcript synchronization for end-to-end verification

## Commit History (Iteration 29)
1. Add --help support for subcommands
2. Add full pipeline example
