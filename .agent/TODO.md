# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 30)

### Critical Bug Fix: Prover/Verifier Transcript Synchronization
- [x] Fix prover to generate commitments BEFORE sumcheck proving
- [x] Prover now absorbs commitments into transcript to bind challenges
- [x] Verifier stages now generate matching pre-challenges
- [x] End-to-end verification now PASSES

### The Core Fix
The issue was that the prover and verifier transcripts were diverging:
1. Prover was running sumcheck first, then generating commitments
2. Verifier was absorbing commitments, then running sumcheck verification

The fix ensures both sides:
1. Absorb public inputs first
2. Generate/absorb commitments in the same order
3. Generate stage-specific challenges in matching sequence

## Completed (Previous Sessions)

### CLI Improvements (Iteration 29)
- [x] Add --help support for subcommands (run, prove, srs, decode)
- [x] Each subcommand now shows usage information when passed --help or -h

### Examples (Iteration 29)
- [x] Add full pipeline example (end-to-end ZK proving workflow)
- [x] Demonstrates preprocessing, proving, and verification

### CLI and API Improvements (Iteration 28)
- [x] Upgrade prove command to actually call prover.prove() and verifier.verify()
- [x] Add timing for each proving step (preprocess, init, prove, verify)
- [x] Show proof summary with commitment status
- [x] Fix toBytes/fromBytes to use toBytesBE/fromBytesBE for consistency
- [x] Fix R1CS proof verification to use eval_claims instead of missing fields
- [x] Add 'srs' command to inspect PTAU ceremony files
- [x] Add preprocessWithSRS() for loading external SRS data

### Examples and Documentation (Iteration 27)
- [x] Add HyperKZG commitment example (hyperkzg_commitment.zig)
- [x] Add sumcheck protocol example (sumcheck_protocol.zig)
- [x] Update build.zig with example-hyperkzg and example-sumcheck targets
- [x] Update README with example usage instructions
- [x] Add BN254Scalar.toU64() helper for debugging
- [x] Fix all 5 examples to match current API

### Core Infrastructure (Iterations 1-26)
- [x] BN254 field and curve arithmetic
- [x] Extension fields (Fp2, Fp6, Fp12)
- [x] Pairing with Miller loop and final exponentiation
- [x] HyperKZG commit, open, verify with batch support
- [x] Dory commit, open, verify with IPA
- [x] Sumcheck protocol
- [x] RISC-V emulator (RV64IMC)
- [x] ELF loader
- [x] MSM operations
- [x] Spartan proof generation and verification
- [x] Lasso lookup argument prover/verifier
- [x] Multi-stage prover (6 stages)
- [x] Host execute
- [x] Preprocessing
- [x] Complete RV64IM instruction coverage (60+ instructions)
- [x] 24 lookup tables
- [x] SRS utilities and PTAU file parsing
- [x] Benchmarking suite

## Working Components

### Lookup Tables (24 total)
- **Bitwise**: And, Or, Xor, Andn
- **Comparison**: Equal, NotEqual, UnsignedLessThan, SignedLessThan, UnsignedGreaterThanEqual, SignedGreaterThanEqual, UnsignedLessThanEqual
- **Arithmetic**: RangeCheck, Sub, Movsign
- **Shifts**: LeftShift, RightShift, RightShiftArithmetic, Pow2
- **Sign Extension**: SignExtend8, SignExtend16, SignExtend32
- **Division**: ValidDiv0, ValidUnsignedRemainder, ValidSignedRemainder

### Complete RV64IM Instruction Coverage
- **Base Integer (I)**: ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU
- **Immediate (I)**: ADDI, ANDI, ORI, XORI, SLTI, SLTIU, SLLI, SRLI, SRAI
- **Immediate Word (RV64I)**: ADDIW, SLLIW, SRLIW, SRAIW
- **Branches**: BEQ, BNE, BLT, BGE, BLTU, BGEU
- **Upper Immediate**: LUI, AUIPC
- **Jumps**: JAL, JALR
- **Loads**: LB, LBU, LH, LHU, LW, LWU, LD
- **Stores**: SB, SH, SW, SD
- **Multiply (M)**: MUL, MULH, MULHU, MULHSU
- **Division (M)**: DIV, DIVU, REM, REMU
- **Word-sized (RV64)**: ADDW, SUBW, SLLW, SRLW, SRAW, MULW, DIVW, DIVUW, REMW, REMUW

## Next Steps (Future Iterations)

### Medium Priority
- [ ] Implement strict sumcheck verification (currently structural only)
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation
- [ ] Download and test with real Ethereum ceremony ptau files

### Low Priority
- [ ] More comprehensive benchmarking
- [ ] Add more example programs

## Test Status
All tests pass (538 tests).
End-to-end verification: PASSED

## Commits This Session (Iteration 30)
1. Fix prover/verifier transcript synchronization for end-to-end verification

