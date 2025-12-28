# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 24)

### SRS Loading Utilities
- [x] Add src/poly/commitment/srs.zig with SRS management:
  - Parse G1/G2 points from uncompressed/compressed formats
  - Load SRS from raw binary format
  - Serialize SRS to raw binary format
  - Generate mock SRS for testing
- [x] Add fromBytesBE()/toBytesBE() to BN254BaseField (MontgomeryField)
- [x] Add fromBytesBE()/toBytesBE() to BN254Scalar
- [x] Add isOnCurve() to AffinePoint

### Integration Tests
- [x] e2e: SRS generation and commitment
- [x] e2e: SRS serialization and deserialization
- [x] e2e: field element big-endian round-trip

### Commitment Verification
- [x] Add verifyCommitmentOpening() to JoltVerifier
- [x] Implement HyperKZG pairing-based verification
- [x] Handle edge cases (no key, empty commitment, constant polynomial)
- [x] Add tests for commitment verification

## Completed (Previous Sessions)

### Iteration 23: Load/Store and Verifier Improvements
- [x] LoadAddressLookup, StoreAddressLookup for memory operations
- [x] All load instructions: LbLookup, LbuLookup, LhLookup, LhuLookup, LwLookup, LwuLookup, LdLookup
- [x] All store instructions: SbLookup, ShLookup, SwLookup, SdLookup
- [x] RV64I immediate word operations: AddiwLookup, SlliwLookup, SrliwLookup, SraiwLookup
- [x] Enhanced verifier with proper transcript binding

### Iterations 1-22: Core Infrastructure
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

### High Priority
- [ ] Import production SRS from Ethereum ceremony (ptau format parsing)

### Medium Priority
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation

### Low Priority
- [ ] Documentation and examples
- [ ] Benchmarking suite

## Test Status
All tests pass (526 tests).

## Commits This Session
1. Add SRS loading utilities for production trusted setups
2. Add big-endian serialization to BN254Scalar and integration tests
3. Add commitment opening verification to JoltVerifier
