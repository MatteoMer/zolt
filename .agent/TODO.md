# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 29)

### CLI Improvements
- [x] Add --help support for subcommands (run, prove, srs, decode)
- [x] Each subcommand now shows usage information when passed --help or -h

## Completed (Iteration 28)

### CLI and API Improvements
- [x] Upgrade prove command to actually call prover.prove() and verifier.verify()
- [x] Add timing for each proving step (preprocess, init, prove, verify)
- [x] Show proof summary with commitment status
- [x] Fix toBytes/fromBytes to use toBytesBE/fromBytesBE for consistency
- [x] Fix R1CS proof verification to use eval_claims instead of missing fields
- [x] Add 'srs' command to inspect PTAU ceremony files
- [x] Add preprocessWithSRS() for loading external SRS data

## Completed (Iteration 27)

### Examples and Documentation
- [x] Add HyperKZG commitment example (hyperkzg_commitment.zig)
- [x] Add sumcheck protocol example (sumcheck_protocol.zig)
- [x] Update build.zig with example-hyperkzg and example-sumcheck targets
- [x] Update README with example usage instructions
- [x] Add BN254Scalar.toU64() helper for debugging
- [x] Fix all 5 examples to match current API

## Completed (Iteration 26)

### CLI Improvements
- [x] Add 'prove' command to CLI (experimental)
- [x] Demonstrate full proving pipeline structure
- [x] Load ELF → Preprocess → Execute → Initialize Prover

### Cleanup and Benchmark Fixes
- [x] Clean up outdated TODO comments
- [x] Fix benchmark suite to compile with Zig 0.15.2
- [x] Use volatile pointer pattern to prevent optimizer interference
- [x] Fix MSM benchmark to use correct type instantiation
- [x] Add HyperKZG commitment benchmark

### Benchmark Results (M1 Mac)
- Field multiplication: 51.5 ns/op
- Field inversion: 13.3 us/op
- Batch inverse (1024): 70.7 us/op
- MSM (256 points): 0.49 ms/op
- HyperKZG commit (1024): 1.5 ms/op

## Completed (Iteration 25)

### PTAU File Format Parser
- [x] Implement snarkjs PTAU file format parser:
  - Magic bytes and version validation
  - Header section parsing (field size, prime, power, ceremony power)
  - Section type enumeration (tauG1, tauG2, alphaTauG1, betaTauG1, betaG2)
- [x] Add little-endian G1/G2 point parsing (snarkjs Montgomery format)
- [x] Create ExtendedSRSData structure for Groth16 SRS data
- [x] Add conversion from ExtendedSRSData to basic SRSData
- [x] Add tests for PTAU parsing and conversion

## Completed (Iteration 24)

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

### Medium Priority
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation
- [ ] Download and test with real Ethereum ceremony ptau files

### Low Priority
- [x] Documentation and examples (added HyperKZG and Sumcheck examples)
- [ ] More comprehensive benchmarking

## Test Status
All tests pass (538 tests).

## Commits This Session (Iteration 29)
1. Add --help support for subcommands

## Commits (Iteration 28)
1. Upgrade prove command to actually generate and verify proofs
2. Add SRS inspection command and preprocessWithSRS method
3. Update README with srs command documentation

## Commits (Iteration 27)
1. Add HyperKZG and Sumcheck protocol examples
2. Fix examples to match current API and add toU64 helper

## Commits (Iteration 26)
1. Clean up outdated TODO comments
2. Fix benchmark to compile with Zig 0.15.2
3. Update tracking files for iteration 26
4. Add 'prove' command to CLI (experimental)
5. Update tracking files with prove command addition
6. Update README with current test count and prove command
