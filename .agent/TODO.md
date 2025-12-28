# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 34)

### Sumcheck Degree Mismatch Fix
- [x] Investigate sumcheck degree mismatch between prover and verifier
- [x] Fix RAF prover to compute [p(0), p(2)] for degree-2 compressed format
- [x] Add `evaluateQuadraticAt3Points` helper for Lagrange interpolation
- [x] Update Stage 2 verifier to use quadratic interpolation with recovered p(1)
- [x] Update Stage 3 verifier to handle Lasso's coefficient form polynomials
- [x] Update Stage 5 verifier to use degree-2 compressed format
- [x] Update Stage 6 verifier to use degree-2 compressed format
- [x] Update Stage 5 prover to send [p(0), p(2)]
- [x] Update Stage 6 prover to send [p(0), p(2)]
- [x] Verify all tests pass
- [x] Verify full pipeline example still works
- [x] Add tests for evaluateQuadraticAt3Points helper (4 new tests)

## Key Insight from This Session

The Jolt protocol uses a **compressed polynomial format** for degree-2 sumchecks:
- Prover sends `[p(0), p(2)]` (evaluations at 0 and 2)
- Verifier uses sumcheck constraint `p(0) + p(1) = claim` to recover p(1)
- Verifier then uses quadratic Lagrange interpolation to evaluate at challenge

This saves 1 field element per round compared to sending all 3 evaluations.

Different stages use different formats:
- **Stage 1 (Spartan)**: Degree 3, sends 4 coefficients
- **Stage 2 (RAF)**: Degree 2, sends [p(0), p(2)]
- **Stage 3 (Lasso)**: Degree 2, sends polynomial coefficients [c0, c1, c2]
- **Stage 4 (Val)**: Degree 3, sends [p(0), p(1), p(2)]
- **Stage 5 (Register)**: Degree 2, sends [p(0), p(2)]
- **Stage 6 (Booleanity)**: Degree 2, sends [p(0), p(2)]

## Known Issues (For Future Iterations)

### Test Interference Issue (Iteration 32)
When adding new integration tests to `src/integration_tests.zig`, seemingly unrelated tests start failing.
**Workaround**: Do not add new e2e integration tests until root cause is found.

## Completed (Previous Sessions)

### Iteration 33 - Module Structure Improvements
- [x] Add claim_reductions module with placeholder types
- [x] Add instruction_lookups module with placeholder types
- [x] Update zkvm/mod.zig to export new modules
- [x] Update README.md with current project structure

### Iteration 32 - Investigation & CLI Improvements
- [x] Improved error handling in CLI to remove stack traces
- [x] Errors now show clean error name and exit with code 1

### Iteration 31 - Strict Verification Mode
- [x] Add `VerifierConfig` struct with `strict_sumcheck` and `debug_output` options

### Iterations 1-30 - Core Implementation
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

### High Priority
- [ ] Investigate test interference issue (see Known Issues)
- [ ] Enable strict_sumcheck mode by default once prover fixes are complete

### Medium Priority
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation
- [ ] Download and test with real Ethereum ceremony ptau files

### Low Priority
- [ ] More comprehensive benchmarking
- [ ] Add more example programs

## Test Status
All tests pass (554 tests).
End-to-end verification: PASSED (lenient mode)
Full pipeline example: WORKING

## Performance (from benchmarks)
- Field addition: 4.0 ns/op
- Field multiplication: 55.5 ns/op
- Field inversion: 11.8 us/op
- MSM (256 points): 0.50 ms/op
- HyperKZG commit (1024): 1.5 ms/op
