# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 22)

### Complete Branch Instruction Lookups
- [x] Added BltLookup, BgeLookup for signed comparisons
- [x] Added BltuLookup, BgeuLookup for unsigned comparisons
- [x] Connected BLT/BGE/BLTU/BGEU to trace collector
- [x] Comprehensive tests for all branch comparison lookups

### Upper Immediate Instructions
- [x] Added LuiLookup for LUI (Load Upper Immediate)
- [x] Added AuipcLookup for AUIPC (Add Upper Immediate to PC)
- [x] Connected LUI/AUIPC to trace collector
- [x] Tests for upper immediate operations

### Jump Instructions
- [x] Added JalLookup for JAL (Jump and Link)
- [x] Added JalrLookup for JALR (Jump and Link Register)
- [x] Both support compressed instruction mode (PC+2 vs PC+4)
- [x] Connected JAL/JALR to trace collector
- [x] Tests for jump target computation and link address

## Completed (Previous Sessions)

### Iteration 21: Division/Remainder and RV64 Word Operations
- [x] ValidDiv0, ValidUnsignedRemainder, ValidSignedRemainder tables
- [x] DivLookup, DivuLookup, RemLookup, RemuLookup instruction lookups
- [x] AddwLookup, SubwLookup, SllwLookup, SrlwLookup, SrawLookup
- [x] MulwLookup, DivwLookup, DivuwLookup, RemwLookup, RemuwLookup
- [x] All W-suffix operations sign-extend 32-bit results to 64 bits

### Iteration 20: Shift Lookup Tables and Instructions
- [x] LeftShift, RightShift, RightShiftArithmetic tables
- [x] Pow2 table
- [x] SignExtend8, SignExtend16, SignExtend32 tables
- [x] SllLookup, SrlLookup, SraLookup, SlliLookup, SrliLookup, SraiLookup

### Iterations 1-19: Core Infrastructure
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

## Working Components

### Fully Working (24 Lookup Tables)
- **Bitwise**: And, Or, Xor, Andn
- **Comparison**: Equal, NotEqual, UnsignedLessThan, SignedLessThan, UnsignedGreaterThanEqual, SignedGreaterThanEqual, UnsignedLessThanEqual
- **Arithmetic**: RangeCheck, Sub, Movsign
- **Shifts**: LeftShift, RightShift, RightShiftArithmetic, Pow2
- **Sign Extension**: SignExtend8, SignExtend16, SignExtend32
- **Division**: ValidDiv0, ValidUnsignedRemainder, ValidSignedRemainder

### Full Instruction Coverage (RV64IM)
- **Base Integer (I)**: ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU
- **Immediate (I)**: ADDI, ANDI, ORI, XORI, SLTI, SLTIU, SLLI, SRLI, SRAI
- **Branches**: BEQ, BNE, BLT, BGE, BLTU, BGEU
- **Upper Immediate**: LUI, AUIPC
- **Jumps**: JAL, JALR
- **Multiply (M)**: MUL, MULH, MULHU, MULHSU
- **Division (M)**: DIV, DIVU, REM, REMU
- **Word-sized (RV64)**: ADDW, SUBW, SLLW, SRLW, SRAW, MULW, DIVW, DIVUW, REMW, REMUW

## Next Steps (Future Iterations)

### High Priority
- [ ] Import production SRS from Ethereum ceremony
- [ ] Implement proper verifier for bytecode/memory/registers proofs

### Medium Priority
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation

### Low Priority
- [ ] Documentation and examples
- [ ] Benchmarking suite

## Test Status
All tests pass (450+ tests).

## Commits This Session
1. Add missing branch instruction lookups (BLT, BGE, BLTU, BGEU)
2. Add LUI and AUIPC instruction lookups
3. Add JAL and JALR jump instruction lookups
