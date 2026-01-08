# Zolt-Jolt Compatibility - Current Status

## Summary

**Stage 1**: PASSES (outer sumcheck) ✓
**Stage 2**: PASSES (product virtualization + RAM RAF) ✓
**Stage 3**: FAILS - Zero polynomials produce wrong output claim
**Stage 4-7**: Untested (blocked on Stage 3)

## Session Progress

### Completed
1. ✓ Added EqPlusOnePolynomial to poly/mod.zig
2. ✓ Added static EqPolynomial.mle() method
3. ✓ Created stage3_prover.zig framework
4. ✓ Analyzed Stage 3 expected_output_claim formulas

### Stage 3 Architecture Understanding

Stage 3 is a batched sumcheck with 3 instances:

1. **ShiftSumcheck** (degree 2, n_cycle_vars rounds)
   - Proves: `Σ_j eq+1(r_outer, j) * (upc + γ*pc + γ²*virt + γ³*first) + γ⁴*(1-noop) * eq+1(r_prod, j)`
   - Opening claims at `SumcheckId::SpartanShift`:
     - UnexpandedPC, PC, OpFlags(VirtualInstruction), OpFlags(IsFirstInSequence), InstructionFlags(IsNoop)

2. **InstructionInputSumcheck** (degree 3, n_cycle_vars rounds)
   - Proves: `(eq(r, r_stage1) + γ²*eq(r, r_stage2)) * (right + γ*left)`
   - Where:
     - `left = left_is_rs1 * rs1_value + left_is_pc * unexpanded_pc`
     - `right = right_is_rs2 * rs2_value + right_is_imm * imm`
   - Opening claims at `SumcheckId::InstructionInputVirtualization`:
     - InstructionFlags(LeftOperandIsRs1Value, LeftOperandIsPC, RightOperandIsRs2Value, RightOperandIsImm)
     - Rs1Value, Rs2Value, UnexpandedPC, Imm

3. **RegistersClaimReduction** (degree 2, n_cycle_vars rounds)
   - Proves: `eq(r, r_spartan) * (rd + γ*rs1 + γ²*rs2)`
   - Opening claims at `SumcheckId::RegistersClaimReduction`:
     - RdWriteValue, Rs1Value, Rs2Value

### Remaining Work for Stage 3

1. **InstructionFlags per cycle**: Need to compute LeftOperandIsRs1Value, etc. from instruction opcode
   - These are stored in Jolt's `InstructionFlags` enum
   - Our `InstructionFlags` enum exists but isn't stored per cycle in R1CSCycleInputs
   - Solution: Compute from instruction opcode in buildInstructionInputMLEs()

2. **Proper round polynomial computation**:
   - Current implementation is simplified (doesn't use eq/eq+1 properly)
   - Need to sum over hypercube with correct weighting

3. **Transcript flow matching**:
   - Verify the exact order matches Jolt's verifier
   - 5 gamma powers (shift) + 1 gamma (instr) + 1 gamma (regs)
   - 3 input claims → 3 batching coeffs
   - Rounds: compressed poly + challenge
   - 16 opening claims (5+8+3)

### Files Modified

- `src/poly/mod.zig`: Added EqPlusOnePolynomial, EqPolynomial.mle()
- `src/zkvm/spartan/mod.zig`: Export stage3_prover
- `src/zkvm/spartan/stage3_prover.zig`: New Stage 3 prover framework

### Testing Commands

```bash
# Build Zolt
zig build

# Run tests (714 should pass)
zig build test --summary all

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```

## Next Steps

1. [ ] Add instruction flags storage to R1CSCycleInputs or compute per-cycle
2. [ ] Implement proper round polynomial computation with eq/eq+1 weighting
3. [ ] Integrate Stage 3 prover into proof_converter
4. [ ] Test Stage 3 verification
5. [ ] Implement Stages 4-7 (similar pattern)

## Commits This Session

- fdb7698: feat: Add EqPlusOnePolynomial and Stage 3 prover framework
