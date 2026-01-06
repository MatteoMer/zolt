# Zolt-Jolt Compatibility TODO

## Current Status: Session 60 - January 6, 2026

**STATUS: Instruction input semantics fixed. Opening claims still mismatch.**

### Progress This Session

1. ✅ Fixed instruction input semantics (`computeInstructionInputs`)
   - Now correctly maps left/right instruction inputs per instruction type
   - ADD: left=rs1, right=rs2
   - ADDI: left=rs1, right=imm
   - JAL: left=PC, right=imm
   - AUIPC: left=PC, right=imm
   - LUI: left=0, right=imm
   - etc.

2. ✅ Fixed updateClaim to use scaled evaluations
   - The verifier uses eval_from_hint with SCALED coefficients and hint
   - Prover now tracks scaled claims for consistency

3. ✅ All sumcheck challenges match (verified all 11 challenges byte-by-byte)

4. ⏳ Opening claims (r1cs_input_evals) still don't match
   - Witness values look correct for first instruction (AUIPC: left=PC, right=imm)
   - EqPolynomial evaluation should be correct
   - Need to verify the exact evaluation matches Jolt's

### Current Issue

Zolt r1cs_input_evals[0] ≠ Jolt r1cs_input_evals[0]

Both sides use:
```
r1cs_input_evals[i] = Σ_t eq(r_cycle, t) * witness[t].values[i]
```

Possible causes:
1. EqPolynomial::evals indexing differs
2. Cycle index mapping differs
3. Field element byte ordering differs during evaluation
4. r_cycle challenges are in different order

### Debug Info

Witness[0] values (first cycle):
- LeftInstructionInput = PC = 0x80000000 ✓
- RightInstructionInput = imm = 0x1000 ✓
- Product = PC * imm (computed in field)
- PC = 0x80000000 ✓

r_cycle ordering:
- r_cycle.len = 10 (cycle challenges)
- r_cycle[0] = challenge[10] reversed (last cycle challenge)
- r_cycle[last] = challenge[1] reversed (first cycle challenge)
- This matches Jolt's BIG_ENDIAN conversion

### Next Steps

1. **Add debug output to compare eq_evals[0..5]** - Verify EqPolynomial produces same values as Jolt
2. **Check Jolt's accumulator semantics** - How are signed/unsigned values handled?
3. **Verify field element representation** - Are values stored the same way?
4. **Test with simpler trace** - Maybe 4 cycles instead of 1024

---

## Files Involved

- `src/zkvm/r1cs/constraints.zig:fromTraceStep` - R1CS witness generation (FIXED)
- `src/zkvm/r1cs/constraints.zig:computeInstructionInputs` - Instruction input semantics (NEW)
- `src/zkvm/r1cs/evaluation.zig:computeClaimedInputs` - MLE evaluation
- `src/poly/mod.zig:EqPolynomial::evals` - Eq polynomial evaluation

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
