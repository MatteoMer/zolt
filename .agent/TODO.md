# Zolt-Jolt Compatibility TODO

## Current Status: Stage 3 InstructionInput Mismatch

### Completed (This Session)
1. **Stage 1** - PASSES verification
2. **Stage 2** - PASSES verification
3. **Stage 3 Shift sumcheck** - shift_match = true (final claim correct)
4. **eq+1 polynomial** - eq+1_outer match = true, eq+1_prod match = true
5. **Fixed computeInstructionInputs** - Returns 0 for ECALL, FENCE, and unknown opcodes

### Current Issue: InstructionInput/Registers Claim Mismatch

**Root Cause Identified:**
The InstructionInput prover computes right operand as:
```
right = right_is_rs2 * rs2_value + right_is_imm * imm
```

But for 10/256 cycles, this formula doesn't equal `RightInstructionInput` stored in R1CS witness.

**Mismatching Cycles:**
- Cycle 1: computed=0, witness=1
- Cycles 8, 13, 18, 23: computed=0, witness=0x43E1F593F0000000 (large value, likely address)
- And 5 more...

**Why It Matters:**
The eq-weighted sum `Σ eq(r, i) * right_computed[i]` must equal the opening claim
`RightInstructionInput @ r`. If individual values differ, the sum won't match.

### Investigation Findings
1. Stage 3 debug shows `shift_match = true` - Shift sumcheck works correctly
2. After all rounds, `left_match = true, right_match = true` (using sumcheck challenges)
3. At initialization, `right_match = false` because eq-weighted sum at r_outer differs
4. The flags (`FlagRightOperandIsRs2`, `FlagRightOperandIsImm`) may not be set consistently
   with what `computeInstructionInputs` returns for the `RightInstructionInput` value

### Next Steps
1. [ ] Add debug to trace what instruction types are at mismatching cycles
2. [ ] Ensure setFlagsFromInstruction flags match computeInstructionInputs values
3. [ ] Consider: are cycles 8,13,18,23 virtual instruction sequences?
4. [ ] Fix all mismatches so that `right_is_rs2*rs2 + right_is_imm*imm == RightInstructionInput`

### Pending
- [ ] Fix Registers sumcheck claim (reg_match = false)
- [ ] Achieve sum_equals_claim = true
- [ ] Pass Stage 3 verification

## Progress Summary

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASSES | Using ZOLT preprocessing |
| 2 | ✅ PASSES | |
| 3 | ❌ FAILS | instr_match=false, reg_match=false |
| 4-7 | Blocked | |

## Key Files
- `src/zkvm/spartan/stage3_prover.zig` - Stage 3 prover with InstructionInput
- `src/zkvm/r1cs/constraints.zig` - computeInstructionInputs, setFlagsFromInstruction
- Jolt: `jolt-core/src/zkvm/spartan/instruction_input.rs` - Reference implementation

## Testing
```bash
bash scripts/build_verify.sh
```
