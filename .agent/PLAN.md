# Plan: W-Extension Instruction Decomposition for Vanilla Jolt Compatibility

## Problem
Zolt treats W-extension instructions (ADDIW, ADDW, SUBW, etc.) as single first-class
instructions. Vanilla Jolt decomposes them into virtual sequences before proving.
When Jolt's verifier loads Zolt's preprocessing bytecode and encounters ADDIW,
it panics with "Unexpected instruction: ADDIW" because ADDIW is not in
Jolt's `define_rv32im_trait_impls!` macro.

## Scope: fibonacci.elf only uses ADDIW (4x) and ADDW (2x)

### ADDIW decomposition → ADDI + VirtualSignExtendWord (2 cycles)
### ADDW decomposition → ADD + VirtualSignExtendWord (2 cycles)

## Changes Required

### 1. Preprocessing (src/zkvm/preprocessing.zig)
- Add VirtualSignExtendWord to InstructionVariant enum
- Modify decodeToJoltInstruction to return multiple instructions
- For ADDIW: emit ADDI(rd, rs1, imm) + VirtualSignExtendWord(rd, rd, 0)
- For ADDW: emit ADD(rd, rs1, rs2) + VirtualSignExtendWord(rd, rd, 0)
- Set virtual_sequence_remaining and is_first_in_sequence properly

### 2. Tracer (src/tracer/mod.zig)
- For ADDIW: generate 2 trace steps
  - Step 1: full 64-bit add (no truncation), write to rd
  - Step 2: sign-extend rd[31:0], write to rd
- For ADDW: generate 2 trace steps
  - Step 1: full 64-bit add, write to rd
  - Step 2: sign-extend rd[31:0], write to rd
- Both steps need virtual_sequence_remaining set

### 3. Lookup Trace (src/zkvm/instruction/lookup_trace.zig)
- Add VirtualSignExtendWord entry constructor
  - lookup table: index 21 (SignExtendHalfWord)
  - operands: (rs1_val, 0)
  - index: rs1_val as u128 (just the input value)
  - result: sign-extended value
  - circuit_flags: AddOperands, WriteLookupOutputToRD, VirtualInstruction, DoNotUpdateUnexpandedPC
  - instruction_flags: LeftOperandIsRs1Value

### 4. R1CS Witness (src/zkvm/r1cs/constraints.zig)
- Virtual sequence steps need proper flags in witness

### 5. Stage5 Prover (src/zkvm/spartan/stage5_prover.zig)
- getLookupTableIndex must handle VirtualSignExtendWord → table index 21
- Need synthetic instruction encoding for virtual instructions

### 6. SignExtendHalfWord MLE (src/zkvm/lookup_table/mod.zig)
- Implement table index 21 MLE (currently returns F.zero())
- Formula: lower_half + upper_half * 2^(XLEN/2), where upper_half = sign_bit * Σ 2^i

## Implementation Order
1. Add VirtualSignExtendWord to preprocessing enum + bytecode decomposition
2. Modify tracer for 2-cycle execution
3. Add VirtualSignExtendWord lookup entry
4. Update R1CS witnesses
5. Implement SignExtendHalfWord MLE
6. Update getLookupTableIndex
7. Test
