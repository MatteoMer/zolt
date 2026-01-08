# Zolt-Jolt Compatibility - Session 25 Status

## Summary

**Status**: Stage 1 passes, Stage 2 fails due to factor claim mismatches.

## Current Issue: Factor Claims Mismatch

After fixing padding cycle handling for NextIsNoop, 5 of 8 factors now match:

| Factor | Description | Status | Notes |
|--------|-------------|--------|-------|
| 0 | LeftInstructionInput | ✓ Match | MLE evaluation correct |
| 1 | RightInstructionInput | ✓ Match | MLE evaluation correct |
| 2 | IsRdNotZero | ✗ Small diff (~2^22) | Instruction-type-specific |
| 3 | WriteLookupOutputToRD | ✓ Match | Fixed |
| 4 | Jump | ✓ Match | Circuit flag correct |
| 5 | LookupOutput | ✗ Large diff | Instruction-type-specific |
| 6 | Branch | ✗ Large diff | Instruction-type-specific |
| 7 | NextIsNoop | ✓ Match | Fixed with padding handling |

## Root Cause

The remaining mismatches (factors 2, 5, 6) are caused by Zolt not having instruction-type-specific information that Jolt has:

1. **Instruction type enum** - Jolt knows if a cycle is ADD, ADDI, BEQ, etc.
2. **Virtual instruction tracking** - Jolt expands some instructions into virtual sequences
3. **Per-instruction LookupOutput** - Each instruction type computes to_lookup_output() differently

## Completed ✓

1. All 712 Zolt internal tests pass
2. Stage 1 verification PASSES in Jolt
3. Fixed padding cycle handling for NextIsNoop (factor 7)
4. Factors 0, 1, 3, 4, 7 now match

## Required Fix

To fix the remaining factor claims, Zolt needs:

### Option 1: Enhance Trace Format
Add instruction type information to the trace:
- Instruction type enum (ADD, ADDI, BEQ, etc.)
- IsVirtual flag
- Pre-computed LookupOutput value

### Option 2: Match Jolt's Computation
For each instruction type, implement Jolt's exact:
- `to_lookup_output()` logic
- `instruction_flags()` logic
- `circuit_flags()` logic

## Files to Modify

- `src/zkvm/r1cs/constraints.zig` - R1CSCycleInputs.fromTraceStep
- `src/zkvm/proof_converter.zig` - computeProductFactorEvaluations
- `src/vm/interpreter.zig` - trace generation

## Next Steps

1. [ ] Understand how Jolt determines instruction types from RISC-V bytes
2. [ ] Implement instruction type detection in Zolt
3. [ ] Implement per-instruction LookupOutput computation
4. [ ] Update witness generation with correct flag values
5. [ ] Verify Stage 2 passes

## Recent Changes

- Added padding cycle handling in `computeProductFactorEvaluations`
- Fixed NextIsNoop factor by adding eq_eval contributions for padding cycles
- Updated NOTES.md with detailed analysis
