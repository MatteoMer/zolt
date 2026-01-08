# Zolt-Jolt Compatibility - Current Status

## Summary

**Stage 1**: PASSES ✓
**Stage 2**: FAILS - factor claims mismatch

## Progress Made

1. **All 712 Zolt tests pass** ✓
2. **Stage 1 verification passes in Jolt** ✓
3. **Fixed padding cycle handling for NextIsNoop** ✓
4. **5 of 8 factor claims now match** ✓

## Factor Claims Status

| Factor | Name | Status | Notes |
|--------|------|--------|-------|
| 0 | LeftInstructionInput | ✓ Match | MLE evaluation correct |
| 1 | RightInstructionInput | ✓ Match | MLE evaluation correct |
| 2 | IsRdNotZero | ✗ diff=4194304 | Need investigation |
| 3 | WriteLookupOutputToRD | ✓ Match | |
| 4 | Jump | ✓ Match | |
| 5 | LookupOutput | ✗ Large diff | Need investigation |
| 6 | Branch | ✗ Large diff | Need investigation |
| 7 | NextIsNoop | ✓ Match | Fixed with padding |

## Technical Details

### What Works
- Transcript handling
- Field arithmetic
- Sumcheck round polynomial generation
- Opening claim serialization
- Eq polynomial evaluation
- Factor MLE evaluation (5/8 correct)

### What Doesn't Work
The remaining 3 factor mismatches suggest:
1. Some cycles have incorrect `IsRdNotZero` values
2. Some cycles have incorrect `LookupOutput` values
3. Some cycles have incorrect `Branch` flag values

### Code Locations
- Factor evaluation: `src/zkvm/proof_converter.zig:computeProductFactorEvaluations`
- Witness generation: `src/zkvm/r1cs/constraints.zig:fromTraceStep`
- LookupOutput computation: `src/zkvm/r1cs/constraints.zig:computeLookupOutput`
- Flag computation: `src/zkvm/r1cs/constraints.zig:setFlagsFromInstruction`

## Next Steps

1. **Add cycle-by-cycle debug output** for factor computation
   - Print instruction opcode for each cycle
   - Print computed values vs expected
   - Identify specific cycles with mismatches

2. **Compare Zolt vs Jolt witness values**
   - For a small trace (like Fibonacci)
   - Identify where values diverge

3. **Fix identified mismatches**
   - Update `computeLookupOutput` if needed
   - Update `setFlagsFromInstruction` if needed

## Files to Check

- Jolt: `jolt-core/src/zkvm/r1cs/inputs.rs` - how Jolt computes witness values
- Jolt: `jolt-core/src/zkvm/instruction/*.rs` - instruction-specific implementations
- Zolt: `src/zkvm/r1cs/constraints.zig` - witness generation
- Zolt: `src/zkvm/proof_converter.zig` - factor evaluation

## Testing Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Run Jolt verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```
