# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 LookupsReadRaf Sumcheck Mismatch

## Current Session Progress (Session 92)

### Investigation Summary

The Stage 5 LookupsReadRaf sumcheck has a mismatch between computed_sum and lookups_input:
- `lookups_input = { 20, 143, 132, 65, 153, 197, 213, 8 }` (expected claim from Stage 2)
- `computed_sum = { 22, 41, 182, 247, 67, 123, 60, 180 }` (actual computed)

### Verified Correct:

1. **Witness Values**: Stage 2 and Stage 5 use the same witness values
   - For LUI (j=0): output=0x8000, left=0x0, right=0x8000
   - Both compute RightLookupOperand = imm = instr & 0xFFFFF000

2. **r_reduction Extraction**: Stage 5 correctly extracts r_reduction from Stage 2 challenges[16..24]
   - challenges[16..24] are the InstructionClaimReduction sumcheck binding challenges
   - Reversed to BIG_ENDIAN order for eq polynomial computation

3. **Field Arithmetic**: fromU64() and toBytesBE() work correctly
   - Small values (like 0x8000) appear in the LOW bytes of a 32-byte big-endian representation
   - Earlier debug output only showed first 8 bytes, which were zeros - this was a display issue

### Individual Sum Analysis:

```
output_sum (Σ eq*output) = { 21, 235, 230, 149, 140, 204, 118, 74, ... }
rv_claim (from Stage 2)  = { 32, 124, 218, 9, 175, 211, 55, 216, ... }

left_sum (Σ eq*left)     = { 4, 15, 48, 43, 92, 191, 180, 111, ... }
left_op_claim (Stage 2)  = { 10, 101, 186, 89, 71, 250, 14, 48, ... }

right_sum (Σ eq*right)   = { 19, 178, 103, 53, 33, 26, 181, 6, ... }
right_op_claim (Stage 2) = { 17, 107, 3, 246, 180, 227, 72, 40, ... }
```

None of the three component sums match their Stage 2 claims!

### Possible Causes (to investigate):

1. **Eq polynomial computation**: computeEqAtIndex may be computing eq(j, r) incorrectly
   - Check bit extraction order in computeEqAtIndex
   - Verify r_reduction elements are in the right order

2. **Sumcheck binding order mismatch**:
   - InstructionLookupsProver uses LowToHigh binding (LSB first)
   - Stage 5 expects BIG_ENDIAN (MSB first)
   - The reversal should handle this, but there may be an off-by-one or indexing issue

3. **Witness value computation differences**:
   - Stage 2 uses fromTraceStep() → setFlagsFromInstruction()
   - Stage 5 computes directly from trace step
   - Need to verify they produce identical values for ALL instructions

### Files Changed This Session
- `/home/vivado/projects/zolt/src/zkvm/r1cs/constraints.zig`: Removed debug output
- `/home/vivado/projects/zolt/src/field/mod.zig`: Removed debug output
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig`: Added debug for Stage 2 challenges extraction

### Next Steps

1. Add debug to compare actual eq polynomial values:
   - Print eq_evals[0..5] from Stage 5
   - Compute expected eq values manually and compare

2. Verify MLE evaluation formula:
   - Stage 2 claims are MLE(poly) evaluated at r_cycle
   - Stage 5 computes Σ_j eq(j, r_cycle) * poly[j]
   - These SHOULD be equivalent by MLE definition

3. Check if there's a padding or indexing issue:
   - Stage 2 may have padded witness to power of 2
   - Stage 5 may be using different padding

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
