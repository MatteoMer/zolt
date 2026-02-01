# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Expected Output Claim Mismatch

## Current Session Progress (Session 91)

### Key Fix Applied: r_reduction source corrected! ✓

**BUG FOUND AND FIXED**: Zolt was using `stage3_result.challenges` for `r_reduction`, but the correct source is `stage2_result.challenges` (last n_cycle_vars = 8 challenges from Stage 2's InstructionClaimReduction).

**Fix applied** in `proof_converter.zig`:
- Changed `r_reduction_be` to be extracted from Stage 2 challenges (indices 16-23 for max_rounds=24, n_cycle_vars=8)
- Commit: `d8b87bd fix(proof_converter): use Stage 2 challenges for r_reduction in Stage 5`

### Verification: eq_r_reduction now matches! ✓
- Zolt's eq_r_reduction: `8349e6eb71ecb0c07088c5aa7c4d7b5a` (BE)
- Jolt's eq_eval_r_reduction: `[5a, 7b, 4d, 7c, aa, c5, 88, 70, c0, b0, ec, 71, eb, e6, 49, 83]` (LE)
- These are the SAME value, just reversed endianness!

### Remaining Issue: Stage 5 expected_output_claim still doesn't match

The sumcheck rounds all pass (polynomial verification succeeds), but the final claim comparison fails:
- `output_claim:   [bd, 7a, 64, 13, 7c, 97, 3f, 42, ...]`
- `expected_claim: [b2, d7, 91, f3, d5, d1, 0e, 0e, ...]`

### Analysis: Opening Claims Comparison

Comparing Zolt and Jolt's opening claims for InstructionReadRaf:

**ra_claims** (LE compressed bytes):
- Jolt ra_claims[0]: `[c6, 9f, 23, 72, 31, a5, 93, 82, de, ad, 8c, 8b, 7a, a9, d5, 1f]`
- Zolt ra_chunks[0] (BE first 8): `{ 31, 213, 169, 122, 139, 140, 173, 222 }` = `1f d5 a9 7a 8b 8c ad de`
- The high 8 bytes match (reversed endian), but Jolt has non-zero low bytes `c6 9f 23 72...`

**table_flags** (non-zero entries):
- Jolt table_flag[0]: `[b9, c9, 4a, 6b, 36, b3, 7c, df, dc, 29, 98, b5, 86, 47, 13, 18]`
- Zolt table_flags[0]: `{ 24, 19, 71, 134, 181, 152, 41, 220 }` (BE first 8 bytes)
- Converting Zolt BE to LE: `dc 29 98 b5 86 47 13 18` - matches Jolt's high bytes!

**raf_flag_claim**:
- Jolt: `[82, 81, f4, 7a, b6, c5, 75, cf, d4, 75, d4, 59, c0, a2, 34, d8]`
- Zolt: `{ 24, 19, 71, 134, 181, 152, 41, 220, ...}` - different!

### Key Finding

The high bytes of the field elements match between Zolt and Jolt, but Jolt's values have additional non-zero LOW bytes (bytes 0-7 in LE). This suggests:

1. Zolt is computing values with only 64-128 bits of entropy
2. Jolt expects full 254-bit field element values
3. The Montgomery conversion or field arithmetic might be truncating values

### Hypothesis

The issue is in how Zolt computes the ra_chunk and table_flag values. The computation uses:
- `ra_chunk_weights[i][0]` - final value after binding
- The weights start as F.one() and get multiplied by factors

If the factors or multiplication is losing precision in the high bits, that would explain the mismatch.

### Next Steps
1. Add full 32-byte debug output for ra_chunks and table_flags
2. Compare Zolt's values after fromMontgomery() conversion
3. Check if the field arithmetic is preserving all 254 bits
4. Investigate ra_chunk_weights initialization and multiplication

### Files Changed This Session
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig`:
  - Fixed r_reduction_be to use Stage 2 challenges (InstructionClaimReduction)
  - Added debug output for r_reduction limbs

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
