# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - ra_chunks Mismatch

## Current Issue

The Stage 5 InstructionRa claims from Zolt don't match what Jolt expects.

### Debug Evidence

1. **Suffix MLEs fixed** - LsbSuffix, Pow2Suffix, Pow2WSuffix, TwoLsbSuffix now return 1 for len==0
2. **Table values now match** - table[0] != table[1] in both Zolt and Jolt (fixed by suffix changes)
3. **Cycle challenges match** - Round 128-135 challenges identical between Zolt and Jolt
4. **Prefix values match** - left_op, right_op, identity prefixes all match

### ra_chunks Mismatch

**Zolt produces:**
```
ra_chunks[0] = 90fa96e636b607e1e46f2c8bff8e00be → [be, 00, 8e, ff, ...]
ra_chunks[1] = e0d59b8773d1c294f6f9472abb40ea58 → [58, ea, 40, bb, ...]
...
```

**Jolt expects:**
```
ra_claims[0] = [a5, 5e, c7, 72, 66, 8e, 13, 27, 21, 0d, f3, 0e, 35, 26, 9b, 11]
ra_claims[1] = [8c, 4c, 83, cb, e9, 16, 6c, a3, 90, 82, 4c, d4, 71, d3, f2, 2b]
...
```

### Root Cause Analysis

The ra_chunks are computed by:
1. At round 128, materialize ra_chunk_weights[i][j] = ∏_{phase in chunk_i} expanding_table[phase][k_phase(j)]
2. Bind through cycle rounds 128-135
3. Final ra_chunks[i] = ra_chunk_weights[i][0]

Potential issues:
1. expanding_tables may have different values than Jolt's at round 128
2. The bit extraction for k_phase might use wrong bit ordering
3. The phase-to-chunk mapping might be incorrect

### Investigation Required

1. Compare expanding_table[phase][k] values between Zolt and Jolt at round 128
2. Verify the EQ polynomial construction during address rounds 0-127
3. Check if Zolt's HighToLow vs LowToHigh binding order matches Jolt

## Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cp /tmp/zolt_*.bin /home/vivado/projects/jolt/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Key Files

- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover
- `/home/vivado/projects/zolt/src/zkvm/lookup_table/suffixes.zig` - Suffix MLEs (FIXED)
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Jolt's InstructionReadRaf
