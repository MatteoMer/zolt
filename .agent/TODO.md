# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 InstructionReadRaf ra_chunk Mismatch

## Session 125 Summary

### Progress Made

1. **Fixed combined_vals rematerialization bug**
   - Cycles without lookup tables were getting `combined_val = 0`
   - Jolt ALWAYS adds RAF contribution regardless of table
   - Fixed: Now cycles without tables get `raf_interleaved` or `raf_identity`

2. **Verified eq_evals match**
   - After eq_evals reinitialization fix, `eq_evals[0] == eq_eval_r_reduction` ✓
   - The eq polynomial is now correct after all bindings

### Current Issue

**ra_chunk values don't match between Zolt and Jolt**

Debug output shows:
```
Zolt ra_chunks[0] = 72c54fffc84783cff5628ecc74b37775
Jolt ra_claims[0] = 12109e5de8bae83db5b8fca2612309a8
```

These are completely different!

### Analysis

1. **ra_chunk_weights computation** (Zolt):
   - Initialized to 1 for all cycles
   - During address rounds, multiplied by `eq(bit[j][round], challenge[round])`
   - Formula: `ra_chunk[chunk][j] = Π_{round ∈ chunk} eq(bit[j][round], challenge[round])`

2. **ra_poly computation** (Jolt):
   - Created from expanding tables after all address rounds
   - `ra_poly[chunk][j] = Π_{phase ∈ chunk} v[phase][lookup_index_chunk[j][phase]]`
   - Where `v[phase][k] = eq(k, challenges_in_phase)`

3. **Theoretical equivalence**:
   - Both should compute `eq(lookup_index_chunk[j], challenges_in_chunk)`
   - But they're giving different results!

4. **Observation**: All ra_chunk values are IDENTICAL across cycles 0-255:
   ```
   ra_chunk[0][0:4] = [72c54fff..., 72c54fff..., 72c54fff..., 72c54fff...]
   ```
   This is suspicious because lookup_indices DO differ between cycles.

### Potential Root Causes

1. **Bit extraction order mismatch**
   - Zolt: `bit_index = LOOKUPS_LOG_K - 1 - round` → processes MSB first
   - Need to verify this matches Jolt's expanding table approach

2. **Chunk assignment difference**
   - Zolt: `chunk_idx = round / chunk_size`
   - Jolt: Groups phases by chunk differently

3. **Phase vs Round confusion**
   - Jolt uses phases with `log_m` bits each (typically 16)
   - Zolt uses individual rounds
   - The grouping might not align

4. **Lookup index storage order**
   - lookup_indices are stored as (lo, hi) u64 pairs
   - Need to verify bit extraction is correct

### Debug Areas

1. Print exact bit patterns from lookup_indices for first few cycles
2. Print which bits are being accessed in each round
3. Compare with Jolt's expanding table bit access pattern
4. Verify chunk boundaries match

### Key Files

**Zolt:**
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck prover
- `src/zkvm/lookup_table/prefix_suffix_prover.zig` - Prefix-suffix decomposition

**Jolt:**
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf sumcheck

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy to /tmp for Jolt test
cp logs/zolt_*.bin /tmp/

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Next Steps

1. Add detailed debug to show exact bit values being accumulated
2. Compare Jolt's expanding table `v[phase][k]` values with Zolt's per-round `eq(bit, challenge)` values
3. Verify the ra_chunk values at intermediate points during address rounds
4. Check if there's an off-by-one or endianness issue in bit extraction
