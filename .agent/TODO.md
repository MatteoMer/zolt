# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 ra_claim Mismatch

## Session 134 Progress

### Key Finding: ra_chunks values don't match!

After extensive debugging, confirmed that:

1. **Challenges match exactly between Zolt and Jolt** ✓
2. **LowerWord prefix checkpoint computation matches** ✓
3. **Table MLE values match** ✓
4. **table_flag claims match** ✓
5. **BUT ra_claims DON'T MATCH!** ✗

### Debug Evidence

**Jolt's ra_claims[0]** (from verifier reading proof):
```
[a5, 5e, c7, 72, 66, 8e, 13, 27, 21, 0d, f3, 0e, 35, 26, 9b, 11]
```

**Zolt's InstructionRa(0)** (serialized to proof):
```
bytes[16..32] = [90 fa 96 e6 36 b6 07 e1 e4 6f 2c 8b ff 8e 00 be]
```

These are COMPLETELY different values!

### Root Cause Analysis

The ra_chunk values are computed by accumulating `eq(lookup_index, challenge)` factors.

In Zolt:
```zig
const bit = getBit128(lookups_indices_lo[j], lookups_indices_hi[j], bit_index);
const factor = if (bit == 0) one_minus_r else challenge;
ra_chunk_weights[chunk_idx][j] = ra_chunk_weights[chunk_idx][j].mul(factor);
```

This formula is correct! But something else must be wrong:
1. The bit_index mapping might be wrong
2. The lookup_indices might be wrong
3. The challenge values might be used in wrong order

### Jolt's Approach

Jolt computes ra_polys using expanding tables in `init_log_t_rounds`:
```rust
let first_idx = ((v >> shift) as usize) & m_mask;
let mut acc = first[first_idx];
for table in iter {
    shift -= log_m;
    let idx = ((v >> shift) as usize) & m_mask;
    acc *= table[idx];
}
```

The expanding table stores products of eq factors accumulated during address rounds.

### Next Steps

1. **Debug lookup_index values**
   - Print first few lookup indices from Zolt
   - Compare with what Jolt would expect for same trace

2. **Verify bit_index ordering**
   - Check if Zolt processes bits in same order as Jolt
   - Round 0 = bit 127 or bit 0?

3. **Trace expanding table vs direct computation**
   - Add debug to compare Jolt's expanding table values
   - Verify Zolt's direct eq computation gives same result

### Test Commands

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

## Previous Sessions Summary

- Session 133: Confirmed challenges match, identified table MLE mismatch
- Session 134: Confirmed table MLEs match, identified ra_claims mismatch

## Files Modified This Session

- `/home/vivado/projects/zolt/src/zkvm/lookup_table/prefixes.zig` - Added detailed debug for LowerWord updates
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/lookup_table/prefixes/lower_word.rs` - Added debug for comparison
