# Session Notes - Stage 5 ra_chunks Investigation

## Summary

Fixed suffix MLEs (LsbSuffix, Pow2Suffix, Pow2WSuffix, TwoLsbSuffix) to return 1 when len==0.
This fixed the table[0] != table[1] issue.

However, ra_chunks still don't match between Zolt and Jolt.

## Key Findings

### Expanding Tables
- Both Zolt and Jolt use HighToLow binding order for expanding tables
- Both reset to F::one() at phase start
- Both update with `update(challenge)` pattern

### ra_polys Structure
- Jolt: `ra_polys[chunk_i][j] = ∏_{phase in chunk_i} expanding_table[phase][k_phase(j)]`
- Zolt: `ra_chunk_weights[chunk_i][j] = ∏_{phase in chunk_i} expanding_table[phase][k_phase(j)]`

### Potential Issues to Investigate

1. **Phase-to-chunk mapping**: Verify that the mapping from phases to chunks is correct
   - Jolt: `chunk_size = v.len() / n = phases / 8`
   - Zolt: `phases_per_chunk = num_phases / ra_num_chunks = 16 / 8 = 2`

2. **Bit extraction for k_phase**: The formula uses:
   - `shift = (phases - 1 - phase) * log_m`
   - This extracts address bits for each phase

3. **Expanding table values**: Need to compare actual v[phase][k] values at round 128

4. **Challenge accumulation**: The challenges during address rounds 0-127 must match
   between Zolt and Jolt for the expanding tables to produce identical values

## Next Steps

1. Add debug output to Jolt to print expanding_table[phase][k] values at round 128
2. Compare with Zolt's values
3. If they differ, trace back to find where the divergence starts

## Code References

- Zolt ra materialization: `stage5_prover.zig` lines 2748-2814
- Jolt ra materialization: `read_raf_checking.rs` lines 626-694
- Expanding table update: `expanding_table.rs` (Jolt), `prefix_suffix_prover.zig` (Zolt)
