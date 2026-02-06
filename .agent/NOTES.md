# Session 94 Notes - Root Cause of Stage 5 Failure Found

## Summary

Identified that the prefix-suffix decomposition in `proverMsgReadChecking` produces incorrect polynomial evaluations from round 0. The brute-force computation gives different values than the prefix-suffix decomposition.

## Key Finding

At round 0 of address binding:

```
Brute-force (bf_val_eval_0) = 136276d9c9f325b23b5bbcc2806aaa88
Prefix-suffix (read_checking[0]) = 986acce18b14b46fcb6e1544d9c065f1
MISMATCH!

Brute-force (bf_raf_eval_0) = 9bac6bba3a49394b7c88153904b17e3d
Prefix-suffix (raf_evals[0]) = 8d6b9084167d72aef843768ce0e84c94
MISMATCH!
```

This divergence starts at round 0 and accumulates through all 128 address rounds.

## What's Working

1. **Total sum at round 0 is correct**: `total_sum(eq*combined_vals) = lookups_claim` ✓
2. **All virtual claims match Jolt's expected values**:
   - ra_claims[0..7] match exactly
   - table_flag[0,1,9] match exactly
   - val_claim, raf_claim, eq_r_reduction all match
3. **Transcript/challenge handling is correct**: All challenges match between Zolt prover and Jolt verifier

## What's Broken

The `proverMsgReadChecking` function computes `eval_0` using prefix-suffix decomposition:
1. For each `b` in 0..half_len:
   - Compute `prefixes_c0[i]` for all prefix types
   - Compute `prefixes_c2[i]` for all prefix types
   - For each table, combine with suffix Q values
2. Sum up contributions to get `eval_0`, `eval_2_left`, `eval_2_right`

The problem is that this produces `eval_0 = 986a...` but brute-force gives `eval_0 = 1362...`.

## Possible Issues

1. **Q polynomial initialization**: The suffix Q polynomials might not be correctly initialized from the trace data.

2. **Prefix MLE evaluation**: The `prefixMle` functions might be evaluated with wrong parameters.

3. **Table combine**: The `tableCombine` function might have an issue.

4. **Bit ordering**: The b-index iteration might use wrong bit ordering.

## Comparison: Jolt vs Zolt

### proverMsgReadChecking structure

**Jolt**:
```rust
let [eval_0, eval_2_left, eval_2_right] = (0..len/2)
    .into_par_iter()
    .flat_map_iter(|b| {
        let b = LookupBits::new(b, log_len - 1);
        let prefixes_c0: Vec<_> = Prefixes::iter()
            .map(|prefix| prefix.prefix_mle(&checkpoints, r_x, 0, b, j))
            .collect();
        // ...
        lookup_tables.iter().zip(suffix_polys.iter())
            .map(|(table, suffixes)| {
                let suffixes_left = suffixes[b];
                let suffixes_right = suffixes[b + len/2];
                [
                    table.combine(&prefixes_c0, &suffixes_left),
                    table.combine(&prefixes_c2, &suffixes_left),
                    table.combine(&prefixes_c2, &suffixes_right),
                ]
            })
    })
    .fold_with(zeros).reduce(...);
```

**Zolt**:
```zig
for (0..half_len) |b_idx| {
    const b = LookupBits(128).new(b_idx, log_len - 1);
    // Compute prefixes_c0[i] and prefixes_c2[i]
    for (0..NUM_TABLES) |table_idx| {
        if (suffix_polys.tables[table_idx]) |table| {
            const suffixes_left = table.polys[s_idx][b_idx];
            const suffixes_right = table.polys[s_idx][b_idx + half_len];
            // Combine...
        }
    }
}
```

The structure looks similar. The issue is likely in:
- How suffix_polys is initialized
- How tableCombine works
- How the prefix checkpoints are updated

## Next Steps

1. Add unit test that compares prefix-suffix output with brute-force for a simple case
2. Debug the Q polynomial initialization to verify values
3. Step through tableCombine for table 0 (RangeCheck) with known values

## Test Commands

```bash
# Build and run
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
