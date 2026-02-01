# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Debugging

## Session 106 Final Summary

### Commits Made
1. `edd31ef` - fix: handle shift overflow in rightShiftPrefixMle and rightShiftWPrefixMle
2. `42378e1` - chore: update TODO with Stage 5 debugging progress

### Verification Status
- **Stage 1-4**: PASSING ✓
- **Stage 5**: FAILING - Sumcheck output_claim doesn't match expected_claim

### Jolt Debug Output (First Round Coefficients)
From Jolt verifier for Stage 5:
```
first round coeffs_except_linear:
  [0]: [e2, ee, 6f, c7, e9, ff, ea, e2, 93, 3a, 36, dd, 78, 31, 47, 9d, ...]
  [1]: [f6, 50, 28, 04, 08, f4, ed, ad, af, 77, b5, 4b, 95, 9a, d3, 49, ...]
  [2]: [00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, ...]
```

### Next Steps for Debugging
1. Add logging in Zolt to print first round coefficients in same format
2. Compare c0 and c2 values between Zolt and Jolt
3. If they differ at round 0, issue is in:
   - Q (suffix) polynomial initialization
   - Prefix MLE computation for phase 0
   - RAF operand computation
4. If they match at round 0 but diverge later:
   - Check phase transition logic
   - Check prefix checkpoint updates
   - Check suffix polynomial binding

### Key Formula Comparison

**Jolt's eval_0 for address round:**
```
eval_0 = Σ_b Σ_table table.combine(prefixes_c0, suffixes_left[b])
```

**Zolt's eval_0 (in proverMsgReadChecking):**
```zig
for (0..half_len) |b_idx| {
    for (0..NUM_TABLES) |table_idx| {
        combined_0 = tableCombine(F, table_idx, &prefixes_c0, suffixes_left[0..n]);
        eval_0 = eval_0.add(combined_0);
    }
}
```

These formulas match structurally, but actual values may differ due to:
1. Different cycle_table_indices assignment
2. Different suffix_bits extraction
3. Different prefix checkpoint values

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof (takes several minutes due to SRS generation)
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Files to Investigate
1. `src/zkvm/lookup_table/prefix_suffix_prover.zig:117-182` - initPhase Q computation
2. `src/zkvm/lookup_table/prefix_suffix_prover.zig:229-302` - proverMsgReadChecking
3. `src/zkvm/lookup_table/prefixes.zig` - All prefix MLE implementations
4. `src/zkvm/spartan/stage5_prover.zig` - cycle_table_indices assignment

### Remaining Work
1. Debug Stage 5 sumcheck polynomial mismatch
2. Once Stage 5 passes, verify complete proof
3. Test with additional programs beyond fibonacci
4. Consider performance optimizations (SRS caching)
