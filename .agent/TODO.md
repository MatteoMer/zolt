# Zolt-Jolt Compatibility Implementation

## Status: Session 94 - Found root cause of Stage 5 failure

## Current Issue: Stage 5 sumcheck verification fails

### Root Cause (Session 94 Finding)

**The prefix-suffix decomposition is computing incorrect polynomial evaluations.**

At round 0 of the address binding:
- Brute-force `bf_val_eval_0` = `136276d9c9f325b23b5bbcc2806aaa88`
- Prefix-suffix `read_checking[0]` = `986acce18b14b46fcb6e1544d9c065f1`
- **MISMATCH!**

And for RAF:
- Brute-force `bf_raf_eval_0` = `9bac6bba3a49394b7c88153904b17e3d`
- Prefix-suffix `raf_evals[0]` = `8d6b9084167d72aef843768ce0e84c94`
- **MISMATCH!**

The divergence starts at round 0 and accumulates throughout the 128 address rounds. By round 128, the polynomial chain has completely diverged from the correct values.

### What's Working

1. **Transcript handling**: Challenges match between Zolt prover and Jolt verifier
2. **Opening claims**: All virtual claims (ra_chunks, table_flags) serialize correctly
3. **Sumcheck polynomial property**: p(0) + p(1) = claim holds for all rounds
4. **Final claim components match Jolt's expected values**:
   - `ra_product` matches
   - `val_claim` matches
   - `raf_claim` matches
   - `eq_r_reduction` matches

### What's Broken

The `proverMsgReadChecking` and `proverMsgRaf` functions return incorrect polynomial evaluations. This causes the polynomial chain to evolve incorrectly, even though it maintains the sumcheck property internally.

### Next Steps

1. **Fix `proverMsgReadChecking`**:
   - Compare Zolt's implementation with Jolt's `prover_msg_read_checking`
   - Check the prefix MLE evaluation formula
   - Check the suffix Q polynomial indexing

2. **Fix `proverMsgRaf`**:
   - Compare with Jolt's RAF decomposition
   - Verify the operand polynomial formulas (left, right, identity)

3. **Test with simpler case**:
   - Add unit tests for prefix-suffix decomposition
   - Compare step-by-step with Jolt's values

### Key Files

- `src/zkvm/lookup_table/prefix_suffix_prover.zig`: Contains `proverMsgReadChecking` and `proverMsgRaf`
- `src/zkvm/lookup_table/prefixes.zig`: Contains `prefixMle` function
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`: Jolt's reference implementation

### Test Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64 2>&1 | tee /tmp/zolt_stage5_debug.log

cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture 2>&1 | tee /tmp/jolt_verify_debug.log
```

### Test Results

- Stage 1: PASSES ✅
- Stage 2: PASSES ✅
- Stage 3: PASSES ✅
- Stage 4: PASSES ✅
- Stage 5: FAILS ❌ (prefix-suffix decomposition bug)
- Stages 6-7: Not reached
