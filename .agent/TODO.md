# Zolt-Jolt Compatibility Implementation

## Status: Session 95 - Fixed UpperWord prefix, continuing Stage 5 debugging

## Current Issue: Stage 5 sumcheck verification fails

### Fixes Applied in This Session

1. **Fixed UpperWord prefix shift formulas**
   - Bug: UpperWord was using `2*XLEN - j` instead of `XLEN - j`
   - This caused overflow at j=0 (trying to shift by 128 bits)
   - Also fixed the suffix handling to match Jolt's upper word extraction

2. **Previously fixed suffix_len bug** (Session 94)
   - 9 prefix functions were computing suffix_len AFTER popMsb() instead of BEFORE

### Root Cause Analysis

At round 0 of the address binding:

- **Brute-force computation** (correct):
  - `bf_val_eval_0 = 136276d9...`
  - `bf_raf_eval_0 = 9bac6bba...`
  - `bf_val + bf_raf = af0ee294...` (matches lookups_claim)

- **Prefix-suffix decomposition** (incorrect):
  - `read_checking[0] = 986acce1...`
  - `raf_evals[0] = 8d6b9084...`
  - `ps_val + ps_raf = fda2751d...` (DOES NOT MATCH!)

The per-table values match between brute-force and prefix-suffix:
- `bf_val_per_table[0] = 821c547e...` = `eval_0_per_table[0]`

This suggests the tableCombine is working correctly but there's an issue with how the Q polynomials are being accumulated or bound.

### Hypothesis

The issue is that at round 0 with small lookup indices:
1. All cycles have `prefix_bits = 0` (since k values are < 2^120)
2. All Q values accumulate at index 0, Q[1..255] = 0
3. At c=1, the suffix interpolation uses Q[b + half_len] which is all zeros
4. So eval_1_indep = 0, and we rely on sumcheck property for eval_1

This should be mathematically correct, but the prefix MLEs might not be computing the right values at c=0.

### Next Steps

1. **Debug prefix_mle at round 0**:
   - For c=0, b=0 at round 0, trace through each prefix type
   - Verify the returned values match Jolt's expected behavior
   - Check if the checkpoints are correctly initialized

2. **Verify Q polynomial accumulation**:
   - Check that suffix_mle values are being computed correctly
   - Verify the tableCombine formula matches Jolt for each table

3. **Compare with Jolt's prover output**:
   - Run Jolt prover with same trace and compare Q sums
   - Check if the initialization of suffix polys matches

### Key Files

- `src/zkvm/lookup_table/prefixes.zig`: Prefix MLE implementations
- `src/zkvm/lookup_table/prefix_suffix_prover.zig`: Q polynomial init and proverMsgReadChecking
- `src/zkvm/lookup_table/suffixes.zig`: Suffix MLE implementations

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
- Stage 5: FAILS ❌ (prefix-suffix decomposition produces wrong eval_0)
- Stages 6-7: Not reached
