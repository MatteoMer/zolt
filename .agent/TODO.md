# Zolt-Jolt Compatibility Implementation

## Status: Session 96 - Fixed Load/Store operands, debugging read_checking sum

## Current Issue: Stage 5 sumcheck verification fails

### Fixes Applied in This Session

1. **Fixed Load/Store operand computation** (lines 924-930)
   - Bug: Load (0x03) and Store (0x23) were falling through to `else` branch
   - This caused left_op=left_input, right_op=right_input (interleaved format)
   - But Load/Store use identity path with left=0, right=address
   - Fixed: Added explicit cases for 0x03 and 0x23 with `left_op = F.zero()` and `right_op = left_input.add(right_input)`

2. **Previously fixed I-type ALU/W-type ALU handling** (Session 95)
   - Only ADDI/ADDIW (funct3=0) use identity path
   - Other I-type like SLLI use interleaved path

### Root Cause Analysis

After the Load/Store fix, the brute-force RAF computation now matches:
- `bf_raf_eval_0 = 8d6b9084...`
- `bf_raf_reconstructed = 8d6b9084...`
- `bf_raf_from_operands = 8d6b9084...`

But the read_checking (val) component still doesn't match:
- `bf_val_eval_0 = 136276d9...` (brute-force)
- `read_checking_evals[0] = 986acce1...` (prefix-suffix)

The per-table values DO match:
- `bf_val_per_table[0] = 821c547e...` = `eval_0_per_table[0]`
- `bf_val_per_table[1] = 92544e9d...` = `eval_0_per_table[1]`
- `bf_val_per_table[9] = ac2e120d...` = `eval_0_per_table[9]`

### Next Steps

1. **Debug why eval_0 != sum of eval_0_per_table**
   - In proverMsgReadChecking, both should accumulate the same values
   - Need to add debug output to verify

2. **Check tableCombine formula**
   - May be an issue with how prefix*suffix values are combined

3. **Verify the debugging approach**
   - The mismatch might be in how we're printing/comparing values
   - Full 32-byte comparison needed

### Key Files

- `src/zkvm/spartan/stage5_prover.zig`: Stage 5 prover with Load/Store fix at lines 924-930
- `src/zkvm/lookup_table/prefix_suffix_prover.zig`: proverMsgReadChecking at line 284

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
- Stage 5: FAILS ❌ (read_checking total doesn't match brute-force)
- Stages 6-7: Not reached
