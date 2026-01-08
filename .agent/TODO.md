# Zolt-Jolt Compatibility - Current Status

## Summary

**Stage 1**: PASSES ✓
**Stage 2**: FAILS - OutputSumcheck zero-check fails

## Root Cause Found!

The OutputSumcheck prover fails the sumcheck soundness check `s(0) + s(1) = current_claim` in EVERY round.

This is because the zero-check sum is NOT zero:
```
sum_k eq(r_address, k) * io_mask(k) * (val_final(k) - val_io(k)) ≠ 0
```

### Why the sum is non-zero:

1. IO region: indices [1024, 4096) (addresses 0x7FFFA000 to 0x80000000)
2. io_mask = 1 for this entire range
3. val_io only has:
   - panic (0) at index 2048
   - termination (1) at index 2049
4. val_final has non-zero values at indices 3584-3598 (addresses 0x7FFFF000-0x7FFFF0F8)
5. Since val_io[3584..3598] = 0 but val_final[3584..3598] ≠ 0, the difference is non-zero!

### Key Question

What are the values at addresses 0x7FFFF000-0x7FFFF0F8?

Looking at the values:
- k=3584: val=282579962709375 (0x101010101FF = appears to be ABI padding or stack setup)
- k=3586: val=4310892546
- k=3590: val=8070450545133223943 (0x7000000000000007)
- etc.

These look like stack frame setup or initial register spills before the stack pointer is properly initialized.

### Possible Causes

1. **Zolt is incorrectly classifying these addresses as being in the IO region**
   - But the addresses ARE in [input_start, RAM_START), so this is correct per Jolt's definition

2. **Zolt's RAM trace includes spurious writes that Jolt doesn't see**
   - Possible if trace serialization/deserialization differs

3. **The Fibonacci program writes to these addresses legitimately, but Jolt handles it differently**
   - Perhaps Jolt's val_io includes advice data that Zolt doesn't?

### Next Steps

1. [ ] Run Jolt's native Fibonacci prove/verify to see if it passes
2. [ ] Compare Jolt's and Zolt's val_final at the problematic indices
3. [ ] Check if these addresses are in the advice region (which might be handled differently)
4. [ ] Investigate whether val_io should include more data than just input/output/panic/termination

### Individual Claims That Match

- val_final_claim ✓
- val_io_eval ✓
- eq_eval ✓
- io_mask_eval ✓
- expected_output_claim (computed from above) = 4629... ✓

But the prover's current_claim = 11607... ≠ 4629... because the underlying sum is not zero.

## Testing Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Run Jolt verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```

## Files Modified in This Session

1. `/Users/matteo/projects/zolt/src/zkvm/ram/output_check.zig` - Added debug output
2. `/Users/matteo/projects/zolt/src/zkvm/proof_converter.zig` - Added inst3 debug tracking
