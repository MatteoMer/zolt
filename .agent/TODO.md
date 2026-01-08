# Zolt-Jolt Compatibility - Iteration 13 Status

## Summary

**Major Progress**: Stage 1 passes, Stage 2 has eq polynomial evaluation mismatch identified.

### Completed ✓
1. All 712 Zolt internal tests pass
2. All 6 verification stages pass in Zolt's internal verifier
3. Compressed G1/G2 serialization implemented
4. Preprocessing loads successfully in Jolt
5. Proof deserialization works in Jolt
6. Transcript initial states match between Zolt and Jolt
7. Fixed double-prove issue in --jolt-format mode
8. Jolt-format proof generated successfully (32KB)
9. Preprocessing export works (93KB)
10. **Stage 1 verification PASSES!**
11. **OutputSumcheck now uses program I/O data for val_io polynomial**
12. **Identified root cause: EQ polynomial evaluation ordering mismatch**

### ROOT CAUSE IDENTIFIED ✅

Added debug output to compare Zolt vs Jolt OutputSumcheck values:

| Value | Zolt | Jolt | Match? |
|-------|------|------|--------|
| val_final[0] | ✅ | ✅ | YES |
| val_io[0] | ✅ | ✅ | YES |
| io_mask[0] | ✅ | ✅ | YES |
| (val_final - val_io)[0] | ✅ | ✅ | YES |
| **eq_r_address[0]** | ❌ | ❌ | **NO** |
| expected | ❌ | ❌ | NO (due to eq mismatch) |

**The EQ polynomial evaluations don't match!**

### Root Cause Analysis

The EQ polynomial mismatch is due to **challenge ordering**:

1. **Jolt** computes: `eq_eval = EqPolynomial::mle(r_address, r_address_prime)`
   - `r_address` = original challenge (generated before sumcheck)
   - `r_address_prime` = sumcheck challenges **normalized to BIG_ENDIAN** (reversed)

2. **Zolt** computes: `eq_r_address` bound with sumcheck challenges
   - Challenges bound in **LITTLE_ENDIAN** order (not reversed)
   - Final `eq_r_address[0] = eq(r_address, [s_0, s_1, ..., s_15])`

**Math difference**:
- Jolt: `eq(r_address, [s_15, s_14, ..., s_0])` (BIG_ENDIAN)
- Zolt: `eq(r_address, [s_0, s_1, ..., s_15])` (LITTLE_ENDIAN)

These produce different values because the eq polynomial is:
```
eq(x, y) = Π_i (x_i * y_i + (1-x_i)*(1-y_i))
```

Swapping the pairing order changes the result!

### Fix Required

Options:
1. **Reverse challenges in Zolt**: After sumcheck, compute eq_eval using reversed challenges
2. **Change binding order in Zolt**: Bind variables in reverse order (breaks sumcheck)
3. **Post-process eq_r_address**: Compute correct value after all rounds

Option 1 or 3 seems most feasible.

### Next Steps

1. [ ] Implement eq polynomial evaluation with reversed challenges in OutputSumcheck
2. [ ] Ensure the expected_output_claim matches Jolt's computation
3. [ ] Test and verify Instance 3 passes
4. [ ] Investigate remaining Instances (0, 4) if needed

### Debug Commands

```bash
# Generate Jolt-format proof with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
    --export-preprocessing /tmp/zolt_preprocessing.bin \
    -o /tmp/zolt_proof_dory.bin 2>&1 | grep "OUTPUT_CHECK"

# Verify in Jolt (with debug output)
cd /Users/matteo/projects/jolt
cargo test --release -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture 2>&1 | grep "OUTPUT_CHECK"
```

### Files Modified

- `src/zkvm/ram/output_check.zig` - Added debug output, updated init signature
- `src/zkvm/proof_converter.zig` - Updated ConversionConfig for program I/O
- `src/zkvm/mod.zig` - Updated call sites to pass program I/O
- `jolt-core/src/zkvm/ram/output_check.rs` - Added debug output (Jolt side)
