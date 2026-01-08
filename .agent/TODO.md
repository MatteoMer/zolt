# Zolt-Jolt Compatibility - Iteration 13 Status

## Summary

**Major Progress**: Stage 1 passes, Stage 2 OutputSumcheck now produces non-zero claims but still has mismatch.

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

### Current Status: Stage 2 Expected Output Claim Mismatch

Instance 3 (OutputSumcheck) now produces non-zero claims, but they don't match:
```
output_claim:          360423619528169532790606316037632682400105269297297952787160974999665739278
expected_output_claim: 1517077064984070876282582683460369922057346739017387399119058233283595484647
```

### Key Changes Made This Session

1. Updated `OutputSumcheckProver.init()` to accept program inputs, outputs, and panic flag
2. Modified `val_io` construction to match Jolt's `ProgramIOPolynomial`:
   - Populates input bytes at `input_start` index
   - Populates output bytes at `output_start` index
   - Sets panic bit at `panic` index
   - Sets termination bit at `termination` index (if not panicking)
3. Updated `ConversionConfig` to pass program I/O data
4. Updated call sites in `mod.zig` to pass device.inputs, device.outputs, device.panic

### Root Cause Analysis (In Progress)

The issue appears to be polynomial size/evaluation differences:

1. **Zolt**: Creates `val_io` as a 65536-element polynomial (16 variables)
2. **Jolt**: `ProgramIOPolynomial` has only 4096 elements (12 variables)
   - Evaluates as: `MLE(coeffs, r_lo) * Π(1 - r_hi[i])` where r_lo has 12 elements, r_hi has 4 elements

For a polynomial that's only non-zero for indices < 4096:
- The MLE over 16 variables should theoretically equal the 12-variable MLE times `Π(1-r_hi)`
- But the implementation details (bit ordering, indexing) matter!

### Next Steps

1. [ ] Verify that Zolt's `val_io` polynomial structure matches Jolt's `ProgramIOPolynomial` exactly
2. [ ] Check bit ordering in EQ polynomial evaluations (Jolt uses big-endian indexing)
3. [ ] Compare `val_final_claim` values between Zolt and Jolt debug output
4. [ ] Compare `val_io_eval` computation between prover and verifier
5. [ ] Consider implementing ProgramIOPolynomial-style evaluation in Zolt

### Debug Commands

```bash
# Generate Jolt-format proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
    --export-preprocessing /tmp/zolt_preprocessing.bin \
    -o /tmp/zolt_proof_dory.bin

# Verify in Jolt (with debug output)
cd /Users/matteo/projects/jolt
cargo test --release -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Latest Jolt Verification Output (Key Lines)

```
[JOLT] INSTANCE[3]: num_rounds=16, claim=21020514113593014076443849375132710987374427750385355553672158660451917205135
...
=== SUMCHECK VERIFICATION FAILED ===
output_claim:          360423619528169532790606316037632682400105269297297952787160974999665739278
expected_output_claim: 1517077064984070876282582683460369922057346739017387399119058233283595484647
```

### Files to Investigate

- `src/zkvm/ram/output_check.zig` - OutputSumcheckProver (updated)
- `jolt-core/src/poly/program_io_polynomial.rs` - ProgramIOPolynomial
- `jolt-core/src/zkvm/ram/output_check.rs` - OutputSumcheckVerifier
- `jolt-core/src/poly/eq_poly.rs` - EqPolynomial (big-endian indexing)
