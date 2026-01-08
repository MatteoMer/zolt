# Zolt-Jolt Compatibility - Iteration 13 Status

## Summary

**Major Progress**: Stage 1 passes, investigating Stage 2 expected_output_claim mismatch.

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

### Current Status: Stage 2 Expected Output Claim Mismatch

When Jolt verifies Zolt's proof:
- Stage 1 (26 rounds) completes successfully ✓
- Stage 2 fails with `output_claim != expected_output_claim`

```
output_claim:          6490144552088470893406121612867210580460735058165315075507596046977766530265
expected_output_claim: 21082316018007705420574862052777536378229816237024856529615376176100426073034
```

### Per-Instance Analysis

Stage 2 batches 5 instances:

| Instance | Name | num_rounds | Jolt expected claim | Jolt contribution |
|----------|------|------------|---------------------|-------------------|
| 0 | ProductVirtualRemainder | 10 | 15183... | 4498... |
| 1 | RamRafEvaluation | 16 | 0 | 0 |
| 2 | RamReadWriteChecking | 26 | 0 | 0 |
| 3 | OutputSumcheck | 16 | 3821... | 5597... |
| 4 | InstructionClaimReduction | 10 | 15033... | 10986... |

**Jolt expected sum**: 4498... + 0 + 0 + 5597... + 10986... = **21082...**

### Key Findings from Investigation

1. **Input claims match**: Zolt and Jolt have the same input_claims for all 5 instances
2. **Sumcheck rounds are consistent**: Each round satisfies `s(0) + s(1) = claim`
3. **Final output_claim matches between Zolt and Jolt**: 6490...
4. **BUT expected_output_claim differs**: Jolt expects 21082...

### Root Cause Hypothesis

The issue is that **Instance 3 (OutputSumcheck)** is producing the wrong polynomial:

1. OutputSumcheck proves: `Σ_k eq(r_address, k) * io_mask(k) * (val_final(k) - val_io(k)) = 0`
2. In Zolt, `val_io` is set to `val_final` for the IO region
3. This makes `val_final[k] - val_io[k] = 0` for all k
4. BUT Jolt's verifier expects `val_io_eval` to come from `ProgramIOPolynomial`
5. `ProgramIOPolynomial` contains the actual program inputs/outputs from `program_io`

**The difference**: Zolt copies from final RAM state; Jolt uses the program I/O data.

For a correctly executing program, these SHOULD be equal. But the polynomials must evaluate to the same thing at the challenge point.

### Next Steps

1. [ ] Check if `ProgramIOPolynomial` evaluation differs from what Zolt sets for `val_io`
2. [ ] Verify the expected_output_claim calculation for Instance 3
3. [ ] Add debug output to compare val_io_eval values between Zolt and Jolt
4. [ ] Fix OutputSumcheckProver to use the correct val_io polynomial

### Commands to Test

```bash
# Generate Jolt-format proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
    --export-preprocessing /tmp/zolt_preprocessing.bin \
    -o /tmp/zolt_proof_dory.bin

# Verify in Jolt (with debug output)
cd /Users/matteo/projects/jolt
cargo test --release -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Files to Investigate

- `src/zkvm/ram/output_check.zig` - OutputSumcheckProver
- `jolt-core/src/poly/program_io_polynomial.rs` - ProgramIOPolynomial
- `jolt-core/src/zkvm/ram/output_check.rs` - OutputSumcheckVerifier
