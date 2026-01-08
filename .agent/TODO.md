# Zolt-Jolt Compatibility - Iteration 12 Status

## Summary

**Major Progress**: Fixed double-prove issue, Stage 1 now passes, investigating Stage 2 issue.

### Completed ✓
1. All 712 Zolt internal tests pass
2. All 6 verification stages pass in Zolt's internal verifier
3. Compressed G1/G2 serialization implemented
4. **Preprocessing loads successfully in Jolt**
5. Proof deserialization works in Jolt
6. Transcript initial states match between Zolt and Jolt
7. **Fixed double-prove issue in --jolt-format mode**
8. **Jolt-format proof generated successfully (32KB)**
9. **Preprocessing export works (93KB)**
10. **Stage 1 verification PASSES!**

### Current Status: Stage 2 Sumcheck Mismatch

When Jolt verifies Zolt's proof (using Zolt's preprocessing):
- Stage 1 completes successfully ✓
- **Stage 2 fails** with output_claim != expected_output_claim

```
output_claim:          6490144552088470893406121612867210580460735058165315075507596046977766530265
expected_output_claim: 21082316018007705420574862052777536378229816237024856529615376176100426073034
```

### Per-Instance Breakdown (from Jolt debug)

Stage 2 has 5 batched instances:

| Instance | Name | num_rounds | claim | contribution |
|----------|------|------------|-------|--------------|
| 0 | ProductVirtualRemainder | 10 | 15183... | 4498... |
| 1 | RamRafEvaluation | 16 | 0 | 0 |
| 2 | RamReadWriteChecking | 26 | 0 | 0 |
| 3 | OutputSumcheck | 16 | 3821... | 5597... |
| 4 | InstructionClaimReduction | 10 | 15033... | 10986... |

**Expected sum**: 4498... + 0 + 0 + 5597... + 10986... = **21082...**
**Actual output_claim**: **6490...**

### Root Cause Analysis

The sumcheck proof's output_claim (6490...) is significantly less than the expected sum (21082...). This indicates the proof's round polynomials are not correctly combining all instance contributions.

Key observations:
1. Instances 0, 3, 4 have non-zero claims
2. The proof's output_claim is less than Instance 4's contribution alone (10986...)
3. This suggests the batched sumcheck combining logic might be wrong

### Potential Issues

1. **Instance round polynomial scaling**: When instances start at different rounds (e.g., Instance 0 starts at round 16, Instance 3 starts at round 10), their contributions need proper 2^k scaling.

2. **Combined polynomial evaluation**: The combined polynomial for each round should be:
   ```
   combined(x) = Σ_i coeff_i * instance_i(x) * 2^(max_rounds - instance_rounds[i])
   ```
   If this scaling is wrong, the final claim will be wrong.

3. **OutputSumcheck (Instance 3) round polynomials**: Need to verify these are being computed and combined correctly starting at round 10.

### Files Changed This Session
- `src/main.zig` - Fixed double-prove issue for --jolt-format

### Next Steps

1. [ ] Add debug logging to show all 26 round polynomial values
2. [ ] Verify Instance 3 (OutputSumcheck) is contributing to rounds 10-25
3. [ ] Check the 2^k scaling factor for each instance
4. [ ] Compare Zolt's combined_evals with what Jolt expects

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

### Architecture Notes

Stage 2 structure:
- UniSkip first round (5 product factors)
- Batched sumcheck with 5 instances, each with different num_rounds
- max_num_rounds = 26 (log_ram_k + n_cycle_vars = 16 + 10)
- Each instance starts at round (max_num_rounds - instance.num_rounds):
  - Instance 0: starts at round 16 (26 - 10)
  - Instance 1: starts at round 10 (26 - 16)
  - Instance 2: starts at round 0 (26 - 26)
  - Instance 3: starts at round 10 (26 - 16)
  - Instance 4: starts at round 16 (26 - 10)
