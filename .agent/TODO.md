# Zolt-Jolt Compatibility - Iteration 12 Status

## Summary

**Progress made**: Fixed double-prove issue, improved proof generation pipeline.

### Completed ✓
1. All 712 Zolt internal tests pass
2. All 6 verification stages pass in Zolt's internal verifier
3. Compressed G1/G2 serialization implemented (32 bytes / 64 bytes)
4. **Preprocessing loads successfully in Jolt**
5. Proof deserialization works in Jolt
6. Transcript initial states match between Zolt and Jolt
7. **Fixed double-prove issue in --jolt-format mode**
8. **Jolt-format proof generated successfully (32KB)**
9. **Preprocessing export works (93KB)**
10. **Stage 1 appears to pass** (no longer the first failure point)

### Current Status: Stage 2 Sumcheck Mismatch

When Jolt verifies Zolt's proof (using Zolt's preprocessing):
- Stage 1 completes successfully
- **Stage 2 fails** with output_claim != expected_output_claim

```
output_claim:          6490144552088470893406121612867210580460735058165315075507596046977766530265
expected_output_claim: 21082316018007705420574862052777536378229816237024856529615376176100426073034
```

### Root Cause Analysis

Stage 2 batches 5 sumcheck instances:
1. **ProductVirtualRemainderVerifier** - Verifies product virtualization from Stage 1
2. **RamRafEvaluation** - RAM read-address-flag evaluation
3. **RamReadWriteChecking** - RAM read/write consistency
4. **OutputCheck** - Verifies output constraints
5. **InstructionClaimReduction** - Reduces instruction lookup claims

The mismatch could be due to:
1. Batching coefficient computation differences
2. Opening point alignment issues
3. Virtual polynomial evaluation differences
4. Instance ordering or scaling issues

### Technical Changes Made (This Iteration)

1. **Fixed double-prove issue**:
   - Modified `src/main.zig` to use single proving path for --jolt-format
   - Previously called `prove()` then `proveJoltCompatibleWithDoryAndSrsAtAddress()` separately
   - Now only calls `proveJoltCompatibleWithDoryAndSrsAtAddress()` when --jolt-format specified

2. **Proof size**: 32,966 bytes (vs ~12KB before, includes more complete Stage 2 data)

3. **Preprocessing export**: 93,456 bytes

### Files Modified
- `src/main.zig` (double-prove fix)

### Next Steps

#### High Priority
1. [ ] Debug Stage 2 batching coefficient computation
2. [ ] Add logging to Jolt's verify_stage2() to see:
   - Input claims from each batched instance
   - Batching coefficients
   - Per-instance expected_output_claims
3. [ ] Compare with Zolt's generateStage2BatchedSumcheckProof()

#### Investigation Approach
1. Add debug output to Jolt's BatchedSumcheck::verify()
2. Add corresponding output to Zolt's Stage 2 generation
3. Compare claim values and batching coefficients

#### Key Files to Examine
- `/jolt/jolt-core/src/zkvm/verifier.rs` - verify_stage2()
- `/jolt/jolt-core/src/zkvm/spartan/product.rs` - ProductVirtualRemainderVerifier
- `/jolt/jolt-core/src/subprotocols/batched_sumcheck.rs` - BatchedSumcheck::verify()
- `/zolt/src/zkvm/proof_converter.zig` - generateStage2BatchedSumcheckProof()

### Commands to Test

```bash
# Generate Jolt-format proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
    --export-preprocessing /tmp/zolt_preprocessing.bin \
    -o /tmp/zolt_proof_dory.bin

# Verify in Jolt
cd /Users/matteo/projects/jolt
cargo test --release -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Architecture Notes

Stage 2 structure:
- UniSkip first round (5 product factors)
- Batched sumcheck with 5 instances
- Each instance has different num_rounds and degree
- Opening claims set during verification for Stage 3+
