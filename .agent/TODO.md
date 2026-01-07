# Zolt-Jolt Compatibility - Session 17 Progress

## Status Summary

### ✅ COMPLETED
- All 712 internal tests pass
- Stage 1 passes Jolt verification completely
- Stage 2 sumcheck constraint (s(0)+s(1) = claim) passes ALL 26 rounds
- All 5 instance provers integrated and producing correct round polynomials
- ProductVirtualRemainder final claim matches Jolt's expected Instance 0 claim

### ❌ REMAINING ISSUE
Stage 2 verification fails because:
- `output_claim` (21589049...) ≠ `expected_output_claim` (9898116...)
- Our prover produces non-zero contributions from ALL 5 instances
- Jolt's verifier expects only Instance 0 to contribute (Instances 1-4 show claim=0)

**Root Cause**: We're inserting `F.zero()` for polynomial opening claims like `RamRa`, `RamVal`, etc.
This makes Jolt's expected_output_claim computation return 0 for Instances 1-4.

## Fixes Made This Session

1. Fixed lagrangeC2 formula in output_check.zig
2. Added bindChallenge/updateClaim for InstructionLookups prover
3. Fixed claim_before capture for ProductVirtualRemainder
4. Verified all sumcheck rounds satisfy s(0)+s(1) = claim

## Opening Claims Needed (Currently F.zero())

To fix Stage 2, we need to compute actual values for:
- `RamRa` @ `RamRafEvaluation` (Instance 1)
- `RamVal` @ `RamReadWriteChecking` (Instance 2)
- `RamRa` @ `RamReadWriteChecking` (Instance 2)
- `RamValFinal` @ `RamOutputCheck` (Instance 3)
- `LookupOutput` @ `InstructionClaimReduction` (Instance 4)
- etc.

These are polynomial evaluations at the sumcheck challenge points.

## Next Steps

1. Compute RAF opening claim (ra_input_claim at opening point)
2. Compute RWC opening claims (val, ra at opening point)
3. Compute Output opening claims
4. Compute InstructionLookups opening claims
5. Verify Stage 2 passes

## Verification Commands

```bash
# Build and test Zolt
zig build test --summary all

# Generate Jolt-compatible proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Verify with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```
