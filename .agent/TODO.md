# Zolt-Jolt Compatibility - Session 16 Progress

## Status Summary

### ✅ COMPLETED
- All 712 internal tests pass
- Stage 1 passes Jolt verification completely
- Stage 2 UniSkip r0 matches Jolt (transcript aligned)
- Stage 2 UniSkip polynomial coefficients verified identical
- RafEvaluationProver integrated (Instance 1)

### ❌ REMAINING
- Stage 2 batched sumcheck fails (2 missing provers)

## Instance Timing in Stage 2

Understanding when each instance becomes "active" in the batched sumcheck:

| Instance | Prover | Rounds | Start Round | Status |
|----------|--------|--------|-------------|--------|
| 0 | ProductVirtualRemainder | 10 | 16 | ✅ Working |
| 1 | RamRafEvaluation | 16 | 10 | ✅ RAF prover integrated |
| 2 | RamReadWriteChecking | 26 | **0** | ❌ **CRITICAL - needs prover** |
| 3 | OutputSumcheck | 16 | 10 | ✅ Working |
| 4 | InstructionLookupsClaimReduction | 10 | 16 | ❌ Needs prover |

## Critical Issue

**Instance 2 (RamReadWriteChecking) starts at round 0 and has non-zero input claim!**

This is the most complex prover (3-phase, 26 rounds) and needs to be implemented
for the batched sumcheck to pass.

Current error at round 0: `s(0)+s(1) != old_claim` because instance 2's
fallback constant polynomial doesn't satisfy the batched constraint.

## Implementation Progress

### Completed
- ✅ RAF prover (Instance 1) produces cubic [s(0), s(1), s(2), s(3)]
- ✅ RAF prover initializes at round 10 with r_cycle from challenges[0..10]
- ✅ Memory trace passed through ConversionConfig
- ✅ RAF prover binds challenges and updates claims properly

### In Progress
- 🔄 RamReadWriteCheckingProver (Instance 2) - CRITICAL

### Pending
- ⏳ InstructionLookupsClaimReductionProver (Instance 4)
- ⏳ Full Jolt verification pass

## Next Steps

1. Study Jolt's RamReadWriteChecking implementation
2. Implement 3-phase prover:
   - Phase 1: read_checking (eq over cycles × ra × val)
   - Phase 2: write_checking (eq × wa × next_val)
   - Phase 3: init_final_checking
3. Integrate into Stage 2 batched sumcheck starting at round 0
4. Test with Jolt verification

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
