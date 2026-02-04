# Zolt-Jolt Compatibility Implementation

## Status: Session 60 - Fixed LT Polynomial, Investigating r_cycle Mismatch

## Current Issue

Stage 4 sumcheck verification fails. After fixing the LT polynomial to use BE formula, the lt_eval values still don't match between Zolt and Jolt.

### Changes Made This Session

1. **Fixed LT polynomial to use BIG_ENDIAN formula** (val_evaluation.zig):
   - Rewrote `LtPolynomial` to precompute evaluations using Jolt's algorithm
   - Changed `init` to take r_cycle in BE order
   - Updated `evaluateAtIndex` to use the precomputed evals table

2. **Changed r_cycle passed to ValEvaluation** (proof_converter.zig):
   - Now passes `r_cycle_be` instead of `r_cycle_le`
   - This should match Jolt's verifier which uses BE for LT computation

### Debug Output Analysis

**What MATCHES:**
1. Zolt's lt_eval_computed using Jolt formula matches lt_eval_prover (from binding)
2. Stage 2 challenges appear to produce correct r_cycle_be values

**What DOESN'T MATCH:**
1. Jolt's opening_point for RamVal @ RamReadWriteChecking:
   - Jolt shows point.r[0] = [00...00, 0d, 8d, 89, b0, c0, ef, 00, b0, 84, a4, 8a, 1b, 0b, 14, 34, 07]
   - Zolt's Stage 2 Round 7: { 118, 202, 103, 155, ...} = { 0x76, 0xCA, 0x67, 0x9B, ...}
   - These are DIFFERENT!

2. ValEvaluation's lt_eval:
   - Jolt verifier: [2e, 62, dc, c0, ...]
   - Zolt prover: { 32, 215, 253, 66, ...} = { 0x20, 0xD7, 0xFD, 0x42, ...}
   - These are DIFFERENT!

### Root Cause Hypothesis

The opening_point stored in the verifier's accumulator for RamVal @ RamReadWriteChecking has different values than what Zolt computed from Stage 2 challenges. This affects:
1. The r_cycle used by ValEvaluation's verifier for LT computation
2. The r_address used for various eq polynomial computations

### Puzzling Observation

If the Stage 2 challenges were different between Zolt and Jolt, the sumcheck verification would fail at Stage 2. But it fails at Stage 4, meaning Stages 1-3 passed.

Possible explanations:
1. The opening_point is reconstructed from a different source than the sumcheck challenges
2. There's additional processing that differs between prover and verifier
3. The debug output is misleading in some way

### Next Steps

1. **Add more debug to Jolt** to see what r_cycle the verifier uses in ValEvaluation's expected_output_claim
2. **Compare Stage 2 challenges byte-by-byte** between Zolt and Jolt
3. **Check if opening_points are stored/retrieved correctly** in the accumulator
4. **Verify the normalize_opening_point logic** matches between Zolt and Jolt

### Key Files

1. `/home/vivado/projects/zolt/src/zkvm/ram/val_evaluation.zig` - Fixed LT polynomial
2. `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 prover, r_cycle handling
3. `/home/vivado/projects/jolt/jolt-core/src/zkvm/ram/val_evaluation.rs` - Jolt's expected_output_claim
4. `/home/vivado/projects/jolt/jolt-core/src/zkvm/ram/read_write_checking.rs` - normalize_opening_point

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Session 60 Summary

Fixed the LT polynomial implementation to use Jolt's BE formula with precomputed evaluations. Changed ValEvaluation prover to use r_cycle_be instead of r_cycle_le.

The lt_eval values still don't match because the r_cycle used by Jolt's verifier (from the stored opening_point) differs from what Zolt computes from Stage 2 challenges. Need to investigate why the opening_points differ.
