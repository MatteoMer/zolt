# Zolt-Jolt Compatibility Implementation

## Status: Session 47 - Stage 4 r_cycle Mismatch (Progress!)

## Progress Summary

### MILESTONE: Stage 2 Now Passes!
After re-enabling termination bit in val_final and val_io, Stage 2 OutputSumcheck verification now passes!

### Fixes Applied This Session
1. **Disabled synthetic termination writes** - Causing R1CS/RAF mismatch
2. **Removed Stage 5 "correction" hack** - Fixed
3. **Fixed ValFinal input_claim calculation** - Using prover's computeInitialClaim()
4. **Re-enabled termination bit in val_final and val_io** - Required for OutputSumcheck

### Current Issue: Stage 4 r_cycle Mismatch

Stage 4 RegistersRWC verification fails because:
```
r_cycle (from sumcheck): [74, e0, 9c, f6...]
params.r_cycle (from Stage 3): [b8, 79, 98, ad...]
```

These should match! The r_cycle comes from Stage 4's sumcheck challenges, but it's different from params.r_cycle passed from Stage 3.

**Analysis**:
- Stage 4 has 3 instances: RegistersRWC, ValEvaluation, ValFinal
- ValEvaluation and ValFinal have zero claims (expected - no RAM writes)
- RegistersRWC uses r_cycle from sumcheck challenges for eq_eval computation
- The r_cycle mismatch causes wrong expected_output_claim

### Key Technical Details

**Stage 4 Instance Structure**:
1. RegistersRWC: Uses r_cycle from sumcheck challenges
2. ValEvaluation: inc*wa*lt = 0 (no RAM ops)
3. ValFinal: inc*wa = 0 (no RAM ops)

**The r_cycle Issue**:
- RegistersRWC's `expected_output_claim` needs `eq(r_cycle, j)` evaluation
- This r_cycle comes from sumcheck challenges via normalize_opening_point
- But params.r_cycle (from Stage 3) is different
- Need to investigate how r_cycle flows from Stage 3 -> Stage 4

### Next Steps
1. Trace r_cycle from Stage 3 to Stage 4
2. Check if normalize_opening_point is correct
3. Verify phase ordering in Stage 4 sumcheck

## Test Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Key Files
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 prover
- `/home/vivado/projects/zolt/src/zkvm/ram/output_check.zig` - OutputSumcheck (now working!)
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/registers/read_write_checking.rs` - RegistersRWC verifier

## Session Commits
1. `39d8386` - Fix Stage 5 claim mismatch: disable synthetic termination writes

## SESSION_ENDING
- Stage 2 is now passing! This is significant progress.
- Stage 4 fails on r_cycle mismatch between sumcheck and params
- Need to investigate r_cycle flow from Stage 3 -> Stage 4 -> RegistersRWC
