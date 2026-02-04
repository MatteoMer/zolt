# Zolt-Jolt Compatibility Implementation

## Status: Session 62 - Fixed ValEvaluation, now debugging Registers RWC

## Current Issue

Stage 4 sumcheck verification fails with output_claim vs expected_claim mismatch.

### Progress This Session

**FIXED: ValEvaluation start_address mismatch**
- Problem: ValEvaluation prover was using `memory_layout.getLowestAddress() = 0x7FFF8000`
- But RWC only tracks RAM region (0x80000000+)
- Fix: Changed ValEvaluation to use `constants.RAM_START_ADDRESS`
- Result: `Match val_eval? true` - prover's initial claim now matches input_claim

**VERIFIED: LT polynomial and r_cycle values match between Zolt and Jolt**
- Stage 2 challenges match perfectly
- r_cycle values derived from challenges also match
- lt_eval computed by Zolt (after binding) matches Jolt's expected value (just byte ordering difference in debug output)

### Current Issue: Stage 4 Instance 0 (Registers RWC)

The sumcheck final claim doesn't match Jolt's expected claim for Instance 0:
```
output_claim:   [14, 98, cf, e7, c2, ee, 31, 57, de, 0a, c1, 6e, 89, 0a, c6, 61, ...]
expected_claim: [2a, 83, f2, 6d, 32, 1a, db, 6d, 4f, 3b, 1c, da, fd, 8a, 76, 36, ...]
```

Instance 0 expected_output_claim:
- `claim: [e7, d0, 3d, c6, ...]`
- `coeff: [53, dd, 21, 20, ...]`
- `claim*coeff: [2a, 83, f2, 6d, ...]` (this is expected_claim for Instance 0)

ValEvaluation and ValFinal now have:
- `inc_claim = 0`
- `wa_claim = 0`
- `result = 0` (correct since no RAM operations in tracked region)

### Key Files

1. `/home/vivado/projects/zolt/src/zkvm/ram/val_evaluation.zig` - LT polynomial
2. `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 prover, start_address fix
3. `/home/vivado/projects/jolt/jolt-core/src/zkvm/ram/val_evaluation.rs` - Jolt's expected_output_claim

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Session 62 Summary

1. Identified that ValEvaluation prover was using wrong start_address (0x7FFF8000 vs 0x80000000)
2. Fixed by using `constants.RAM_START_ADDRESS` for ValEvaluation
3. ValEvaluation now works correctly (Match val_eval? true)
4. Remaining issue: Registers RWC (Stage 4 Instance 0) expected_output_claim mismatch

## Next Steps

1. Debug why Registers RWC expected_output_claim doesn't match
2. Compare Zolt's Registers RWC prover output with Jolt's expectations
3. Check if the polynomial round values are being computed correctly

