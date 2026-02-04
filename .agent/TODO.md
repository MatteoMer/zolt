# Zolt-Jolt Compatibility Implementation

## Status: Session 63 - Fixed ValFinal start_address, now debugging expected_output_claim

## Current Issue

Stage 4 sumcheck verification still fails with output_claim vs expected_claim mismatch.

### Progress This Session

**FIXED: ValFinal start_address mismatch**
- Problem: ValFinal prover was using `RAM_START_ADDRESS` (0x80000000)
- But Jolt's ValFinal uses `getLowestAddress()` (0x7FFF8000) to include termination bit
- Fix: Changed ValFinal to use `getLowestAddress()` via memory_layout
- Result: `Match val_final? true` - prover's initial claim now matches input_claim

**VERIFIED: ValEvaluation working correctly**
- `Match val_eval? true`
- inc_claim = 0, wa_claim = 0, lt_eval computed correctly (no RAM operations)

**CURRENT ISSUE: ValFinal expected_output_claim mismatch**
- Zolt's ValFinal inc*wa = `{ 17, 181, 185, 137, ... }` (first 8 bytes)
- Jolt's ValFinal inc*wa = `[dc, e8, fd, 7c, ...]`
- These don't match!

The root cause appears to be that the `inc_eval` and `wa_eval` values after Stage 4 sumcheck binding are different between Zolt and Jolt.

### Analysis

The ValFinal sumcheck proves: `Σ_j inc(j) * wa(r_address, j) = val_final(r_address) - val_init(r_address)`

After binding through Stage 4 sumcheck challenges:
- `inc_eval = inc(r_cycle_prime)` - inc polynomial at Stage 4's opening point
- `wa_eval = wa(r_address, r_cycle_prime)` - wa polynomial at combined point

The issue is that the final polynomial openings don't match what Jolt expects.

Possible causes:
1. Different opening point construction (r_address, r_cycle_prime)
2. Different polynomial initialization (trace data, start_address)
3. Different binding order or formula

### Key Files

1. `/home/vivado/projects/zolt/src/zkvm/ram/val_final.zig` - ValFinal prover
2. `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 prover, ValFinal init
3. `/home/vivado/projects/jolt/jolt-core/src/zkvm/ram/val_final.rs` - Jolt's ValFinal expected_output_claim

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Session 63 Summary

1. Fixed ValFinal prover to use `getLowestAddress()` instead of `RAM_START_ADDRESS`
2. ValFinal now includes termination bit contribution (at 0x7FFFC008)
3. `Match val_final? true` - input_claim matches prover's initial claim
4. Remaining issue: ValFinal's final polynomial openings (inc_eval, wa_eval) don't match Jolt's expected values

## Next Steps

1. Debug why ValFinal's inc_eval and wa_eval after binding don't match Jolt's expectations
2. Verify the opening point construction is correct (r_address from Stage 2, r_cycle_prime from Stage 4)
3. Compare polynomial binding order with Jolt (LowToHigh)
4. Test the complete fix with Jolt verification
