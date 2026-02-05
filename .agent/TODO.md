# Zolt-Jolt Compatibility Implementation

## Status: Session 87 - Stage 5 brute force FIXED, sumcheck still fails

## Current Issue: Stage 5 sumcheck output_claim ≠ expected_claim

### Root Cause Found and Fixed (this session)
The termination store used virtual register slots (k=32,33) in val_poly which broke
the ValEvaluation identity `Val(r) = Σ inc*wa*LT`. Virtual slots had non-zero val
but no corresponding inc/wa entries.

**Fix**: Replaced synthetic `SB x0, 0(x0)` with 4 synthetic instructions:
1. NoOp (satisfies j.'s NextIsNoop=1)
2. LUI x31, upper20(addr) — real register write
3. ADDI x30, x0, 1 — real register write
4. SB x30, lower12(x31) — real store with real register values

This fixes the val_claim mismatch between Stage 4 and Stage 5:
- `[STAGE5 BRUTE] match? true` ✅

### Remaining Issue
Stage 5 sumcheck verification still fails with different output_claim vs expected_claim.
This is likely due to one of the other two instances:
- Instance 1: RamRaClaimReduction
- Instance 2: LookupsReadRaf

The claim values are close (first byte matches: 73) but diverge after that.

### Changes Made This Session
1. **tracer/mod.zig**: Rewrote `recordTerminationWrite()` to emit 4 synthetic trace steps
   (noop + LUI x31 + ADDI x30 + SB) using real register values
2. **stage4_gruen_prover.zig**: Removed virtual register slot handling (lines 205-240)
   - Simplified to just `if (step.is_noop) continue`
3. **stage4_prover.zig**: Same simplification

### Results After Fix
- Stage 1: PASSES ✅
- Stage 2: PASSES ✅
- Stage 3: PASSES ✅
- Stage 4: PASSES ✅
- Stage 5: Brute force FIXED ✅, but sumcheck output_claim still mismatches
  - Instance 0 (RegistersValEvaluation): Fixed
  - Instance 1 (RamRaClaimReduction): Needs investigation
  - Instance 2 (LookupsReadRaf): Needs investigation
- Stages 6-7: Not yet reached

### Next Steps
1. Debug Stage 5 sumcheck mismatch — likely Instance 1 or 2
2. Check if the new termination steps affect RamRaClaimReduction or LookupsReadRaf
3. Once Stage 5 passes, fix Stages 6-7
4. Clean up diagnostic prints

## Debug Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
