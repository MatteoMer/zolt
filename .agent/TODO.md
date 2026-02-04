# Zolt-Jolt Compatibility Implementation

## Status: Session 67 - Instance 1 Hint Mechanism Fix (In Progress)

## Current Issue

Stage 5 sumcheck verification fails. The combined polynomial doesn't satisfy `p(0) + p(1) = current_batched_claim`.

**Root cause identified**: The batched claim tracking is inconsistent with individual instance claims.

## Progress Made This Session

### 1. Identified the Batched Claim Mismatch
- Debug shows `current_batched_claim matches expected: false` for round 128
- The `expected_batched` (from individual claims) matches `hint_expected` (from polynomial)
- But `current_batched_claim` (the hint) doesn't match

### 2. Fixes Implemented
1. **Scaled claim initialization**: Changed `regs_val_current_claim` and `ram_ra_current_claim` to use SCALED values at initialization
2. **Individual claim tracking for address rounds**: Added halving for inactive instances during address rounds (0-127)
3. **PhaseAddress claim update**: Added `ram_ra_current_claim` update after PhaseAddress binding (rounds 112-127)
4. **Batched claim recomputation**: Moved to after RamRaClaimReduction binding in address rounds

### 3. Current State
- Rounds 129-135: `current_batched_claim matches expected: true` ✓
- Round 128: `current_batched_claim matches expected: false` ✗

The issue is that the batched claim at round 128 START was set at the END of round 127 (address round), but there's still a mismatch.

## Next Steps

1. **Debug round 127**: Add explicit debug to see what values are being used to compute `current_batched_claim` at the end of round 127

2. **Verify Instance 1 claim evolution**: Trace `ram_ra_current_claim` through PhaseAddress rounds 112-127 to ensure it's being updated correctly

3. **Check code path**: Round 127 is the last address round. Verify that the batched claim recomputation is using the UPDATED `ram_ra_current_claim` value

## Analysis

At round 128 start, the debug shows:
- `regs_val_current_claim = 3c9c41f47d8f524d9a5f5c701f596169` (unscaled, after 128 halvings)
- `ram_ra_current_claim = 1199386937d5c9ad1891fe9b6031d657` (after PhaseAddress rounds)
- `lookups_claim = f47fe430cf070d741a203164625e7bab` (after address rounds)

Expected batched claim: `{ 220, 42, 164, 10, ...}` = batch0*c0 + batch1*c1 + batch2*c2
Actual batched claim: `{ 34, 28, 172, 85, ...}` ≠ expected

This suggests the batched claim wasn't correctly computed at the end of round 127.

## Files Modified

- `src/zkvm/spartan/stage5_prover.zig`:
  - Line 1199: Changed `regs_val_current_claim = regs_scaled` (was `regs_val_input`)
  - Line 1408: Changed `ram_ra_current_claim = ram_ra_scaled_corrected`
  - Lines 2412-2425: Added individual claim updates for address rounds
  - Line 2517-2524: Added PhaseAddress claim update
  - Line 2604-2609: Moved batched claim recomputation to after RamRaClaimReduction binding

## Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Previous Progress

- Stage 1-4: WORKING
- Stage 5 address rounds (0-127): Sumcheck passes individually
- Stage 5 cycle rounds (128-135): Batched claim consistency achieved for rounds 129-135
- Round 128 transition: Still has batched claim mismatch
