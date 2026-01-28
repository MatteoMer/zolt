# Zolt-Jolt Compatibility: Stage 4 Fix

## Current Status
Stage 4 RegistersRWC sumcheck verification fails with `output_claim != expected_claim`.

## Root Cause Found ✅
**Zolt's Stage 4 prover doesn't respect the phase configuration!**

In `stage4_gruen_prover.zig`, line 489:
```zig
if (round < self.log_T) {
    // Phase 1: Binding cycle variables using Gruen
    return self.phase1ComputeMessage(current_claim);
}
```

This treats ALL first log_T rounds as Phase 1 (cycle binding), but the actual config is:
- `phase1_num_rounds = 4` (from ReadWriteConfig)
- `phase2_num_rounds = 7` (LOG_REGISTER_COUNT = 7)
- `phase3_cycle_len = 4` (remaining cycle vars)
- `phase3_address_len = 0` (all address vars handled in phase2)

The correct structure should be:
- Phase 1 (rounds 0-3): Bind cycle vars via Gruen
- Phase 2 (rounds 4-10): Bind address vars via Gruen
- Phase 3 (rounds 11-14): Bind remaining cycle vars (dense)

## Verified ✅
1. Stage 3 challenges match perfectly between Zolt and Jolt
2. Stage 4's `params.r_cycle` (from Stage 3) matches between Zolt and Jolt
3. Stage 4 input claims match
4. Stage 4 batching coefficients match
5. Stage 4 sumcheck challenges match (all 15 rounds)
6. Stage 4 final claims match (val, rs1_ra, rs2_ra, rd_wa, inc)

## Issue
The eq polynomial binding order doesn't match Jolt's because Zolt binds:
- Rounds 0-7: ALL cycle vars (wrong!)
- Rounds 8-14: address vars

But Jolt binds:
- Rounds 0-3: FIRST 4 cycle vars (Phase 1)
- Rounds 4-10: ALL 7 address vars (Phase 2)
- Rounds 11-14: REMAINING 4 cycle vars (Phase 3)

This causes `eq_eval = EqPolynomial::mle_endian(&r_cycle, &params.r_cycle)` to produce different values because `r_cycle` is constructed from Stage 4's sumcheck challenges using `normalize_opening_point`:

```rust
// Jolt's normalize_opening_point for 15 rounds:
r_cycle = [c14, c13, c12, c11, c3, c2, c1, c0]  // NOT [c7, c6, ..., c0]
```

## Fix Required
Modify `stage4_gruen_prover.zig` to:
1. Accept `phase1_num_rounds` and `phase2_num_rounds` from ReadWriteConfig
2. Use 3-phase structure:
   - Phase 1: first `phase1_num_rounds` rounds bind cycle vars via Gruen
   - Phase 2: next `phase2_num_rounds` rounds bind address vars (new!)
   - Phase 3: remaining rounds bind more cycle vars (dense)
3. Ensure the eq polynomial is bound in the same order as Jolt

## Tasks
- [x] Fix ReadWriteConfig to use correct LOG_REGISTER_COUNT (7, not 5)
- [x] Verify Stage 3 challenges match
- [x] Verify params.r_cycle matches
- [x] Identify root cause: phase config not used in Stage 4 prover
- [ ] **IN PROGRESS**: Fix Stage 4 prover to use correct 3-phase structure
- [ ] Verify eq_eval matches after fix
- [ ] Run full verification test

## Key Files
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage4_gruen_prover.zig` - Needs phase config
- `/home/vivado/projects/zolt/src/zkvm/jolt_types.zig` - ReadWriteConfig (fixed)
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/registers/read_write_checking.rs` - Reference implementation

## Commands
```bash
# Generate proof
zig build -Doptimize=ReleaseFast run -- prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Test verification
cd /home/vivado/projects/jolt && cargo test --package jolt-core --no-default-features --features "minimal,zolt-debug" test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
