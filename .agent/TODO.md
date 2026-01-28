# Zolt-Jolt Compatibility: Stage 4 Fix

## Current Status
Stage 4 RegistersRWC sumcheck verification still fails with `output_claim != expected_claim`.

## Progress Made ✅

### 1. Implemented 3-Phase Structure
The Stage 4 prover now correctly uses:
- **Phase 1** (rounds 0-3): Bind first 4 cycle vars via Gruen
- **Phase 2** (rounds 4-10): Bind 7 address vars (eq NOT bound)
- **Phase 3** (rounds 11-14): Bind remaining 4 cycle vars via merged dense eq

### 2. Key Code Changes
- `stage4_gruen_prover.zig`:
  - Added `phase1_num_rounds` and `phase2_num_rounds` fields
  - Added `merged_eq` field for dense eq after Phase 1
  - Implemented `phase2ComputeMessage` for address binding
  - Implemented `phase3ComputeMessage` for remaining cycle binding
  - Updated `bindPolynomials` with 3-phase binding logic

- `gruen_eq.zig`:
  - Added `merge()` function to convert Gruen eq to dense polynomial

### 3. Verified Working
- Stage 3 challenges match perfectly between Zolt and Jolt ✅
- Phase structure debug output shows correct phase transitions ✅
- Phase 3 produces non-zero polynomial coefficients ✅

## Current Issue

**Round 14 polynomial mismatch:**
- Zolt computes non-zero coefficients for round 14:
  - `c0 = {15, 233, 235, 113, ...}` (non-zero)
  - `c1 = {105, 121, 0, 145, ...}` (non-zero)
  - `c2 = {137, 157, 19, 237, ...}` (non-zero)
- But Jolt deserializes all zeros for round 14

**Possible causes:**
1. Serialization issue for round 14
2. Different degree bounds between implementations
3. The actual polynomial SHOULD be zero but Zolt is computing wrong values

**Note:** Looking at earlier rounds in Jolt's output, Phase 3 rounds (11-13) have `c3 = 0` (degree 2), which Zolt is producing correctly. But round 14's c0 is also zero in Jolt, suggesting a fundamental difference.

## Investigation Needed

1. Compare how Jolt handles the final round of Phase 3
2. Check if there's special handling for when `current_T = 2`
3. Verify the polynomial evaluation formula is correct for the final round
4. Check if Jolt's `inc.len() == 1` case applies here (it shouldn't for RegistersRWC)

## Next Steps
- [ ] Add detailed debug for round 14 serialization
- [ ] Compare Zolt and Jolt polynomial values at round 14
- [ ] Verify merged_eq binding is correct in Phase 3
- [ ] Check if final eq_eval matches between implementations

## Commands
```bash
# Generate proof
zig build -Doptimize=ReleaseFast run -- prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Test verification
cd /home/vivado/projects/jolt && cargo test --package jolt-core --no-default-features --features "minimal,zolt-debug" test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Key Files
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage4_gruen_prover.zig`
- `/home/vivado/projects/zolt/src/zkvm/spartan/gruen_eq.zig`
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig`
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/registers/read_write_checking.rs`
