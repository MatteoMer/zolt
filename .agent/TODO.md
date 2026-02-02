# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Expected Claim Mismatch

## Session 129 Summary

### Progress Made

1. **ra_claims NOW MATCH** - After all binding rounds, the InstructionRa chunks are correct:
   - Zolt: `InstructionRa(0) = 119b26350ef30d2127138e6672c75ea5` (matches Jolt!)
   - The initial value `90fa96e6...` transforms to `119b2635...` after 8 binding rounds

2. **Instance 0 (RegistersValEvaluation) CORRECT**:
   - `expected_product = inc_claim * wa_claim * lt_eval` matches
   - Zolt: `2e133a8eb83ed52d70c91f47fe5c8d8c118ac4a7969d53212f19f47fa1b9a265`
   - Jolt: `2e133a8eb83ed52d70c91f47fe5c8d8c118ac4a7969d53212f19f47fa1b9a265` ✓

3. **Instance 1 (RamRaClaimReduction) ra_claim_reduced MATCHES**:
   - Zolt ram_ra_claim: `0956072d38428d511d5342e39c916fe33a948b3e88eff3755eb97c124ab471ff`
   - Jolt ra_claim_reduced: matches (same value)
   - But eq_combined * ra_claim computation needs verification

4. **Instance 2 (InstructionReadRaf) final_result MATCHES**:
   - Jolt final_result: `[b1, 30, 62, bc, a8, dd, 8c, d2, 53, f9, 1f, 69, f5, 7b, e1, 72]`
   - This matches Instance 2's claim in the debug output

### The Remaining Issue

The sumcheck output_claim doesn't match expected_claim:
```
output_claim:   [43, 99, b3, 5d, b4, 2d, 6b, ae, bf, 91, 7b, dd, d4, 96, ac, ce, ...]
expected_claim: [a5, 06, a9, 32, b6, e9, 68, 44, 5c, f2, 37, 59, df, fa, 57, c6, ...]
```

But:
- Zolt's final batched claim matches output_claim
- All three instance claims individually appear correct

This suggests the issue is either:
1. The sumcheck polynomial coefficients in some round
2. A challenge mismatch between prover and verifier
3. Different round offsets or slicing

### Debug Data

**Instance Claims (all appear correct):**
- Instance 0 claim: `[65, a2, b9, a1, 7f, f4, 19, 2f, ...]`
- Instance 1 claim: `[1b, bf, d3, c0, 17, 46, 79, 99, ...]`
- Instance 2 claim: `[b1, 30, 62, bc, a8, dd, 8c, d2, ...]`

**Batch Coefficients:**
- Instance 0 coeff: `[04, 97, 3d, 64, ...]`
- Instance 1 coeff: `[50, 2a, 19, a0, ...]`
- Instance 2 coeff: `[45, 50, 75, e2, ...]`

### Next Steps

1. Compare sumcheck polynomial coefficients for each round between Zolt and Jolt
2. Verify the round offset calculations are correct
3. Check if batching coefficients match between prover and verifier
4. Trace a few specific rounds to identify where the divergence happens

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy and verify
cp logs/zolt_*.bin /tmp/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
