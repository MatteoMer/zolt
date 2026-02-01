# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Cycle Round Polynomial Computation

## Current Session Progress (Session 89)

### Key Findings

1. **Opening claims ARE being transmitted correctly**
   - Verified ra_chunks match between Zolt and Jolt
   - table_flags and raf_flag are stored correctly
   - r_reduction from Stage 2 is passed correctly to Stage 5

2. **Sumcheck challenges match**
   - Round 128-135 challenges are identical between Zolt prover and Jolt verifier
   - This confirms the polynomial coefficients and transcript are synchronized

3. **The ISSUE: Final output_claim doesn't match expected_claim**
   - Zolt's final batched claim (big-endian): `2e1261a71e90c9cb48db6fadd357eee783dca70b9327673139f69c9a18d595df`
   - Jolt's output_claim (little-endian): `7ca6b67cbde8581df753e63ab4aba00389...`
   - These are completely different!

4. **Root Cause: Cycle round polynomial computation is wrong**
   - The prover computes polynomials for rounds 128-135 (cycle variables)
   - These polynomials should evaluate to values that, when summed across all cycles, give the correct claim
   - But the verifier expects: `eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)`
   - The polynomial doesn't match this formula

### Debug Output Summary
- Jolt verifier expected Instance 2 claim: `[d8, ab, 4b, 5b, c9, 3e, db, 30, e7, a8, 71, e9, 21, ba, c9, f9, ...]`
- Jolt verifier expected_claim (batched): `[ae, 77, e6, c2, 75, bd, 38, 88, ...]`
- Jolt verifier output_claim (from sumcheck): `[7c, a6, b6, 7c, bd, e8, 58, 1d, ...]`
- Output != Expected → Verification fails

### Files Changed This Session
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`
  - Added debug for ra_chunk claims, table_flag claims, raf_flag_claim
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig`
  - Updated debug print condition for cycle rounds

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Next Steps for Future Session
1. **Investigate the cycle round polynomial formula in Zolt**
   - Current implementation uses `finishMlesProductSumFromEvals` which may not match Jolt
   - Need to verify the eq*ra*combined product computation

2. **Compare Zolt's polynomial evaluations with Jolt's expected structure**
   - Jolt's verifier expects: `eq(r_reduction, r_cycle') * ra * (val + gamma * raf)`
   - Zolt's prover computes: `Σ eq(r_reduction, j) * Π_c ra_chunk_c(j) * combined(j)`
   - These need to be equivalent

3. **Check if the ra_chunk binding is correct during cycle rounds**
   - The ra_chunks are bound during address rounds (0-127)
   - During cycle rounds (128-135), the eq and combined_vals are bound
   - The final ra_claim should be the PRODUCT of the bound chunk values

4. **Verify the polynomial degree and Toom-Cook conversion**
   - Cycle rounds should have degree 10 (product of 9 linear factors + eq)
   - The compressed format must match Jolt's expectations

SESSION_ENDING - Context getting long, key findings documented above.
