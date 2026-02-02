# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Expected Output Claim Mismatch

## Session 3 Progress

### Fixed: r_address_raf and r_address_rw Mismatch

**Root Cause**: Zolt was using the pre-sampled `r_address` (from OutputSumcheck) for `r_address_raf`, but this is incorrect. Both `r_address_raf` and `r_address_rw` should be computed from the Stage 2 sumcheck challenges.

**Fix**: In `proof_converter.zig`, changed `r_address_raf` to be computed the same way as `r_address_rw` - from the Stage 2 sumcheck challenges, reversed to big-endian order.

Now both `eq_addr_1` and `eq_addr_2` are equal in Jolt's verifier, matching the expected behavior.

## Current Issue

Stage 5 sumcheck verification still fails with:
- `output_claim`: `[c8, d4, 1b, fc, ...]`
- `expected_claim`: `[e5, b7, 3b, 32, ...]`

### Verified Components (All Match)

1. **r_address_1 and r_address_2** ✓ - Now identical (both derived from Stage 2 sumcheck challenges)
2. **eq_addr_1 and eq_addr_2** ✓ - Now identical: `[d5, c4, 7d, 93, ...]`
3. **Round polynomial coefficients** ✓ - First 3 rounds match exactly
4. **Sumcheck challenges** ✓ - All 136 challenges match
5. **InstructionRa claims** ✓ - All 8 chunks match
6. **ra_product** ✓ - Matches
7. **LookupTableFlag claims** ✓ - All 42 flags match
8. **raf_claim** ✓ - Matches
9. **eq_eval_r_reduction** ✓ - Matches
10. **batching coefficients** ✓ - batch0, batch1, batch2 all match
11. **input_claims** ✓ - RegistersVal and other Stage 5 inputs match
12. **r_reduction** ✓ - Stage 3 challenges match
13. **ram_ra_claim** ✓ - RamRa@RamRaClaimReduction value matches

### Per-Instance Expected Output Claims from Jolt

| Instance | Claim (LE hex) | Weighted |
|----------|----------------|----------|
| 0 (RegistersValEvaluation) | `[30, 55, 62, 42, ...]` | `[eb, 4c, de, b3, ...]` |
| 1 (RamRaClaimReduction) | `[9a, fa, 32, 06, ...]` | `[65, 1d, 0a, f6, ...]` |
| 2 (InstructionReadRaf) | `[3b, 96, 39, b1, ...]` | `[96, 4d, 53, 78, ...]` |

### Next Steps

1. Compare the weighted sum of individual instance claims to the expected_claim
2. Check if there's a mismatch in Zolt's polynomial computation during later rounds
3. Verify that Stage 5 prover generates correct round polynomials for all 136 rounds

## Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cp /tmp/zolt_*.bin /home/vivado/projects/jolt/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Key Files

- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Proof converter (r_address fix made here)
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - Jolt sumcheck verifier
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf verifier
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/claim_reductions/ram_ra.rs` - RamRaClaimReduction verifier
