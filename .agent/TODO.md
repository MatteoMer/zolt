# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Expected Output Claim Mismatch

## Current Issue (Session 2 - Continued Investigation)

Stage 5 sumcheck verification fails with:
- `output_claim` (from proof): `[ce, b6, 9e, 59, 52, e3, 16, ed, ...]`
- `expected_claim` (computed): `[b7, c0, 2e, 8e, c4, 3e, bd, 5d, ...]`

### Verified Components (All Match)

1. **Round polynomial coefficients** ✓ - First 3 rounds match exactly
2. **Sumcheck challenges** ✓ - All 136 challenges match (verified rounds 128-135)
3. **InstructionRa claims** ✓ - All 8 chunks match
4. **ra_product** ✓ - Product matches (`[7f, 43, eb, f8, ...]`)
5. **LookupTableFlag claims** ✓ - All 42 flags match
6. **raf_claim** ✓ - Formula computation matches (`[92, 5a, 0b, 79, ...]`)
7. **eq_eval_r_reduction** ✓ - Matches (`[f6, fa, a0, 07, ...]`)
8. **left/right/identity prefix evaluations** ✓ - All match
9. **batching coefficients** ✓ - batch0, batch1, batch2 all match
10. **input_claims** ✓ - RegistersVal and other Stage 5 inputs match
11. **r_reduction** ✓ - Stage 3 challenges match
12. **ram_ra_claim** ✓ - RamRa@RamRaClaimReduction value matches

### Per-Instance Expected Output Claims from Jolt

| Instance | Claim (LE hex first 16 bytes) | Coeff (first 8) | Weighted |
|----------|-------------------------------|-----------------|----------|
| 0 (RegistersValEvaluation) | `[4a, 87, c9, 11, 88, 1e, 02, 9a, ...]` | `[04, 97, 3d, 64, ...]` | `[79, ab, e0, 49, ...]` |
| 1 (RamRaClaimReduction) | `[5d, c0, 20, b8, ab, a2, 44, 3e, ...]` | `[50, 2a, 19, a0, ...]` | `[57, e5, bc, c3, ...]` |
| 2 (InstructionReadRaf) | `[10, fe, 87, dc, b4, 79, 2b, 97, ...]` | `[45, 50, 75, e2, ...]` | `[e7, 2f, 91, 80, ...]` |

### Key Observation

The output_claim is computed correctly from the round polynomials - it's the result of evaluating the sumcheck polynomial at all 136 challenge points.

The expected_claim is computed from the instance-specific formulas. Since they don't match, the prover's polynomial doesn't match what the verifier expects.

**But** round polynomial coefficients match for the first 3 rounds. This is puzzling - if the polynomial is wrong, why would coefficients match?

### Root Cause Hypothesis

The discrepancy may be in later rounds (not the first 3). The polynomial construction for each round must match exactly what the verifier formula expects. Even if early rounds match, a divergence in later rounds (especially cycle rounds 128-135) would cause the final output_claim to differ from expected_claim.

### Next Steps

1. Compare more round coefficients (especially rounds 128-135 where all 3 instances are active)
2. Verify Instance 0's polynomial computation during cycle rounds
3. Verify Instance 1's PhaseCycle2 polynomial computation
4. Check if there's a mismatch in how Instance 2's high-degree polynomial is combined

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
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - Jolt sumcheck verifier
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf verifier
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/claim_reductions/ram_ra.rs` - RamRaClaimReduction verifier
