# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Expected Output Claim Mismatch

## Session 4 Progress

### Fixed: ra_chunks Computation (COMMITTED in this session)

**Root Cause**: Zolt was using `ra_chunk_weights[i][0]` which is just the weight for cycle 0. The correct formula is:
```
ra_claims[i] = Σ_j eq(r_cycle', j) * ra_chunk_weights[i][j]
```
This is the sumcheck claim for each ra chunk polynomial.

**Fix**: Changed the computation to sum over all cycles weighted by `eq(r_cycle', j)`.

**Result**: The ra_product now matches between Zolt and Jolt (`d6 68 c5 b2 ...`)

## Current Issue

Stage 5 sumcheck verification still fails:
- `output_claim`: `[c8, d4, 1b, fc, ...]` (polynomial evaluation at challenges)
- `expected_claim`: `[b8, ef, bc, e1, ...]` (sum of per-instance expected claims)

### What Matches ✓

1. **Input claims** ✓ - All 3 instances match
2. **Batching coefficients** ✓ - batch0, batch1, batch2 all match
3. **Round polynomial coefficients** ✓ - All 136 rounds match
4. **Sumcheck challenges** ✓ - All 136 challenges match
5. **r_address_1 and r_address_2** ✓ - Now identical after previous fix
6. **ra_product** ✓ - Now matches (`d6 68 c5 b2 ...`)

### Per-Instance Expected Claims

- Instance 0 (RegistersValEvaluation) weighted: `[eb, 4c, de, b3, ...]`
- Instance 1 (RamRaClaimReduction) weighted: `[65, 1d, 0a, f6, ...]`
- Instance 2 (InstructionReadRaf) weighted: `[69, 85, d4, 27, ...]`

### The Problem

The polynomial's output_claim doesn't match the sum of the three expected claims. Since:
- The round polynomial coefficients match
- The ra_product matches
- But expected_claim != output_claim

The remaining issue must be in how one of the expected_output_claim formulas is computed by the verifier using the opening claims.

Specifically, Instance 2's formula is:
```
expected = eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
```

Components from Jolt debug:
- eq_eval_r_reduction: `[01, 29, 32, 86, ...]`
- ra_claim: `[d6, 68, c5, b2, ...]` ✓
- raf_flag_claim: `[6e, e5, 67, de, ...]`
- raf_claim: `[0a, 84, af, 51, ...]`
- val_claim: `[f6, 6e, 76, c0, ...]`
- gamma: `[5a, b9, a0, 12, ...]`
- final_result: `[33, b4, 2d, 87, ...]`

### Next Steps

1. Verify val_claim computation:
   - val_claim = Σ table_flags[i] * table_eval[i]
   - Need to check if Zolt's table_flags match Jolt's expectations

2. Verify raf_claim computation:
   - raf_claim = (1 - raf_flag) * (left_op + gamma*right_op) + raf_flag * gamma * identity
   - Need to check raf_flag_claim matches

3. Check if the problem is in the **polynomial construction** rather than the opening claims
   - The polynomial output_claim is computed from the round coefficients
   - If opening claims are correct but output doesn't match expected, the prover polynomial might have a structural error

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
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Proof converter
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf verifier
