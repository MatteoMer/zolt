# Zolt-Jolt Compatibility Implementation

## Status: Session 65 - Stage 4 FIXED, Stage 5 investigation started

## Current Issue

Stage 5 (InstructionReadRaf) sumcheck verification fails with output_claim vs expected_claim mismatch.

## Session 65 Progress

### Stage 4 Fix (COMPLETED)

Fixed several critical issues in Stage 4 sumcheck:

1. **ValEvaluation hint mechanism**: Changed from `p(1) = claim - p(0)` to `p(0) = claim - p(1)` to match Jolt's convention.

2. **ValFinal polynomial evaluation**: Changed from Toom-Cook to Lagrange interpolation (degree-2).

3. **ValFinal hint mechanism**: Keep actual p(2) unchanged when applying hint to p(1).

4. **ValFinal combined polynomial contribution**: Don't add c2 to c3 slot (degree-2 has no cubic term).

### Stage 5 Investigation (STARTED)

Stage 5 (InstructionReadRaf) is a complex batched sumcheck with:
- Total 136 rounds: 128 address + 8 cycle rounds
- Degree: 10 (with default config: 8 virtual RA polys + 2)

Two phases:
1. **Address rounds (0-127)**: Uses `from_evals_and_hint` with [p(0), p(2)] - degree-2
2. **Cycle rounds (128-135)**: Uses `finish_mles_product_sum_from_evals` with Gruen-split EQ optimization

The cycle rounds use a different hint mechanism:
- Input `sum_evals` contains evaluations on `[1, 2, ..., d-1, ∞]` (NOT including 0!)
- Uses hint to compute `eval_at_0 = (claim - eq_eval_at_1 * eval_at_1) / eq_eval_at_0`
- Then interpolates with `from_evals_toom`
- Multiplies result by EQ factor

This is more complex than Stage 4 - need to analyze Zolt's Stage 5 prover to find mismatches.

### Key Files

**Jolt Stage 5**:
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/mles_product_sum.rs` - `finish_mles_product_sum_from_evals`

**Zolt Stage 5**:
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig`
- `/home/vivado/projects/zolt/src/zkvm/spartan/instruction_read_raf.zig`

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Next Steps

1. Compare Zolt's Stage 5 prover with Jolt's read_raf_checking.rs
2. Focus on:
   - How cycle rounds compute evaluations
   - How the EQ polynomial (Gruen-split) is handled
   - The hint mechanism in `finish_mles_product_sum_from_evals`
3. May need to implement similar Gruen-split EQ handling in Zolt

## Technical Notes

### Hint Mechanisms by Stage

**Stage 4**:
- ValFinal (degree-2): `p(1) = claim - p(0)`, keep actual p(0) and p(2)
- ValEvaluation (degree-3): `p(0) = claim - p(1)`, keep actual p(1), p(2), c3
- Registers (degree-3): standard `p(1) = claim - p(0)`

**Stage 5** (InstructionReadRaf):
- Address rounds: `from_evals_and_hint` with [p(0), p(2)]
- Cycle rounds: `finish_mles_product_sum_from_evals` with special EQ handling
  - `eval_at_0 = (claim - eq(1,r) * eval_at_1) / eq(0,r)`
  - where `eq(0,r) = 1-r` and `eq(1,r) = r`

### Polynomial Degrees

- Stage 4: Mixed (3, 3, 2) → combined degree 3
- Stage 5: Mixed (variable) → combined degree 10+ for cycle rounds
