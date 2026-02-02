# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Opening Claims Expected Output

## Session 138 Progress

### Fixed: Polynomial Evaluation Bug

The Stage 5 sumcheck was using incorrect Lagrange interpolation. Fixed by using Horner's method.

### Current Status

The sumcheck produces matching output_claim between Zolt and Jolt:
- output_claim: `[84, 83, e6, 0a, ...]` (matches)
- expected_claim: `[c6, 19, df, ae, ...]` (computed from opening claims - MISMATCH)

### Verified Components

1. **Polynomial evaluation** - Fixed, sumcheck output matches
2. **Opening claims serialization** - Correct (LookupTableFlag, InstructionRa match)
3. **Table MLE evaluations** - Correct (table[0], table[1], table[9] match between Zolt and Jolt)
4. **Challenge serialization** - Values appear to be serialized/deserialized correctly

### Remaining Issue

The expected_output_claim computation in Jolt's verifier produces a different result than
what the sumcheck proves. Since individual components match, the issue might be in:

1. How components are combined in expected_output_claim
2. The gamma/batching coefficient computation
3. The EQ polynomial evaluation (r_reduction, r_cycle_prime)

### Debug Output Analysis

From Jolt verification:
```
Stage 5 Instance 2 expected_output_claim:
  claim: [15, 9b, 8f, 56, ...]
  final_result: [15, 9b, 8f, 56, 56, fe, 8f, 9a, 46, 83, ff, 8b, a2, 52, 61, 10]

InstructionReadRaf expected_output_claim debug:
  left_operand_eval:  [6f, af, 08, 28, ...]
  right_operand_eval: [c1, 8f, 3e, c0, ...]
  identity_poly_eval: [7b, 1d, 43, db, ...]
  gamma:              [5a, b9, a0, 12, ...]
  eq_eval_r_reduction: [e0, 9c, 4e, 2a, ...]
  ra_claim:           [a9, c2, 35, c8, ...]
  raf_flag_claim:     [e0, af, 49, 51, ...]
  raf_claim:          [5b, 62, ca, 7d, ...]
  val_claim:          [75, 20, 72, 98, ...]
```

These values come from Jolt reading Zolt's proof. Need to compare what Zolt computes for
these components during proof generation.

### Next Steps

1. Add debug output in Zolt to print these same values during proof generation
2. Compare each component to find the mismatch
3. Focus on: gamma, eq_eval_r_reduction, raf_claim, val_claim

### Test Commands

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
- `/home/vivado/projects/zolt/src/poly/mod.zig` - UniPoly with evaluateToomCookAt
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Instance 2 verification
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - Sumcheck verifier
