# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Output Claim Mismatch

## Session 109 Summary

### Progress Made
The Stage 5 sumcheck loop now completes all 136 rounds! The proof generation finishes successfully but verification still fails.

### Current Issue
Stage 5 sumcheck verification fails because the output_claim doesn't match the expected_claim:
```
output_claim:   [eb, 1c, 1a, 7c, 50, c5, 1b, 64, dd, 58, 39, 41, a8, d8, 94, 28, ...]
expected_claim: [76, 19, 2f, 98, 45, 38, 7b, 09, b3, 3c, 7f, 8b, b0, ac, cd, b0, ...]
```

The expected_claim is computed by Jolt's verifier using:
```
expected = batch0*inst0_claim + batch1*inst1_claim + batch2*inst2_claim

where inst2_claim (InstructionReadRaf) =
    eq(r_reduction, r_cycle_prime) × Π_i ra_claim[i] × (val_claim + γ × raf_claim)
```

### Debug Values from Jolt Verification
```
Instance 2 (InstructionReadRaf):
  left_operand_eval:  [1b, d2, a2, 65, ...]
  right_operand_eval: [82, 6f, 6c, 13, ...]
  identity_poly_eval: [c7, 09, e1, 93, ...]
  gamma:              [5a, b9, a0, 12, ...]
  eq_eval_r_reduction: [03, c3, dc, 21, ...]
  ra_claim:           [66, f1, ef, 21, ...]
  raf_flag_claim:     [b9, e9, 0d, 00, ...]
  raf_claim:          [35, 15, 48, 0b, ...]
  val_claim:          [ed, 65, fd, 2d, ...]
  final_result:       [59, 7c, c0, 15, ...]
```

### Key Components Implemented

1. **Prefix-Suffix Decomposition** (`src/zkvm/lookup_table/`)
   - All 46 prefix types implemented in `prefixes.zig`
   - All 43 suffix types implemented in `suffixes.zig`
   - Table-specific combine functions in `prefix_suffix_prover.zig`
   - RAF decomposition for left/right/identity operands

2. **Stage 5 Prover** (`src/zkvm/spartan/stage5_prover.zig`)
   - Three-instance batched sumcheck
   - Instance 0: RegistersValEvaluation (8 rounds)
   - Instance 1: RamRaClaimReduction (24 rounds) - simplified for fibonacci
   - Instance 2: LookupsReadRaf (136 rounds) with prefix-suffix decomposition
   - Phase transitions every 8 rounds (16 phases total)
   - Expanding table condensation working

### Investigation Areas

The polynomial produced by Zolt's sumcheck doesn't sum to the correct value. Possible issues:

1. **Prefix MLE computation errors**
   - The prefix checkpoint updates may not match Jolt's formula
   - The prefix_mle evaluation at c=0 and c=2 may differ

2. **Suffix polynomial initialization**
   - The Q[suffix_idx][prefix_idx] accumulation may use wrong bits
   - Suffix MLE values may be computed incorrectly

3. **Phase transition logic**
   - The condenseUEvals function may not correctly extract k_bound
   - The expanding table multiplication may be wrong

4. **RAF decomposition**
   - The left/right operand interleaving may differ from Jolt
   - The identity path vs operand path split may be incorrect

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
timeout 600 ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o logs/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Next Steps

1. **Add round-by-round comparison logging**
   - Export each round's polynomial coefficients from Zolt
   - Compare with Jolt's expected round polynomials

2. **Verify prefix MLE implementations one by one**
   - Start with the simple prefixes (Eq, LowerWord) used by fibonacci
   - Add unit tests comparing Zolt vs Jolt prefix evaluations

3. **Verify suffix MLE implementations**
   - Check that suffix_mle values match for the same lookup indices

4. **Debug the opening claims computation**
   - The InstructionRa, LookupTableFlag, InstructionRafFlag claims
