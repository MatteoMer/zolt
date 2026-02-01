# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Debugging

## Session 107 Summary

### Current Issue
Stage 5 sumcheck verification fails because Zolt's polynomial coefficients don't match Jolt's expected values.

Jolt debug output shows expected first round coefficients:
```
first round coeffs_except_linear:
  [0]: [e2, ee, 6f, c7, e9, ff, ea, e2, 93, 3a, 36, dd, 78, 31, 47, 9d, ...] = c0 (constant term)
  [1]: [f6, 50, 28, 04, 08, f4, ed, ad, af, 77, b5, 4b, 95, 9a, d3, 49, ...] = c2 (x^2 coefficient)
  [2]: [00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, ...] = c3 (should be 0 for degree-2)
```

### Key Understanding Gained

1. **CompressedUniPoly format**:
   - `coeffs_except_linear_term[0]` = c0 (constant term = p(0))
   - `coeffs_except_linear_term[1]` = c2 (x^2 coefficient)
   - `coeffs_except_linear_term[2]` = c3 (x^3 coefficient)
   - The linear term c1 is recovered using: `c1 = hint - 2*c0 - c2 - c3 - ...`

2. **Stage 5 sumcheck structure**:
   - Batched sumcheck with 3 instances:
     - Instance 0: RegistersValEvaluation (8 rounds)
     - Instance 1: RamRaClaimReduction (24 rounds)
     - Instance 2: LookupsReadRaf (136 rounds)
   - For first 128 rounds, only Instance 2 (LookupsReadRaf) is active
   - Other instances contribute constant polynomials

3. **Prefix-suffix decomposition**:
   - First 128 rounds use prefix-suffix decomposition for address variables
   - Read-checking: `Σ_tables Σ_b table.combine(prefix(c,b), Q_suffix[b])`
   - RAF: `γ*left + γ²*(identity + right)`
   - Polynomial is degree-2: computed as `eval_0` and `eval_2`

4. **Phase configuration**:
   - For small traces (log_T < 24): 16 phases with log_m = 8 bits per phase
   - For large traces: 8 phases with log_m = 16 bits per phase

### Investigation Areas

1. **Q polynomial initialization** - Is the accumulation `Q[prefix] = Σ u_evals[j] * suffix_mle(suffix_bits[j])` correct?
2. **Prefix MLE evaluations** - Are prefix checkpoints being updated correctly?
3. **RAF decomposition** - Is the bound_value tracking correct?
4. **Phase transitions** - Is the u_evals condensation working correctly?

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof (takes several minutes due to SRS generation)
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Files to Investigate
1. `src/zkvm/lookup_table/prefix_suffix_prover.zig` - Q initialization, proverMsgReadChecking
2. `src/zkvm/lookup_table/prefixes.zig` - All prefix MLE implementations
3. `src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover implementation
4. Compare with Jolt's `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`

### Formula Reference

**Jolt's expected_output_claim for InstructionReadRaf:**
```
expected = eq(r_reduction, r_cycle') * ra_claim * (val_claim + γ * raf_claim)

where:
  val_claim = Σ table[i].evaluate_mle(r_address) * LookupTableFlag[i]
  raf_claim = (1 - raf_flag) * (left_op_eval + γ * right_op_eval) + raf_flag * γ * identity_eval
  ra_claim = Π InstructionRa[i](r_sumcheck)
```

### Remaining Work
1. Debug Stage 5 sumcheck polynomial mismatch
2. Verify Q initialization matches Jolt exactly
3. Test complete proof verification
4. Test with additional programs beyond fibonacci
