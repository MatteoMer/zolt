# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 InstructionReadRaf Sumcheck Debug

## Session 124 Summary

### Current Issue

**Stage 5 sumcheck verification fails** for Instance 2 (InstructionReadRaf).

The debug output shows:
```
Sumcheck verification failed!
  output_claim:   [8a, 08, 54, 0f, ...]
  expected_claim: [02, c5, 72, ac, ...]
```

### Analysis

1. **Structure**: Stage 5 is a batched sumcheck with 3 instances:
   - Instance 0: RegistersValEvaluation (8 rounds)
   - Instance 1: RamRaClaimReduction (24 rounds)
   - Instance 2: LookupsReadRaf (136 rounds) - 128 address + 8 cycle

2. **Key Equation** being proved for Instance 2:
   ```
   rv(r_reduction) + γ·left_op(r_reduction) + γ²·right_op(r_reduction)
   = Σ_{j=0}^{T-1} Σ_{k=0}^{K-1} [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ · RafVal_j(k)) ]
   ```

3. **Verifier's expected output claim** (Jolt line 1267):
   ```rust
   let eq_eval_r_reduction = EqPolynomial::<F>::mle(&r_reduction, &r_cycle_prime.r);
   eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
   ```

   Where:
   - `r_reduction` is from Stage 3 (InstructionClaimReduction)
   - `r_cycle_prime` is the last 8 challenges from Stage 5

4. **Zolt's prover** (after sumcheck):
   ```
   lookups_eq_evals[0] * lookups_ra_weights[0] * lookups_combined_vals[0]
   ```

### Potential Root Causes

1. **Address round prefix-suffix decomposition mismatch**
   - Zolt uses a complex prefix-suffix decomposition ported from Jolt
   - Any bug in prefix/suffix computation would cause polynomial mismatch

2. **Cycle round polynomial formula mismatch**
   - Jolt uses `GruenSplitEqPolynomial` for the eq factor
   - Zolt uses direct eq_evals array binding
   - The product structure (eq * val * ra_chunks) may differ

3. **combined_val rematerialization**
   - After address rounds, Jolt rematerializes combined_val using table MLEs at r_address
   - Zolt does this at line 2538-2563 but may have formula differences

4. **Transcript/coefficient format mismatch**
   - The compressed polynomial format must match exactly for transcript alignment

### Debug Areas

1. Check `eq_eval_r_reduction` computation in Zolt vs Jolt
2. Check that `lookups_eq_evals[0]` after binding equals `eq(r_reduction, r_cycle_prime)`
3. Verify `combined_vals[0]` matches verifier's `val_claim + gamma * raf_claim`
4. Verify `ra_chunks` product matches verifier's `ra_claim`

### Key Files

**Zolt:**
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck prover
- `src/zkvm/lookup_table/prefix_suffix_prover.zig` - Prefix-suffix decomposition

**Jolt:**
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf sumcheck

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy to /tmp for Jolt test
cp logs/zolt_*.bin /tmp/

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Next Steps

1. Add debug output comparing:
   - Zolt's `lookups_eq_evals[0]` vs verifier's `eq_eval_r_reduction`
   - Zolt's `lookups_combined_vals[0]` vs verifier's computed value
   - Each ra_chunk vs verifier's ra_claims

2. If values differ, trace back to find where mismatch occurs

3. Likely fix areas:
   - Combined_val rematerialization formula
   - eq binding during cycle rounds
   - ra_chunk computation from expanding tables
