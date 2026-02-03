# Zolt-Jolt Compatibility Implementation

## Status: Session 23 - Stage 5 Deep Analysis

## Current Issue

Stage 5 verification fails - sumcheck output_claim doesn't match expected_claim.

**From Jolt debug:**
- Sumcheck output_claim: `[ed, a5, f6, bf, ...]`
- Expected_claim: `[b2, 8f, 91, 24, ...]`

## Deep Analysis Findings

### Verified Correct:
1. **Batched sumcheck structure**: 3 instances (RegistersValEval, RamRaClaimReduction, InstructionReadRaf)
2. **LowToHigh binding order**: Matches Jolt
3. **eq decomposition formula**: `eq_prefix = eq_0 / (1 - r_round)` is correct
4. **r_round source**: Using `r_reduction[n_cycle_vars - 1 - lookups_round]` which is the w value (correct)
5. **computeEqAtIndex**: BIG_ENDIAN handling is correct
6. **Product of 9 factors**: `eq_prefix * combined * ra_chunk[0..7]` structure is correct

### Potential Issues to Investigate:
1. **Transcript synchronization**: Do the polynomial coefficients from Zolt match Jolt's expectations?
2. **Opening claims vs polynomial evaluation**: The stored claims may not match what the polynomial actually evaluates to
3. **Challenge multiplication**: Using `mulHiBigIntU128` for F * Challenge - is this correct?

### Key Formula for Instance 2 (InstructionReadRaf):
**Prover computes** (per cycle round):
```
p(X) = Σ_j eq(X, w[round]) * eq_prefix(j) * combined[j] * Π_c ra_chunk[c][j]
```

**Verifier expects**:
```
expected = eq(r_reduction, r_cycle') * ra_claim * (val_claim + gamma * raf_claim)
```

where:
- `ra_claim = Π_c ra_chunks[c]` (product of stored claims)
- `val_claim = Σ_i table_flag[i] * table_eval[i]`
- `raf_claim` = computed from raf_flag, operand evals, identity eval

## Next Steps

1. **Compare polynomial coefficients**: Run Zolt and capture first few round coefficients, compare with Jolt
2. **Trace lookups_claim evolution**: Verify claim updates correctly through all 8 cycle rounds
3. **Verify opening claim storage**: Check that ra_chunks[i] = ra_chunk_weights[i][0] after binding matches Jolt's ra_poly.final_sumcheck_claim()
4. **Check combined_vals rematerialization**: Verify table_values_at_r_addr computation matches Jolt

## Test Commands
```bash
# Jolt verification with debug
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# Zolt proof generation
cd zolt && zig build test
```

## Key Files
- Zolt Stage 5: `src/zkvm/spartan/stage5_prover.zig`
- Jolt Stage 5 verifier: `jolt-core/src/zkvm/verifier.rs`
- Jolt InstructionReadRaf: `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`
