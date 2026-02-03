# Session 18 Notes - Stage 5 Deep Dive

## Summary

Investigated the Stage 5 sumcheck polynomial mismatch. The output_claim from sumcheck doesn't match expected_claim from opening claims.

## Key Findings

### 1. Code Structure
- Stage 5 has 136 rounds: 128 address + 8 cycle
- Three instances batched: RegistersValEvaluation, RamRaClaimReduction, LookupsReadRaf
- Cycle rounds use `evalLinearProd9` for 9-factor products

### 2. Polynomial Degree
- 9 linear factors → product polynomial is degree 9
- `finishMlesProductSumFromEvals` multiplies by eq(X, r) → degree 10
- 11 coefficients total, which is correct

### 3. combined_val Construction
At rematerialization (start of cycle rounds):
```
combined_val[j] = table_values_at_r_addr[table(j)] + raf_val(j)
```
where:
- `table_values_at_r_addr` = table MLE at bound r_address
- `raf_val(j)` = γ*left + γ²*right (interleaved) or γ²*identity

### 4. Opening Claims vs Sumcheck Output

The verifier expects:
```
expected = eq_r_reduction * ra_claim * (val_claim + γ * raf_claim)
```

The sumcheck computes:
```
output = Σ_j eq(r_reduction, j) * ra(j) * combined_val(j)
```

After binding, these should equal if combined_val is correct.

## Potential Issues to Investigate

1. **raf_val formula**: Is `γ*left + γ²*right` correct for interleaved operands?
   - Jolt might use a different batching formula

2. **eq_prefix computation**: The extraction `eq_0 / (1 - r_round)` assumes a specific eq structure

3. **ra_chunk_weights materialization**: Need to verify expanding table product matches Jolt

4. **Transcript ordering**: Any subtle difference in coefficient serialization

## Next Session Tasks

1. Add debugging to print exact coefficients for rounds 128-135
2. Compare byte-by-byte with Jolt's debug output
3. Verify raf_val formula matches Jolt's implementation
4. Check if combined_val properly incorporates all terms
