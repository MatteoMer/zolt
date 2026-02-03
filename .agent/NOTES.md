# Session 25 Notes - Stage 5 Polynomial Coefficient Analysis

## Summary

Deep investigation of why Stage 5 sumcheck fails. The key finding is that all round verifications pass (polynomial is self-consistent), but the final output_claim differs from the expected_claim.

## The Mismatch

**Output claim (from Zolt sumcheck):**
```
[ed, a5, f6, bf, 30, c4, 10, f8, 59, ce, db, ef, ee, 23, 2f, 96, ...]
```

**Expected claim (from Jolt verifier):**
```
[b2, 8f, 91, 24, 33, 0c, b4, 56, b9, 08, 89, 4c, fd, af, 54, 11, ...]
```

## Individual Instance Expected Claims

The verifier computes expected claims per instance:
- Instance 0 (RegistersValEvaluation): `[74, f7, 8e, 8c, ...]`
- Instance 1 (RamRaClaimReduction): `[c9, 1b, b9, ac, ...]`
- Instance 2 (InstructionReadRaf): `[02, ad, 67, 08, ...]`

Then batches them:
```
expected = coeff[0] * inst0 + coeff[1] * inst1 + coeff[2] * inst2
```

With coefficients:
- coeff[0]: `[04, 97, 3d, 64, ...]`
- coeff[1]: `[50, 2a, 19, a0, ...]`
- coeff[2]: `[45, 50, 75, e2, ...]`

## Key Technical Finding: GruenSplitEqPolynomial

Jolt uses a sophisticated split eq polynomial for cycle rounds:

```rust
// Jolt's approach (read_raf_checking.rs lines 790-834)
for (j_out, e_out) in eq_r_reduction.E_out_current() {
    for (j_in, e_in) in eq_r_reduction.E_in_current() {
        let j = group_index(j_out, j_in);
        // e_in is the eq factor for inner unbound variables
        val_pair = (e_in * v_at_0, e_in * v_at_1);
        // ... ra_pairs ...
        eval_linear_prod_accumulate(&pairs, &mut evals_acc);
    }
    // Multiply accumulated result by e_out
    result.iter_mut().for_each(|v| *v *= e_out);
}
// Then multiply everything by current_scalar
sum_evals.iter_mut().for_each(|v| *v *= current_scalar);
finish_mles_product_sum_from_evals(&sum_evals, claim, &eq_r_reduction)
```

Key components:
- `E_in_current()`: eq values for inner unbound variables
- `E_out_current()`: eq values for outer unbound variables
- `current_scalar`: accumulated eq for already-bound variables (from bind())

## Zolt's Approach

Zolt uses a simpler approach:

```zig
// Zolt's approach (stage5_prover.zig lines 2890-2920)
const eq_0 = lookups_eq_evals[2 * j];
const eq_prefix = eq_0.mul(inv_one_minus_r_round);  // eq_0 / (1 - r_round)
pairs[0][0] = eq_prefix.mul(lookups_combined_vals[2 * j]);
pairs[0][1] = eq_prefix.mul(lookups_combined_vals[2 * j + 1]);
// ... ra_chunk_weights ...
const prod_evals = UniPoly(F).evalLinearProd9(pairs);
sum_evals[k] = sum_evals[k].add(prod_evals[k]);
```

Differences:
- Zolt keeps full eq evaluations in `lookups_eq_evals[]`
- Divides by `(1 - r_round)` to extract eq_prefix
- No separate E_in/E_out/current_scalar structure

## MontU128Challenge Arithmetic

Verified that both Jolt and Zolt handle MontU128Challenge the same way:

1. **Storage**: `[0, 0, low, high]` as Montgomery representation
2. **Subtraction**: `F::one() - Challenge` converts Challenge to F first
3. **Multiplication**: `F * Challenge` uses `mul_by_hi_2limbs(low, high)`

Zolt's implementation matches:
- `F.one().sub(rj)` for (1 - r)
- `result.mulHiBigIntU128(rj.limbs)` for multiplication

## Why Sumcheck "Passes" But Claim Differs

The sumcheck verification at each round checks:
```
p(0) + p(1) == claim
```

If Zolt computes polynomial p(X) such that this holds for the current claim, verification passes. But the NEXT claim becomes `p(challenge)`.

If Zolt's polynomial differs from what Jolt would compute:
- Each round still satisfies p(0) + p(1) = claim
- But p(challenge) produces a DIFFERENT next claim
- After 136 rounds, the accumulated difference becomes the mismatch

## Likely Root Causes

1. **eq_prefix computation**: The division approach may not match Jolt's split eq structure

2. **combined_val rematerialization**: At cycle round start, Zolt rematerializes combined_vals. This might differ from Jolt's `init_log_t_rounds()`.

3. **ra_chunk_weights**: The expanding table lookup for RA polynomial chunks might have issues.

4. **Binding order**: Although both claim LowToHigh, the details of how variables are bound might differ.

## Next Steps

1. Add debug to print Jolt prover's polynomial coefficients during cycle rounds
2. Run a test that exercises Jolt's prover (not just verifier)
3. Compare eq_prefix values between Zolt and Jolt
4. Check combined_val values at cycle round start
5. Verify ra_chunk_weights match between implementations

## Files to Study

- Jolt: `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` lines 600-836
- Jolt: `jolt-core/src/poly/split_eq_poly.rs` (GruenSplitEqPolynomial)
- Zolt: `src/zkvm/spartan/stage5_prover.zig` lines 2700-3000
