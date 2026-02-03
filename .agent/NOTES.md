# Session 25/26 Notes - Stage 5 Polynomial Coefficient Analysis

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

## Session 26 Analysis

### Mathematical Equivalence Investigation

Traced through the eq_prefix computation:

**For cycle round 0:**
- Zolt: `eq_prefix[j] = eq_evals[2*j] / (1 - r_round)` where `r_round = r_reduction[7]`
- This extracts `eq(j, r_reduction[0:7])` - the eq factor without the current variable

**For Jolt:**
- `current_scalar = 1` (no bound variables at start)
- `E_in * E_out` covers all 8 cycle variables in split form
- Total eq = `current_scalar * E_out[j_out] * E_in[j_in]` = `eq(j, r_reduction)`

After factoring out the current round's eq factor, these should be equivalent.

### Key Insight: r_round Source

Both Jolt and Zolt use `r_reduction` (from Stage 3 claim reduction) for the eq polynomial, NOT the sumcheck challenges. This is correct because:
- `r_reduction` is the "target" point the prover is proving evaluation at
- Sumcheck challenges are different - they determine WHERE to evaluate each round's polynomial

### Binding vs Current Scalar

**Jolt's bind() method:**
- Multiplies `current_scalar *= eq(w[i], r)`
- Decrements `current_index`
- Pops E_in or E_out tables

**Zolt's bindLookupsChallenge:**
- `eq_evals[i] = (1-r) * eq_evals[2*i] + r * eq_evals[2*i+1]`
- Standard multilinear binding

These are mathematically equivalent but structured differently.

### The Core Question

Why does `sum_evals` differ between Zolt and Jolt?

Possibilities:
1. `eq_prefix` computation error (division by zero? wrong r_round?)
2. `combined_val` rematerialization error
3. `ra_chunk_weights` computation error
4. Structural difference in how sums are accumulated (E_in/E_out vs flat loop)

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

## ROOT CAUSE FOUND - Session 26 (Late)

### The Bug: eq_evals Binding

**Zolt incorrectly binds `eq_evals` with sumcheck challenges, while Jolt does NOT.**

In Jolt's GruenSplitEqPolynomial:
- `E_in_vec` and `E_out_vec` are computed from `r_reduction` (original challenges)
- `bind(r_j)` only updates `current_scalar`, NOT E_in/E_out
- `current_scalar` accumulates `eq(w[i], r_j)` where w is original, r_j is sumcheck challenge
- Total eq contribution = `current_scalar * E_out[j_out] * E_in[j_in]`

In Zolt:
- `eq_evals[j]` starts as `eq(j, r_reduction)`
- `bindLookupsChallenge` modifies eq_evals: `eq[i] = (1-c)*eq[2*i] + c*eq[2*i+1]`
- This MIXES original r_reduction with sumcheck challenge c
- Result: eq_prefix no longer represents eq(j, r_reduction[remaining])

### Why This Causes the Bug

After round 0 binding with challenge c_0:
- **Jolt**: `current_scalar = eq(r_reduction[n-1], c_0)`, E_in/E_out unchanged
- **Zolt**: `eq_evals` contains mixed values, NOT pure eq(j, r_reduction)

For round 1 polynomial:
- **Jolt**: uses `current_scalar * eq(j, r_reduction[0:n-2])` - correct
- **Zolt**: uses `eq_prefix` derived from mixed eq_evals - WRONG

### The Fix

Zolt needs to:
1. NOT bind `eq_evals` with sumcheck challenges for polynomial computation
2. Instead, maintain `current_scalar` that accumulates `eq(r_reduction[i], challenge_i)`
3. Use `current_scalar * eq_prefix` in polynomial computation
4. Still bind eq_evals for CLAIM computation (to verify p(0)+p(1)=claim)

Alternatively, restructure to match Jolt's GruenSplitEqPolynomial approach exactly.

## Next Steps

1. Implement `current_scalar` accumulation in Zolt's cycle round computation
2. Modify eq_prefix usage to include current_scalar factor
3. Test fix with Jolt verification

## Files to Study

- Jolt: `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` lines 600-836
- Jolt: `jolt-core/src/poly/split_eq_poly.rs` (GruenSplitEqPolynomial)
- Zolt: `src/zkvm/spartan/stage5_prover.zig` lines 2700-3000

## Test Commands

```bash
# Jolt verification with debug
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# NOTE: Zolt tests OOM on this machine - use Jolt verification for debugging
```
