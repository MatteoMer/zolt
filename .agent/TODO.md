# Zolt-Jolt Compatibility Implementation

## Status: Session 25 - Stage 5 Polynomial Coefficient Investigation

## Current Issue

Stage 5 verification fails - sumcheck output_claim doesn't match expected_claim.

**From Jolt test (Session 25):**
- output_claim (from sumcheck): `[ed, a5, f6, bf, 30, c4, 10, f8, 59, ce, db, ef, ee, 23, 2f, 96, ...]`
- expected_claim (verifier computed): `[b2, 8f, 91, 24, 33, 0c, b4, 56, b9, 08, 89, 4c, fd, af, 54, 11, ...]`

**Expected claims per instance:**
- Instance 0 (RegistersValEvaluation): `[74, f7, 8e, 8c, ...]`
- Instance 1 (RamRaClaimReduction): `[c9, 1b, b9, ac, ...]`
- Instance 2 (InstructionReadRaf): `[02, ad, 67, 08, ...]`

**Batching coefficients:**
- coeff[0]: `[04, 97, 3d, 64, ...]`
- coeff[1]: `[50, 2a, 19, a0, ...]`
- coeff[2]: `[45, 50, 75, e2, ...]`

## Key Findings This Session

### 1. MontU128Challenge Arithmetic Verified

When Jolt does `F::one() - r_round` where r_round is Challenge:
- The `-` operator converts Challenge to F via `Into::<F>::into()`
- This uses `from_bigint_unchecked([0, 0, low, high])` - no Montgomery conversion
- The result is a proper F value

When Jolt does `F * Challenge`:
- Uses `mul_by_hi_2limbs(Challenge.low, Challenge.high)`
- This is the optimized multiplication

Zolt's implementation matches this:
- `F.one().sub(rj)` for (1 - r)
- `result.mulHiBigIntU128(rj.limbs)` for multiplication

### 2. GruenSplitEqPolynomial Structure

Jolt uses a sophisticated split eq polynomial that maintains:
- `E_in_vec`: precomputed eq tables for inner variables
- `E_out_vec`: precomputed eq tables for outer variables
- `current_scalar`: accumulated eq for already-bound variables

The cycle round computation (lines 790-834 in read_raf_checking.rs):
```rust
for (j_out, e_out) in self.eq_r_reduction.E_out_current() {
    for (j_in, e_in) in self.eq_r_reduction.E_in_current() {
        // Use e_in directly in pairs
        *val_pair = (*e_in * v_at_0, *e_in * v_at_1);
    }
    // Multiply by e_out at the end
    result *= e_out;
}
// Then multiply by current_scalar
sum_evals *= current_scalar;
finish_mles_product_sum_from_evals(...)
```

### 3. Zolt's Simplified Approach

Zolt uses a simpler approach:
- `lookups_eq_evals[]` contains full eq evaluations for all cycles
- For each round, extracts `eq_prefix = eq_0 / (1 - r_round)`
- Uses `eq_prefix * combined_val` in the polynomial

This SHOULD be mathematically equivalent but the structure differs.

### 4. Sumcheck Passes But Final Claim Differs

All 136 sumcheck rounds verify correctly (the verifier doesn't fail early).
This means the polynomial coefficients are **self-consistent** (p(0) + p(1) = claim).
But the final claim after all rounds differs from what Jolt expects.

This indicates the issue is in **how Zolt computes the polynomial during the sumcheck**,
not in the verification of those coefficients.

## Likely Root Cause

The polynomial computation for Instance 2 (InstructionReadRaf) during cycle rounds
produces different coefficients than Jolt. The candidates are:

1. **eq_prefix computation** - The division `eq_0 / (1 - r_round)` might not match
   Jolt's split eq polynomial structure.

2. **combined_val rematerialization** - At the start of cycle rounds, combined_vals
   are rematerialized. This might differ from Jolt's approach.

3. **ra_chunk_weights materialization** - The expanding table lookup for RA chunks
   might be computing wrong values.

4. **Batching formula** - The way Instance 2's degree-10 polynomial is batched with
   Instance 0 and 1's degree-3 polynomials.

## Next Steps

1. Add debug to compare Zolt's cycle round polynomial coefficients with Jolt prover
2. Compare eq_prefix values with Jolt's E_in_current() values
3. Verify combined_val rematerialization matches Jolt's init_log_t_rounds()
4. Check if the ra_chunk_weights from expanding tables are correct

## Test Commands
```bash
# Jolt verification with debug
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture 2>&1 | grep -E "Stage5|output_claim|expected_claim"

# NOTE: Zolt test OOMs on this machine - use Jolt verification for debugging
```

## Key Files
- Zolt Stage 5: `src/zkvm/spartan/stage5_prover.zig` (lines 2870-2934 for cycle round poly)
- Jolt InstructionReadRaf: `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` (lines 775-836)
- Jolt split eq poly: `jolt-core/src/poly/split_eq_poly.rs` (lines 473-501 for gruen_poly_from_evals)
- Jolt mles_product_sum: `jolt-core/src/subprotocols/mles_product_sum.rs` (lines 235-269)
