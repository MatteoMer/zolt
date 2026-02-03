# Zolt-Jolt Compatibility Implementation

## Status: Session 26 - Stage 5 Deep Analysis

## Current Issue

Stage 5 verification fails - sumcheck output_claim doesn't match expected_claim.

**From Jolt test:**
- output_claim (from sumcheck): `[ed, a5, f6, bf, 30, c4, 10, f8, 59, ce, db, ef, ee, 23, 2f, 96, ...]`
- expected_claim (verifier computed): `[b2, 8f, 91, 24, 33, 0c, b4, 56, b9, 08, 89, 4c, fd, af, 54, 11, ...]`

## Key Analysis This Session

### Mathematical Verification

Traced through the cycle round polynomial computation in both Jolt and Zolt:

1. **Jolt's GruenSplitEqPolynomial**:
   - Uses `E_in_current()`, `E_out_current()`, `current_scalar`
   - Nested loop structure: inner loop accumulates with E_in, outer multiplies by E_out
   - Finally multiplies by `current_scalar`

2. **Zolt's approach**:
   - Uses flat `lookups_eq_evals[]` array
   - Computes `eq_prefix = eq_0 / (1 - r_round)` to extract prefix without current variable
   - Single loop over all indices

3. **Mathematical equivalence**:
   - Should be equivalent: `eq_prefix[j] = current_scalar * E_out[j_out] * E_in[j_in]`
   - Both use `r_reduction` (from Stage 3) for the eq polynomial
   - Both bind polynomials with sumcheck challenges

### Verified Components

- MontU128Challenge arithmetic matches between Jolt and Zolt
- `raf_interleaved` and `raf_identity` formulas match
- Binding operations are mathematically equivalent

### Suspected Remaining Issues

1. **combined_val rematerialization** - The lookup table values at r_addr might differ
2. **ra_chunk_weights** - The expanding table lookups might have subtle issues
3. **Structural difference** - The nested E_in/E_out loop vs flat loop might not be equivalent after binding

## Next Steps (Priority Order)

1. [IN PROGRESS] Add detailed debug to Zolt to print `sum_evals` at cycle round 0
2. [PENDING] Run Jolt prover to capture expected `sum_evals` for comparison
3. [PENDING] Compare `combined_val` values between Zolt and Jolt
4. [PENDING] Verify `ra_chunk_weights` from expanding tables match

## Test Commands
```bash
# Jolt verification with debug
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# NOTE: Zolt tests OOM on this machine - use Jolt verification for debugging
```

## Key Files
- Zolt Stage 5: `src/zkvm/spartan/stage5_prover.zig` (lines 2870-2934 for cycle round poly)
- Jolt InstructionReadRaf: `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` (lines 775-836)
- Jolt split eq poly: `jolt-core/src/poly/split_eq_poly.rs` (lines 473-501 for gruen_poly_from_evals)
- Jolt mles_product_sum: `jolt-core/src/subprotocols/mles_product_sum.rs` (lines 235-269)

## Debug Locations in Code

### Jolt (already has zolt-debug feature)
- `read_raf_checking.rs:835-864` - prints sum_evals, current_scalar, r_round
- `read_raf_checking.rs:868-878` - prints polynomial coefficients

### Zolt (needs more debug)
- `stage5_prover.zig:2950-2963` - basic cycle round debug
- Need to add: sum_evals values, eq_prefix for first few j, combined_val for first few j
