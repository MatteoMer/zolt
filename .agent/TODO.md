# Zolt-Jolt Compatibility - Status Update

## Current Status: Session 12

### Stage 2 Sumcheck Failure - Deep Investigation

- **Stage 1: PASSING ✅** - Sumcheck output_claim matches expected
- **Stage 2: FAILING ❌** - output_claim != expected_output_claim

```
output_claim:          15813746968267243297450630513407769417288591023625754132394390395019523654383
expected_output_claim: 21370344117342657988577911810586668133317596586881852281711504041258248730449
```

### Progress This Session

1. **Fixed computeEqEvalsGeneric** - Now uses BIG_ENDIAN indexing to match Jolt:
   - r[0] controls MSB of index (not LSB)
   - This aligns with Jolt's EqPolynomial::evals() convention

2. **Verified split_eq construction** - E_out_vec and E_in_vec use correct big-endian indexing

3. **Verified polynomial storage** - left/right polynomials stored sequentially matching Jolt

4. **Verified Gruen cubic** - computeCubicRoundPoly formula matches Jolt's gruen_poly_deg_3

5. **Verified binding** - bindLow formula matches Jolt's bound_poly_var_bot

### Mathematical Analysis

**The Key Identity:**
`MLE_LE(poly, r) = MLE_BE(poly, r_reversed)` when using the same sequential polynomial storage.

This means:
- Prover binds with LowToHigh, computing `MLE_LE(left, r)`
- Factor claims compute `MLE_BE(factor, r_reversed)` using big-endian eq tables
- These should be equal!

**Why Still Failing:**
The eq polynomial part and polynomial evaluation part should both match. But the
sumcheck output doesn't match the expected. This suggests either:

1. **Round polynomial computation issue** - t0/t_inf values differ
2. **First round handling** - Jolt handles first round differently
3. **Claim update logic** - Evaluation at challenge differs
4. **Something in the batched sumcheck** - Coefficient combining issue

### What Matches (Verified)
1. ✅ tau_high matches between Zolt and Jolt
2. ✅ Initial batched claim matches
3. ✅ All 26 Stage 2 challenges match byte-for-byte
4. ✅ Round polynomial coefficients match (c0, c2, c3 for rounds 0, 25)
5. ✅ Virtual polynomial factor evaluations match (l_inst, r_inst, etc.)
6. ✅ Opening claims serialize correctly
7. ✅ EqPolynomial formulas match Jolt
8. ✅ split_eq binding formula matches Jolt (eq(tau, r) accumulation)
9. ✅ computeCubicRoundPoly formula matches Jolt's gruen_poly_deg_3
10. ✅ bindLow formula matches Jolt's bound_poly_var_bot
11. ✅ Big-endian eq indexing in factor claim computation (fixed this session)

### Next Steps

1. **Debug t0/t_inf values** - Add output to compare per-round values with Jolt
2. **Check first round special handling** - Jolt's `compute_first_quadratic_evals_and_bound_polys`
3. **Trace claim updates** - Verify s(r_challenge) computation matches
4. **Compare intermediate claims** - After each round, are prover and verifier claims same?

### Key Files

- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainderProver
- `src/poly/split_eq.zig` - GruenSplitEqPolynomial
- `src/poly/mod.zig` - DensePolynomial.bindLow
- `src/zkvm/proof_converter.zig` - Stage 2 batched sumcheck

## Previous Sessions

### Session 11
- Verified extensive formula matching between Zolt and Jolt
- Identified that polynomial indexing/storage must match exactly

### Session 10
- Fixed output-sumcheck r_address_prime reversal
- Stage 1 started passing

### Session 9
- Fixed transcript challenge sampling
- Aligned MontU128Challenge representation
