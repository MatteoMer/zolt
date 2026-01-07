# Zolt-Jolt Compatibility - Status Update

## Current Status: Session 11

### Stage 2 Sumcheck Failure

- **Stage 1: PASSING ✅** - Sumcheck output_claim matches expected
- **Stage 2: FAILING ❌** - output_claim != expected_output_claim

```
output_claim:          15813746968267243297450630513407769417288591023625754132394390395019523654383
expected_output_claim: 21370344117342657988577911810586668133317596586881852281711504041258248730449
```

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

### Root Cause Analysis

The sumcheck verification formula mismatch is subtle:

**Prover side (Zolt):**
After binding challenges r[0], r[1], ..., r[n-1] with LowToHigh binding:
- split_eq accumulates: `Eq(tau_reversed, r)` = `Eq(tau, r_reversed)` (due to index reversal)
- left/right polynomials evaluate at: `left(r)`, `right(r)` (in LE convention)
- Final claim: `Eq(tau, r_reversed) * left(r) * right(r)`

**Verifier side (Jolt):**
- Computes: `tau_bound_r_reversed = Eq(tau, r_reversed)`
- Computes factor claims: `fused_left(r_reversed)`, `fused_right(r_reversed)`
- Expected: `Eq(tau, r_reversed) * fused_left(r_reversed) * fused_right(r_reversed)`

**The Mismatch:**
- Prover: `left(r) * right(r)`
- Verifier expects: `left(r_reversed) * right(r_reversed)`

The eq part matches, but the polynomial evaluations differ!

### Why Jolt Works

Jolt's prover and verifier must be consistent because:
1. The polynomial indexing uses BIG_ENDIAN convention
2. When bound with LowToHigh challenges, the BIG_ENDIAN indexing causes the
   polynomial to effectively evaluate at the reversed point
3. The factor claims use normalize_opening_point which also reverses

Zolt may have a mismatch in how the polynomial is stored or indexed compared to Jolt.

### Next Steps

1. Check Zolt's DensePolynomial storage order vs Jolt's
2. Verify that Zolt's fused_left/fused_right construction matches Jolt's
3. Compare the polynomial values at cycle index 0, 1, etc. between Zolt and Jolt
4. Consider if bindLow in Zolt produces LE indexing while Jolt uses BE

### Key Files

- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainderProver
- `src/poly/split_eq.zig` - GruenSplitEqPolynomial
- `src/poly/mod.zig` - DensePolynomial.bindLow
- `src/zkvm/proof_converter.zig` - Stage 2 batched sumcheck

## Previous Sessions

### Session 10
- Fixed output-sumcheck r_address_prime reversal
- Stage 1 started passing

### Session 9
- Fixed transcript challenge sampling
- Aligned MontU128Challenge representation
