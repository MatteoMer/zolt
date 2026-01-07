# Zolt-Jolt Compatibility Notes

## Current Status (Session 13 - January 7, 2026)

### Summary

**Stage 1 PASSES, Stage 2 FAILS at sumcheck verification**

The Stage 2 batched sumcheck produces `output_claim` that doesn't match the verifier's `expected_output_claim`.

### ROOT CAUSE IDENTIFIED: r0 Mismatch

The Stage 2 failure is caused by mismatched `r0` values between Zolt and Jolt:

- **Zolt r0**: `5629772851639812945906736172593031815056148939881883788449064297659372967906`
- **Jolt r0**: `16176819525807790011525369806787798080841445107270164702191186390206256879646`

This causes `tau_high_bound_r0` to differ:
- Zolt's prover computes with one r0
- Jolt's verifier expects a different r0
- The `expected_output_claim` formula depends on `tau_high_bound_r0`

### r0 Derivation Path

In Jolt, `r0` for Stage 2 comes from:
```rust
// In ProductVirtualRemainderParams::new()
let (r_uni_skip, _) = opening_accumulator.get_virtual_polynomial_opening(
    VirtualPolynomial::UnivariateSkip,
    SumcheckId::SpartanProductVirtualization,
);
let r0 = r_uni_skip[0];
```

This `r_uni_skip` is the **opening point** (not the claim) that was stored when the UniSkip verification appended to the accumulator.

### Problem: Opening Points vs Claims

Zolt's `OpeningClaims` only stores claim values, NOT opening points! The Jolt accumulator stores both:
- `(OpeningPoint, claim_value)` for each opening

But Zolt only stores:
- `claim_value`

So when Stage 2 needs the opening point for UnivariateSkip at SpartanProductVirtualization, it can't get it because Zolt didn't store it!

### Verified Matching Values

These values MATCH between Zolt and Jolt:
- `fused_left`: 15479400476700808083175193706386825626005767142779158246159402270795992278944
- `fused_right`: 16089746625107921886379479343676619567444150947455675976379397017146086344498
- All polynomial coefficients (c0, c2, c3) for all 26 rounds
- All round challenges for all 26 rounds

### What Differs

- `r0` (Stage 2 UniSkip challenge) - from transcript state divergence
- Consequently `tau_high_bound_r0`
- Consequently `expected_output_claim` for instance 0

### Error Values

```
output_claim:          15813746968267243297450630513407769417288591023625754132394390395019523654383
expected_output_claim: 21370344117342657988577911810586668133317596586881852281711504041258248730449
```

### Secondary Issue: Constant Polynomials

For instances 1, 2, 4 (RafEvaluation, RamReadWriteChecking, InstructionLookupsClaimReduction):
- Zolt uses constant polynomials but these instances have non-zero input claims
- The expected_output_claim is 0 for these (because ra=0, val=0)
- Constant polynomials don't reduce to 0, they reduce to input_claim/2^N
- This is a secondary issue that becomes visible only after fixing the r0 issue

### Next Steps

1. **Fix r0 derivation for Stage 2**:
   - Track transcript state between rounds 54 and 176
   - Find where Zolt and Jolt transcript states diverge
   - The r0 is sampled at round 177

2. **Once r0 is fixed**, address the constant polynomial issue for instances 1, 2, 4:
   - Option A: Implement proper RAF/RWC/Instruction provers
   - Option B: Use zero-polynomial with hint approach

---

## Previous Sessions

### Session 12
- Identified individual claims match but claim trajectory diverges
- Found that s(0) + s(1) != claim for combined polynomial

### Session 11
- Fixed Stage 1 output-sumcheck r_address_prime reversal
- Stage 1 started passing

### Session 10
- Implemented Stage 1 streaming outer prover
- Fixed constraint evaluation ordering

### Earlier Sessions
- Established transcript compatibility with Jolt
- Fixed field element serialization (LE format)
- Fixed polynomial coefficient compression format
