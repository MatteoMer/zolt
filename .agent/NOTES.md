# Zolt-Jolt Compatibility Notes

## Current Status (Session 49 - January 3, 2026)

### Summary

Working on challenge representation. Round polynomials match, but challenges still diverge between Zolt prover and Jolt verifier.

### Key Finding: Challenge Limb Representation

**IMPORTANT - DO NOT CHANGE THIS:**

The 128-bit (actually 125-bit masked) challenge is stored in limbs[0:1], NOT limbs[2:3]:

```zig
// CORRECT - represents the value (low + high * 2^64)
const raw = F{ .limbs = .{ masked_low, masked_high, 0, 0 } };
const result = raw.toMontgomery();
```

**Why NOT `[0, 0, low, high]`?**

- `[0, 0, low, high]` would represent `(low * 2^128 + high * 2^192)` - a value shifted by 2^128
- This is NOT what we want for the field element value
- Jolt's `MontU128Challenge` stores `[0, 0, low, high]` for its INTERNAL optimized representation
- But when converted to Fr via `from_bigint_unchecked`, the VALUE is `low * 2^128 + high * 2^192`
- In Zolt, we want the challenge to represent the actual 128-bit value, not the shifted value

**Serialization Format (toBytesBE):**

When a challenge stored as `[low, high, 0, 0]` is serialized:
1. `fromMontgomery()` converts to standard form: `[low, high, 0, 0]`
2. Serialize to LE bytes: bytes[0..8]=low, bytes[8..16]=high, bytes[16..32]=zeros
3. Reverse to BE: zeros in first 16 bytes, value in last 16 bytes

This is expected and correct - the challenge IS a 128-bit value, so the upper 128 bits of the 256-bit field element are zero.

**Jolt's Challenges Look Different:**

Jolt's verifier prints challenges as full 256-bit field elements. This is because Jolt stores `MontU128Challenge` as `[0, 0, low, high]` and uses `from_bigint_unchecked` which interprets this as a shifted value. When printed, the Montgomery-form field element has non-zero bytes everywhere.

**CURRENT ISSUE:** The challenge VALUES themselves differ between Zolt and Jolt - not just the representation. Need to trace back to find where the transcript state diverges.

### Session 49 Investigation (cont.)

**Key Finding: tau values don't match**

- Zolt's tau[11] (as shifted field element) = 3700036982749152569148447626330409071994233407396770520073887757500189507584
- Jolt's tau_high = 4949406946327026266101368757399851829917753905666081332932170379465433058454
- These are DIFFERENT!

Since tau is derived from the transcript, this means the transcript states diverge somewhere before tau is derived (i.e., during or before the Dory commitment phase).

**Round polynomial coefficients DO match:**
- Zolt c0_le matches Jolt c0_bytes exactly
- This confirms the sumcheck computation is correct
- The issue is in the transcript/challenge derivation

### CRITICAL DISCOVERY: from_bigint_unchecked behavior

**ROOT CAUSE FOUND:**

Arkworks' `from_bigint_unchecked(BigInt::new([0, 0, low, high]))` interprets the BigInt as ALREADY in Montgomery form! It does NOT multiply by R to convert to Montgomery.

**What this means:**
- Jolt stores `[0, 0, low, high]` as the Montgomery representation directly
- When printing Fr, arkworks divides by R to get the canonical value
- The canonical value is `(low * 2^128 + high * 2^192) / R mod p`

**The bug in Zolt:**
- Zolt was calling `toMontgomery()` on `[0, 0, low, high]`
- This MULTIPLIED by R, giving `(low * 2^128 + high * 2^192) * R mod p`
- This is the WRONG value!

**The fix:**
- Do NOT call `toMontgomery()`
- Use `F{ .limbs = .{ 0, 0, masked_low, masked_high } }` directly
- The limbs ARE the Montgomery representation

**Verification:**
```
bigint = 3649381361935200060066435842883492086020528436178408344296001395814532382720
fr_printed = 3350198182347904564092445461553703484396816537342330508621481301908782066970
bigint / R mod p = 3350198182347904564092445461553703484396816537342330508621481301908782066970  ✓
```

---

## Previous Status (Session 40 - January 2, 2026)

### Summary

Fixed the batching coefficient issue. Now round polynomials are scaled by `batching_coeff` before output. Challenges now match between prover and verifier. However, the final output_claim still doesn't match expected_output_claim.

### Session 40 Key Progress

1. **Batching coefficient applied**: Round polynomials are now multiplied by `batching_coeff` before being written to proof and transcript. The prover's internal state uses unscaled claims.

2. **Transcript consistency verified**: The sumcheck challenges now match exactly between Zolt prover and Jolt verifier.

3. **Lagrange kernel symmetry confirmed**: `L(x, y) = L(y, x)`, so argument order doesn't matter.

4. **Eq polynomial binding verified**: Both implementations bind all 11 elements of tau_low.

### Current Values

```
output_claim:          11745972059365673324717055336378505103382790433770080606002230314528714321637
expected_output_claim: 13147110630967021857497758076978613720325907259294229523986769287815268967658
```

### Expected Output Claim Formula

From Jolt's `OuterRemainingSumcheckVerifier::expected_output_claim`:

```rust
let tau_high = &tau[tau.len() - 1];
let tau_low = &tau[..tau.len() - 1];

// L(τ_high, r0) - Lagrange kernel at UniSkip challenge
let tau_high_bound_r0 = LagrangePolynomial::lagrange_kernel(tau_high, &self.params.r0);

// eq(τ_low, r_tail_reversed)
let r_tail_reversed = sumcheck_challenges.iter().rev().collect();
let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);

// Az(rx_constr) * Bz(rx_constr) where rx_constr = [r_stream, r0]
let inner_sum_prod = key.evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);

let result = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod;
// expected_output_claim = result * batching_coeff
```

### Remaining Hypothesis

The output_claim from sumcheck should equal:
```
output_claim = eq_factor * (Az * Bz) * batching_coeff
```

Where `eq_factor = L(τ_high, r0) * eq(τ_low, r)` after all bindings.

The mismatch suggests either:
1. The eq_factor doesn't match (split_eq or Lagrange kernel issue)
2. The Az*Bz product doesn't match (constraint evaluation issue)

### Next Debug Steps

1. Print `split_eq.current_scalar` at the end of sumcheck to verify eq_factor
2. Print the final Az*Bz value from the prover
3. Compare with Jolt's `tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod`

---

## Previous Status (Session 39 - January 2, 2026)

### Summary

The Stage 1 sumcheck output_claim doesn't match the expected_output_claim. All 712 unit tests pass, and the transcript challenges are identical between prover and verifier.

### Session 39 Investigation

1. **Round zero materialization**: Updated `materializeLinearPhasePolynomials` to use the simple `full_idx = grid_size * i + j` indexing (matching Jolt's `fused_materialise_polynomials_round_zero`).

2. **Transcript consistency**: The challenges are correctly shared between prover and verifier.

3. **E_out/E_in tables**: These are correctly initialized with `[1]` at index 0.

4. **Multiquadratic expansion**: The `expandGrid` function correctly computes `[f0, f1, f1-f0]` for window_size=1.

---

## Architecture Notes

### Sumcheck Structure

Stage 1 has:
- 1 UniSkip round (produces r0)
- 1 + num_cycle_vars remaining rounds

For trace_length = 1024:
- num_cycle_vars = 10
- num_rows_bits = 12
- tau.len = 12
- tau_low.len = 11
- Remaining rounds = 11
- r_tail_reversed = [r_10, r_9, ..., r_1, r_stream]

### Big-Endian Convention

From Jolt's eq_poly.rs:
```
evals(r)[i] = eq(r, b₀…b_{n-1})
where i has MSB b₀ and LSB b_{n-1}
```

### Key Insight: MLE of Product vs Product of MLEs

The sumcheck produces `Az_MLE(r) * Bz_MLE(r)` at the final point, NOT `MLE(Az*Bz)(r)`:
- These are mathematically different: `MLE(f*g)(r) ≠ MLE(f)(r) * MLE(g)(r)`
- After binding all variables to r, Az(r) and Bz(r) become single-point evaluations
- The sumcheck correctly reduces to the product of these evaluations

### Code Paths

**Zolt prover flow:**
1. `StreamingOuterProver.initWithScaling` - sets up split_eq with tau and scaling_factor
2. `bindFirstRoundChallenge(r0)` - sets r_stream, current_round=1
3. `computeRemainingRoundPoly()` calls:
   - `materializeLinearPhasePolynomials()` - fills az_poly, bz_poly, t_prime_poly
   - `computeTEvals()` - gets (t_zero, t_infinity) from t_prime_poly
   - `split_eq.computeCubicRoundPoly(t_zero, t_infinity, previous_claim)` - builds round polynomial
4. After each round: `bindRemainingRoundChallenge(r)` binds split_eq, t_prime_poly, az_poly, bz_poly

**Jolt prover flow:**
1. `OuterSharedState::new` - sets up split_eq_poly, r0
2. `OuterLinearStage::initialize` - builds az, bz, t_prime_poly
3. `next_round` - computes (t_zero, t_infinity) and cubic round poly
4. After each round: `ingest_challenge` binds all polynomials
