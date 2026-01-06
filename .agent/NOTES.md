# Zolt-Jolt Compatibility Notes

## Current Status (Session 56 - January 6, 2026)

### Summary

Root cause confirmed: The sumcheck verification fails because the **inner_sum_prod** computed by the verifier
uses the two-group blending with `r_stream`, but Zolt's prover doesn't compute the same blended product.

**Key Insight from Jolt Verifier:**
```rust
// In evaluate_inner_sum_product_at_point:
let az_final = az_g0 + r_stream * (az_g1 - az_g0);
let bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0);
inner_sum_prod = az_final * bz_final
```

The verifier expects the prover to produce a claim where the two constraint groups are blended
using the `r_stream` challenge. The streaming round (first remaining round) produces `r_stream`.

**Debug Output Analysis:**
- r1cs_input_evals match between prover and verifier ✓
- sumcheck challenges match (same Fiat-Shamir transcript) ✓
- But output_claim ≠ expected_output_claim

**Verifier Debug Values (from test run):**
```
r_stream = 403453513528045288561805897703363889300374796249554293903824070113366263553
r0 = 9956720580376218385912772229440842054804584136250960391272783931628043427152
az_g0 = 7784823065365355150329898612995661848060669911039813026488804049763027733277
bz_g0 = 17295306602198664491678763456607120003277220072842593579631007088433375750847
az_g1 = 1819540214425536184764541679660107604875800049998949590309674377395141916961
bz_g1 = 7283832083962387142906994059148710843912079143868948223318847447205362507259
az_final = 18118550305323548719991270363692126563213624719115355489360394285597267782328
bz_final = 189353324837795743110663437939657740667583580724624394002588005226092056829
inner_sum_prod = 3428564744352329898278095955238265070037131657307455691194697055242544749299
```

**Next Step:** Add debug output to Zolt prover to compute the SAME values and compare

**Verification formula:**
```
expected_output_claim = inner_sum_prod * eq_eval * lagrange_tau_r0
```

Where `inner_sum_prod` uses the blended Az*Bz product from two constraint groups.

---

## Session 55 - rebuildTPrimePoly Bug Investigation (SUPERSEDED)

### Session 55 - rebuildTPrimePoly Bug Investigation

**Key Findings:**

1. **Verified correct components:**
   - Polynomial coefficients (c0, c2, c3) match exactly at ALL rounds ✓
   - Challenges match exactly (same Fiat-Shamir transcript) ✓
   - eq_factor computation is correct (lagrange_tau_r0 * eq bindings) ✓
   - Scaling model is consistent (prover unscaled, verifier scaled by batching_coeff) ✓
   - bindLow operation matches Jolt's bound_poly_var_bot ✓

2. **BUG IDENTIFIED in rebuildTPrimePoly:**
   ```
   ROUND 2 AFTER REBUILD: t_prime[0] = {29, 42, 85, 202, ...}
   EXPECTED t_prime[0] = az[0]*bz[0] = {6, 206, 182, 204, ...} (DIFFERENT!)
   ```
   - rebuildTPrimePoly calls buildTPrimePoly
   - buildTPrimePoly iterates over E_out * E_in pairs and accesses az/bz by index
   - After binding, the polynomial sizes reduce: az.boundLen() = 1024 (from 2048)
   - The eq tables also change: E_out.len=32, E_in.len=16 (instead of 32*32)
   - The product E_out.len * E_in.len * grid_size = 32*16*2 = 1024 matches az.boundLen() ✓
   - BUT the values computed are still wrong

3. **Debug EXPECTED is misleading:**
   - t_prime[0] should be weighted sum: Σ E_out[i] * E_in[j] * az[idx] * bz[idx]
   - NOT just az[0]*bz[0]
   - However the actual sum is still producing wrong values

4. **Comparison with Jolt:**
   - Jolt's `compute_evaluation_grid_from_polynomials_parallel` uses same formula:
     ```rust
     let index = grid_size * i + j;
     az_grid[j] = az[index];
     ```
   - Need to verify Zolt's iteration matches exactly

**Root Cause Hypothesis:**
The claim drift starts at round 2 when rebuildTPrimePoly is first called. The rebuilt t_prime polynomial doesn't match the actual bound polynomial state, causing the sumcheck claim to diverge from eq_factor * Az*Bz.

---

### Session 54 - Claim Propagation Investigation

**Key Findings:**

1. **Polynomial coefficients match exactly** - Verified that c0, c2, c3 bytes in Zolt's proof match what Jolt reads (accounting for endianness).

2. **Batched sumcheck scaling model:**
   - Prover computes UNSCALED polynomial from actual t_zero, t_infinity
   - Polynomial is scaled by batching_coeff before sending to proof
   - Verifier tracks SCALED claims (initial_claim * batching_coeff)
   - Prover tracks UNSCALED claims internally

3. **The mismatch:** Verified that `implied_inner_sum_prod (output_claim / eq_factor) != az_final * bz_final`, even though the sumcheck should give this equality.

4. **Root cause hypothesis:** The Gruen polynomial q(X) is drifting from the true bound polynomial t'(X) over multiple rounds. While q(X) satisfies the sumcheck constraint s(0)+s(1)=claim, evaluating q(r) ≠ t'(r) in general.

---

## Session 53 - Batching Coefficient Fix (COMPLETED)

### Summary

Fixed the batching coefficient Montgomery form bug. Initial claim now matches Jolt exactly. Sumcheck still fails due to Gruen polynomial claim divergence (original issue from Session 52).

### Session 53 - Batching Coefficient Fix

**Key Finding:**

Jolt uses TWO different challenge scalar derivation methods:

1. **`challenge_scalar_optimized()`** → Returns `F::Challenge` (MontU128Challenge)
   - Stores 128-bit value as `[0, 0, low, high]` directly
   - Used for sumcheck challenges (r0, r1, etc.)
   - Has special multiplication semantics

2. **`challenge_scalar()`** (via `challenge_vector`) → Returns proper `F`
   - Uses `F::from_bytes()` which converts to proper Montgomery form
   - Used for batching coefficients
   - Uses standard field multiplication

**The Bug:**

Zolt was using the `[0, 0, low, high]` representation for ALL challenges, which:
- ✓ Correct for sumcheck challenges (matches MontU128Challenge)
- ✗ Wrong for batching coefficients (should use proper Montgomery form)

**Values Before Fix:**
```
uni_skip_claim: 6819845662591168690540177426014504904478356236997774174463102166949757228057 ✓ (matched)
batching_coeff (Zolt): 3585365310819910961476179832490187488669617511825727803093062673748144578813 (WRONG - 256-bit)
batching_coeff (Jolt): 38168636090528866393074519943917698662 (correct - 128-bit)
initial_claim: MISMATCH
```

**Values After Fix:**
```
uni_skip_claim: 6819845662591168690540177426014504904478356236997774174463102166949757228057 ✓
batching_coeff: 38168636090528866393074519943917698662 ✓ (now matches!)
initial_claim: 21674923214564316833547681277109851767489952526125883853786217589527714841889 ✓ (now matches!)
```

**The Fix:**

Added `challengeScalarFull()` method to transcript that:
1. Gets 16 bytes from transcript
2. Reverses them (matching Jolt)
3. Interprets as 128-bit value
4. Masks to 125 bits
5. Stores in lower limbs `[low, high, 0, 0]`
6. Converts to Montgomery form via `toMontgomery()`

Updated `proof_converter.zig` to use `challengeScalarFull()` for batching_coeff.

**Key Insight - Why `[0, 0, low, high]` Works for Sumcheck:**

In Jolt, sumcheck challenges use `MontU128Challenge` which has special multiplication that handles the `[0, 0, low, high]` representation correctly. When Zolt uses this same representation with standard Montgomery multiplication, it still works because:
- Both representations give the same result in polynomial evaluation
- The `uni_skip_claim = poly.evaluate(r0)` matched even with the "wrong" representation

But for batching coefficients:
- `initial_claim = uni_skip_claim * batching_coeff`
- Standard multiplication with `[0, 0, low, high]` gives wrong result
- Need proper Montgomery form for correct multiplication

---

## Session 52 - Gruen Polynomial Claim Divergence

**Key Finding:**

The Gruen polynomial q(X) is constructed to satisfy the sumcheck constraint s(0)+s(1)=previous_claim, but it does NOT equal the bound multiquadratic polynomial. This causes the claim to diverge from eq_factor * (Az*Bz at bound point).

**All These Values Match Jolt:**
1. eq_factor (split_eq.current_scalar): 11957315549363330504202442373139802627411419139285673324379667683258896529103
2. Az_final * Bz_final: 12979092390518645131981692805702461345196587836340614110145230289986137758183
3. r1cs_input_evals (opening claims): Match exactly (verified byte-by-byte)

**But This Doesn't Match:**
- Zolt implied_inner_sum_prod (output_claim / eq_factor): differs from expected
- Expected: should equal Az_final * Bz_final

**Root Cause Analysis:**

The Gruen polynomial construction:
1. q(0) = t_zero = Σ eq * Az(0) * Bz(0) ✓
2. q(∞) = t_infinity = Σ eq * slope_products ✓
3. q(1) is SOLVED from s(0)+s(1)=previous_claim constraint

But q(1) ≠ Σ eq * Az(1) * Bz(1) in general!

So after evaluating at challenge r:
- q(r) ≠ bound_t_prime(r)
- final_claim ≠ eq_factor * Az_final * Bz_final

**Next Step:** Investigate how Jolt's prover ensures the claim tracks correctly. There may be a different mechanism or state that we're missing.

---

## Session 51 - Round Offset Fix

**Root Cause Found and Fixed:**

Jolt's verification flow after UniSkip:
1. `UniSkipFirstRoundProof::verify` derives r0
2. `cache_openings` calls `accumulator.append_virtual()` which appends uni_skip_claim to transcript
3. `BatchedSumcheck::verify` appends uni_skip_claim again as input_claim
4. Batching coefficient derived

Zolt was missing step 2. Added `transcript.appendScalar(uni_skip_claim)` after r0.

---

## CRITICAL - Challenge Scalar Methods

### MontU128Challenge-style (for sumcheck challenges)

```zig
// Used by challengeScalar() → challengeScalar128Bits()
const result = F{ .limbs = .{ 0, 0, masked_low, masked_high } };
```

- Stores 128-bit value in upper limbs
- DO NOT call toMontgomery()
- Matches Jolt's MontU128Challenge representation
- Used for r0, r1, r2, ... sumcheck challenges

### Proper Montgomery form (for batching coefficients)

```zig
// Used by challengeScalarFull()
const standard = F{ .limbs = .{ masked_low, masked_high, 0, 0 } };
const result = standard.toMontgomery();
```

- Stores 128-bit value in lower limbs as standard form
- MUST call toMontgomery() to convert
- Matches Jolt's F::from_bytes() behavior
- Used for batching coefficients

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
