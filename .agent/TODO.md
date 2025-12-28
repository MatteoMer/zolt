# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: DEBUGGING STAGE 1 SUMCHECK - CLAIM MISMATCH**

### Latest Progress (2024-12-28, Agent Iteration)

**✅ Implemented Core Fixes:**
1. Added `LagrangeHelper.shiftCoeffsI32` - matches Jolt's Lagrange coefficient computation
2. Precomputed `COEFFS_PER_J[9][10]` for extended domain evaluation
3. Fixed `SpartanOuterProver.computeUniskipFirstRoundPoly` to use COEFFS_PER_J:
   - Evaluates Az(y_j) and Bz(y_j) separately at extended points
   - Multiplies them together for the Az*Bz product
   - This gives non-zero results even when base Az*Bz products are zero

**✅ Test Verified:**
- Unit test with non-trivial Az/Bz values produces non-zero coefficients ✅
- The algorithm is correct for non-zero inputs

**Current Blocker:**
- UniSkip polynomial is STILL all zeros in the proof file
- Unit tests for `outer.zig` aren't registered in build.zig (not running)
- Need to investigate why production code path produces zeros

**Possible Root Causes:**
1. The constraint evaluators (AzFirstGroup/BzFirstGroup.fromWitness) might produce all zeros
2. The cycle_witnesses slice might be empty
3. There might be an early return path that bypasses the computation

**Next Steps:**
1. Register outer.zig tests in build.zig so they run
2. Add logging to trace actual Az/Bz values during proof generation
3. Verify the R1CSCycleInputs are properly populated from the execution trace

---

### Previous Progress (2024-12-28, Iteration 25 - Agent)

1. ✅ **Found Root Cause of UniSkip All-Zeros Issue**

   The problem is in how we compute the extended domain evaluations:

   **Jolt's Approach**:
   - For base domain points y ∈ {-4,...,5}: Az*Bz = 0 for satisfied constraints
   - For extended domain points y ∈ {-9,...,-5} ∪ {6,...,9}: Uses **precomputed Lagrange coefficients** (`COEFFS_PER_J[j]`) to evaluate the constraint polynomials directly
   - Even if base evaluations are zero, extended evaluations are NON-ZERO because the constraint polynomial has specific structure

   **Zolt's Bug**:
   - `evaluators.zig::computeExtendedEvals()` calls `computeBaseWindowEvals()` which returns zeros
   - Then interpolates from zeros → gets more zeros
   - This gives an all-zero UniSkip polynomial, which is WRONG

   **Fix Required**:
   - Implement precomputed Lagrange coefficients for extended domain evaluation
   - Evaluate Az and Bz at extended points using the polynomial structure, NOT by interpolating from base values

2. ✅ **Understood UniSkip Evaluation Flow**:
   - UniSkip polynomial s1(Y) = L(τ_high, Y) · t1(Y)
   - t1(Y) = Σ_x eq(τ, x) · Az(x, Y) · Bz(x, Y)
   - Even when constraints are satisfied (base window = 0), the polynomial has non-zero evaluations at extended points
   - The evaluation at r0 (transcript challenge) becomes input_claim for remaining sumcheck rounds

3. ✅ **Previous fixes still apply**:
   - R1CS input evaluation point correction
   - Transcript integration
   - Gruen polynomial implementation

### Current Issue Analysis

The UniSkip polynomial is all zeros because of incorrect extended evaluation computation.

### Path Forward: Implement Precomputed Lagrange Coefficients

**Required Changes**:

1. **Precompute `COEFFS_PER_J` array**:
   - For each extended domain point j ∈ {0,...,8} (mapping to {-9,...,-5, 6,...,9})
   - Compute Lagrange coefficients L_i(y_j) for i ∈ {0,...,9} (base domain indices)
   - These are the weights for evaluating Az and Bz at extended points

2. **Use constraint structure directly**:
   - For first group (10 constraints): evaluate Az_i and Bz_i for each constraint
   - Weight by COEFFS_PER_J[j][i] to get evaluation at extended point y_j
   - Compute Az(y_j) * Bz(y_j)

3. **Sum over cycles with eq weights**:
   - extended_evals[j] = Σ_x eq(τ, x) * Az(x, y_j) * Bz(x, y_j)

4. **Build UniSkip polynomial**:
   - Fill t1_vals with base (zeros) and extended (non-zero) evaluations
   - Interpolate to get t1 coefficients
   - Multiply by Lagrange kernel L(τ_high, Y)
   - Result is UniSkip polynomial with non-zero coefficients

---

## Major Milestones

### Completed ✅
1-35. All previous items plus:
36. R1CS input evaluation point correction (r_cycle endianness)
37. Analysis of Jolt's expected_output_claim computation

### In Progress 🔄
- Stage 1 Sumcheck Verification

### Pending ⏳
- Fix streaming prover polynomial interpolation
- Stages 2-7 verification
- Full proof verification test

---

## Test Status

### Zolt: All tests passing ✅

### Jolt Cross-Verification

| Test | Status | Details |
|------|--------|---------|
| `test_serialization_vectors` | ✅ PASS | Field/GT serialization matches |
| `test_zolt_compatibility_vectors` | ✅ PASS | Blake2b transcript compatible |
| `test_deserialize_zolt_proof` | ⏳ | Requires proof file |
| `test_verify_zolt_proof` | ⏳ | Requires preprocessing |

---

## Key Technical Insights

### R1CS Input Evaluation Point

In Jolt's `OuterRemainingSumcheckVerifier::expected_output_claim`:
```rust
let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);
// where normalize_opening_point returns challenges[1..] in BIG_ENDIAN
```

And for the eq polynomial:
```rust
let r_tail_reversed: Vec<F::Challenge> = sumcheck_challenges.iter().rev().copied().collect();
let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);
```

The key is that `r_cycle` for opening claims and the `r_tail_reversed` for eq computation are different! The opening claims use `challenges[1..]` in BIG_ENDIAN, while the eq computation uses ALL challenges reversed.

### Streaming Prover Fix Priority

1. `interpolateFirstRoundPoly` - Must use proper Newton/Lagrange interpolation
2. `computeLagrangeEvalsAtR0` - Must use domain {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
3. Remaining round polynomials - Must correctly sum over bound variables

---

## Summary

**Serialization: COMPLETE** ✅
**Transcript Format: CORRECT** ✅
**Dory Commitment: MATCHING** ✅
**R1CS Constraints: STRUCTURE OK** ✅
**Opening Point: FIXED** ✅
**Stage 1 Sumcheck: PROVER BUGS** ❌

The streaming prover has fundamental issues in polynomial interpolation that must be fixed for proper sumcheck verification.
