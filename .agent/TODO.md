# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: DEBUGGING STAGE 1 SUMCHECK - CLAIM MISMATCH**

### Latest Progress (2024-12-28, Iteration 24)

1. ✅ **Fixed R1CS Input Evaluation Point** - Corrected `r_cycle` computation:
   - Skip first challenge (`r_stream`)
   - Reverse remaining challenges (LITTLE_ENDIAN → BIG_ENDIAN)
   - Match Jolt's `normalize_opening_point` behavior

2. ✅ **Analyzed Jolt's Expected Output Claim** - Understood verification equation:
   ```
   expected_output_claim = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
   ```
   where `inner_sum_prod = (Σ_y A(rx,y)·z(y)) * (Σ_y B(rx,y)·z(y))`

3. ❌ **Streaming Prover Issues Identified**:
   - `interpolateFirstRoundPoly` is a placeholder (copies evals as coeffs instead of proper interpolation)
   - Lagrange basis computed for wrong domain ({0..9} instead of {-4..5})

### Current Issue Analysis

The sumcheck verification fails because:

1. **Opening Claims Must Match Sumcheck**: The `r1cs_input_evals` in opening_claims are the MLE evaluations at `r_cycle`. The verifier uses these to compute `expected_output_claim`.

2. **Challenge Ordering**:
   - Jolt: `sumcheck_challenges = [r_stream, r_1, ..., r_n]`
   - `r_cycle = challenges[1..]` reversed to BIG_ENDIAN
   - Our code now does this correctly

3. **Streaming Prover Bugs**:
   - The `interpolateFirstRoundPoly` function doesn't actually interpolate
   - This means first-round polynomial is incorrect
   - All subsequent challenges derived from it are wrong

### Path Forward

**Option A: Fix Streaming Prover** (Complex)
1. Implement proper polynomial interpolation from evaluations to coefficients
2. Fix Lagrange basis for domain {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
3. Ensure Az*Bz products match Jolt's constraint structure

**Option B: Zero Proof Mode** (Simple)
1. Use all-zero witnesses
2. All polynomials become zero
3. All opening claims are zero
4. `expected_output_claim = 0` (since inner_sum_prod = 0)
5. Verification should pass (but proves nothing useful)

**Option C: Port Jolt's Prover** (Most Reliable)
1. Translate Jolt's `OuterLinearStage` to Zig exactly
2. Use same round polynomial computation
3. Ensure byte-for-byte compatibility

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
