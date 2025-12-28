# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: DEBUGGING STAGE 1 SUMCHECK - CLAIM MISMATCH**

### Latest Progress (2024-12-28, Iteration 23)

1. ✅ **Fixed Az*Bz Computation** - Now computes (Σ L_i * az_i) * (Σ L_i * bz_i)
2. ✅ **Fixed Evals-to-Coeffs Conversion** - Proper interpolation for compressed polys
3. ✅ **Fixed Transcript Markers** - "UncompressedUniPoly_begin/end" for UniSkip, "UniPoly_begin/end" for compressed
4. ✅ **Fixed Tau Ordering** - Now derives tau in same order as Jolt (not reversed)
5. ❌ **Stage 1 Sumcheck** - output_claim ≠ expected_output_claim

### Current Issue Analysis

The sumcheck verification fails at the final check:
```rust
if output_claim != expected_output_claim {
    return Err(ProofVerifyError::SumcheckVerificationError);
}
```

The values are completely different, indicating the sumcheck polynomials don't correctly encode the R1CS constraint satisfaction.

### Fixes Applied This Session

1. **Polynomial Interpolation** - Added `UniPoly.interpolateDegree3()` to convert evaluations `[s(0), s(1), s(2), s(3)]` to coefficients `[c0, c1, c2, c3]`, and `evalsToCompressed()` for Jolt's format `[c0, c2, c3]`.

2. **Az*Bz Product** - Fixed to compute `(Σ L_i * condition_i) * (Σ L_i * magnitude_i)` instead of `Σ L_i * (condition_i * magnitude_i)`.

3. **Transcript Markers** - Use `UncompressedUniPoly_begin/end` for UniSkip (28 coefficients), `UniPoly_begin/end` for compressed round polys (3 coefficients).

4. **Tau Order** - Changed from `tau[n-1-i] = challenge` to `tau[i] = challenge` to match Jolt's `challenge_vector_optimized`.

### Remaining Investigation

The prover generates non-zero polynomial coefficients, but verification fails. Possible causes:

1. **Initial Claim Mismatch** - The Stage 1 sumcheck input claim comes from UnivariateSkip output
2. **Round Polynomial Computation** - The Gruen method may not correctly encode partial sums
3. **R1CS Input Evaluations** - The opening_claims may not match what the verifier expects

### Key Insight

Even with all transcript fixes, the core issue is that the streaming prover's round polynomial computation must satisfy:
- `p(0) + p(1) = current_claim` (hint relation)
- After all rounds: `p_{final}(r_{final}) = expected_output_claim`

The second condition depends on the actual R1CS witness polynomial evaluations at the random sumcheck point.

---

## Major Milestones

### Completed ✅
1-31. All previous items plus:
32. Polynomial interpolation (evals → coeffs)
33. Correct Az*Bz computation
34. Correct transcript markers
35. Correct tau ordering

### In Progress 🔄
- Stage 1 Sumcheck Verification

### Pending ⏳
- Stages 2-7 verification
- Full proof verification test

---

## Test Status

### Zolt: All 618 tests passing ✅

### Jolt Cross-Verification

| Test | Status | Details |
|------|--------|---------|
| `test_deserialize_zolt_proof` | ✅ PASS | Deserializes correctly |
| `test_verify_zolt_proof` | ❌ FAIL | Stage 1 - claim mismatch |

---

## Summary

**Serialization: COMPLETE** ✅
**Transcript Format: CORRECT** ✅
**Dory Commitment: MATCHING** ✅
**R1CS Constraints: SATISFIED** ✅
**Stage 1 Sumcheck: CLAIM MISMATCH** ❌

The streaming prover generates polynomial coefficients, but the final verification check fails. The output claim from polynomial evaluation doesn't match the expected output claim from R1CS evaluation.
