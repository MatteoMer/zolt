# Zolt-Jolt Cross-Verification Progress

## Session 46 Summary - Stage 4 Round Polynomial Mismatch (2026-01-18)

### Verified Facts
1. **gamma matches** between Zolt and Jolt ✓
2. **input_claim matches** ✓
3. **params.r_cycle (Stage 3 challenges) matches** ✓
4. **All 5 opening claims match** (val, rs1_ra, rs2_ra, rd_wa, inc) ✓
5. **Sumcheck univariate checks pass** (internally consistent)

### The Problem
Round 0 polynomial coefficients are completely different:

**Jolt's round 0 coefficients (from native prover):**
```
coeffs[0] = [c9, bd, 24, 80, fd, 25, 4f, 66, ...]
coeffs[1] = [c4, 31, 54, 6f, f7, a8, 04, 71, ...]
coeffs[2] = [28, 07, 8a, 46, 27, 62, fd, c6, ...]
coeffs[3] = [64, ab, e8, 66, ad, 84, 4b, b8, ...]
```

**Zolt's round 0 coefficients:**
```
coeffs[0] = [77, bb, 08, ec, 88, 15, 06, 86, ...]
coeffs[1] = [a9, 39, 38, 6a, dc, db, 0f, 2a, ...]
coeffs[2] = [d6, 96, 40, 95, c4, 1f, e8, b7, ...]
```

### Root Cause: Different Prover Implementations

**Jolt's Stage 4 Prover:**
1. Uses `GruenSplitEqPolynomial` with `LowToHigh` binding
2. Uses `ReadWriteMatrixCycleMajor` sparse matrix representation
3. Computes via `prover_message_contribution` on sparse entries only
4. Applies Gruen optimization: eq = E_out * E_in

**Zolt's Stage 4 Prover:**
1. Uses `computeEqEvalsBE` dense eq polynomial
2. Uses dense arrays: `val_poly[k * T + j]`
3. Iterates over all (j, k) pairs directly
4. No Gruen optimization

### Key Structural Differences

1. **Sparse vs Dense**:
   - Jolt: Only stores non-zero entries (sparse matrix)
   - Zolt: Dense arrays with many zeros

2. **Eq Polynomial**:
   - Jolt: GruenSplitEqPolynomial (splits into E_in and E_out for efficiency)
   - Zolt: Dense array of size T

3. **Round Poly Computation**:
   - Jolt: `prover_message_contribution` processes sparse entries
   - Zolt: Direct iteration over all pairs

### Why They Compute Different Polynomials

Both should represent the same polynomial:
```
P(j, k) = eq(r_cycle, j) * combined(j, k)
```

But the round polynomial coefficients differ because:
1. Jolt's Gruen optimization reorders the summation
2. Different sparse matrix processing vs dense iteration
3. Possibly different zero-handling

**This causes the sumcheck challenges to diverge**, which causes eq_val to differ, which causes the verification to fail.

### Files to Study
- Jolt sparse matrix: `jolt-core/src/subprotocols/read_write_matrix/`
- Jolt Gruen eq: `jolt-core/src/poly/gruen_split_eq_poly.rs`
- Jolt prover: `jolt-core/src/zkvm/registers/read_write_checking.rs`
- Zolt prover: `src/zkvm/spartan/stage4_prover.zig`

### Potential Fixes
1. **Full port**: Rewrite Zolt's Stage 4 prover to match Jolt's sparse + Gruen approach
2. **Debug deeper**: Add step-by-step comparison to find exact divergence

---

## Session 42 Summary - Sumcheck Polynomial Issue (2026-01-17)

### New Findings

#### Opening Claims ARE Being Computed
The opening claims ARE using actual polynomial evaluations (not zeros):
- Zolt's `r1cs_input_evals[0]` (LeftInstructionInput) computes to ~20050671788...
- This matches Jolt's expected value
- 36 R1CS input claims are appended to transcript in correct order
- The `addSpartanOuterOpeningClaimsWithEvaluations` function is being called

#### The Real Issue: Sumcheck Polynomial Coefficients
The sumcheck verification failure is NOT due to opening claims, but the **sumcheck proof polynomials**:
- `output_claim` (from proof): 10634556229438437044377436930505224242122955378672273598842115985622605632100
- `expected_output_claim` (from evaluations): 17179895292250284685319038554266025066782411088335892517994655937988141579529

These values should match. The verifier computes `expected_output_claim` from:
1. R1CS input evaluations (opening claims) - CORRECT
2. R1CS constraint matrix evaluations
3. Tau/challenge parameters

The `output_claim` comes from evaluating the proof's polynomial at challenges. The mismatch means:
1. The sumcheck proof polynomials are incorrectly generated
2. OR the challenges used during proving differ from verification
3. OR the Az*Bz computation is wrong

### Root Cause: Streaming Outer Prover Bug

**Key Discovery:**
- The round polynomial coefficients ARE non-zero (correct)
- The batching_coeff MATCHES between Zolt and Jolt (185020165269464640985840804826040774859)
- The challenges appear to be derived correctly from transcript
- BUT the final output_claim doesn't match expected_output_claim

**The Problem:**
The streaming outer prover generates polynomials that are **internally consistent** (sum(p(0), p(1)) == claim passes) but encode the **wrong computation**. The Az*Bz evaluation in the streaming prover differs from what Jolt's verifier expects.

---

## Session 41 Summary - Montgomery Fix & Serialization (2026-01-17)

### Major Accomplishments

1. **Fixed Stage 4 Montgomery Conversion** ✅
   - Root cause: Jolt's MontU128Challenge stores [0, 0, L, H] as BigInt representation
   - When converted to Fr: represents 2^128 * original_value (NOT the original 125-bit value)
   - OLD Zolt behavior: directly stored [0, 0, L, H] as Montgomery limbs (WRONG)
   - FIX: Store [0, 0, L, H] as standard form, then call toMontgomery()
   - Result: All 6 Zolt internal verification stages now PASS

2. **Fixed Proof Serialization Format** ✅
   - Issue: Using `--jolt` flag instead of `--jolt-format`
   - Issue: Proof was missing the claims count header
   - Result: Jolt can now deserialize Zolt proofs successfully

### Stage Status (as of Session 46)

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ✅ PASS | Fixed MontU128Challenge |
| 2 | ✅ PASS | ✅ PASS | - |
| 3 | ✅ PASS | ✅ PASS | - |
| 4 | ✅ PASS | ❌ FAIL | Round poly coefficients differ (sparse vs dense) |
| 5 | ✅ PASS | - | Blocked by Stage 4 |
| 6 | ✅ PASS | - | Blocked by Stage 4 |

---

## Technical References

- Jolt MontU128Challenge: `jolt-core/src/field/challenge/mont_ark_u128.rs`
- Jolt BatchedSumcheck verify: `jolt-core/src/subprotocols/sumcheck.rs:180`
- Jolt Stage 4 prover: `jolt-core/src/zkvm/registers/read_write_checking.rs`
- Jolt sparse matrix: `jolt-core/src/subprotocols/read_write_matrix/`
- Zolt Blake2b transcript: `src/transcripts/blake2b.zig`
- Zolt Stage 4 prover: `src/zkvm/spartan/stage4_prover.zig`
