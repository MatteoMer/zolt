# Zolt-Jolt Cross-Verification Progress

## Session 77 Summary - Config Format Fixed, Polynomial Mismatch Found (2026-01-29)

### Major Progress

1. **Config Serialization Fixed** - trace_length, ram_K, bytecode_K, ReadWriteConfig (4 u8s), OneHotConfig (2 u8s), DoryLayout (1 u8) now match Jolt format exactly.

2. **Proof Deserialization Works** - 91 opening claims, 37 commitments parsed correctly.

3. **Stage 1 Passes** - Outer Spartan sumcheck verification succeeds!

4. **Stage 2 Fails** - `output_claim != expected_claim`

### Stage 2 Analysis

**What Matches:**
- `initial_claim`: Zolt `fd 01 cb 55...` = Jolt `[fd, 01, cb, 55, ...]` ✓
- `batching_coeff[0]`: Zolt `de 49 43 bd...` = Jolt `[de, 49, 43, bd, ...]` ✓
- `input_claim[0]`: Zolt `86 a8 80 d3...` = Jolt `[86, a8, 80, d3, ...]` ✓

**What Doesn't Match:**
- Jolt `first round coeffs_except_linear[0]`: `[97, 3f, b6, 7c, c2, de, 38, c7, ...]`
- Zolt `combined_evals[0]`: `[0e, 82, 58, f7, 16, 29, e4, 34, ...]` (different!)

### Format Difference

Jolt uses `CompressedPoly<F>` which stores `coeffs_except_linear` = [c0, c2, c3] (skipping c1).
Zolt's Stage 2 outputs `combined_evals` = [s(0), s(1), s(2), s(3)] evaluations.

The conversion from evaluations to compressed coefficients might be wrong, OR the evaluations themselves are computed incorrectly.

### Key Issue Identified

The round polynomial values differ at round 0. This cascades through all subsequent rounds, causing the final output_claim to mismatch.

### Investigation Needed

1. Verify `evalsToCompressed` conversion matches Jolt's format
2. Check if ProductVirtualRemainder's `computeRoundPolynomial` produces correct values
3. Verify the split_eq polynomial initialization

---

## Session 75 Summary - Challenge Type Analysis (2026-01-29)

### Challenge Type Mapping Verified

| Jolt Function | Returns | Zolt Equivalent | Use Case |
|---------------|---------|-----------------|----------|
| `challenge_scalar::<F>()` | Fr (Montgomery) | `challengeScalarFull()` | Batching coeffs, gamma values |
| `challenge_scalar_optimized::<F>()` | MontU128Challenge (125-bit, `[0,0,L,H]`) | `challengeScalar()` | tau_high, r0, sumcheck r_i |
| `challenge_vector(n)` | Vec<Fr> | n × `challengeScalarFull()` | Batching coeffs |
| `challenge_vector_optimized(n)` | Vec<MontU128Challenge> | n × `challengeScalar()` | r_address |
| `challenge_scalar_powers(n)` | Vec<Fr> (1, q, q², ...) | `challengeScalarPowers()` | Gamma powers |

### Stage 2 Challenge Sampling Order (Verified Correct)

1. `ProductVirtualUniSkipParams::new` → `challenge_scalar_optimized` → `tau_high_stage2`
2. UniSkip proof: append poly → `challenge_scalar_optimized` → `r0_stage2`
3. UniSkip `cache_openings`: `append_virtual(uni_skip_claim)`
4. `RamReadWriteCheckingParams::new` → `challenge_scalar` → `gamma_rwc`
5. `OutputSumcheckParams::new` → `challenge_vector_optimized(log_k)` → `r_address`
6. `InstructionLookupsClaimReductionSumcheckParams::new` → `challenge_scalar` → `gamma_instr`
7. `BatchedSumcheck::verify` → append input_claims → `challenge_vector(5)` → batching_coeffs

**Zolt uses matching functions for all of these ✓**

---

## Session 74 Summary - Stage 2 Deep Dive (2026-01-29)

### Key Finding: Zolt Prover is INTERNALLY CONSISTENT

**Evidence:**
- `STAGE2_FINAL: output_claim` = `{ 181, 30, 249, 122, ... }` (LE bytes)
- `expected_batched (from provers)` = `{ 35, 43, 4, 85, ... }` (BE bytes)
- Converting LE→BE: These ARE the same value ✓

**Implication:** The prover computes correct round polynomials that evaluate to what it expects. The issue is that Jolt's verifier computes a DIFFERENT expected value.

### Stage 2 Architecture Analysis

| Instance | Verifier | Rounds | Start | input_claim |
|----------|----------|--------|-------|-------------|
| 0 | ProductVirtualRemainder | 8 | 16 | uni_skip_claim |
| 1 | RamRafEvaluation | 16 | 8 | RamAddress@SpartanOuter |
| 2 | RamReadWriteChecking | 24 | 0 | RamReadValue + γ*RamWriteValue |
| 3 | OutputSumcheck | 16 | 8 | 0 |
| 4 | InstructionLookupsClaimReduction | 8 | 16 | LookupOutput + γ*Left + γ²*Right |

### Instance 0 (ProductVirtualRemainder) Expected Formula

```
expected = tau_high_bound_r0 * eq(tau_low, r_tail_reversed) * fused_left * fused_right
```

Where:
- `fused_left = w[0]*l_inst + w[1]*is_rd_not_zero + w[2]*is_rd_not_zero + w[3]*lookup_out + w[4]*j_flag`
- `fused_right = w[0]*r_inst + w[1]*wl_flag + w[2]*j_flag + w[3]*branch_flag + w[4]*(1-next_is_noop)`
- `w[i]` = Lagrange weights at r0 over domain [-2,-1,0,1,2]

---

## Session 73 Summary - Deserialization Complete! (2026-01-29)

### Critical Fix: SumcheckId Mismatch

**Root Cause:** Zolt had 24 SumcheckId values, Jolt has 22.

The extra values were:
- `AdviceClaimReductionCyclePhase = 20`
- `AdviceClaimReduction = 21`

**Fix:** Removed extra values, renumbered:
- `IncClaimReduction = 20`
- `HammingWeightClaimReduction = 21`
- `COUNT = 22`

### Deserialization Result: COMPLETE SUCCESS

All 40544 bytes parse correctly.

---

## Previous Sessions

### Session 72 (2026-01-28)
- 714/714 unit tests passing
- Stage 3 sumcheck mathematically correct
- Opening claims storage verified

### Session 71 (2026-01-28)
- Instance 0 (RegistersRWC) verified correct
- Synthetic termination write discovery

### Session 70 (2026-01-28)
- Stage 4 final claim mismatch found
- Phase 2/3 from_evals_and_hint pattern applied
