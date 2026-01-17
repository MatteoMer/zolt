# Zolt-Jolt Cross-Verification Progress

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

**Evidence:**
- Jolt's `inner_sum_prod (Az*Bz)` = 14279035532130326282759614533689080459036208928223103610768541756919699764986
- But proof produces output_claim = 10634556229438437044377436930505224242122955378672273598842115985622605632100
- These should be related by tau factors, but the mismatch suggests Az*Bz is computed differently

**Likely Bug Location:**
`src/zkvm/spartan/streaming_outer.zig`:
- `materializeLinearPhasePolynomials()` - Az/Bz computation
- `computeRemainingRoundPoly()` - Round polynomial generation
- Constraint group splitting (FIRST_GROUP_INDICES, SECOND_GROUP_INDICES)
- Lagrange kernel application

### Next Steps

1. **Debug Az/Bz values**: Add logging to show actual Az*Bz products during streaming prover
2. **Compare constraint evaluation**: Check if R1CS constraint evaluation matches Jolt
3. **Verify split_eq handling**: Ensure eq polynomial factorization is correct
4. **Check Lagrange kernel**: Verify lagrange_tau_r0 is applied correctly

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

### Stage Status

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ❌ Sumcheck mismatch | Polynomial coefficients wrong |
| 2 | ✅ PASS | - | - |
| 3 | ✅ PASS | - | - |
| 4 | ✅ PASS | Montgomery fix applied |
| 5 | ✅ PASS | - | - |
| 6 | ✅ PASS | - | - |

---

## Session 40 Summary - Stage 4 Investigation (2026-01-17)

### Stage 2 Fix
Removed synthetic termination write from memory trace. In Jolt, the termination bit
is set directly in val_final during OutputSumcheck, NOT in the execution/memory trace.
The RWC sumcheck only includes actual LOAD/STORE instructions.

### Stage 4 Deep Investigation

#### Verified Matches
1. **Transcript state**: IDENTICAL between Zolt and Jolt at all checkpoints
2. **Challenge bytes**: IDENTICAL (f5 ce c4 8c b0 64 ba b5 ce 4d a4 2a db 38 f8 ac)
3. **Input claims**: ALL THREE match exactly
4. **Batching coefficients**: MATCH
5. **Polynomial coefficients in proof**: MATCH

### Root Cause Analysis

The challenge in Zolt was stored as:
```zig
result = F{ .limbs = .{ 0, 0, masked_low, masked_high } };
```

This is NOT proper Montgomery form. Jolt's MontU128Challenge stores the same bytes
but interprets them as a BigInt that gets converted to Montgomery via from_bigint_unchecked.

**Fix applied in Session 41**: Convert properly using toMontgomery().

---

## Previous Sessions

### Stage 3 Fix (Session 35)
- Fixed prefix-suffix decomposition convention (r_hi/r_lo)

### Stage 1 Fix
- Fixed NextPC = 0 issue for NoOp padding

---

## Technical References

- Jolt MontU128Challenge: `jolt-core/src/field/challenge/mont_ark_u128.rs`
- Jolt BatchedSumcheck verify: `jolt-core/src/subprotocols/sumcheck.rs:180`
- Zolt Blake2b transcript: `src/transcripts/blake2b.zig`
- Zolt Stage 4 proof: `src/zkvm/proof_converter.zig` line ~1700
- Zolt sumcheck generation: `src/zkvm/proof_converter.zig:generateStreamingOuterSumcheckProofWithTranscript`
