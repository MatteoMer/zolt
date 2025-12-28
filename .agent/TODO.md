# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: STAGE 1 SUMCHECK - NON-ZERO POLYNOMIALS**

### Latest Progress (2024-12-28)

1. ✅ **Fixed Az*Bz Computation** - Now computes (Σ L_i * az_i) * (Σ L_i * bz_i)
   - Was incorrectly computing Σ L_i * (az_i * bz_i)
   - Now produces non-zero products for valid traces

2. ✅ **Fixed Evals-to-Coeffs Conversion** - Proper interpolation for compressed polys
   - Added interpolateDegree3() and evalsToCompressed() functions
   - Converts [s(0), s(1), s(2), s(3)] evaluations to [c0, c2, c3] coefficients

3. ❌ **Stage 1 Sumcheck Verification** - Still fails with non-zero polynomials
   - Issue: Likely transcript challenge mismatch
   - The streaming prover generates polynomials, but verification fails

---

## Major Milestones

### Completed ✅
1. Blake2b Transcript
2. JoltProof 7-stage structure
3. Arkworks serialization
4. GT element serialization
5. Cross-deserialization
6. UniSkip infrastructure (28-coefficient polynomial)
7. 48 opening claims
8. 19 R1CS constraints matching Jolt
9. JoltDevice support
10. Fiat-Shamir preamble
11. Byte reversal for transcript
12. GT elements in transcript
13. JoltProofWithDory bundle
14. SRS loading from file
15. --srs CLI option
16. Polynomial-based matrix dimensions
17. arkworks flag bit masking
18. G1 MSM matching Jolt
19. G2 generator matching arkworks
20. ATE_LOOP_COUNT from arkworks
21. Projective Miller loop implementation
22. fp6MulBy01 sparse multiplication
23. fp12MulBy034 sparse multiplication
24. **arkworks final exponentiation algorithm**
25. **Pairing matching Jolt**
26. **Dory commitment matching Jolt**
27. **IsCompressed detection from trace**
28. **R1CS constraint satisfaction (Az*Bz = 0)**
29. **Polynomial interpolation (evals → coeffs)**
30. **Correct Az*Bz computation (product of sums, not sum of products)**

### In Progress 🔄
- **Stage 1 Sumcheck Verification**
  - Non-zero polynomial coefficients being generated
  - Need to debug challenge derivation mismatch

### Pending ⏳
- Stages 2-7 verification
- Full proof verification test

---

## Current Issue: Stage 1 Sumcheck

The streaming outer prover now generates non-zero polynomial coefficients, but verification still fails.

### Current Output

Proof shows:
- `round 0 coeffs: [10048208656370426965890505746999157938..., 3429443408412640508..., 10951947682441407657...]`
- Non-zero Az*Bz values (e.g., 13877885662807731494)
- PC = 2147483648 (0x80000000) - valid entry point

### Hypothesis

The transcript challenge derivation differs between Zolt's prover and Jolt's verifier:
1. Zolt computes round polynomials using challenges from its transcript
2. Jolt verifies using challenges from its transcript
3. If these transcripts diverge, the final claim won't match

### Possible Causes

1. **UniSkip polynomial transcription** - How coefficients are appended
2. **Round polynomial format** - What gets appended to transcript each round
3. **Byte ordering** - LE vs BE in transcript operations
4. **Input claim value** - What the initial sumcheck claim should be

### Key Files
- `src/zkvm/proof_converter.zig` - generateStreamingOuterSumcheckProofWithTranscript
- `src/zkvm/spartan/streaming_outer.zig` - computeRemainingRoundPoly
- `/jolt-core/src/subprotocols/sumcheck.rs` - SumcheckInstanceProof::verify

---

## Test Status

### Zolt: All 618 tests passing ✅

### Jolt Cross-Verification

| Test | Status | Details |
|------|--------|---------|
| `test_deserialize_zolt_proof` | ✅ PASS | Deserializes correctly |
| `test_debug_zolt_format` | ✅ PASS | All claims valid |
| `test_export_dory_srs` | ✅ PASS | SRS exported |
| `test_export_dory_commitment_debug` | ✅ PASS | All values match |
| `test_verify_zolt_proof` | ❌ FAIL | Stage 1 sumcheck fails |

---

## Next Steps

1. **Debug transcript synchronization**
   - Add logging to compare transcript state between prover and verifier
   - Verify what gets appended to transcript and in what order

2. **Verify initial claim**
   - Should Stage 1 sumcheck start with claim = 0?
   - Or should it use the UnivariateSkip output claim?

3. **Check expected_output_claim computation**
   - Ensure R1CS input evaluations match
   - Verify Lagrange/eq polynomial evaluations

---

## Summary

**Serialization: COMPLETE** ✅
**Transcript: COMPLETE** ✅
**Dory Commitment: MATCHING** ✅
**R1CS Constraints: SATISFIED** ✅
**UniSkip Polynomial: CORRECT** ✅
**Streaming Outer Prover: GENERATES NON-ZERO POLYS** ✅
**Stage 1 Sumcheck: VERIFICATION FAILING** 🔄
