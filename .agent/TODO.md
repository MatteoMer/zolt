# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: WORKING ON STAGE 1 SUMCHECK VERIFICATION**

### Latest Progress (2024-12-28)

1. ✅ **R1CS Constraints Satisfied** - All 19 constraints now pass (Az*Bz = 0)
   - Fixed IsCompressed flag detection from trace
   - Added `is_compressed` field to TraceStep

2. ✅ **UniSkip Polynomial Correct** - All-zero polynomial (correct for satisfied constraints)

3. ❌ **Stage 1 Sumcheck Verification** - Fails at BatchedSumcheck::verify
   - Issue: Opening accumulator protocol mismatch
   - The verifier's `cache_openings` expects specific evaluation relationships

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

### In Progress 🔄
- **Stage 1 Sumcheck Verification**
  - Understanding opening accumulator protocol
  - Matching prover-verifier relationship for opening claims

### Pending ⏳
- Stages 2-7 verification
- Full proof verification test

---

## Current Issue: Stage 1 Sumcheck

The verification passes UniSkip but fails during the batched sumcheck verification.

### Root Cause Analysis

Looking at Jolt's `BatchedSumcheck::verify`:

1. **Input claim** = `accumulator.get_virtual_polynomial_opening(UnivariateSkip, SpartanOuter)`
2. **Verification loop** computes `e = compressed_poly.eval_from_hint(&e, &r_i)`
3. **After rounds**, calls `cache_openings(...)` then `expected_output_claim(...)`

The `cache_openings` function modifies the accumulator using sumcheck challenges.
The `expected_output_claim` uses R1CS input evaluations from the accumulator.

For satisfied constraints (Az*Bz = 0):
- `inner_sum_prod = 0` → `expected_output_claim = 0`
- With all-zero polynomials and hint=0 → `output_claim = 0`

These should match, but something in the protocol is misaligned.

### Key Files
- `/jolt-core/src/subprotocols/sumcheck.rs:200-252` - BatchedSumcheck::verify
- `/jolt-core/src/zkvm/spartan/outer.rs:438-453` - cache_openings
- `/jolt-core/src/zkvm/spartan/outer.rs:407-436` - expected_output_claim

---

## Test Status

### Zolt: All 614 tests passing ✅

### Jolt Cross-Verification

| Test | Status | Details |
|------|--------|---------|
| `test_deserialize_zolt_proof` | ✅ PASS | Deserializes correctly |
| `test_debug_zolt_format` | ✅ PASS | All claims valid |
| `test_export_dory_srs` | ✅ PASS | SRS exported |
| `test_export_dory_commitment_debug` | ✅ PASS | All values match |
| `test_verify_zolt_proof` | ❌ FAIL | Stage 1 sumcheck fails |

---

## Summary

**Serialization: COMPLETE** ✅
**Transcript: COMPLETE** ✅
**Dory Commitment: MATCHING** ✅
**R1CS Constraints: SATISFIED** ✅
**UniSkip Polynomial: CORRECT** ✅
**Stage 1 Sumcheck: IN PROGRESS** 🔄
