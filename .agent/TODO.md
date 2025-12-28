# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: TRANSCRIPT COMMITMENT TYPE MISMATCH**

Key achievements:
1. **R1CS Input MLE Evaluations** - Opening claims now contain actual computed values
2. **Correct Round Polynomial Count** - Stage 1 has proper 1 + num_cycle_vars polynomials
3. **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs
4. **Non-zero UniSkip Coefficients** - UniSkip polynomial has non-trivial values
5. **JoltDevice from File** - Can read JoltDevice from Jolt-generated file
6. **Byte Reversal Fix** - Commitments now reverse bytes before appending to transcript

Current issue: **Commitment type mismatch**
- Zolt uses G1 points (64 bytes) for commitments in transcript
- Jolt uses GT elements (384 bytes, Dory/Fp12) for commitments in transcript
- This causes tau values to differ, which causes UniSkip verification to fail

---

## Root Cause Analysis

### The Core Issue

Jolt uses **Dory commitments** which are GT elements (Fp12, 384 bytes), while Zolt uses **HyperKZG commitments** which are G1 points (64 bytes).

When the verifier runs:
1. **Preamble**: Appends memory layout, I/O, ram_K, trace_length ✅ (matches)
2. **Commitments**: Appends each commitment using `append_serializable` ❌ (MISMATCH)
   - Jolt: `serialize_uncompressed(GT) -> 384 bytes -> reverse -> append`
   - Zolt: `toBytes(G1) -> 64 bytes -> reverse -> append`
3. **Tau derivation**: Gets challenges from transcript ❌ (different due to step 2)
4. **UniSkip verification**: Uses tau to check power sum ❌ (fails)

### Solutions

1. **Full Solution**: Refactor Zolt prover to use Dory natively for commitments
   - Replace `PolyCommitment` (G1) with `DoryCommitment` (GT)
   - Use Dory throughout the proving pipeline

2. **Hybrid Solution**: Generate Dory commitments for transcript only
   - Keep internal HyperKZG for efficiency
   - Compute Dory commitments from polynomial evaluations
   - Append GT elements to transcript

3. **Testing Solution**: Create a transcript test that matches Jolt exactly
   - Use same JoltDevice file
   - Use same commitment values (serialize GT from Jolt)
   - Verify transcript state matches byte-for-byte

---

## Immediate Next Steps

1. [ ] Implement Dory commitment generation in transcript flow
2. [ ] Store polynomial evaluations (not just commitments) in proof structure
3. [ ] Add transcript state debugging to compare with Jolt
4. [ ] Test with known-good GT elements from Jolt

---

## Major Milestones Achieved

1. ✅ **Blake2b Transcript** - Identical Fiat-Shamir challenges for test vectors
2. ✅ **Proof Types** - JoltProof with 7-stage structure
3. ✅ **Arkworks Serialization** - Byte-perfect format compatibility
4. ✅ **Dory Commitment** - GT (Fp12) serialization
5. ✅ **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs!
6. ✅ **Univariate Skip Infrastructure** - Degree-27/12 polynomials
7. ✅ **All 48 Opening Claims** - Including all R1CS inputs + OpFlags
8. ✅ **19 R1CS Constraints** - Matching Jolt's exact structure
9. ✅ **Constraint Evaluators** - Az/Bz for first and second groups
10. ✅ **JoltDevice Type** - Read from Jolt-generated file
11. ✅ **Fiat-Shamir Preamble** - Function implemented
12. ✅ **CLI --device option** - Use JoltDevice from file
13. ✅ **Byte Reversal** - Commitments reversed before transcript append

---

## Test Status

### All 608 Tests Passing (Zolt)

```
zig build test --summary all
Build Summary: 5/5 steps succeeded; 608/608 tests passed
```

### Cross-Verification Tests (Jolt)

| Test | Status | Details |
|------|--------|---------|
| `test_deserialize_zolt_proof` | ✅ PASS | 26558 bytes, 48 claims |
| `test_debug_zolt_format` | ✅ PASS | All claims valid |
| `test_verify_zolt_proof` | ❌ FAIL | UniSkip power sum check fails |

---

## Key Files

### Core Implementation
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript with byte reversal |
| `src/zkvm/jolt_device.zig` | ✅ Done | JoltDevice deserialization |
| `src/zkvm/mod.zig` | 🔄 In Progress | Need Dory commitments for transcript |
| `src/zkvm/proof_converter.zig` | 🔄 In Progress | Stage conversion with tau |
| `src/zkvm/spartan/outer.zig` | ✅ Done | UniSkip computation |
| `src/poly/commitment/dory.zig` | ✅ Done | GT element support |

---

## Summary

**Serialization Compatibility: COMPLETE**
**Transcript Integration: NEEDS DORY COMMITMENTS**
**Verification: BLOCKED ON COMMITMENT TYPE MATCH**

The proof structure is correct and deserializes successfully. The remaining issue is
that the transcript must use Dory commitments (GT elements, 384 bytes) instead of
HyperKZG commitments (G1 points, 64 bytes) to match Jolt's verifier exactly.

Next step: Implement Dory commitment generation in the transcript flow.
