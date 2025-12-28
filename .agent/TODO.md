# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: DEBUGGING FIAT-SHAMIR TRANSCRIPT MISMATCH**

Key achievements:
1. **R1CS Input MLE Evaluations** - Opening claims now contain actual computed values
2. **Correct Round Polynomial Count** - Stage 1 has proper 1 + num_cycle_vars polynomials
3. **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs
4. **Non-zero UniSkip Coefficients** - UniSkip polynomial has non-trivial values
5. **JoltDevice from File** - Can read JoltDevice from Jolt-generated file

Current issue: **UniSkip power sum check fails**
- The UniSkip polynomial doesn't sum to zero over the symmetric domain
- This suggests the tau values derived from transcript don't match Jolt's
- Even with same JoltDevice file, the transcript state differs

---

## Root Cause Analysis

### Likely Issues

1. **Commitment serialization mismatch**
   - We append commitments to transcript in a format
   - Jolt may serialize GT elements differently for transcript

2. **Preamble format differences**
   - Jolt may use different serialization for u64 (arkworks LE vs raw LE)
   - Byte ordering or padding differences

3. **JoltDevice deserialization issue**
   - Our deserialization may not match arkworks format exactly
   - Fields might be in wrong order or have wrong sizes

### Next Debugging Steps

1. Add debug output to both Zolt and Jolt to print:
   - Transcript state after preamble
   - Transcript state after commitments
   - First few tau values derived

2. Compare byte-by-byte what's being appended to transcript

3. Verify JoltDevice deserialization is correct by:
   - Printing memory_layout values from Zolt
   - Comparing with values printed by Jolt

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
11. ✅ **Fiat-Shamir Preamble** - Function implemented (needs debugging)
12. ✅ **CLI --device option** - Use JoltDevice from file

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
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript |
| `src/zkvm/jolt_device.zig` | ✅ Done | JoltDevice deserialization |
| `src/zkvm/mod.zig` | 🔄 Debug | Fiat-Shamir preamble |
| `src/zkvm/proof_converter.zig` | 🔄 Debug | Stage conversion with tau |
| `src/zkvm/spartan/outer.zig` | ✅ Done | UniSkip computation |

---

## Summary

**Serialization Compatibility: COMPLETE**
**Transcript Integration: NEEDS DEBUGGING**
**Verification: BLOCKED ON TRANSCRIPT MATCH**

We're very close - the proof structure is correct, deserialization works, and
we have non-zero values in the UniSkip polynomial. The remaining issue is
ensuring the Fiat-Shamir transcript state matches Jolt's exactly so that tau
values are derived identically.

Next immediate step: Add debug output to compare transcript states.
