# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: TESTING DORY TRANSCRIPT INTEGRATION**

Key achievements:
1. **R1CS Input MLE Evaluations** - Opening claims now contain actual computed values
2. **Correct Round Polynomial Count** - Stage 1 has proper 1 + num_cycle_vars polynomials
3. **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs
4. **Non-zero UniSkip Coefficients** - UniSkip polynomial has non-trivial values
5. **JoltDevice from File** - Can read JoltDevice from Jolt-generated file
6. **Byte Reversal Fix** - Commitments now reverse bytes before appending to transcript
7. **Dory Transcript Integration** - Prover now uses GT elements (384 bytes) in transcript

Current issue: **Testing end-to-end verification**
- Need to generate a new proof with Dory commitments in transcript
- Verify with Jolt to confirm the fix works

---

## Recent Changes

### Dory Transcript Integration (just completed)

The prover now computes Dory commitments (GT elements, 384 bytes) for the transcript:

1. **Build polynomial evaluations** from bytecode, memory trace, and register trace
2. **Setup Dory SRS** with appropriate size
3. **Compute Dory commitments** using `DoryCommitmentScheme.commit()`
4. **Append GT elements to transcript** using `appendGT()` which reverses bytes

This matches what Jolt does:
- `append_serializable` serializes the commitment
- Reverses all bytes for EVM compatibility
- Appends to transcript

### Files Modified

- `src/transcripts/blake2b.zig`: Added `appendSerializable`, updated `appendGT` with byte reversal
- `src/zkvm/mod.zig`: Both `proveJoltCompatible` and `proveJoltCompatibleWithDevice` now use Dory

---

## Next Steps

1. [ ] Generate a proof using the updated prover
2. [ ] Test with Jolt verifier to confirm transcript match
3. [ ] If still failing, add transcript state debugging
4. [ ] Verify SRS generation matches Jolt's seed

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
14. ✅ **Dory Transcript** - GT elements used in transcript

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
| `test_verify_zolt_proof` | 🔄 TESTING | Needs re-test with Dory transcript |

---

## Key Files

### Core Implementation
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript with GT support |
| `src/zkvm/jolt_device.zig` | ✅ Done | JoltDevice deserialization |
| `src/zkvm/mod.zig` | ✅ Done | Dory commitments in transcript |
| `src/zkvm/proof_converter.zig` | ✅ Done | Stage conversion with tau |
| `src/zkvm/spartan/outer.zig` | ✅ Done | UniSkip computation |
| `src/poly/commitment/dory.zig` | ✅ Done | GT element support |

---

## Summary

**Serialization Compatibility: COMPLETE**
**Transcript Integration: COMPLETE (using Dory GT elements)**
**Verification: READY FOR TESTING**

The prover now uses Dory commitments (GT elements, 384 bytes) in the transcript,
matching Jolt's format. All bytes are reversed before appending, as Jolt does.

Next step: Generate a new proof and test with Jolt verifier.
