# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: DORY SRS MISMATCH**

The bundled Dory commitment approach is working (same commitments in transcript and proof),
but the underlying Dory SRS generation in Zolt differs from Jolt's external `dory` crate.

Key achievements:
1. ✅ JoltProofWithDory bundle ensures consistent commitments
2. ✅ Polynomial evaluations stored with proof
3. ✅ Serialization uses bundled commitments
4. ✅ All 608 Zolt tests pass
5. ✅ Cross-deserialization works (Jolt can read Zolt proofs)

Remaining issue: **Dory SRS generation mismatch**
- Zolt's Dory SRS uses custom G1/G2 point generation
- Jolt's Dory uses external `dory` crate with `ArkworksProverSetup::new_from_urs`
- Even with same seed "Jolt Dory URS seed", the resulting points differ
- This causes commitment values to differ, which causes transcript mismatch

---

## Root Cause Analysis

### The SRS Generation Issue

Both Zolt and Jolt use the same seed:
```
SHA3-256("Jolt Dory URS seed") -> ChaCha20Rng seed
```

But:
- **Jolt**: Uses `dory::backends::arkworks::ArkworksProverSetup::new_from_urs(&mut rng, max_num_vars)`
- **Zolt**: Uses custom `generateG1Point(seed, index)` and `generateG2Point(seed, index)`

These produce different curve points, leading to different commitments.

### Solutions

1. **Port Jolt's exact SRS generation**
   - Study `dory` crate's `new_from_urs` implementation
   - Port the exact point generation algorithm to Zig
   - This is complex but ensures compatibility

2. **Use Jolt's SRS as data file**
   - Generate SRS in Jolt, serialize to file
   - Load SRS from file in Zolt
   - Works but requires external dependency

3. **Create transcript test without commitments**
   - Test that transcript produces same challenges after preamble
   - Isolate the commitment issue from transcript logic

---

## Verification Flow Analysis

When Jolt verifies a proof:
1. **Create transcript** with label "Jolt"
2. **Preamble** - append memory layout, I/O, ram_K, trace_length
3. **Append commitments** - for each commitment: serialize GT (384 bytes), reverse, append
4. **Derive tau** - call `challenge_vector_optimized` to get tau values
5. **Verify UniSkip** - check power sum using tau

The failure occurs at step 5 because:
- Jolt recomputes the transcript from the proof file's commitments
- These commitments are GT elements computed by Zolt's Dory
- Since Zolt's Dory uses different SRS, the GT values differ
- Tau values derived from transcript differ
- UniSkip check fails

---

## Next Steps

1. [ ] Study `dory` crate's SRS generation code
2. [ ] Port exact algorithm to Zig
3. [ ] Create test vector: generate SRS in Jolt, export points, compare with Zolt
4. [ ] Alternatively: create hybrid approach where Zolt loads Jolt's SRS

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
9. ✅ **JoltDevice Type** - Read from Jolt-generated file
10. ✅ **Fiat-Shamir Preamble** - Function implemented
11. ✅ **Byte Reversal** - Commitments reversed before transcript append
12. ✅ **Dory Transcript** - GT elements used in transcript
13. ✅ **JoltProofWithDory Bundle** - Consistent commitment handling
14. ✅ **Polynomial Evaluations** - Stored with proof for serialization

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
| `test_deserialize_zolt_proof` | ✅ PASS | Proof deserializes correctly |
| `test_debug_zolt_format` | ✅ PASS | All claims valid |
| `test_verify_zolt_proof` | ❌ FAIL | UniSkip fails (SRS mismatch) |

---

## Key Files

### Core Implementation
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript with GT support |
| `src/zkvm/jolt_device.zig` | ✅ Done | JoltDevice deserialization |
| `src/zkvm/mod.zig` | ✅ Done | JoltProofWithDory bundle |
| `src/zkvm/jolt_types.zig` | ✅ Done | JoltProofWithDory type |
| `src/poly/commitment/dory.zig` | 🔄 Needs Work | SRS generation mismatch |

---

## Summary

**Serialization Compatibility: COMPLETE**
**Transcript Integration: COMPLETE (consistent Dory handling)**
**Verification: BLOCKED ON DORY SRS MATCH**

The proof structure is correct and the commitment handling is consistent.
The remaining blocker is that Zolt's Dory SRS generation produces different
points than Jolt's external `dory` crate, causing commitment values to differ.

Next step: Port exact SRS generation algorithm from `dory` crate to Zig.
