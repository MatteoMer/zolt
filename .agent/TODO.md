# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: DORY COMMITMENT COMPARISON IN PROGRESS**

Created test infrastructure to compare Dory commitments between Zolt and Jolt.
Currently investigating why commitments differ even with same SRS points.

Key achievements:
1. ✅ SRS loading from Jolt-exported file
2. ✅ Polynomial-based matrix dimensions (sigma/nu calculation)
3. ✅ Comparison test infrastructure with Jolt
4. ✅ All 610 Zolt tests pass

Remaining issue: **Commitment values differ**
- Jolt commitment for [1,2,3,4,5,6,7,8]: first bytes `cf 11 82 20 dc 8c 59 10...`
- Zolt commitment for same polynomial: first bytes `88 12 50 7a 66 2c 7d 16...`
- Need to investigate MSM/pairing algorithm differences

---

## Debugging Plan

1. Compare G1 generator points between Jolt and Zolt
2. Compare row MSM results for same inputs
3. Compare individual pairing results
4. Check if polynomial layout differs (row-major vs column-major)

---

## Recent Progress

### Matrix Dimension Fix
Now computing sigma/nu from polynomial length instead of SRS size:
```
For num_vars=3 (8 coeffs): sigma=2, nu=1 → 4 cols × 2 rows
For num_vars=2 (4 coeffs): sigma=1, nu=1 → 2 cols × 2 rows
For num_vars=1 (2 coeffs): sigma=1, nu=0 → 2 cols × 1 row
```

### Test Infrastructure
- Added `test_export_dory_commitment` in Jolt to export reference commitment
- Added `dory commitment with jolt srs - compare matrix layout` test in Zolt
- Can now directly compare commitment bytes between implementations

---

## Major Milestones Achieved

1. ✅ Blake2b Transcript
2. ✅ JoltProof 7-stage structure
3. ✅ Arkworks serialization
4. ✅ GT element serialization
5. ✅ Cross-deserialization
6. ✅ UniSkip infrastructure
7. ✅ 48 opening claims
8. ✅ 19 R1CS constraints
9. ✅ JoltDevice support
10. ✅ Fiat-Shamir preamble
11. ✅ Byte reversal for transcript
12. ✅ GT elements in transcript
13. ✅ JoltProofWithDory bundle
14. ✅ SRS loading from file
15. ✅ --srs CLI option
16. ✅ Polynomial-based matrix dimensions
17. ✅ Commitment comparison test

---

## Test Status

### Zolt: 610/610 tests passing

### Jolt Cross-Verification

| Test | Status | Details |
|------|--------|---------|
| `test_deserialize_zolt_proof` | ✅ PASS | Deserializes correctly |
| `test_debug_zolt_format` | ✅ PASS | All claims valid |
| `test_export_dory_srs` | ✅ PASS | SRS exported |
| `test_export_dory_commitment` | ✅ PASS | Reference commitment exported |
| `test_verify_zolt_proof` | ❌ FAIL | UniSkip fails (commitment mismatch) |

---

## Summary

**Serialization: COMPLETE**
**Transcript: COMPLETE**
**SRS Loading: COMPLETE**
**Matrix Dimensions: FIXED**
**Commitment Algorithm: INVESTIGATING**

Next step: Debug MSM/pairing algorithm to match Jolt's commitment output.
