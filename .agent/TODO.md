# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: PAIRING IMPLEMENTATION REQUIRES MAJOR REWRITE**

### Summary

Successfully aligned:
- ✅ SRS loading with arkworks flag bit handling
- ✅ G1 MSM results matching Jolt exactly
- ✅ G2 generator matching arkworks exactly
- ✅ ATE_LOOP_COUNT matching arkworks (fixed)
- ✅ Miller loop squaring skip on first iteration (fixed)

Still different:
- ❌ Pairing produces different GT element than arkworks

### Root Cause Analysis

The pairing mismatch is caused by fundamental algorithmic differences:

1. **Line Coefficient Format**:
   - Arkworks uses 3 coefficients (c0, c1, c2) from projective formulas
   - Zolt uses 2 coefficients (r0=λ, r1=λ·x-y) from affine formulas
   - The formulas are completely different!

2. **Coordinate System**:
   - Arkworks: Homogeneous projective coordinates (x, y, z)
   - Zolt: Affine coordinates (x, y)
   - Affects all intermediate calculations

3. **Line Evaluation**:
   - Arkworks: `mul_by_034(c0*y_P, c1*x_P, c2)` sparse multiplication
   - Zolt: Full Fp12 multiplication with different coefficients

4. **Precomputation**:
   - Arkworks: Precomputes all line coefficients in G2Prepared
   - Zolt: Computes on-the-fly

---

## Options Forward

### Option A: Full Pairing Rewrite (Effort: High)
Completely rewrite the pairing to match arkworks:
1. Add G2HomProjective struct
2. Implement projective doubling/addition with 3 coefficients
3. Add mul_by_034 and mul_by_01 sparse multiplications
4. Create G2Prepared for precomputation
5. Match final exponentiation algorithm

### Option B: FFI to arkworks (Effort: Medium)
Use Rust FFI to call arkworks pairing:
1. Create small Rust library with pairing function
2. Export C ABI function
3. Call from Zig via extern
4. Only affects pairing, rest of Zolt stays native

### Option C: Focus on Higher Priority Items (Effort: Low)
The commitment scheme ultimately uses:
- MSM (already matching)
- Pairing for verification (could be done on Jolt side)

If Zolt generates proofs and Jolt verifies, the pairing mismatch only matters
if Zolt needs to verify its own proofs. Could defer pairing fix.

---

## Major Milestones

### Completed ✅
1. Blake2b Transcript
2. JoltProof 7-stage structure
3. Arkworks serialization
4. GT element serialization
5. Cross-deserialization
6. UniSkip infrastructure
7. 48 opening claims
8. 19 R1CS constraints
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
20. ATE_LOOP_COUNT from arkworks (fixed)

### In Progress ⏳
21. Pairing implementation matching arkworks (blocked - needs major rewrite)

---

## Test Status

### Zolt: All tests passing

### Jolt Cross-Verification

| Test | Status | Details |
|------|--------|---------|
| `test_deserialize_zolt_proof` | ✅ PASS | Deserializes correctly |
| `test_debug_zolt_format` | ✅ PASS | All claims valid |
| `test_export_dory_srs` | ✅ PASS | SRS exported (max_num_vars=3) |
| `test_export_dory_commitment_debug` | ✅ PASS | MSM matches, pairing differs |
| `test_verify_zolt_proof` | ❌ FAIL | Commitment mismatch (pairing) |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/poly/commitment/dory.zig` | Dory commitment (working) |
| `src/field/pairing.zig` | BN254 pairing (needs rewrite) |
| `src/field/mod.zig` | Montgomery field arithmetic |
| `src/msm/mod.zig` | Multi-scalar multiplication |

---

## Summary

**Serialization: COMPLETE**
**Transcript: COMPLETE**
**SRS Loading: COMPLETE**
**G1 MSM: MATCHING**
**G2 Points: MATCHING**
**Pairing: NEEDS MAJOR REWRITE (projective coords + 3-coeff line function)**
