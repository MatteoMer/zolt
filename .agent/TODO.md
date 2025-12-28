# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: PAIRING IMPLEMENTATION IN PROGRESS**

### Summary

Successfully implemented:
- ✅ SRS loading with arkworks flag bit handling
- ✅ G1 MSM results matching Jolt exactly
- ✅ G2 generator matching arkworks exactly
- ✅ ATE_LOOP_COUNT matching arkworks (65 elements)
- ✅ Miller loop structure (skip first squaring)
- ✅ G2HomProjective struct (projective coordinates)
- ✅ EllCoeff (3-coefficient line format)
- ✅ Projective double_in_place
- ✅ Projective add_in_place
- ✅ fp6MulBy01 sparse multiplication
- ✅ fp12MulBy034 sparse multiplication
- ✅ mulByChar (Frobenius on G2)
- ✅ millerLoopArkworks using projective algorithm

Still different:
- ❌ Pairing output: `f5a1...` instead of expected `950e...`

### Debugging Notes

The pairing algorithm now follows arkworks structure:
1. Uses G2HomProjective with (x, y, z) coordinates
2. Returns 3 line coefficients from double/add steps
3. Evaluates lines via c0*y_P, c1*x_P, c2
4. Uses sparse multiplication fp12MulBy034
5. Adds Frobenius steps at end

The output differs from arkworks. Possible issues:
1. Frobenius coefficients (twistMulByQX, twistMulByQY)
2. COEFF_B value or its usage
3. Final exponentiation differences
4. Subtle tower construction differences

---

## Options Forward

### Option A: Continue Debugging (Current)
Add step-by-step debug output to find divergence point.

### Option B: Component Testing
Create isolated tests for each sparse multiplication function
with known inputs/outputs from arkworks.

### Option C: FFI to arkworks
Call arkworks pairing from Rust as fallback.

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
20. ATE_LOOP_COUNT from arkworks
21. Projective Miller loop implementation
22. Sparse multiplication functions

### In Progress ⏳
23. Pairing matching arkworks (debugging output mismatch)

---

## Test Status

### Zolt: Core tests passing

### Jolt Cross-Verification

| Test | Status | Details |
|------|--------|---------|
| `test_deserialize_zolt_proof` | ✅ PASS | Deserializes correctly |
| `test_debug_zolt_format` | ✅ PASS | All claims valid |
| `test_export_dory_srs` | ✅ PASS | SRS exported |
| `test_export_dory_commitment_debug` | ✅ PASS | MSM matches |
| `test_verify_zolt_proof` | ❌ FAIL | Pairing mismatch |

---

## Key Files Modified

| File | Changes |
|------|---------|
| `src/field/pairing.zig` | Added G2HomProjective, EllCoeff, millerLoopArkworks, fp6MulBy01, fp12MulBy034, mulByChar, twistB |

---

## Summary

**Serialization: COMPLETE**
**Transcript: COMPLETE**
**SRS Loading: COMPLETE**
**G1 MSM: MATCHING**
**G2 Points: MATCHING**
**Pairing: DEBUGGING** (arkworks algorithm implemented, output differs)
