# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: OPENING CLAIMS COMPLETE - SUMCHECK VALUES NEXT**

### Major Milestones Achieved

1. ✅ **Blake2b Transcript** - Identical Fiat-Shamir challenges as Jolt
2. ✅ **Proof Types** - JoltProof with 7-stage structure, OpeningClaims, etc.
3. ✅ **Arkworks Serialization** - Byte-perfect format compatibility
4. ✅ **Dory Commitment** - GT (Fp12) serialization, IPA proof structure
5. ✅ **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs!
6. ✅ **Univariate Skip Infrastructure** - Degree-27/12 polynomials for stages 1-2
7. ✅ **All 48 Opening Claims** - Including all 36 R1CS inputs for SpartanOuter + 13 OpFlags variants
8. ✅ **VirtualPolynomial Ordering** - Fixed payload comparison for OpFlags, InstructionFlags, etc.

### Latest Verification Test Result

```
$ cargo test -p jolt-core test_verify_zolt_proof -- --ignored --nocapture

Loaded all files, attempting verification...
  Proof trace length: 8
  Proof commitments: 5
  Opening claims count: 48

Opening claims include all 13 OpFlags variants (AddOperands through IsFirstInSequence)

Verification failed: Stage 1
Caused by: Sumcheck verification failed
```

### Analysis

The proof structure is now fully correct:
- ✅ 48 opening claims including all R1CS inputs and OpFlags variants
- ✅ UniSkip polynomials have correct degrees (28 and 13 coefficients)
- ✅ UniSkip first-round check passes (sum over domain = 0)
- ❌ Stage 1 sumcheck verification fails - claims don't match expected values

### What's Needed for Full Verification

The sumcheck verification fails because our "zero proofs" don't satisfy the actual
sumcheck equation. The verifier computes expected claims from R1CS constraint
evaluations, which don't match our zeros.

To fix this, we would need to:
1. Implement Jolt's exact R1CS constraint structure in Zolt
2. Compute actual Az(x,y) · Bz(x,y) evaluations during proving
3. Generate proper univariate skip polynomials from constraint evaluations
4. Ensure sumcheck round polynomials satisfy p(0) + p(1) = claim for actual claims

---

## Phase 2: Jolt Compatibility

### 1. Transcript Alignment ✅ COMPLETE

- [x] **Create Blake2bTranscript** (`src/transcripts/blake2b.zig`)
- [x] **Test Vector Validation** - 7 test vectors verified

### 2. Proof Types ✅ COMPLETE

- [x] **JoltProof Structure** (`src/zkvm/jolt_types.zig`)
- [x] **CompressedUniPoly** - Proper compression (remove linear term)
- [x] **UniSkipFirstRoundProof** - High-degree first-round polynomials
- [x] **OpeningClaims** - With proper VirtualPolynomial ordering

### 3. Serialization ✅ COMPLETE

- [x] **Arkworks-Compatible Format** (`src/zkvm/jolt_serialization.zig`)
- [x] **writeDoryProof method** - Matches ark_serde.rs format exactly
- [x] **writeG1Compressed/writeG2Compressed** - Arkworks point compression
- [x] **writeUniSkipFirstRoundProof** - Serializes all coefficients
- [x] **Opening claims serialization** - BTreeMap-compatible order

### 4. Dory Commitment Scheme ✅ COMPLETE

- [x] **Dory Implementation** (`src/poly/commitment/dory.zig`)
- [x] **Full Dory IPA Prover** - reduce-and-fold algorithm
- [x] **Point Compression** - G1/G2 in arkworks format
- [x] **Proof Serialization** - Matches ArkDoryProof format

### 5. Opening Claims ✅ COMPLETE

- [x] **All 36 R1CS inputs for SpartanOuter** - Including all 13 OpFlags variants
- [x] **VirtualPolynomial.orderByPayload** - Correct ordering for tagged unions
- [x] **Additional stage claims** - RamRa, RamVal, RegistersVal, etc.
- [x] **48 total opening claims** - Verified in Jolt deserialization

### 6. Cross-Verification 🔄 PARTIAL

- [x] **Jolt deserializes Zolt proofs** - VERIFIED WORKING
- [x] **48 opening claims** - All R1CS inputs + OpFlags + stage claims
- [x] **UniSkip first-round check** - Passes (sum = 0 for zero polynomial)
- [ ] **Stage 1 sumcheck** - Fails (claims don't match expected values)
- [ ] **Full verification** - Blocked on sumcheck

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
| `test_deserialize_zolt_proof` | ✅ PASS | 27910 bytes, 48 claims |
| `test_debug_zolt_format` | ✅ PASS | All claims and commitments valid |
| `test_verify_zolt_proof` | ❌ FAIL | Stage 1 sumcheck verification |

---

## Key Files

### Zolt (Implemented)
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript |
| `src/zkvm/jolt_types.zig` | ✅ Done | Jolt proof types with VirtualPolynomial ordering |
| `src/zkvm/jolt_serialization.zig` | ✅ Done | Arkworks serialization |
| `src/zkvm/proof_converter.zig` | ✅ Done | 6→7 stage converter with all claims |
| `src/zkvm/mod.zig` | ✅ Done | JoltProver |
| `src/poly/commitment/dory.zig` | ✅ Done | Dory IPA |
| `src/zkvm/r1cs/univariate_skip.zig` | ✅ Done | Univariate skip constants |
| `src/zkvm/spartan/outer.zig` | 🔄 Needs work | Actual constraint evaluation |

---

## Summary

**Serialization Goal: ACHIEVED**
- Zolt produces proofs that Jolt can deserialize
- Byte-perfect arkworks format compatibility
- Dory commitment scheme with GT serialization
- All 48 opening claims preserved

**Next Steps:**
1. Implement actual R1CS constraint evaluation
2. Generate sumcheck round polynomials that satisfy verification
3. Compute proper univariate skip polynomials from constraint values
