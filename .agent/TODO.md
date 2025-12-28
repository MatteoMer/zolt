# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: SERIALIZATION COMPLETE, VERIFICATION BLOCKED**

### Major Milestones Achieved

1. ✅ **Blake2b Transcript** - Identical Fiat-Shamir challenges as Jolt
2. ✅ **Proof Types** - JoltProof with 7-stage structure, OpeningClaims, etc.
3. ✅ **Arkworks Serialization** - Byte-perfect format compatibility
4. ✅ **Dory Commitment** - GT (Fp12) serialization, IPA proof structure
5. ✅ **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs!

### Blocking Issue: Univariate Skip Mismatch

Jolt's verifier uses "univariate skip" optimization for stages 1-2:
- Requires degree-27 polynomials (28 coefficients)
- Zolt produces degree-2/3 polynomials (3-4 coefficients)

This is a **fundamental architectural difference**, not a serialization issue.

---

## Phase 2: Jolt Compatibility

### 1. Transcript Alignment ✅ COMPLETE

- [x] **Create Blake2bTranscript** (`src/transcripts/blake2b.zig`)
- [x] **Test Vector Validation** - 7 test vectors verified

### 2. Proof Types ✅ COMPLETE

- [x] **JoltProof Structure** (`src/zkvm/jolt_types.zig`)
- [x] **CompressedUniPoly** - Proper compression (remove linear term)

### 3. Serialization ✅ COMPLETE

- [x] **Arkworks-Compatible Format** (`src/zkvm/jolt_serialization.zig`)
- [x] **writeDoryProof method** - Matches ark_serde.rs format exactly
- [x] **writeG1Compressed/writeG2Compressed** - Arkworks point compression
- [x] **writeU32** - For num_rounds, nu, sigma

### 4. Dory Commitment Scheme ✅ COMPLETE

- [x] **Dory Implementation** (`src/poly/commitment/dory.zig`)
- [x] **Full Dory IPA Prover** - reduce-and-fold algorithm
- [x] **Point Compression** - G1/G2 in arkworks format
- [x] **Proof Serialization** - Matches ArkDoryProof format

### 5. Prover Wiring ✅ COMPLETE

- [x] **Proof Converter** (`src/zkvm/proof_converter.zig`)
- [x] **JoltProver Integration** (`src/zkvm/mod.zig`)
- [x] **serializeJoltProofDory()** - Generates full Dory proofs

### 6. Cross-Verification ⚠️ BLOCKED

- [x] **Jolt deserializes Zolt proofs** - VERIFIED WORKING
- [x] **Config parameters fixed** - log_k_chunk <= 8
- [x] **CompressedUniPoly fixed** - Proper linear term removal
- [ ] **Jolt verifies Zolt proofs** - BLOCKED by univariate skip mismatch

---

## What's Required for Full Verification

To make Zolt proofs fully verifiable by Jolt, Zolt would need to:

1. **Implement Jolt's R1CS Constraints** (19 constraints)
   - Match `R1CSConstraintLabel` enum exactly
   - Use same variable ordering

2. **Implement Univariate Skip Optimization**
   - `build_uniskip_first_round_poly()`
   - Extended domain evaluation
   - Degree-27 first-round polynomials

3. **Match Stage Structure**
   - Stage 1: SpartanOuter (univariate skip)
   - Stage 2: Product virtualization (univariate skip)
   - Stages 3-7: Standard sumcheck

---

## Test Status

### All 608 Tests Passing

```
zig build test --summary all
Build Summary: 5/5 steps succeeded; 608/608 tests passed
```

### Cross-Verification Tests

| Test | Status |
|------|--------|
| `test_deserialize_zolt_proof` | ✅ PASS |
| `test_debug_zolt_format` | ✅ PASS |
| `test_verify_zolt_proof` | ❌ BLOCKED - UniSkip mismatch |

---

## Key Files

### Zolt (Implemented)
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript |
| `src/zkvm/jolt_types.zig` | ✅ Done | Jolt proof types |
| `src/zkvm/jolt_serialization.zig` | ✅ Done | Arkworks serialization |
| `src/zkvm/proof_converter.zig` | ✅ Done | 6→7 stage converter |
| `src/zkvm/mod.zig` | ✅ Done | JoltProver |
| `src/poly/commitment/dory.zig` | ✅ Done | Dory IPA |

---

## Summary

**Serialization Goal: ACHIEVED**
- Zolt produces proofs that Jolt can deserialize
- Byte-perfect arkworks format compatibility
- Dory commitment scheme with GT serialization

**Verification Goal: BLOCKED**
- Requires adopting Jolt's R1CS constraint structure
- Requires implementing univariate skip optimization
- This is architectural, not just format alignment
