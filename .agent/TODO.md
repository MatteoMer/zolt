# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: UNIVARIATE SKIP IMPLEMENTED - TESTING PHASE**

### Major Milestones Achieved

1. ✅ **Blake2b Transcript** - Identical Fiat-Shamir challenges as Jolt
2. ✅ **Proof Types** - JoltProof with 7-stage structure, OpeningClaims, etc.
3. ✅ **Arkworks Serialization** - Byte-perfect format compatibility
4. ✅ **Dory Commitment** - GT (Fp12) serialization, IPA proof structure
5. ✅ **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs!
6. ✅ **Univariate Skip Infrastructure** - Degree-27/12 polynomials for stages 1-2

### Recently Completed (This Session)

- ✅ **univariate_skip.zig** - Core univariate skip optimization module
  - Constants matching Jolt (NUM_R1CS_CONSTRAINTS=19, DEGREE=9, NUM_COEFFS=28)
  - `buildUniskipFirstRoundPoly()` for degree-27 first-round polynomial
  - Lagrange polynomial interpolation utilities
  - `uniskipTargets()` for extended evaluation points

- ✅ **spartan/outer.zig** - Spartan outer prover with univariate skip
  - `SpartanOuterProver` type with proper constants
  - `computeUniskipFirstRoundPoly()` generating 28-coefficient polynomial
  - Extended evaluation computation framework

- ✅ **proof_converter.zig updates** - Stage-specific univariate skip proofs
  - `createUniSkipProofStage1()` - degree-27 polynomial (28 coefficients)
  - `createUniSkipProofStage2()` - degree-12 polynomial (13 coefficients)

---

## Phase 2: Jolt Compatibility

### 1. Transcript Alignment ✅ COMPLETE

- [x] **Create Blake2bTranscript** (`src/transcripts/blake2b.zig`)
- [x] **Test Vector Validation** - 7 test vectors verified

### 2. Proof Types ✅ COMPLETE

- [x] **JoltProof Structure** (`src/zkvm/jolt_types.zig`)
- [x] **CompressedUniPoly** - Proper compression (remove linear term)
- [x] **UniSkipFirstRoundProof** - High-degree first-round polynomials

### 3. Serialization ✅ COMPLETE

- [x] **Arkworks-Compatible Format** (`src/zkvm/jolt_serialization.zig`)
- [x] **writeDoryProof method** - Matches ark_serde.rs format exactly
- [x] **writeG1Compressed/writeG2Compressed** - Arkworks point compression
- [x] **writeUniSkipFirstRoundProof** - Serializes all coefficients

### 4. Dory Commitment Scheme ✅ COMPLETE

- [x] **Dory Implementation** (`src/poly/commitment/dory.zig`)
- [x] **Full Dory IPA Prover** - reduce-and-fold algorithm
- [x] **Point Compression** - G1/G2 in arkworks format
- [x] **Proof Serialization** - Matches ArkDoryProof format

### 5. Univariate Skip Optimization ✅ INFRASTRUCTURE COMPLETE

- [x] **Constants** - Match Jolt's R1CS constraint structure
- [x] **buildUniskipFirstRoundPoly()** - Polynomial construction
- [x] **LagrangePolynomial utilities** - Interpolation on extended domain
- [x] **Proof converter integration** - Generates proper-degree polynomials
- [ ] **Extended Az*Bz evaluation** - Need full constraint evaluation at extended points

### 6. Cross-Verification 🔄 IN PROGRESS

- [x] **Jolt deserializes Zolt proofs** - VERIFIED WORKING
- [x] **Config parameters fixed** - log_k_chunk <= 8
- [x] **CompressedUniPoly fixed** - Proper linear term removal
- [x] **UniSkip polynomial degrees** - Now producing 28/13 coefficients
- [ ] **Verify polynomial values** - Need to verify constraint evaluation matches Jolt

---

## What's Required for Full Verification

Zolt now produces proofs with the correct structure (degree-27 and degree-12
first-round polynomials). For full verification:

1. **Constraint Evaluation at Extended Points**
   - The univariate skip polynomial values must match Jolt's computation
   - This requires evaluating Az(x,y)*Bz(x,y) at extended domain points
   - Currently using simplified interpolation - may need Jolt's exact algorithm

2. **Verify Polynomial Sum Invariants**
   - s1(Y) summed over symmetric domain should equal initial claim
   - Verifier checks `check_sum_evals::<N, 28>()` on stage 1 polynomial

3. **Test Against Jolt Verifier**
   - Run `cargo test test_verify_zolt_proof` in Jolt
   - Debug any remaining mismatches

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
| `test_verify_zolt_proof` | 🔄 Needs Testing |

---

## Key Files

### Zolt (Implemented)
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript |
| `src/zkvm/jolt_types.zig` | ✅ Done | Jolt proof types |
| `src/zkvm/jolt_serialization.zig` | ✅ Done | Arkworks serialization |
| `src/zkvm/proof_converter.zig` | ✅ Done | 6→7 stage converter with UniSkip |
| `src/zkvm/mod.zig` | ✅ Done | JoltProver |
| `src/poly/commitment/dory.zig` | ✅ Done | Dory IPA |
| `src/zkvm/r1cs/univariate_skip.zig` | ✅ Done | Univariate skip optimization |
| `src/zkvm/spartan/outer.zig` | ✅ Done | Spartan outer prover |

---

## Summary

**Serialization Goal: ACHIEVED**
- Zolt produces proofs that Jolt can deserialize
- Byte-perfect arkworks format compatibility
- Dory commitment scheme with GT serialization

**Univariate Skip Goal: STRUCTURE COMPLETE**
- Proper polynomial degrees (27 for stage 1, 12 for stage 2)
- Infrastructure for extended domain evaluation
- Need to verify polynomial values match Jolt's computation

**Next Steps:**
1. Test with Jolt verifier to see if polynomial values are accepted
2. If not, debug the extended evaluation computation
3. May need to port Jolt's exact constraint evaluation algorithm
