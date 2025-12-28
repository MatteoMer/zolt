# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: STRUCTURE VERIFIED - POLYNOMIAL VALUES NEXT**

### Major Milestones Achieved

1. ✅ **Blake2b Transcript** - Identical Fiat-Shamir challenges as Jolt
2. ✅ **Proof Types** - JoltProof with 7-stage structure, OpeningClaims, etc.
3. ✅ **Arkworks Serialization** - Byte-perfect format compatibility
4. ✅ **Dory Commitment** - GT (Fp12) serialization, IPA proof structure
5. ✅ **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs!
6. ✅ **Univariate Skip Infrastructure** - Degree-27/12 polynomials for stages 1-2
7. ✅ **UnivariateSkip Opening Claims** - Required claims now included

### Latest Verification Test Result

```
$ cargo test -p jolt-core test_verify_zolt_proof -- --ignored --nocapture

Loaded all files, attempting verification...
  Proof trace length: 8
  Proof commitments: 5
Verifier created successfully, running verification...

Verification failed: Stage 1 univariate skip first round
Caused by: UniSkip first-round verification failed
```

### Analysis

The proof structure is now fully correct:
- ✅ Opening claims include UnivariateSkip for SpartanOuter and SpartanProductVirtualization
- ✅ Polynomials have correct degrees (28 and 13 coefficients)
- ❌ Polynomial *values* are placeholders (padded zeros), not actual constraint evaluations

### What's Needed for Full Verification

The univariate skip first-round polynomial requires computing:

```
s1(Y) = L(τ_high, Y) · t1(Y)
```

Where `t1(Y)` is computed by evaluating Az(x,y) · Bz(x,y) at extended domain points.

Currently, Zolt's proof converter pads the original sumcheck polynomial coefficients to
the required length, but doesn't compute the actual constraint values at extended points.

To fix this:
1. Port Jolt's constraint evaluation logic (Az, Bz for all 19 constraints)
2. Evaluate constraints at extended domain points {-9, ..., 9}
3. Use `buildUniskipFirstRoundPoly()` with actual evaluations

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

### 5. Univariate Skip Optimization ✅ STRUCTURE COMPLETE

- [x] **Constants** - Match Jolt's R1CS constraint structure
- [x] **buildUniskipFirstRoundPoly()** - Polynomial construction
- [x] **LagrangePolynomial utilities** - Interpolation on extended domain
- [x] **Proof converter integration** - Generates proper-degree polynomials
- [x] **Stage 1** - Degree-27 polynomial (28 coefficients)
- [x] **Stage 2** - Degree-12 polynomial (13 coefficients)
- [x] **Opening claims** - UnivariateSkip claims for both stages
- [ ] **Extended evaluation** - Need actual Az·Bz evaluations at extended points

### 6. Cross-Verification 🔄 PARTIAL

- [x] **Jolt deserializes Zolt proofs** - VERIFIED WORKING
- [x] **Config parameters correct** - trace=8, RAM_K=65536, Bytecode_K=65536
- [x] **5 Dory commitments** - All valid GT elements
- [x] **13 opening claims** - All valid Fr elements (including UnivariateSkip)
- [x] **UniSkip polynomial degrees** - 28/13 coefficients
- [ ] **Full verification** - Fails at univariate skip polynomial check

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
| `test_deserialize_zolt_proof` | ✅ PASS | 26067 bytes, all fields valid |
| `test_debug_zolt_format` | ✅ PASS | All claims and commitments valid |
| `test_verify_zolt_proof` | ❌ FAIL | UniSkip first-round verification |

---

## Key Files

### Zolt (Implemented)
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript |
| `src/zkvm/jolt_types.zig` | ✅ Done | Jolt proof types |
| `src/zkvm/jolt_serialization.zig` | ✅ Done | Arkworks serialization |
| `src/zkvm/proof_converter.zig` | ✅ Done | 6→7 stage converter with UniSkip claims |
| `src/zkvm/mod.zig` | ✅ Done | JoltProver |
| `src/poly/commitment/dory.zig` | ✅ Done | Dory IPA |
| `src/zkvm/r1cs/univariate_skip.zig` | ✅ Done | Univariate skip optimization |
| `src/zkvm/spartan/outer.zig` | 🔄 In Progress | Needs extended evaluation |

---

## Summary

**Serialization Goal: ACHIEVED**
- Zolt produces proofs that Jolt can deserialize
- Byte-perfect arkworks format compatibility
- Dory commitment scheme with GT serialization
- All opening claims and commitments valid

**Univariate Skip Goal: STRUCTURE COMPLETE, VALUES PENDING**
- Proper polynomial degrees (27 for stage 1, 12 for stage 2)
- Opening claims correctly included
- Need to compute actual constraint evaluations at extended points

**Next Steps:**
1. Implement extended domain constraint evaluation (Az·Bz at {-9,...,9})
2. Use actual evaluations in `buildUniskipFirstRoundPoly()`
3. Re-run verification test
