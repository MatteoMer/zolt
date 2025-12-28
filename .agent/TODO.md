# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: CROSS-DESERIALIZATION VERIFIED - FULL VERIFICATION NEXT**

### Major Milestones Achieved

1. ✅ **Blake2b Transcript** - Identical Fiat-Shamir challenges as Jolt
2. ✅ **Proof Types** - JoltProof with 7-stage structure, OpeningClaims, etc.
3. ✅ **Arkworks Serialization** - Byte-perfect format compatibility
4. ✅ **Dory Commitment** - GT (Fp12) serialization, IPA proof structure
5. ✅ **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs!
6. ✅ **Univariate Skip Infrastructure** - Degree-27/12 polynomials for stages 1-2

### Latest Cross-Verification Test Results

```
$ cargo test -p jolt-core test_deserialize_zolt_proof -- --ignored --nocapture

Read 25999 bytes from Zolt proof
Successfully deserialized Zolt proof!
  Trace length: 8
  RAM K: 65536
  Bytecode K: 65536
  Commitments: 5
test zolt_compat_test::tests::test_deserialize_zolt_proof ... ok
```

And debug format test:
- ✅ 11 opening claims parsed with valid Fr elements
- ✅ 5 GT (Dory) commitments valid
- ✅ 23689 remaining bytes for sumcheck proofs

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

### 5. Univariate Skip Optimization ✅ COMPLETE

- [x] **Constants** - Match Jolt's R1CS constraint structure
- [x] **buildUniskipFirstRoundPoly()** - Polynomial construction
- [x] **LagrangePolynomial utilities** - Interpolation on extended domain
- [x] **Proof converter integration** - Generates proper-degree polynomials
- [x] **Stage 1** - Degree-27 polynomial (28 coefficients)
- [x] **Stage 2** - Degree-12 polynomial (13 coefficients)

### 6. Cross-Verification ✅ DESERIALIZATION VERIFIED

- [x] **Jolt deserializes Zolt proofs** - VERIFIED WORKING
- [x] **Config parameters correct** - trace=8, RAM_K=65536, Bytecode_K=65536
- [x] **5 Dory commitments** - All valid GT elements
- [x] **11 opening claims** - All valid Fr elements
- [x] **UniSkip polynomial degrees** - 28/13 coefficients
- [ ] **Full verification** - Next step: run full Jolt verifier

---

## What's Required for Full Verification

The serialization is complete and working. For full verification:

1. **Generate Jolt preprocessing** (verifier preprocessing data)
   ```bash
   cd /path/to/jolt
   cargo run --example fibonacci -- --save
   ```

2. **Generate IO device** (program I/O)
   - Need matching I/O between Zolt and Jolt execution

3. **Run full verification test**
   ```bash
   cargo test -p jolt-core test_verify_zolt_proof -- --ignored --nocapture
   ```

---

## Test Status

### All 608 Tests Passing (Zolt)

```
zig build test --summary all
Build Summary: 5/5 steps succeeded; 608/608 tests passed
```

### Cross-Verification Tests (Jolt)

| Test | Status |
|------|--------|
| `test_deserialize_zolt_proof` | ✅ PASS |
| `test_debug_zolt_format` | ✅ PASS |
| `test_verify_zolt_proof` | 🔄 Needs preprocessing files |

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
- All opening claims and commitments valid

**Univariate Skip Goal: COMPLETE**
- Proper polynomial degrees (27 for stage 1, 12 for stage 2)
- Infrastructure for extended domain evaluation
- Integrated into proof converter

**Next Steps:**
1. Generate Jolt preprocessing for verification
2. Run full verification test
3. Debug any sumcheck polynomial value mismatches
