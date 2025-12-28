# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: JOLT COMPATIBILITY - Dory Commitment Complete**

The following Jolt-compatibility components are now working:
- Blake2b transcript with identical Fiat-Shamir challenges
- Jolt-compatible proof types (JoltProof, SumcheckInstanceProof, etc.)
- Arkworks-compatible serialization with byte-perfect output
- Proof converter (6-stage Zolt → 7-stage Jolt)
- JoltProver with `proveJoltCompatible()` method
- Jolt proof serialization with `serializeJoltProof()`
- **NEW: Dory commitment scheme with GT (Fp12) serialization**
- **NEW: 384-byte GT element serialization matching arkworks format**
- End-to-end serialization tests verifying format compatibility

---

## Phase 2: Jolt Compatibility

### 1. Transcript Alignment ✅ COMPLETE

- [x] **Create Blake2bTranscript** (`src/transcripts/blake2b.zig`)
  - [x] Blake2b-256 using Zig stdlib
  - [x] 32-byte state with round counter
  - [x] Message padding to 32 bytes
  - [x] Scalar serialization (LE then reverse to BE)
  - [x] Vector append with begin/end markers
  - [x] 128-bit challenges

- [x] **Test Vector Validation** - 7 test vectors verified

### 2. Proof Types ✅ COMPLETE

- [x] **JoltProof Structure** (`src/zkvm/jolt_types.zig`)
  - [x] SumcheckId enum (22 variants matching Jolt)
  - [x] CommittedPolynomial and VirtualPolynomial enums
  - [x] OpeningId with compact encoding
  - [x] CompressedUniPoly for round polynomials
  - [x] SumcheckInstanceProof with compressed_polys
  - [x] UniSkipFirstRoundProof for stages 1-2
  - [x] OpeningClaims as sorted map
  - [x] JoltProof with 7 explicit stages

### 3. Serialization ✅ COMPLETE

- [x] **Arkworks-Compatible Format** (`src/zkvm/jolt_serialization.zig`)
  - [x] Field elements as 32 bytes LE (from Montgomery form)
  - [x] usize as u64 little-endian
  - [x] OpeningId compact encoding
  - [x] Roundtrip deserialization
  - [x] **NEW: GT elements as 384 bytes (12 Fp elements)**
  - [x] **NEW: Dory commitment serialization**

- [x] **Test Vectors Verified**
  - [x] Fr(42) → `[2a, 00, ...]`
  - [x] Fr(0) → `[00, 00, ...]`
  - [x] Fr(1) → `[01, 00, ...]`
  - [x] Fr(0xDEADBEEF) → `[ef, be, ad, de, ...]`
  - [x] usize(1234567890) → `[d2, 02, 96, 49, ...]`
  - [x] GT.one() → `[01, 00, ...] + 352 zeros`

### 4. Dory Commitment Scheme ✅ COMPLETE

- [x] **Dory Implementation** (`src/poly/commitment/dory.zig`)
  - [x] DoryCommitmentScheme with setup/commit
  - [x] SRS generation using "Jolt Dory URS seed"
  - [x] GT (Fp12) commitment type
  - [x] DoryProof structure
  - [x] Commitment serialization (384 bytes)

- [x] **GT Serialization** (`src/field/pairing.zig`)
  - [x] Fp12.toBytes() - 384 bytes arkworks format
  - [x] Fp12.fromBytes() - deserialization
  - [x] GT alias for Fp12

### 5. Prover Wiring ✅ COMPLETE

- [x] **Proof Converter** (`src/zkvm/proof_converter.zig`)
  - [x] Map Zolt 6-stage to Jolt 7-stage format
  - [x] Create UniSkipFirstRoundProof for stages 1-2
  - [x] Populate OpeningClaims with SumcheckId mappings
  - [x] Configuration for bytecode_K, log_k_chunk, etc.

- [x] **JoltProver Integration** (`src/zkvm/mod.zig`)
  - [x] `proveJoltCompatible()` - Generate Jolt-compatible proof
  - [x] `serializeJoltProof()` - Serialize to arkworks format
  - [x] Wire up proof converter with prover

### 6. Integration Testing ✅ COMPLETE

- [x] **End-to-End Serialization Tests** (`src/zkvm/jolt_serialization.zig`)
  - [x] Test JoltProof with all 7 stages
  - [x] Verify opening claims serialization
  - [x] Verify commitments serialization
  - [x] Test empty proof serialization
  - [x] Validate config parameters serialization
  - [x] **NEW: Dory commitment serialization tests**
  - [x] **NEW: GT roundtrip tests**

### 7. Remaining Work (Future Phase)

- [ ] **Full Dory Opening Proof**
  - [ ] Implement inner product argument (IPA)
  - [ ] Match Jolt's ArkDoryProof structure exactly

- [ ] **Cross-Verification Test** (Requires Jolt-side changes)
  - [ ] Create Rust test in Jolt that loads Zolt proof file
  - [ ] Generate proof in Zolt for fibonacci
  - [ ] Verify proof loads and parses correctly in Jolt
  - [ ] Attempt verification

---

## Test Status

### All 596 Tests Passing

```
zig build test --summary all
Build Summary: 5/5 steps succeeded; 596/596 tests passed
```

### New Tests Added

| Component | Tests |
|-----------|-------|
| Fp12 toBytes/fromBytes | Pass |
| Fp12 format verification | Pass |
| GT alias | Pass |
| Dory setup | Pass |
| Dory commit | Pass |
| Dory deterministic | Pass |
| Dory serialization | Pass |
| Dory roundtrip | Pass |
| Jolt serialization GT | Pass |

---

## Key Reference Files

### Zolt (Modified)
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript |
| `src/zkvm/jolt_types.zig` | ✅ Done | Jolt proof types |
| `src/zkvm/jolt_serialization.zig` | ✅ Done | Arkworks serialization + Dory |
| `src/zkvm/proof_converter.zig` | ✅ Done | 6→7 stage converter |
| `src/zkvm/mod.zig` | ✅ Done | JoltProver with Jolt export |
| `src/poly/commitment/dory.zig` | ✅ **NEW** | Dory commitment scheme |
| `src/field/pairing.zig` | ✅ Updated | GT (Fp12) serialization |

### Jolt (Reference Only)
| File | Purpose |
|------|---------|
| `jolt-core/src/transcripts/blake2b.rs` | ✅ Verified |
| `jolt-core/src/zkvm/proof_serialization.rs` | ✅ Analyzed |
| `jolt-core/src/poly/commitment/dory/` | ✅ Analyzed |
| `jolt-core/src/subprotocols/sumcheck.rs` | Reference |
| `jolt-core/src/poly/opening_proof.rs` | Reference |

---

## Success Criteria

1. ✅ `zig build test` passes all 596 tests
2. ✅ Zolt can generate a proof in Jolt format (`proveJoltCompatible`)
3. ✅ Zolt can serialize proofs in arkworks format (`serializeJoltProof`)
4. ✅ E2E serialization tests verify format compatibility
5. ✅ Dory commitment scheme with GT serialization
6. ⏳ The proof can be loaded and verified by Jolt (requires full IPA + Jolt test)
7. ⏳ No modifications needed on the Jolt side (requires Dory alignment)

## Priority Order

1. ✅ **Transcript** - Matching Fiat-Shamir complete
2. ✅ **Proof Types** - JoltProof structure defined
3. ✅ **Serialization** - Byte-perfect compatibility verified
4. ✅ **Dory Commitment** - GT serialization complete
5. ✅ **Prover Wiring** - Connect types to prover
6. ✅ **Integration Tests** - E2E serialization verified
7. ⏳ **Full Dory IPA** - Required for verification
8. ⏳ **Cross-Verification** - Jolt-side test needed
