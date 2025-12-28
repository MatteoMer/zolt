# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: JOLT COMPATIBILITY - Dory IPA Proof Structure Complete**

The following Jolt-compatibility components are now working:
- Blake2b transcript with identical Fiat-Shamir challenges
- Jolt-compatible proof types (JoltProof, SumcheckInstanceProof, etc.)
- Arkworks-compatible serialization with byte-perfect output
- Proof converter (6-stage Zolt → 7-stage Jolt)
- JoltProver with `proveJoltCompatible()` method
- Jolt proof serialization with `serializeJoltProof()`
- Dory commitment scheme with GT (Fp12) serialization
- 384-byte GT element serialization matching arkworks format
- **NEW: Full Dory IPA proof structure (VMVMessage, FirstReduceMessage, SecondReduceMessage, ScalarProductMessage)**
- **NEW: G1/G2 point compression in arkworks format (32/64 bytes)**
- **NEW: Tonelli-Shanks sqrt for Fp and Fp2**
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
  - [x] GT elements as 384 bytes (12 Fp elements)
  - [x] Dory commitment serialization

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
  - [x] Commitment serialization (384 bytes)

- [x] **Dory IPA Proof Structure** (`src/poly/commitment/dory.zig`)
  - [x] VMVMessage (c: GT, d2: GT, e1: G1)
  - [x] FirstReduceMessage (d1_left, d1_right, d2_left, d2_right: GT, e1_beta: G1, e2_beta: G2)
  - [x] SecondReduceMessage (c_plus, c_minus: GT, e1_plus, e1_minus: G1, e2_plus, e2_minus: G2)
  - [x] ScalarProductMessage (e1: G1, e2: G2)
  - [x] DoryProof with all messages + nu/sigma
  - [x] Serialization matching dory-pcs ark_serde.rs format

- [x] **Point Compression** (`src/poly/commitment/dory.zig`)
  - [x] G1 compression to 32 bytes (arkworks format)
  - [x] G1 decompression with Tonelli-Shanks sqrt
  - [x] G2 compression to 64 bytes (arkworks format)
  - [x] Fp2 sqrt for G2 decompression

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
  - [x] Dory commitment serialization tests
  - [x] GT roundtrip tests

- [x] **Dory Proof Serialization Tests** (`src/poly/commitment/dory.zig`)
  - [x] G1 compression/decompression roundtrip
  - [x] G1 identity compression
  - [x] G2 compression format verification
  - [x] G2 identity compression/decompression
  - [x] VMV message serialization
  - [x] Full proof serialization size check

### 7. Remaining Work (Future Phase)

- [ ] **Full Dory IPA Proving Algorithm**
  - [ ] Implement reduce-and-fold rounds
  - [ ] Compute proper VMV message values
  - [ ] Compute FirstReduceMessage values per round
  - [ ] Compute SecondReduceMessage values per round
  - [ ] Compute ScalarProductMessage final values

- [ ] **Cross-Verification Test** (Requires Jolt-side changes)
  - [ ] Create Rust test in Jolt that loads Zolt proof file
  - [ ] Generate proof in Zolt for fibonacci
  - [ ] Verify proof loads and parses correctly in Jolt
  - [ ] Attempt verification

---

## Test Status

### All 608 Tests Passing

```
zig build test --summary all
Build Summary: 5/5 steps succeeded; 608/608 tests passed
```

### New Tests Added (This Iteration)

| Component | Tests |
|-----------|-------|
| G1 compression roundtrip | Pass |
| G1 identity compression | Pass |
| G2 compression format | Pass |
| G2 identity compression | Pass |
| Dory proof serialization | Pass |
| VMV message serialization | Pass |

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
| `src/poly/commitment/dory.zig` | ✅ **Updated** | Full Dory IPA structure + G1/G2 compression |
| `src/field/pairing.zig` | ✅ Done | GT (Fp12) serialization |

### Jolt (Reference Only)
| File | Purpose |
|------|---------|
| `jolt-core/src/transcripts/blake2b.rs` | ✅ Verified |
| `jolt-core/src/zkvm/proof_serialization.rs` | ✅ Analyzed |
| `jolt-core/src/poly/commitment/dory/` | ✅ Analyzed |
| `dory-pcs/src/proof.rs` | ✅ Analyzed - DoryProof structure |
| `dory-pcs/src/messages.rs` | ✅ Analyzed - Message types |
| `dory-pcs/src/backends/arkworks/ark_serde.rs` | ✅ Analyzed - Serialization |

---

## Success Criteria

1. ✅ `zig build test` passes all 608 tests
2. ✅ Zolt can generate a proof in Jolt format (`proveJoltCompatible`)
3. ✅ Zolt can serialize proofs in arkworks format (`serializeJoltProof`)
4. ✅ E2E serialization tests verify format compatibility
5. ✅ Dory commitment scheme with GT serialization
6. ✅ Dory IPA proof structure with G1/G2 compression
7. ⏳ The proof can be loaded and verified by Jolt (requires full IPA + Jolt test)
8. ⏳ No modifications needed on the Jolt side (requires Dory alignment)

## Priority Order

1. ✅ **Transcript** - Matching Fiat-Shamir complete
2. ✅ **Proof Types** - JoltProof structure defined
3. ✅ **Serialization** - Byte-perfect compatibility verified
4. ✅ **Dory Commitment** - GT serialization complete
5. ✅ **Dory IPA Structure** - Proof structure matching Jolt
6. ✅ **G1/G2 Compression** - arkworks format
7. ✅ **Prover Wiring** - Connect types to prover
8. ✅ **Integration Tests** - E2E serialization verified
9. ⏳ **Full Dory IPA** - Required for verification
10. ⏳ **Cross-Verification** - Jolt-side test needed
