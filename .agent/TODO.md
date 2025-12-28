# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: CLI INTEGRATION COMPLETE - Commitment Scheme Mismatch Identified**

The following Jolt-compatibility components are now working:
- Blake2b transcript with identical Fiat-Shamir challenges
- Jolt-compatible proof types (JoltProof, SumcheckInstanceProof, etc.)
- Arkworks-compatible serialization with byte-perfect output
- Proof converter (6-stage Zolt → 7-stage Jolt)
- JoltProver with `proveJoltCompatible()` method
- Jolt proof serialization with `serializeJoltProof()`
- Dory commitment scheme with GT (Fp12) serialization
- **COMPLETE: Full Dory IPA proving algorithm**
  - Multilinear Lagrange basis computation
  - L_vec/R_vec evaluation vectors
  - VMV message (C, D2, E1)
  - First reduce messages (D1L, D1R, D2L, D2R, E1_beta, E2_beta)
  - Second reduce messages (C+, C-, E1+, E1-, E2+, E2-)
  - Scalar product final message (E1, E2)
  - Transcript-integrated version with proper Fiat-Shamir challenges
- G1/G2 point compression in arkworks format
- Tonelli-Shanks sqrt for Fp and Fp2
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
  - [x] GT element appending for Dory
  - [x] G1/G2 compressed point appending

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

- [x] **Full Dory IPA Prover** (`src/poly/commitment/dory.zig`)
  - [x] `open()` - Basic version with deterministic challenges
  - [x] `openWithTranscript()` - Transcript-integrated version
  - [x] Multilinear Lagrange basis computation
  - [x] Evaluation vector computation (left_vec, right_vec)
  - [x] Vector-matrix product for v_vec
  - [x] VMV message computation
  - [x] Reduce-and-fold rounds with beta/alpha challenges
  - [x] Final scalar product message with gamma challenge

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

- [x] **Dory Proof Tests** (`src/poly/commitment/dory.zig`)
  - [x] G1 compression/decompression roundtrip
  - [x] G1 identity compression
  - [x] G2 compression format verification
  - [x] G2 identity compression/decompression
  - [x] VMV message serialization
  - [x] Full proof serialization size check

### 7. Cross-Verification Test Infrastructure ✅ COMPLETE

- [x] **Test Infrastructure in Jolt** (`jolt-core/src/zolt_compat_test.rs`)
  - [x] `test_deserialize_zolt_proof` - Test proof deserialization
  - [x] `test_verify_zolt_proof` - Full verification test
  - [x] `test_jolt_proof_roundtrip` - Basic serialization roundtrip

### 8. CLI Integration ✅ COMPLETE

- [x] **CLI --jolt-format flag** (`src/main.zig`)
  - [x] Add `--jolt-format` option to `prove` command
  - [x] Generate Jolt-compatible proof with `proveJoltCompatible()`
  - [x] Serialize to file using arkworks format
  - [x] Test proof generation for fibonacci.elf

### 9. Remaining Work (Blocking Issue)

**Critical Issue: Commitment Scheme Mismatch**

The Jolt verifier (`RV64IMACProof`) expects **Dory commitments** (GT = Fp12, 384 bytes each),
but Zolt is currently generating **HyperKZG commitments** (G1 points, 32 bytes each).

The proof format analysis shows:
- OpeningClaims: ✅ Correct format, all 11 claims parse correctly
- VirtualPolynomial serialization: ✅ Matches Jolt exactly
- Commitments: ❌ Wrong type (G1 vs GT)
- Opening proofs: ❌ Wrong type (HyperKZG vs Dory)

**Resolution Options:**

1. **Switch Zolt to use Dory** (recommended):
   - Replace HyperKZG commitment scheme with Dory throughout the prover
   - Use `DoryCommitmentScheme` instead of `HyperKZGScheme`
   - Generate GT commitments (384 bytes) instead of G1 points
   - This requires significant refactoring of the prover

2. **Create HyperKZG proof type in Jolt**:
   - Add `RV64IMACProofHyperKZG` type to Jolt
   - Create verifier that accepts HyperKZG commitments
   - Less work on Zolt side, but requires Jolt modifications

3. **Dual-mode prover**:
   - Keep HyperKZG for fast internal verification
   - Add Dory commitment generation as a separate step
   - More complex but maintains both capabilities

- [ ] **Resolve Commitment Scheme Mismatch**
  - [ ] Choose resolution option
  - [ ] Implement chosen solution
  - [ ] Generate proof with correct commitment type
  - [ ] Verify proof parses in Jolt

---

## Test Status

### All 608 Tests Passing

```
zig build test --summary all
Build Summary: 5/5 steps succeeded; 608/608 tests passed
```

### Components Tested

| Component | Tests |
|-----------|-------|
| Blake2bTranscript | 7+ test vectors |
| Dory IPA prover | Proof serialization, VMV messages |
| G1 compression roundtrip | Pass |
| G1 identity compression | Pass |
| G2 compression format | Pass |
| G2 identity compression | Pass |
| GT serialization | Roundtrip verified |

---

## Key Reference Files

### Zolt (Implemented)
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript + Dory helpers |
| `src/zkvm/jolt_types.zig` | ✅ Done | Jolt proof types |
| `src/zkvm/jolt_serialization.zig` | ✅ Done | Arkworks serialization + Dory |
| `src/zkvm/proof_converter.zig` | ✅ Done | 6→7 stage converter |
| `src/zkvm/mod.zig` | ✅ Done | JoltProver with Jolt export |
| `src/poly/commitment/dory.zig` | ✅ **Complete** | Full Dory IPA + G1/G2 compression |
| `src/field/pairing.zig` | ✅ Done | GT (Fp12) serialization |

### Jolt (Reference Only)
| File | Purpose |
|------|---------|
| `jolt-core/src/transcripts/blake2b.rs` | ✅ Verified |
| `jolt-core/src/zkvm/proof_serialization.rs` | ✅ Analyzed |
| `jolt-core/src/poly/commitment/dory/` | ✅ Analyzed |
| `dory-pcs/src/evaluation_proof.rs` | ✅ Analyzed - Prover algorithm |
| `dory-pcs/src/reduce_and_fold.rs` | ✅ Analyzed - State machines |
| `dory-pcs/src/messages.rs` | ✅ Analyzed - Message types |

---

## Success Criteria

1. ✅ `zig build test` passes all 608 tests
2. ✅ Zolt can generate a proof in Jolt format (`proveJoltCompatible`)
3. ✅ Zolt can serialize proofs in arkworks format (`serializeJoltProof`)
4. ✅ E2E serialization tests verify format compatibility
5. ✅ Dory commitment scheme with GT serialization
6. ✅ Dory IPA proof structure with G1/G2 compression
7. ✅ Full Dory IPA reduce-and-fold algorithm
8. ✅ Transcript-integrated Dory prover
9. ✅ Cross-verification test infrastructure in Jolt created
10. ✅ CLI `--jolt-format` flag added to prove command
11. ✅ Opening claims format matches Jolt (verified with debug parsing)
12. ⚠️ Commitment scheme mismatch: HyperKZG vs Dory (blocking issue)
13. ⏳ The proof can be loaded and verified by Jolt (blocked by #12)
14. ⏳ No modifications needed on the Jolt side (blocked by #12)

## Priority Order

1. ✅ **Transcript** - Matching Fiat-Shamir complete
2. ✅ **Proof Types** - JoltProof structure defined
3. ✅ **Serialization** - Byte-perfect compatibility verified
4. ✅ **Dory Commitment** - GT serialization complete
5. ✅ **Dory IPA Structure** - Proof structure matching Jolt
6. ✅ **G1/G2 Compression** - arkworks format
7. ✅ **Prover Wiring** - Connect types to prover
8. ✅ **Integration Tests** - E2E serialization verified
9. ✅ **Full Dory IPA** - Reduce-and-fold algorithm complete
10. ✅ **Transcript Integration** - Fiat-Shamir challenges from transcript
11. ✅ **Cross-Verification Infrastructure** - Jolt test files created
12. ⏳ **Full Proof Generation** - Generate valid proof for cross-verification
