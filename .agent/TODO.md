# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: STREAMING SUMCHECK INTEGRATED**

We have implemented and integrated the streaming sumcheck infrastructure:

1. **GruenSplitEqPolynomial** - Efficient eq polynomial with prefix tables
2. **MultiquadraticPolynomial** - Ternary grid {0, 1, ∞}^d representation
3. **StreamingOuterProver** - Framework for generating non-zero sumcheck proofs
4. **Proof Converter Integration** - Stage 1 remaining rounds use actual evaluations

The next step is to hook up the Fiat-Shamir transcript for consistent challenges.

---

## Major Milestones Achieved

1. ✅ **Blake2b Transcript** - Identical Fiat-Shamir challenges as Jolt (7 test vectors verified)
2. ✅ **Proof Types** - JoltProof with 7-stage structure, OpeningClaims, etc.
3. ✅ **Arkworks Serialization** - Byte-perfect format compatibility
4. ✅ **Dory Commitment** - GT (Fp12) serialization, IPA proof structure
5. ✅ **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs!
6. ✅ **Univariate Skip Infrastructure** - Degree-27/12 polynomials for stages 1-2
7. ✅ **All 48 Opening Claims** - Including all 36 R1CS inputs + OpFlags variants
8. ✅ **VirtualPolynomial Ordering** - Fixed payload comparison for OpFlags, etc.
9. ✅ **19 R1CS Constraints** - Matching Jolt's exact constraint structure
10. ✅ **Constraint Evaluators** - Az/Bz for first and second groups
11. ✅ **GruenSplitEqPolynomial** - Prefix eq tables for efficient factored evaluation
12. ✅ **MultiquadraticPolynomial** - Ternary grid expansion for streaming sumcheck
13. ✅ **StreamingOuterProver** - Framework with degree-27 and degree-3 round polys
14. ✅ **Proof Converter Integration** - Stage 1 uses StreamingOuterProver

---

## Current Work: Full Cross-Verification

### Implemented Components

1. **GruenSplitEqPolynomial** (`src/poly/split_eq.zig`)
   - Prefix eq tables E_out_vec and E_in_vec
   - Variable binding with scalar accumulation
   - Cubic round polynomial computation using Gruen's method

2. **MultiquadraticPolynomial** (`src/poly/multiquadratic.zig`)
   - Base-3 grid encoding with z_0 fastest-varying
   - Expansion from linear {0,1}^d to ternary {0,1,∞}^d
   - f(∞) = f(1) - f(0) (slope extrapolation)

3. **StreamingOuterProver** (`src/zkvm/spartan/streaming_outer.zig`)
   - First-round univariate skip polynomial (degree 27)
   - Remaining rounds with degree-3 polynomials
   - Lagrange basis precomputation at r0

4. **Proof Converter** (`src/zkvm/proof_converter.zig`)
   - `generateStreamingOuterSumcheckProof()` for actual evaluations
   - Falls back to zero proofs on error
   - Currently uses deterministic challenges (not transcript)

### Remaining Work

1. **Hook up Fiat-Shamir Transcript**
   - Pass Blake2bTranscript through proof generation
   - Use transcript for all challenges
   - Ensures proofs match Jolt's verification

2. **End-to-end Cross-Verification**
   - Generate proof for trivial trace
   - Verify with Jolt's verifier
   - Debug any discrepancies

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

### 4. Dory Commitment Scheme ✅ COMPLETE

- [x] **Dory Implementation** (`src/poly/commitment/dory.zig`)
- [x] **Full Dory IPA Prover** - reduce-and-fold algorithm
- [x] **Point Compression** - G1/G2 in arkworks format

### 5. R1CS Constraints ✅ COMPLETE

- [x] **19 constraints matching Jolt** (`src/zkvm/r1cs/constraints.zig`)
- [x] **First group (10 constraints)** - Boolean guards, ~64-bit Bz
- [x] **Second group (9 constraints)** - Mixed guards, ~128-bit Bz
- [x] **Constraint evaluators** - Az/Bz per group

### 6. Streaming Sumcheck ✅ MOSTLY COMPLETE

- [x] **GruenSplitEqPolynomial** - Prefix tables for factored eq evaluation
- [x] **MultiquadraticPolynomial** - Ternary grid for streaming
- [x] **StreamingOuterProver** - Framework for degree-27 and degree-3 rounds
- [x] **Integration with proof converter** - Stage 1 remaining rounds
- [ ] **Transcript integration** - Use Blake2bTranscript for challenges

### 7. Cross-Verification 🔄 IN PROGRESS

- [x] **Jolt deserializes Zolt proofs** - VERIFIED WORKING
- [x] **48 opening claims** - All R1CS inputs + OpFlags + stage claims
- [x] **UniSkip first-round structure** - Correct degree polynomials
- [x] **Streaming sumcheck proofs** - Now using actual evaluations
- [ ] **Fiat-Shamir consistency** - Transcript for challenges
- [ ] **Full verification** - End-to-end with Jolt verifier

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
| `test_verify_zolt_proof` | ❌ FAIL | Stage 1 final claim check |

---

## Key Files

### Zolt (Implemented)
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript |
| `src/zkvm/jolt_types.zig` | ✅ Done | Jolt proof types |
| `src/zkvm/jolt_serialization.zig` | ✅ Done | Arkworks serialization |
| `src/zkvm/proof_converter.zig` | ✅ Done | 6→7 stage converter |
| `src/poly/commitment/dory.zig` | ✅ Done | Dory IPA |
| `src/zkvm/r1cs/constraints.zig` | ✅ Done | 19 R1CS constraints |
| `src/zkvm/r1cs/evaluators.zig` | ✅ Done | Az/Bz constraint evaluators |
| `src/poly/split_eq.zig` | ✅ Done | Gruen's efficient eq polynomial |
| `src/poly/multiquadratic.zig` | ✅ Done | Ternary grid expansion {0, 1, ∞} |
| `src/zkvm/spartan/streaming_outer.zig` | ✅ Done | Streaming outer sumcheck prover |
| `src/zkvm/spartan/outer.zig` | ✅ Done | UniSkip first-round prover |

### Next Steps
| Task | Priority | Complexity |
|------|----------|------------|
| Hook up Fiat-Shamir transcript | High | Medium |
| Test with Jolt verifier | High | Medium |
| Debug any verification failures | High | High |

---

## Summary

**Serialization Compatibility: COMPLETE**
- Zolt produces proofs that Jolt can deserialize
- Byte-perfect arkworks format compatibility
- All structural components in place

**Verification Compatibility: CLOSE**
- Core polynomial types implemented and integrated
- StreamingOuterProver generates actual polynomial evaluations
- Need to connect Fiat-Shamir transcript for consistent challenges
- Expected timeline: 1 more iteration for transcript, then testing

**Architecture Notes:**
- The streaming sumcheck uses Gruen's method with multiquadratic expansion
- Prefix eq tables allow efficient O(n) evaluation instead of O(n log n)
- First round uses degree-27 univariate skip polynomial
- Remaining rounds use degree-3 polynomials
