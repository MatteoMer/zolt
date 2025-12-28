# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: STREAMING SUMCHECK INFRASTRUCTURE IN PROGRESS**

We have implemented the core polynomial types and streaming sumcheck framework:

1. **GruenSplitEqPolynomial** - Efficient eq polynomial with prefix tables for factored evaluation
2. **MultiquadraticPolynomial** - Ternary grid {0, 1, ∞}^d representation for streaming optimization
3. **StreamingOuterProver** - Framework for generating non-zero sumcheck proofs

The next step is to connect these components and generate actual valid proofs.

---

## Major Milestones Achieved

1. ✅ **Blake2b Transcript** - Identical Fiat-Shamir challenges as Jolt
2. ✅ **Proof Types** - JoltProof with 7-stage structure, OpeningClaims, etc.
3. ✅ **Arkworks Serialization** - Byte-perfect format compatibility
4. ✅ **Dory Commitment** - GT (Fp12) serialization, IPA proof structure
5. ✅ **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs!
6. ✅ **Univariate Skip Infrastructure** - Degree-27/12 polynomials for stages 1-2
7. ✅ **All 48 Opening Claims** - Including all 36 R1CS inputs + OpFlags variants
8. ✅ **VirtualPolynomial Ordering** - Fixed payload comparison for OpFlags, etc.
9. ✅ **19 R1CS Constraints** - Matching Jolt's exact constraint structure
10. ✅ **Constraint Evaluators** - Az/Bz for first and second groups
11. ✅ **Verification Analysis** - Understood why zero proofs fail
12. ✅ **GruenSplitEqPolynomial** - Prefix eq tables for efficient factored evaluation
13. ✅ **MultiquadraticPolynomial** - Ternary grid expansion for streaming sumcheck
14. ✅ **StreamingOuterProver** - Framework with degree-27 and degree-3 round polys

---

## Current Work: Full Sumcheck Prover

### Implemented Components

1. **GruenSplitEqPolynomial** (`src/poly/split_eq.zig`)
   - Prefix eq tables E_out_vec and E_in_vec
   - Variable binding with scalar accumulation
   - Cubic round polynomial computation using Gruen's method
   - Window-based eq table access

2. **MultiquadraticPolynomial** (`src/poly/multiquadratic.zig`)
   - Base-3 grid encoding with z_0 fastest-varying
   - Expansion from linear {0,1}^d to ternary {0,1,∞}^d
   - f(∞) = f(1) - f(0) (slope extrapolation)
   - Projection to first variable with eq weights

3. **StreamingOuterProver** (`src/zkvm/spartan/streaming_outer.zig`)
   - First-round univariate skip polynomial (degree 27)
   - Remaining rounds with degree-3 polynomials
   - Lagrange basis precomputation at r0
   - Integration with GruenSplitEqPolynomial

### Remaining Work

1. **Connect StreamingOuterProver to proof converter**
   - Replace JoltOuterProver with StreamingOuterProver
   - Ensure transcript matches Jolt's exactly
   - Generate actual polynomial evaluations

2. **Validate sumcheck correctness**
   - Verify round polynomials satisfy sumcheck equation
   - Verify final claim matches expected output

3. **End-to-end testing**
   - Generate proof for trivial trace (all NOPs)
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
- [x] **writeUniSkipFirstRoundProof** - Serializes all coefficients
- [x] **Opening claims serialization** - BTreeMap-compatible order

### 4. Dory Commitment Scheme ✅ COMPLETE

- [x] **Dory Implementation** (`src/poly/commitment/dory.zig`)
- [x] **Full Dory IPA Prover** - reduce-and-fold algorithm
- [x] **Point Compression** - G1/G2 in arkworks format
- [x] **Proof Serialization** - Matches ArkDoryProof format

### 5. R1CS Constraints ✅ COMPLETE

- [x] **19 constraints matching Jolt** (`src/zkvm/r1cs/constraints.zig`)
- [x] **First group (10 constraints)** - Boolean guards, ~64-bit Bz
- [x] **Second group (9 constraints)** - Mixed guards, ~128-bit Bz
- [x] **Constraint evaluators** - Az/Bz per group

### 6. Streaming Sumcheck 🔄 IN PROGRESS

- [x] **GruenSplitEqPolynomial** - Prefix tables for factored eq evaluation
- [x] **MultiquadraticPolynomial** - Ternary grid for streaming
- [x] **StreamingOuterProver** - Framework for degree-27 and degree-3 rounds
- [ ] **Integration with proof converter** - Connect to JoltProof generation
- [ ] **End-to-end verification** - Generate and verify non-zero proofs

### 7. Cross-Verification 🔄 PARTIALLY COMPLETE

- [x] **Jolt deserializes Zolt proofs** - VERIFIED WORKING
- [x] **48 opening claims** - All R1CS inputs + OpFlags + stage claims
- [x] **UniSkip first-round structure** - Correct degree polynomials
- [ ] **Sumcheck proofs with correct claims** - In progress with StreamingOuterProver
- [ ] **Full verification** - Waiting on correct sumcheck implementation

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
| `test_verify_zolt_proof` | ❌ FAIL | Stage 1 final claim check (expected with zero proofs) |

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
| `src/zkvm/spartan/streaming_outer.zig` | ✅ Done | Streaming outer sumcheck prover framework |

### Next Steps
| Task | Priority | Complexity |
|------|----------|------------|
| Connect StreamingOuterProver to proof converter | High | Medium |
| Validate sumcheck with trivial trace | High | Medium |
| Debug against Jolt verifier | High | High |
| Optimize for larger traces | Low | High |

---

## Summary

**Serialization Compatibility: COMPLETE**
- Zolt produces proofs that Jolt can deserialize
- Byte-perfect arkworks format compatibility
- All structural components in place

**Verification Compatibility: IN PROGRESS**
- Core polynomial types implemented (GruenSplitEqPolynomial, MultiquadraticPolynomial)
- StreamingOuterProver framework implemented
- Need to connect components and validate with Jolt verifier
- Expected timeline: 1-2 more iterations

**Next Session Goals:**
1. Connect StreamingOuterProver to proof converter
2. Generate proof with actual polynomial evaluations
3. Test against Jolt verifier
4. Debug any discrepancies
