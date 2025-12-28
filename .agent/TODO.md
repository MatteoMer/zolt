# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: R1CS CONSTRAINT EVALUATORS COMPLETE - SUMCHECK PROOF GENERATION NEXT**

### Major Milestones Achieved

1. ✅ **Blake2b Transcript** - Identical Fiat-Shamir challenges as Jolt
2. ✅ **Proof Types** - JoltProof with 7-stage structure, OpeningClaims, etc.
3. ✅ **Arkworks Serialization** - Byte-perfect format compatibility
4. ✅ **Dory Commitment** - GT (Fp12) serialization, IPA proof structure
5. ✅ **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs!
6. ✅ **Univariate Skip Infrastructure** - Degree-27/12 polynomials for stages 1-2
7. ✅ **All 48 Opening Claims** - Including all 36 R1CS inputs for SpartanOuter + 13 OpFlags variants
8. ✅ **VirtualPolynomial Ordering** - Fixed payload comparison for OpFlags, InstructionFlags, etc.
9. ✅ **19 R1CS Constraints** - Updated to match Jolt's exact constraint structure (Iteration 13)
10. ✅ **Constraint Evaluators** - AzFirstGroup, BzFirstGroup, AzSecondGroup, BzSecondGroup (Iteration 13)
11. ✅ **Univariate Skip with Real Evaluations** - `createUniSkipProofStage1FromWitnesses()` (Iteration 13)

### Latest Changes (Iteration 13)

#### 1. Updated R1CS Constraints (`src/zkvm/r1cs/constraints.zig`)
- All 19 constraints now match Jolt's exact layout
- Added `FIRST_GROUP_INDICES` (10 constraints for univariate skip domain)
- Added `SECOND_GROUP_INDICES` (9 constraints for separate handling)
- Constraint form: Az * Bz = 0 (equality-conditional)

#### 2. Created Constraint Evaluators (`src/zkvm/r1cs/evaluators.zig`)
- `AzFirstGroup` / `BzFirstGroup` - Evaluate 10 first-group constraints
- `AzSecondGroup` / `BzSecondGroup` - Evaluate 9 second-group constraints
- `UnivariateSkipEvaluator` - Computes Az*Bz products across cycles
- Proper domain point mapping (y ∈ {-4, -3, ..., 5})

#### 3. Updated Spartan Outer Prover (`src/zkvm/spartan/outer.zig`)
- `initFromWitnesses()` - Uses constraint evaluators
- Precomputes base window evaluations
- Lagrange extrapolation for extended domain points

#### 4. Proof Converter with Witness Support (`src/zkvm/proof_converter.zig`)
- `convertWithWitnesses()` - Takes cycle witnesses and tau challenge
- `createUniSkipProofStage1FromWitnesses()` - Computes real Az*Bz products
- Uses SpartanOuterProver for proper polynomial computation

### What's Still Needed for Full Verification

The sumcheck verification fails because:
1. Only Stage 1 univariate skip uses real evaluations
2. Remaining sumcheck rounds still use "zero proofs"
3. Need to generate proper round polynomials satisfying p(0) + p(1) = claim

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

### 5. R1CS Constraints ✅ COMPLETE (NEW)

- [x] **19 constraints matching Jolt** (`src/zkvm/r1cs/constraints.zig`)
- [x] **First group (10 constraints)** - Boolean guards, ~64-bit Bz
- [x] **Second group (9 constraints)** - Mixed guards, ~128-bit Bz
- [x] **Constraint evaluators** - Az/Bz per group (`src/zkvm/r1cs/evaluators.zig`)
- [x] **Univariate skip evaluator** - Computes Az*Bz products

### 6. Cross-Verification 🔄 IN PROGRESS

- [x] **Jolt deserializes Zolt proofs** - VERIFIED WORKING
- [x] **48 opening claims** - All R1CS inputs + OpFlags + stage claims
- [x] **UniSkip first-round with real evaluations** - Stage 1 implemented
- [ ] **Remaining stage sumcheck proofs** - Still using zeros
- [ ] **Full verification** - Blocked on proper sumcheck

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
| `src/zkvm/proof_converter.zig` | ✅ Done | 6→7 stage converter with witness support |
| `src/zkvm/mod.zig` | ✅ Done | JoltProver |
| `src/poly/commitment/dory.zig` | ✅ Done | Dory IPA |
| `src/zkvm/r1cs/constraints.zig` | ✅ Done | 19 R1CS constraints matching Jolt |
| `src/zkvm/r1cs/evaluators.zig` | ✅ Done | Az/Bz constraint evaluators |
| `src/zkvm/r1cs/univariate_skip.zig` | ✅ Done | Univariate skip constants |
| `src/zkvm/spartan/outer.zig` | ✅ Done | Spartan outer with real evaluations |

---

## Summary

**Serialization + Constraint Evaluation: COMPLETE**
- Zolt produces proofs that Jolt can deserialize
- Byte-perfect arkworks format compatibility
- R1CS constraints match Jolt's 19-constraint structure
- Constraint evaluators compute real Az*Bz products
- Stage 1 univariate skip uses actual evaluations

**Next Steps:**
1. Generate proper sumcheck round polynomials for all stages
2. Integrate witness-based evaluation into remaining stages
3. Test full cross-verification with Jolt
