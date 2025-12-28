# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: SUMCHECK VERIFICATION ANALYSIS COMPLETE**

The investigation revealed why zero proofs don't work for cross-verification:

1. **Univariate skip first round passes** (sum over domain = 0 with zero coefficients)
2. **Remaining sumcheck rounds pass** (p(0) + p(1) = 0 with zero coefficients)
3. **Final claim check FAILS** (expected_output_claim ≠ 0)

The verifier computes `expected_output_claim` from R1CS constraints, which include
constant terms that evaluate to non-zero even with zero input claims.

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

---

## What's Needed for Full Verification

### The Core Problem

The sumcheck protocol requires:
- Round polynomials that satisfy `p(0) + p(1) = previous_claim`
- Final claim that equals `expected_output_claim` (computed from witnesses)

Zero proofs satisfy the first but not the second requirement.

### Implementation Requirements

To generate valid proofs, we need:

1. **Full Sumcheck Prover** (Spartan Outer)
   - Materialize Az(x) and Bz(x) over the cycle hypercube
   - Compute `Σ_x eq(τ, x) * Az(x) * Bz(x)` using streaming sumcheck
   - Handle multiquadratic expansion for efficient streaming
   - File: `src/zkvm/spartan/outer_prover.zig` (to create)

2. **MLE Evaluation at Challenge Point**
   - Compute r1cs_input_evals[i] = MLE_i(r_cycle)
   - These are the evaluations of R1CS input polynomials at the random point
   - Used by verifier to compute expected_output_claim

3. **Consistent Prover/Verifier Transcript**
   - Challenges derived from actual round polynomials
   - Must match Jolt's transcript exactly

### Complexity Estimate

The Jolt prover uses:
- `GruenSplitEqPolynomial` (~500 lines Rust)
- `MultiquadraticPolynomial` (~400 lines Rust)
- `OuterSharedState` and `OuterLinearStage` (~800 lines Rust)
- Streaming sumcheck infrastructure (~300 lines Rust)

Total: ~2000 lines of complex parallel code

A faithful port to Zig would be a significant undertaking.

---

## Alternative Approaches

### Option 1: Native Sumcheck Only
- Use Zolt's existing sumcheck prover for Zolt's own verification
- Accept that cross-verification with Jolt won't work
- Simplest, but defeats the purpose of compatibility

### Option 2: Format Compatibility Only
- Focus on serialization/deserialization compatibility
- Document that proofs are format-compatible but not verifiable
- Useful for tooling and debugging

### Option 3: Incremental Implementation
- Start with a simplified sumcheck that works for trivial traces
- Gradually add streaming/windowed optimizations
- Most realistic path forward

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

### 6. Cross-Verification 🔄 PARTIALLY COMPLETE

- [x] **Jolt deserializes Zolt proofs** - VERIFIED WORKING
- [x] **48 opening claims** - All R1CS inputs + OpFlags + stage claims
- [x] **UniSkip first-round structure** - Correct degree polynomials
- [ ] **Sumcheck proofs with correct claims** - BLOCKED on full prover
- [ ] **Full verification** - BLOCKED on correct sumcheck

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

### Needed for Full Verification
| File | Status | Purpose |
|------|--------|---------|
| `src/zkvm/spartan/outer_prover.zig` | ❌ TODO | Full Spartan outer sumcheck prover |
| `src/poly/multiquadratic.zig` | ❌ TODO | Tertiary grid expansion |
| `src/poly/split_eq.zig` | ❌ TODO | Gruen's efficient eq polynomial |

---

## Summary

**Serialization Compatibility: COMPLETE**
- Zolt produces proofs that Jolt can deserialize
- Byte-perfect arkworks format compatibility
- All structural components in place

**Verification Compatibility: BLOCKED**
- Zero proofs don't satisfy the final claim check
- Need to implement full sumcheck prover with actual polynomial evaluations
- This requires ~2000 lines of complex parallel computation code

**Recommended Next Steps:**
1. Document the current compatibility level
2. Consider if full verification is required for the use case
3. If needed, plan incremental implementation of streaming sumcheck
