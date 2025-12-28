# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: OPENING CLAIMS COMPUTED, SUMCHECK VERIFICATION FAILING**

Key achievements:
1. **R1CS Input MLE Evaluations** - Opening claims now contain actual computed values
2. **Correct Round Polynomial Count** - Stage 1 has proper 1 + num_cycle_vars polynomials
3. **Cross-Deserialization** - Jolt successfully deserializes Zolt proofs

Current issue: Sumcheck verification fails because round polynomials are not correctly
computed from Az*Bz products. The opening claims are correct (non-zero PC values, OpFlags)
but the sumcheck polynomials themselves are placeholder values.

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
15. ✅ **Transcript Integration** - Blake2bTranscript in proof generation
16. ✅ **R1CS Input MLE Evaluation** - Compute actual evaluations at challenge point
17. ✅ **Correct Round Count** - Stage 1 has 1 + num_cycle_vars round polynomials

---

## Current Work: Sumcheck Verification

### Verified Working

1. **Opening Claims** - Non-zero values for PC, OpFlags(AddOperands), OpFlags(WriteLookupOutputToRD)
2. **Round Polynomial Count** - 4 polynomials for trace length 8 (3 cycle vars)
3. **Proof Deserialization** - Jolt correctly parses all proof components

### Failing: Sumcheck Polynomial Values

The sumcheck verification fails because:
- Round polynomials are generated from computeRemainingRoundPoly() but values are incorrect
- The formula: s(X) = Σ eq(τ, i) × Az(i, X) × Bz(i, X) needs proper implementation
- Current streaming prover uses simplified evaluations

### Root Cause Analysis

Looking at Jolt verification output:
```
Virtual(PC, SpartanOuter) => 470923325918454702788286590928955227900599927949267948307234034664185460615
Virtual(OpFlags(AddOperands), SpartanOuter) => 14116661703799451320060418720194240191430100414874762526722692778591556927761
```

These claims are non-zero (good!) but the sumcheck relation:
```
output_claim == expected_output_claim(r_cycle)
```
is not satisfied because our round polynomials don't correspond to the correct Az*Bz sums.

---

## Remaining Work

### High Priority

1. **Fix StreamingOuterProver.computeRemainingRoundPoly()**
   - Compute actual Az*Bz products per cycle
   - Use constraint evaluators for first/second groups
   - Generate correct degree-3 polynomials

2. **Fix UniSkip First Round Polynomial**
   - The degree-27 polynomial needs proper constraint evaluation
   - Currently using placeholder values

### Medium Priority

3. **Stages 2-7** - Currently all zeros (placeholder)
4. **Joint Opening Proof** - Dory batch opening proof

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
| `test_deserialize_zolt_proof` | ✅ PASS | 26558 bytes, 48 claims |
| `test_debug_zolt_format` | ✅ PASS | All claims and commitments valid |
| `test_verify_zolt_proof` | ❌ FAIL | Sumcheck verification failed |

---

## Key Files

### Core Implementation
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript |
| `src/zkvm/jolt_types.zig` | ✅ Done | Jolt proof types |
| `src/zkvm/jolt_serialization.zig` | ✅ Done | Arkworks serialization |
| `src/zkvm/proof_converter.zig` | ✅ Done | 6→7 stage converter + transcript |
| `src/poly/commitment/dory.zig` | ✅ Done | Dory IPA |
| `src/zkvm/r1cs/constraints.zig` | ✅ Done | 19 R1CS constraints |
| `src/zkvm/r1cs/evaluators.zig` | ✅ Done | Az/Bz constraint evaluators |
| `src/zkvm/r1cs/evaluation.zig` | ✅ Done | MLE evaluation at r_cycle |
| `src/poly/split_eq.zig` | ✅ Done | Gruen's efficient eq polynomial |
| `src/poly/multiquadratic.zig` | ✅ Done | Ternary grid expansion {0, 1, ∞} |
| `src/zkvm/spartan/streaming_outer.zig` | 🔄 WIP | Streaming outer sumcheck prover |
| `src/zkvm/spartan/outer.zig` | 🔄 WIP | UniSkip first-round prover |

### Next Steps
| Task | Priority | Complexity |
|------|----------|------------|
| Fix Az*Bz computation in streaming prover | High | High |
| Fix UniSkip polynomial computation | High | High |
| Implement Stages 2-7 | Medium | High |
| Debug verification with trace | Low | Medium |

---

## Summary

**Serialization Compatibility: COMPLETE**
- Zolt produces proofs that Jolt can deserialize
- Byte-perfect arkworks format compatibility
- All structural components in place

**Transcript Integration: COMPLETE**
- Blake2bTranscript fully integrated in proof generation
- Stage 1 uses transcript-derived challenges
- convertWithTranscript() method for full integration

**Opening Claims: WORKING**
- Non-zero values computed from MLE evaluation
- Proper mapping from Zolt's R1CSInputIndex to Jolt's VirtualPolynomial

**Verification Compatibility: CLOSE BUT FAILING**
- Sumcheck polynomial values are incorrect
- Need proper Az*Bz computation in streaming prover
- Expected timeline: 1-2 more iterations

**Architecture Notes:**
- The streaming sumcheck uses Gruen's method with multiquadratic expansion
- Prefix eq tables allow efficient O(n) evaluation instead of O(n log n)
- First round uses degree-27 univariate skip polynomial
- Remaining rounds use degree-3 polynomials
- All challenges derived from Blake2b Fiat-Shamir transcript
- Opening claims computed using eq polynomial MLE evaluation
