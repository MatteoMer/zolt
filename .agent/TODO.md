# Zolt-Jolt Compatibility TODO

## Completed ✅

### Phase 1-5: Core Infrastructure
1. **Transcript Compatibility** - Blake2b transcript matches Jolt
2. **Proof Structure Refactoring** - 7-stage proof with UniSkip
3. **Serialization Alignment** - Arkworks-compatible serialization
4. **Commitment Scheme** - Dory with Jolt-compatible SRS
5. **Verifier Preprocessing Export** - DoryVerifierSetup exports correctly

### Stage 1 Fixes
6. **Lagrange Interpolation Bug** - Fixed dead code corrupting basis array
7. **UniSkip Verification** - Domain sum check passes
8. **UnivariateSkip Claim** - Correctly set to uni_poly.evaluate(r0)
9. **Montgomery Form Fix** - appendScalar converts from Montgomery form
10. **MontU128Challenge Compatibility** - Challenge scalars match Jolt's format
11. **Symmetric Lagrange Domain** - Uses {-4,...,5} matching Jolt
12. **Streaming Round Logic** - Separate handling for constraint group selection
13. **MultiquadraticPolynomial** - Implemented in src/poly/multiquadratic.zig
14. **Multiquadratic Round Polynomial** - Added computeRemainingRoundPolyMultiquadratic()
15. **r0 Not Bound in split_eq** - Uses Lagrange scaling instead
16. **Claim Update** - Converts evaluations to coefficients properly
17. **Factorized Eq Weights** - eq[i] = E_out[i>>5] * E_in[i&0x1F]
18. **getWindowEqTables** - Matches Jolt's E_out_in_for_window logic
19. **Window eq tables sizing** - 32*32=1024 factorization verified
20. **ExpandingTable** - Added for incremental eq polynomial computation
21. **Constraint Group Indices** - Fixed to match Jolt's ordering
22. **r_grid Integration** - Added to streaming prover for bound challenge weights
23. **Product of Slopes Fix** - t'(∞) = (Az_g1-Az_g0)*(Bz_g1-Bz_g0), NOT t'(1)-t'(0)

---

## Current Status: Stage 1 Sumcheck Final Claim Mismatch

### What Works ✅
- UniSkip verification passes
- All 11 remaining sumcheck rounds pass (p(0)+p(1)=claim)
- Challenge derivation matches between Zolt and Jolt
- Streaming round (round 1) uses product of slopes

### What Doesn't Work ❌
- Final output_claim ≠ expected_output_claim

### Latest Values (After Session 13 Fixes)
```
output_claim (from sumcheck):     6342589437459870311969131907974809364188732687625165290542906571219779431047
expected_output_claim (from R1CS): 10815232497405550102099453343964744311420855254768592892790152336464556005907
```

### Root Cause Analysis

The issue is likely in the **cycle rounds** (rounds 2-11). Possible causes:

1. **r_grid weighting** - The weight computation might not match Jolt's
2. **Cycle bit indexing** - The way cycles are split by current_bit might be wrong
3. **eq weight factorization** - E_out/E_in tables might not be computed correctly for cycle rounds
4. **current_scalar accumulation** - split_eq.current_scalar might not be updated properly

### Next Steps

1. [ ] Compare r_grid values at each round between Zolt and Jolt
2. [ ] Verify E_out/E_in table sizes match at each round
3. [ ] Check if cycle rounds should use multiquadratic (product of slopes) like streaming round
4. [ ] Consider implementing Jolt's LinearStage approach (materialize Az/Bz polynomials)

---

## Later Stages

### Stage 2: Outer Product
- Product virtualization sumcheck
- Uses similar multiquadratic method

### Stage 3-7: Various Sumchecks
- Will need verification after Stage 1 passes

---

## Test Commands

```bash
# Generate proof in Zolt
cd /Users/matteo/projects/zolt
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Run cross-verification in Jolt
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
