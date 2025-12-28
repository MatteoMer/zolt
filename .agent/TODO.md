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

---

## Current Status: Stage 1 Sumcheck Final Claim Mismatch

### What Works ✅
- UniSkip verification passes
- All 11 remaining sumcheck rounds pass (p(0)+p(1)=claim)
- Challenge derivation matches between Zolt and Jolt

### What Doesn't Work ❌
- Final output_claim ≠ expected_output_claim

### Key Values
```
output_claim (from sumcheck):    1974116927555899330558670899680846593837010021404266128199797046376479596381
expected_output_claim (from R1CS): 15883159918150614947002691188000461030180166032167382519626147847666314949749
```

### Root Cause Analysis

The sumcheck rounds are internally consistent, but polynomial evaluations at challenge points produce incorrect intermediate claims.

The issue is likely in one of:
1. Cycle index to eq weight mapping (E_out, E_in, r_grid)
2. How the multiquadratic t'(0) and t'(∞) values are computed
3. Subtle difference in how Jolt structures the prover vs how Zolt implements it

### Next Steps

1. [ ] Add detailed debug output comparing Zolt and Jolt values at each round
2. [ ] Trace exact cycle-to-weight mappings in both implementations
3. [ ] Consider implementing a minimal test case with known values
4. [ ] Verify r_grid update timing matches Jolt's r_grid.update(r_j) placement

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
