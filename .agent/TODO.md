# Zolt-Jolt Compatibility TODO

## Completed ✅

### Phase 1-5: Core Infrastructure
1. **Transcript Compatibility** - Blake2b transcript matches Jolt
2. **Proof Structure Refactoring** - 7-stage proof with UniSkip
3. **Serialization Alignment** - Arkworks-compatible serialization
4. **Commitment Scheme** - Dory with Jolt-compatible SRS
5. **Verifier Preprocessing Export** - DoryVerifierSetup exports correctly

### Stage 1 Fixes (Session 11-14)
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
23. **Product of Slopes Fix** - t'(∞) = (Az_g1-Az_g0)*(Bz_g1-Bz_g0)
24. **Multiquadratic Sum of Products** - t'(∞) = Σ (slope_Az * slope_Bz), NOT (Σ slope_Az) * (Σ slope_Bz)
25. **current_scalar double-counting fix** - Removed from t' computation, only applied in l(X)

---

## Current Status: ~10x Discrepancy in Stage 1

### What Works ✅
- UniSkip verification passes
- All 11 remaining sumcheck rounds pass (p(0)+p(1)=claim)
- Challenge derivation matches
- Streaming round uses multiquadratic with product of slopes

### Remaining Issue
- output_claim ≈ 10x expected_output_claim
- The computeCubicRoundPoly logic matches Jolt's gruen_poly_deg_3

### Possible Causes
1. Eq table indexing (E_out, E_in) might be wrong
2. r_grid weighting in cycle rounds differs from Jolt
3. Streaming vs cycle round r_grid update timing
4. Cycle index bit manipulation for pairs

### Key Insight from Jolt
In Jolt's streaming sumcheck:
- `OuterStreamingWindow::ingest_challenge` updates r_grid
- `OuterLinearStage::ingest_challenge` does NOT update r_grid
- This suggests r_grid captures eq weights from streaming phase only

### Next Steps for fix
1. [ ] Add detailed debug output comparing intermediate values
2. [ ] Trace E_out, E_in table access patterns
3. [ ] Compare r_grid values between Zolt and Jolt at each round
4. [ ] Consider implementing Jolt's LinearStage approach (materialize Az/Bz)

---

## Later Tasks

### Stage 2-7 Verification
After Stage 1 exact match, verify:
- Stage 2: Outer Product
- Stage 3: Inner Sumcheck
- Stage 4: RAM R/W Checking
- Stage 5: Lookup
- Stage 6: Register
- Stage 7: Dory Batched Opening

### End-to-End Verification
Complete Jolt proof verification of Zolt-generated proofs.

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
