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
26. **r_grid HalfSplitSchedule** - Fixed to match Jolt's streaming/linear phase split
27. **Dory MSM length fixes** - Fixed row_commitments vs v_vec/left_vec length mismatches

---

## Current Status: ~28x Discrepancy in Stage 1

### Session 6 Progress
- Fixed r_grid to use Jolt's HalfSplitSchedule (streaming/linear phase split)
- Fixed Dory open crashes (MSM length mismatches)
- Successfully generating Jolt-format proofs with `--jolt-format`
- Proof values changed after r_grid fix (confirming fix is applied)

### Remaining Issue
- `output_claim = 10802353943536118619191613488565009513754763340520674309069454666556780486960`
- `expected_claim = 382352852595393953063479719277902514423598561439194590455879357973322296027`
- Ratio: ~28x (was ~10x before r_grid fix)

### Key Observations from Jolt Test
- `inner_sum_prod` = sum over cycles of Az*Bz weighted by eq
- `tau_high_bound_r0` = eq(tau[11], r0) for constraint group selection
- `tau_bound_r_tail` = product of eq values for bound cycle challenges
- Expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod

### Possible Causes
1. The `inner_sum_prod` computation in Zolt may have different cycle indexing
2. The eq polynomial weights (E_out, E_in, r_grid) may be applied differently
3. The tau/r_tail values may be used in wrong order

### Next Steps
1. [ ] Compare intermediate values: E_out, E_in, r_grid at each round
2. [ ] Verify tau is used correctly in split_eq initialization
3. [ ] Check if cycle indexing matches between Zolt and Jolt
4. [ ] Consider adding detailed debug output to trace eq weights per cycle

---

## Test Commands

```bash
# Generate Jolt-format proof in Zolt
cd /Users/matteo/projects/zolt
zig build
./zig-out/bin/zolt prove examples/sum.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Run cross-verification in Jolt
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```

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
