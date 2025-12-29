# Zolt-Jolt Compatibility TODO

## Completed ✅

### Phase 1-5: Core Infrastructure
1. **Transcript Compatibility** - Blake2b transcript matches Jolt
2. **Proof Structure Refactoring** - 7-stage proof with UniSkip
3. **Serialization Alignment** - Arkworks-compatible serialization
4. **Commitment Scheme** - Dory with Jolt-compatible SRS
5. **Verifier Preprocessing Export** - DoryVerifierSetup exports correctly

### Stage 1 Fixes (Session 11-14+)
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
24. **Multiquadratic Sum of Products** - t'(∞) = Σ (slope_Az * slope_Bz)
25. **current_scalar double-counting fix** - Only applied in l(X), not t'
26. **r_grid HalfSplitSchedule** - Fixed streaming/linear phase split
27. **Dory MSM length fixes** - Proper padding for row_commitments

---

## Current Status: ~28x Discrepancy in Stage 1

### Session 6 Progress
- ✅ Fixed r_grid to match Jolt's HalfSplitSchedule
  - Streaming phase (rounds 1-5): r_grid updated, total 32 entries
  - Linear phase (rounds 6-11): r_grid frozen
- ✅ Fixed Dory open crashes (MSM length mismatches)
- ✅ Successfully generating Jolt-format proofs with `--jolt-format`
- ✅ All 632 Zolt tests pass

### Remaining Issue
```
output_claim (from sumcheck):    10802353943536118619191613488565009513754763340520674309069454666556780486960
expected_output_claim (from R1CS): 382352852595393953063479719277902514423598561439194590455879357973322296027
Ratio: ~28x
```

### Analysis
The expected output claim is computed as:
```
expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
```

Where:
- `tau_high_bound_r0` = Lagrange kernel L(r0, tau[11]) ✅ verified correct
- `tau_bound_r_tail` = Π eq(tau[i], r[i]) for bound challenges
- `inner_sum_prod` = Σ_cycle Az*Bz without eq weighting

In Zolt:
- `lagrange_tau_r0` is passed as initial scaling to split_eq ✅
- `current_scalar` accumulates bound eq values via `bind()` ✅
- t_zero/t_infinity include E_out * E_in * r_grid weighting

### Possible Causes
1. **E_out/E_in indexing** - May not match Jolt's bit ordering
2. **r_grid vs r_tail** - Streaming phase weighting might be off
3. **Cycle iteration** - current_bit_pos calculation might differ
4. **Window size mismatch** - Linear mode uses window size 1

### Verified Correct
- tau vector matches Jolt
- tau_high = tau[11]
- r0 challenge derivation
- Lagrange kernel computation

### Next Steps
1. [ ] Add debug output to print E_out, E_in, r_grid values
2. [ ] Compare per-cycle eq weights between Zolt and Jolt
3. [ ] Verify bit ordering in cycle iteration
4. [ ] Check if linear mode computes eq differently

---

## Test Commands

```bash
# Generate Jolt-format proof
cd /Users/matteo/projects/zolt
zig build
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Run Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```

---

## Later Tasks

### Stage 2-7 Verification
After Stage 1 exact match, verify remaining stages:
- Stage 2: Outer Product
- Stage 3: Inner Sumcheck
- Stage 4: RAM R/W Checking
- Stage 5: Lookup
- Stage 6: Register
- Stage 7: Dory Batched Opening

### End-to-End Verification
Complete Jolt proof verification of Zolt-generated proofs.
