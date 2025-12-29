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

---

## Current Status: Testing r_grid fix

### What's Fixed (Session 6)
- **r_grid update schedule** matches Jolt's HalfSplitSchedule:
  - Streaming phase (rounds 1 to switch_over): r_grid is updated
  - Linear phase (after switch_over): r_grid stays frozen
  - For 11 remaining rounds: switch_over = 5, so r_grid has 32 entries (2^5)

### Blocking Issue: Dory Open Crash
- The `--jolt-format` proof generation crashes in Dory.open()
- The MSM function hits an unreachable
- Need to fix before testing with Jolt

### Next Steps
1. [ ] Fix Dory open crash (MSM issue with evaluation point)
2. [ ] Generate proof with `--jolt-format` flag
3. [ ] Export preprocessing to /tmp/zolt_preprocessing.bin
4. [ ] Run Jolt test to verify Stage 1 output claim matches

---

## Test Commands

```bash
# Generate Jolt-format proof in Zolt
cd /Users/matteo/projects/zolt
zig build -Doptimize=ReleaseFast
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
