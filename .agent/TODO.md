# Zolt-Jolt Compatibility TODO

## Completed ✅

### Phase 1-5: Core Infrastructure
1. **Transcript Compatibility** - Blake2b transcript matches Jolt
2. **Proof Structure Refactoring** - 7-stage proof with UniSkip
3. **Serialization Alignment** - Arkworks-compatible serialization
4. **Commitment Scheme** - Dory with Jolt-compatible SRS
5. **Verifier Preprocessing Export** - DoryVerifierSetup exports correctly

### Stage 1 Fixes (Sessions 11-15+)
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
28. **Jolt Index Structure** - Use full_idx = x_out|x_in|x_val|r_idx, step_idx = full_idx >> 1
29. **Selector from full_idx** - Use selector = full_idx & 1 for constraint group in cycle rounds

---

## Current Status: ~1.23x Discrepancy in Stage 1

### Session 16 (December 29, 2024)

**Values:**
- output_claim: 15155108253109715956971809974428807981154511443156768969051245367813784134214
- expected:     18643585735450861043207165215350408775243828862234148101070816349947522058550
- Ratio: 0.813 (output/expected ≈ 13/16)

**Fixed This Session:**
30. **r0 not in challenges** - r0 should NOT be added to the challenges list
   - challenges = [r_stream, r_1, ..., r_10] (11 elements)
   - cycle_challenges = challenges[1..] = [r_1, ..., r_10] (10 elements)

**Still Failing:**
- Even after fix, output_claim != expected_claim
- The sumcheck polynomials produce different final claim than R1CS expects

**Verified Components (All Match Jolt):**
- ExpandingTable update: `values[i] = (1-r)*old`, `values[i+len] = r*old`
- eq table factorization: E_out 5 bits (32 entries), E_in 5 bits (32 entries)
- Lagrange domain: {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5} for 10 constraints
- Multiquadratic expansion: [f(0), f(1), f(∞)=slope]
- Compressed poly format: [c0, c2, c3] (linear term omitted)
- Index mapping: out_idx = i >> 5, in_idx = i & 31
- Streaming round: Both constraint groups for same cycle

**Key Insights:**
- Jolt expected_output_claim formula:
  expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
  where inner_sum_prod = (az_g0 + r_stream*(az_g1-az_g0)) * (bz_g0 + r_stream*(bz_g1-bz_g0))
- r_tail = sumcheck_challenges.reversed()

**Next Investigation Areas:**
1. Per-round polynomial values - compare actual s(0), s(1) with expected
2. Transcript consistency - are challenges the same as Jolt computes?
3. Input claim (uni_skip_claim) value
4. Add debugging to first few rounds to find divergence point

---

## Test Commands

```bash
# Run Zolt tests (all 632)
zig build test --summary all

# Generate Jolt-format proof
zig build -Doptimize=ReleaseFast
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
