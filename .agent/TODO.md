# Zolt-Jolt Compatibility TODO

## Completed

### Phase 1-5: Core Infrastructure
1. **Transcript Compatibility** - Blake2b transcript matches Jolt
2. **Proof Structure Refactoring** - 7-stage proof with UniSkip
3. **Serialization Alignment** - Arkworks-compatible serialization
4. **Commitment Scheme** - Dory with Jolt-compatible SRS
5. **Verifier Preprocessing Export** - DoryVerifierSetup exports correctly

### Stage 1 Fixes (Sessions 11-18)
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
30. **r0 not in challenges** - r0 should NOT be added to the challenges list
31. **Debug test for streaming round** - Added test to inspect intermediate values
32. **Big-Endian Eq Tables** - E_out/E_in tables now use big-endian indexing
33. **Split Logic** - Simplified to match Jolt's exact split (m = tau.len/2, skip last element)

---

## Current Status: ~1.2 Ratio Discrepancy

### Session 18 (December 29, 2024) - Investigation Continues

**Current Values:**
- output_claim: 17544243885955816008056628262847401707989885215135853123958675606975515887014
- expected:     14636075186748817511857284373650752059613754347411376791236690874143105070933
- Ratio: ~1.1987 (consistent across multiple runs)

**Verified Matching Components:**
1. gruen_poly_deg_3 / computeCubicRoundPoly formula ✓
2. eq_eval_0, eq_eval_1 computation ✓
3. q_2 and q_3 formulas ✓
4. inner_sum_prod = az_final * bz_final ✓
5. tau_high_bound_r0 and tau_bound_r_tail match Jolt's formulas ✓
6. Split logic: m = tau.len/2, w_last skipped ✓
7. current_index = tau.len ✓

**Key Insight from Jolt Code:**

In Jolt, for streaming round with window_size=1:
- `num_unbound = current_index = 11` (tau_low.len)
- `head_len = 11 - 1 = 10`
- `m = 11 / 2 = 5`
- `head_out_bits = min(10, 5) = 5` → E_out has 32 entries
- `head_in_bits = 10 - 5 = 5` → E_in has 32 entries
- Total: 32 * 32 = 1024 cycles ✓

**Next Investigation Steps:**

1. **Add debug output to streaming round**
   - Print t_zero and t_infinity before computeCubicRoundPoly
   - Print round polynomial coefficients
   - Compare with Jolt's values

2. **Trace through claim propagation**
   - Initial claim = uni_skip_claim
   - Each round: new_claim = poly.evaluate(challenge)
   - Final claim should match expected

3. **Check if there's an off-by-one in cycle iteration**
   - Zolt: iterate 0..padded_trace_len
   - Jolt: i = out_idx * e_in_len + in_idx
   - These should be equivalent

---

## Test Commands

```bash
# Run Zolt tests (all 656)
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
