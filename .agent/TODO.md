# Zolt-Jolt Compatibility TODO

## Completed

### Phase 1-5: Core Infrastructure
1. **Transcript Compatibility** - Blake2b transcript matches Jolt
2. **Proof Structure Refactoring** - 7-stage proof with UniSkip
3. **Serialization Alignment** - Arkworks-compatible serialization
4. **Commitment Scheme** - Dory with Jolt-compatible SRS
5. **Verifier Preprocessing Export** - DoryVerifierSetup exports correctly

### Stage 1 Fixes (Sessions 11-16)
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

---

## Current Status: ~0.8129 Ratio Discrepancy (Close to 13/16)

### Session 16 (December 29, 2024) - Comprehensive Investigation

**Values:**
- output_claim: 15155108253109715956971809974428807981154511443156768969051245367813784134214
- expected:     18643585735450861043207165215350408775243828862234148101070816349947522058550
- Ratio: 0.8129 (close to 13/16 = 0.8125)

**Verified Matching Components:**
- Constraint group ordering (first: 10 constraints, second: 9 constraints)
- Lagrange weight array usage (both groups use w[0..N])
- Eq table factorization (E_out 5 bits, E_in 5 bits = 32×32 = 1024)
- tau_low/tau_high split (tau_high = last element, tau_low = rest)
- Lagrange kernel formula: K(x,y) = Σ L_i(x)·L_i(y)
- Index mapping: out_idx = i >> head_in_bits, in_idx = i & mask
- r_cycle construction: skip r_stream, reverse for BIG_ENDIAN
- Gruen polynomial: l(X) = eq_0 + (eq_1 - eq_0) * X

**Remaining Investigation:**
The systematic ratio suggests a missing/extra factor somewhere in t'(0) and t'(∞):

1. **Az/Bz evaluation at single cycle** - Compare exact values per constraint
2. **Lagrange weight order** - Verify w[i] maps to correct constraint
3. **Product computation** - Ensure t'(0) = Σ eq * Az_g0 * Bz_g0 exactly

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
