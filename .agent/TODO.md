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

### Session 7 Analysis (December 29, 2024)

**Key Finding**: The ratio is ~28.25, which is very close to 1024/36 ≈ 28.44

This suggests a scaling issue related to trace_length (1024) and R1CS inputs (36).

**Verified Correct:**
- E_out/E_in factorization: indexes match Jolt's structure
- Lagrange weighting in Az/Bz computation
- split_eq initialization with tau_low and scaling factor
- streaming vs linear phase r_grid updates

**Potential Issues to Investigate:**

1. **Index bit ordering during cycle rounds**
   - Zolt iterates over base_idx and reconstructs cycle_idx_0, cycle_idx_1
   - Need to verify the bit insertion matches Jolt's index structure

2. **remaining_idx calculation in cycle rounds**
   - `remaining_idx = high_bits` (bits above current_bit_pos)
   - May not match Jolt's E_out/E_in indexing after streaming phase

3. **r_grid mask calculation**
   - Zolt uses `cycle_idx_0 & r_grid_mask`
   - Need to verify this matches Jolt's k indexing

4. **Window variable handling**
   - In streaming round, window variable = constraint group selector
   - In cycle rounds, window variable = current cycle bit
   - The transition between these may be incorrect

### Debug Plan

Add debug output to compare:
1. t_zero and t_infinity values each round
2. E_out and E_in table contents
3. r_grid contents after each round
4. current_scalar value after each bind

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
