# Zolt-Jolt Compatibility TODO

## Completed ✅

### Phase 1-5: Core Infrastructure
1. **Transcript Compatibility** - Blake2b transcript matches Jolt
2. **Proof Structure Refactoring** - 7-stage proof with UniSkip
3. **Serialization Alignment** - Arkworks-compatible serialization
4. **Commitment Scheme** - Dory with Jolt-compatible SRS
5. **Verifier Preprocessing Export** - DoryVerifierSetup exports correctly

### Stage 1 Fixes
6. **Lagrange Interpolation Bug** - Dead code was corrupting basis array
7. **UniSkip Verification** - Domain sum check passes
8. **UnivariateSkip Claim** - Now correctly set to uni_poly.evaluate(r0)
9. **Montgomery Form Fix** - appendScalar now converts from Montgomery form
10. **MontU128Challenge Compatibility** - Challenge scalars now match Jolt's format
11. **Symmetric Lagrange Domain** - Fixed to use {-4,...,5} matching Jolt
12. **Streaming Round Logic** - Separate handling for constraint group selection
13. **MultiquadraticPolynomial** - Already implemented in src/poly/multiquadratic.zig
14. **Multiquadratic Round Polynomial** - Added computeRemainingRoundPolyMultiquadratic()
15. **r0 Not Bound in split_eq** - Jolt uses Lagrange scaling, not binding
16. **Claim Update** - Now converts evaluations to coefficients properly
17. **Factorized Eq Weights** - eq[i] = E_out[i>>5] * E_in[i&0x1F] with bit shifting
18. **getWindowEqTables** - Fixed to match Jolt's E_out_in_for_window logic
19. **Window eq tables sizing** - 32*32=1024 factorization verified correct
20. **ExpandingTable** - Added for incremental eq polynomial computation
21. **Constraint Group Indices** - Fixed to match Jolt's FIRST_GROUP/SECOND_GROUP

---

## Current Status: Stage 1 Sumcheck Output Mismatch ❌

### Observations

- UniSkip passes ✅
- All 11 remaining sumcheck rounds pass (p(0)+p(1)=claim) ✅
- Final output_claim ≠ expected_output_claim ❌

```
output_claim (from sumcheck):    14155882678837957064...
expected_output_claim (from R1CS): 1206125418311750210...
```

### Key Insight

The sumcheck rounds are internally consistent (each p(0)+p(1) = claim), but the polynomial evaluations at challenge points produce wrong intermediate claims.

This means the round polynomial COEFFICIENTS are wrong, even though their sums are correct.

For a degree-3 polynomial:
- p(0) = c_0
- p(1) = c_0 + c_1 + c_2 + c_3
- p(r) = c_0 + c_1*r + c_2*r^2 + c_3*r^3

The constraint p(0) + p(1) = claim doesn't fully determine the polynomial. We need:
- t'(0) = correct value for first half sum
- t'(1) = correct value for second half sum
- t'(∞) = correct quadratic coefficient

If any of these are wrong, p(r) will be wrong even if p(0)+p(1) is correct.

### Likely Root Cause

The split_eq polynomial's `computeCubicRoundPoly` function may not match Jolt's formula exactly.

Jolt's formula uses:
```rust
// l(X) = current_scalar * eq(tau[current_index-1], X)
// q(X) = quadratic factor from Az*Bz
// p(X) = l(X) * q(X)
```

The cubic polynomial has the form:
- c_0 = l(0) * q(0)
- c_1 + c_2 + c_3 = l(1) * q(1) - l(0) * q(0)
- c_3 = l_slope * q_quadratic_coeff (the cubic term)

### Next Steps

1. [ ] Compare computeCubicRoundPoly output with Jolt's equivalent
2. [ ] Verify eq weight computations match exactly
3. [ ] Add detailed logging to trace intermediate values

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
