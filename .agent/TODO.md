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

---

## Current Status: Stage 1 Sumcheck Mismatch ❌

The Jolt cross-verification runs but Stage 1 sumcheck output_claim doesn't match expected.

```
output_claim (from sumcheck):      9328088438419821762178329852958014809003674147304165221608390320629231184085
expected_output_claim (from R1CS): 15770715866241261093869584783304477941139842654876630627419092129570271411009
```

### Key Observations

1. **UniSkip passes** - Domain sum is zero, polynomial evaluation correct
2. **All 11 sumcheck rounds pass** - p(0) + p(1) = claim for each round
3. **Final claim mismatch** - The output doesn't match expected

### Root Cause Analysis (In Progress)

The expected_output_claim formula is:
```
expected = L(tau_high, r0) * eq(tau_low, r_reversed) * Az(rx_constr, z) * Bz(rx_constr, z)
```

Where:
- `L(tau_high, r0)` = Lagrange kernel from UniSkip ✅
- `eq(tau_low, r_reversed)` = eq polynomial at reversed challenges
- `Az`, `Bz` = constraint evaluations using R1CS input MLE evaluations

The sumcheck computes:
```
output = eq_factor * Σ_c eq(r, c) * Az(c) * Bz(c)
```

**Key finding:** The verifier expects:
- `Az(z) * Bz(z)` where z = MLE evaluations
- = `(Σ_c eq(r,c) * Az_c) * (Σ_c eq(r,c) * Bz_c)`

But the sumcheck produces:
- `Σ_c eq(r,c) * Az_c * Bz_c`

These are mathematically different expressions! However:
- For satisfied R1CS constraints: `Az_c * Bz_c = 0` for all c
- This means both expressions should be 0

### Likely Issues

1. **Missing r_grid**: Jolt uses `ExpandingTable` (r_grid) to weight cycles by bound challenges. Zolt doesn't have this.

2. **Constraint evaluation formula**: The Az/Bz computation might differ from Jolt's `R1CSEval`.

3. **MLE evaluation point**: The r_cycle point for MLE evaluations might be computed differently.

---

## Next Steps

1. [ ] Add ExpandingTable equivalent to Zolt's streaming prover
2. [ ] Verify Az/Bz computation matches Jolt's R1CSEval exactly
3. [ ] Add debug output comparing intermediate values
4. [ ] Check r_cycle point computation and endianness

---

## Test Commands

```bash
# Generate proof in Zolt
cd /Users/matteo/projects/zolt
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Run cross-verification in Jolt
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# Run detailed debug test
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```

---

## Technical Reference

### Jolt Files
- `jolt-core/src/zkvm/spartan/outer.rs` - Outer sumcheck prover/verifier
- `jolt-core/src/utils/expanding_table.rs` - r_grid implementation
- `jolt-core/src/poly/split_eq_poly.rs` - GruenSplitEqPolynomial
- `jolt-core/src/zkvm/r1cs/key.rs` - evaluate_inner_sum_product_at_point

### Zolt Files
- `src/zkvm/spartan/streaming_outer.zig` - Streaming outer prover
- `src/poly/split_eq.zig` - GruenSplitEqPolynomial
- `src/zkvm/r1cs/constraints.zig` - Constraint evaluation
