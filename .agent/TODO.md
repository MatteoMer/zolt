# Zolt-Jolt Compatibility TODO

## Completed ✅

1. **Phase 1: Transcript Compatibility** - Blake2b transcript matches Jolt
2. **Phase 2: Proof Structure Refactoring** - 7-stage proof with UniSkip
3. **Phase 3: Serialization Alignment** - Arkworks-compatible serialization
4. **Phase 4: Commitment Scheme** - Dory with Jolt-compatible SRS
5. **Phase 5: Verifier Preprocessing Export** - DoryVerifierSetup exports correctly
6. **Fix Lagrange Interpolation Bug** - Dead code was corrupting basis array
7. **Stage 1 UniSkip Verification** - Domain sum check passes
8. **UnivariateSkip Claim** - Now correctly set to uni_poly.evaluate(r0)

## MAJOR MILESTONE: Stage 1 UniSkip PASSES! 🎉

The Stage 1 UniSkip first-round verification passes:
```
Domain sum check:
  Input claim (expected domain sum): 0
  Computed domain sum: 0
  Sum equals input_claim: true

✓ Stage 1 UniSkip verification PASSED!
```

## In Progress 🚧

### Stage 1 Regular Sumcheck Fails

After UniSkip passes, the remaining sumcheck rounds fail:
```
Verification failed: Stage 1
Caused by: Sumcheck verification failed
```

### Root Cause Analysis

The `StreamingOuterProver` in Zolt produces round polynomials, but:
1. These polynomials don't reduce to the correct `expected_output_claim`
2. The verifier computes `expected_output_claim` from R1CS input evaluations
3. The prover's round polynomials must be mathematically consistent

### What Works Now

- UnivariateSkip claim = `uni_poly.evaluate(r0)` ✅
- R1CS input evaluations computed via MLE ✅
- Opening claims have correct non-zero values ✅
- Round polynomials have non-zero coefficients ✅

### What Needs Fixing

The round polynomials need to satisfy the sumcheck relation:
1. `p(0) + p(1) = claim` for each round
2. After all rounds, `output_claim` must equal `expected_output_claim`
3. The `expected_output_claim` is:
   ```
   tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
   ```
   where `inner_sum_prod = Az(rx) * Bz(rx)`

### Required Changes in streaming_outer.zig

1. Fix `GruenSplitEqPolynomial` binding logic
2. Fix `computeRemainingRoundPoly` to compute correct t'(0), t'(∞)
3. Track `current_claim` updates correctly
4. Ensure Az*Bz products use properly bound Lagrange coefficients

---

## Git History (Key Commits)

- Latest: fix UnivariateSkip claim to use uni_poly.evaluate(r0)
- `0c5f8c6` - fix: remove dead code in Lagrange interpolation
- `178232d` - test: add buildUniskipFirstRoundPoly domain sum test
- `62a5675` - test: add interpolation preserves zeros test
- `cb406ec` - feat: implement proper Lagrange interpolation for UniSkip

---

## Test Commands

```bash
# Run all 632 tests
zig build test --summary all

# Build release
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/sum.elf --jolt-format \
    --export-preprocessing /tmp/zolt_preprocessing.bin \
    -o /tmp/zolt_proof_dory.bin

# Run Jolt debug test (Stage 1 sumcheck structure)
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_sumcheck -- --ignored --nocapture

# Run Jolt full verification test
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

---

## Progress Indicators

- [x] UniSkip verification passes
- [x] UnivariateSkip claim correctly set
- [x] R1CS input claims correctly computed
- [ ] Stage 1 remaining rounds verify
- [ ] Stage 2 UniSkip verification
- [ ] Stage 2 remaining rounds verify
- [ ] Stages 3-7 verify
- [ ] Full proof verification passes
