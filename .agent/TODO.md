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
9. **Montgomery Form Fix** - appendScalar now converts from Montgomery form
10. **MontU128Challenge Compatibility** - Challenge scalars now match Jolt's format
11. **Symmetric Lagrange Domain** - Fixed to use {-4,...,5} matching Jolt
12. **Streaming Round Logic** - Separate handling for constraint group selection

## MAJOR MILESTONE: Stage 1 UniSkip Claims Match! 🎉🎉

The Stage 1 UniSkip verification now has matching claims.

---

## CURRENT ISSUE: Stage 1 Expected Output Claim Mismatch ❌

The sumcheck polynomial equations pass (p(0) + p(1) = claim for all rounds), but:
```
output_claim != expected_output_claim
```

### The Verification Formula

The verifier computes:
```rust
expected_output_claim = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
```

Where:
- `inner_sum_prod = Az_final * Bz_final` using R1CS input evaluations
- `tau_high_bound_r0` = Lagrange kernel at r0 with tau_high
- `tau_bound_r_tail` = eq polynomial at reversed sumcheck challenges

### Key Issues to Investigate

1. **Gruen Method / Split Eq**
   - The `computeCubicRoundPoly()` method may not integrate correctly
   - The split eq polynomial tables might not be structured correctly for outer sumcheck

2. **Tau Integration**
   - The eq polynomial binding needs to include tau values
   - Current implementation may not correctly combine tau with the sumcheck

3. **Round Polynomial Structure**
   - The polynomial s(X) should equal eq(τ, x) * Az(x) * Bz(x) summed over x
   - Current t_zero/t_infinity computation may miss the tau_high factor

### Debug Output

Latest test shows:
- Stage 1 sumcheck equations all pass
- R1CS input evaluations are being computed
- But final claim doesn't match expected

---

## Progress Indicators

- [x] UniSkip verification passes (domain sum = 0)
- [x] Transcript states match after UniSkip poly append
- [x] UnivariateSkip claim formula is correct
- [x] R1CS input claims correctly computed via MLE
- [x] n_rounds counter matches
- [x] r0 challenge matches
- [x] Stage 1 UniSkip claims match
- [x] Symmetric Lagrange domain (fixed)
- [x] Streaming round logic (added)
- [ ] Stage 1 expected_output_claim matches
- [ ] Stages 2-7 verify
- [ ] Full proof verification passes

---

## Test Commands

```bash
# Run all 632 tests
zig build test --summary all

# Build release
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Run Jolt debug test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture

# Run Jolt full proof test
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Next Steps

1. **Investigate tau integration in sumcheck**
   - How does Jolt integrate tau into the round polynomial computation?
   - Does `split_eq` already handle this?

2. **Compare Gruen method implementation**
   - Read Jolt's `gruen_poly_deg_3` carefully
   - Ensure Zolt's version matches

3. **Trace through a single round**
   - Print intermediate values in both Zolt and Jolt
   - Find where divergence occurs
