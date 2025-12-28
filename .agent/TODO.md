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
13. **MultiquadraticPolynomial** - Already implemented in src/poly/multiquadratic.zig
14. **Multiquadratic Round Polynomial** - Added computeRemainingRoundPolyMultiquadratic()
15. **r0 Not Bound in split_eq** - Jolt uses Lagrange scaling, not binding
16. **Claim Update** - Now converts evaluations to coefficients properly
17. **Factorized Eq Weights** - eq[i] = E_out[i/E_in.len] * E_in[i%E_in.len]
18. **getWindowEqTables** - Fixed to match Jolt's E_out_in_for_window logic

---

## Current Status: Test Runs But Verification Fails ❌

The Jolt cross-verification test runs without crashing, but **Stage 1 sumcheck verification fails**.

### Latest Test Output (2024-12-28)

```
output_claim (from sumcheck):    18276273718881795230055698751155029867368575945533636768878307520109710274366
expected_output_claim (from R1CS): 12713752026862396773059828773864138237897786023884325663325072498443735660914
Match: false
```

### Remaining Issues to Investigate

1. **Eq polynomial indexing** - The factorized eq approach is implemented, but there may be an off-by-one or ordering issue

2. **Binding order mismatch** - The tau elements may be indexed differently than the sumcheck challenges

3. **current_index tracking** - The split_eq.current_index tracks how many variables are unbound, but this may not be updated correctly

4. **Az/Bz computation** - The constraint evaluations may differ from Jolt's expectations

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
