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

---

## Current Status: Test Runs But Verification Fails ❌

The Jolt cross-verification test runs without crashing, but **Stage 1 sumcheck verification fails**.

### Latest Test Output (2024-12-28)

```
Verification failed: Stage 1

Caused by:
    Sumcheck verification failed
```

Debug test shows:
```
output_claim (from sumcheck):      15494770952016151805100679853636026761384296060550815442753024133495164390908
expected_output_claim (from R1CS): 1634089052370213054875543818155028295629058109509108648770993566620247586716
Match: false
```

### Next Priority: Debug Stage 1 Sumcheck Mismatch

The round polynomials are being generated but the final claim doesn't match what the verifier expects from R1CS evaluation.

Likely areas to investigate:
1. **Eq weight indexing** - How E_out is indexed for first/second half cycles
2. **Streaming round (round 0)** - Uses `computeRemainingRoundPoly()`, cycle rounds use multiquadratic
3. **Gruen split eq factorization** - May have subtle differences from Jolt's implementation
4. **tau_low vs tau ordering** - Verify tau vector is constructed identically to Jolt

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
