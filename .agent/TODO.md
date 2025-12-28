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

### Verified Components

- ✅ Factorized eq tables: E_out.len=32, E_in.len=32, head_in_bits=5
- ✅ Coverage: 32*32=1024 cycles
- ✅ Bit shifting for indexing: i >> 5, i & 0x1F
- ✅ Gruen polynomial sumcheck constraint: s(0) + s(1) = previous_claim
- ✅ UniSkip polynomial serialization to transcript
- ✅ Lagrange kernel L(tau_high, r0) initialization
- ✅ current_scalar used correctly in computeCubicRoundPoly

### Remaining Issues to Investigate

1. **Az/Bz computation per cycle** - The constraint evaluations (Az and Bz) need verification against Jolt's `R1CSEval::from_cycle_inputs`.

2. **Constraint group separation** - Jolt uses a selector bit (`full_idx & 1`) to separate 19 constraints into two groups. Zolt might be computing groups differently.

3. **r_grid pattern** - Jolt uses an expanding r_grid for bound challenge accumulation, though for the streaming round it's just [1]. This may affect later rounds.

4. **Tau ordering in eq evaluation** - The pairing between tau elements and challenges during binding may be subtly different.

---

## Next Steps

1. Add debug output to compare specific values:
   - t'(0) and t'(1) for streaming round
   - Az and Bz for individual cycles
   - eq weights for specific cycle indices

2. Compare first round polynomial coefficients between Zolt and Jolt

3. Verify constraint evaluation formulas match exactly

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
