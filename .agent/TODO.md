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

## Current Status: Still Failing ❌

### Latest Values

```
output_claim (from sumcheck):    7413306969080172833518326335080771468018799697078015495953466550648276143147
expected_output_claim (from R1CS): 12608356760442883687804722211164748832204938436522852816835407207687023899913
Match: false
```

The output_claim changed (was 11612...), indicating the multiquadratic computation is having an effect, but still not matching.

### Likely Issues

1. **Eq Weight Indexing**
   - Currently indexing eq_tables.E_out by cycle index
   - May need different indexing for first/second half

2. **Streaming Round Handling**
   - Round 0 uses old method (constraint group selection)
   - May need special handling

3. **Window Size**
   - Jolt uses window-based streaming
   - Current implementation processes one cycle at a time

---

## Test Commands

```bash
zig build test --summary all
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin

cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
