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

## MAJOR MILESTONE: Stage 1 UniSkip Claims Match! 🎉🎉

The Stage 1 UniSkip verification now has matching claims:
```
r0 and r0_fr evaluations match: true
Claims match: true
✓ Stage 1 UniSkip verification PASSED!
```

### Key Fix: MontU128Challenge-Compatible Arithmetic

Jolt uses `MontU128Challenge` for 128-bit challenges, which stores values as `[0, 0, low, high]`
in a BigInt (NOT in Montgomery form). When used in polynomial evaluation, this raw BigInt
is multiplied with Montgomery-form coefficients.

The fix required:
1. **Challenge Format**: Store challenges as `[0, 0, low, high]` to match Jolt's `from_bigint_unchecked`
2. **Polynomial Evaluation**: Use raw BigInt multiplication (not Montgomery conversion)
   so `REDC(raw * mont)` produces correct results
3. **Byte Interpretation**: Interpret reversed bytes as big-endian u128 (matching Rust's `u128::from_be_bytes`)

---

## Progress Indicators

- [x] UniSkip verification passes (domain sum = 0)
- [x] Transcript states match after UniSkip poly append
- [x] UnivariateSkip claim formula is correct (uni_poly.evaluate(r0))
- [x] R1CS input claims correctly computed via MLE
- [x] n_rounds counter matches between Zolt and Jolt
- [x] r0 challenge matches
- [x] Stage 1 UniSkip claims match
- [ ] Stage 1 remaining rounds verify
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
```

---

## Next Steps

1. Verify Stage 1 remaining sumcheck rounds
2. Implement Stage 2 verification test
3. Continue through Stages 3-7
4. Full proof verification
