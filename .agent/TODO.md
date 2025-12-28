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

### Transcript State Matches, But Challenge Derivation Differs

After fixing Montgomery form serialization:
- Transcript states NOW MATCH after appending UniSkip polynomial
- Jolt state: `[51, 28, c0, 92, ab, 81, 34, c6, ...]`
- Zolt state: `5128c092ab8134c6178d3b18...` (same!)

But the r0 challenge values still differ:
- Jolt r0: `3203159906685754656633863192913202159923849199052541271036524843387280424960`
- Zolt r0: ~268 trillion (different!)

### Root Cause

The `n_rounds` counter likely differs between Zolt and Jolt. The round counter is mixed into the Blake2b hash during `challengeBytes32`:
```
hasher() = Blake2b256(state || [0u8; 28] || n_rounds.to_be_bytes())
```

When deriving r0:
- Zolt: n_rounds = 55
- Jolt: n_rounds = ? (likely different)

### Next Step

Compare how n_rounds is incremented between the two implementations:
- Jolt increments in `update_state()` called from `challenge_bytes32()`
- Zolt should match this behavior

The difference might be in how many times `update_state` is called during:
1. Preamble (memory layout, I/O)
2. Commitment appending
3. Tau derivation
4. UniSkip poly appending

---

## Git History (Key Commits)

- `39991b8` - fix: convert from Montgomery form in Blake2b appendScalar
- Previous: fix UnivariateSkip claim to use uni_poly.evaluate(r0)
- `0c5f8c6` - fix: remove dead code in Lagrange interpolation

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

# Run Jolt debug test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```

---

## Progress Indicators

- [x] UniSkip verification passes (domain sum = 0)
- [x] Transcript states match after UniSkip poly append
- [x] UnivariateSkip claim formula is correct (uni_poly.evaluate(r0))
- [x] R1CS input claims correctly computed via MLE
- [ ] n_rounds counter matches between Zolt and Jolt
- [ ] r0 challenge matches
- [ ] Stage 1 remaining rounds verify
- [ ] Stages 2-7 verify
- [ ] Full proof verification passes
