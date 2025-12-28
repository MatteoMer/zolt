# Zolt-Jolt Compatibility TODO

## Completed ✅

1. **Phase 1: Transcript Compatibility** - Blake2b transcript matches Jolt
2. **Phase 2: Proof Structure Refactoring** - 7-stage proof with UniSkip
3. **Phase 3: Serialization Alignment** - Arkworks-compatible serialization
4. **Phase 4: Commitment Scheme** - Dory with Jolt-compatible SRS
5. **Phase 5: Verifier Preprocessing Export** - DoryVerifierSetup exports correctly
6. **Polynomial Construction** - Lagrange interpolation and L*t1 multiplication verified

## In Progress 🚧

### Issue: R1CS Constraints Not Satisfied

**Root Cause Identified:**

The polynomial domain sum is non-zero because `base_evals` are non-zero, meaning the R1CS constraints are NOT being satisfied.

**Test Evidence:**
- Added test `buildUniskipFirstRoundPoly domain sum is zero when base evals are zero` ✓ PASSES
- This proves the polynomial construction is CORRECT
- The issue is in the witness data or constraint evaluation

**Analysis:**

For a valid R1CS execution:
- Each constraint: `Az(x,y) * Bz(x,y) = 0` for all (x, y) in base window
- Base window evaluations: `t1(y) = Σ_x eq(τ,x) * Az(x,y) * Bz(x,y) = 0`

But in the actual proof:
- `base_evals` are computed as non-zero values
- This means some constraints are NOT satisfied

**Possible Causes:**

1. **Witness values incorrect**: R1CSCycleInputs populated with wrong values
2. **Constraint evaluators wrong**: Az/Bz computation doesn't match Jolt
3. **Constraint definitions wrong**: The constraint conditions/left/right don't match Jolt's

**Next Steps:**

1. Check if R1CSCycleInputs are correctly populated from execution trace
2. Compare constraint definitions with Jolt's UNIFORM_CONSTRAINTS
3. Debug a single cycle: print Az and Bz values for all 10 first-group constraints
4. Verify that for a satisfied constraint, Az=1 implies Bz=0 (or Az=0)

---

## Commands

```bash
# Run tests (632/632 passing)
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

## Recent Commits

- `178232d` - test: add buildUniskipFirstRoundPoly domain sum test
- `62a5675` - test: add interpolation preserves zeros test
- `9346ccd` - refactor: simplify extended Az*Bz evaluation
- `cb406ec` - feat: implement proper Lagrange interpolation for UniSkip
