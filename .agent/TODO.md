# Zolt-Jolt Compatibility TODO

## Current Status: Session 60 - January 6, 2026

**STATUS: Round 0 polynomial correct, Round 1+ polynomials WRONG**

### Progress This Session

1. ✅ Fixed instruction input semantics (`computeInstructionInputs`)
   - LUI/AUIPC now use sign-extended U-type immediates
   - Matches Jolt's FormatU::parse sign extension behavior

2. ✅ Fixed r1cs_input_evals - now match Jolt PERFECTLY
   - r1cs_input_evals[0] matches ✓
   - r1cs_input_evals[1] matches ✓
   - r1cs_input_evals[2] matches ✓

3. ✅ Round 0 polynomial is CORRECT
   - All coefficients (c0, c2, c3) match Jolt
   - Challenge matches
   - next_claim matches

4. ❌ Round 1+ polynomial computation is WRONG
   - Round 1 s(0) correct, but s(1), s(2), s(3) wrong
   - Sumcheck constraint `s(0) + s(1) = previous_claim` NOT satisfied

### Root Cause Analysis

**The Bug:**
```
Zolt computes: s(0) + s(1) = 2127849171702515242032736530567016046776512160048301414144827652734113940861
Expected:      s(0) + s(1) = 18977235297014647440385331877696957720245438231586084249713843600848329563216
```

The polynomial at Round 1 doesn't satisfy the sumcheck constraint!

**Where the bug is:**

The polynomial is computed by:
1. `computeTEvals()` returns `(t_zero, t_infinity)` from bound t_prime polynomial
2. `split_eq.computeCubicRoundPoly(t_zero, t_infinity, previous_claim)` constructs s(X)

The issue must be one of:
- t_prime polynomial isn't being bound correctly after round 0
- eq tables aren't updated correctly
- The `computeTEvals` function extracts wrong values

### Key Files

- `src/zkvm/spartan/streaming_outer.zig:computeRemainingRoundPoly` - Polynomial computation
- `src/zkvm/spartan/streaming_outer.zig:bindRemainingRoundChallenge` - Binding logic
- `src/zkvm/spartan/streaming_outer.zig:computeTEvals` - t_zero/t_infinity extraction
- `src/poly/split_eq.zig:computeCubicRoundPoly` - Gruen polynomial construction

### Previous Session Progress

- Sessions 51-58: Various fixes (batching, round offset, transcript, UniSkip)
- Session 59: Verified sumcheck polynomials match, identified r1cs_input_evals mismatch

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
