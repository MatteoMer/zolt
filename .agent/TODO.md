# Zolt-Jolt Compatibility TODO

## Current Status: Session 38 - January 2, 2026

**712 tests pass. Stage 1 sumcheck output_claim mismatch remains.**

---

## Summary of Findings

### What Works ✅
1. EqPolynomial - partition of unity holds, sum equals 1
2. Individual Az and Bz MLE evaluations match between prover and verifier
3. All 712 unit tests pass
4. Proof generation completes successfully

### What Doesn't Work ❌
1. Stage 1 sumcheck output_claim doesn't match expected_output_claim
   - Prover's inner product is ~79.5% of expected
   - Difference: output=21656... vs expected=4977...

### Root Cause Hypothesis

The issue is in how `t_prime_poly` accumulates the product `Az * Bz` across cycles.

Jolt has two materialization paths:
1. `round_zero` - simpler, no r_grid scaling
2. `general` - complex, with r_grid scaling

Zolt has one path that always uses r_grid, but at round 1, r_grid = [1.0], so this should be equivalent.

The indexing formulas use different styles:
- Jolt: `full_idx = grid_size * i + j` (multiplication)
- Zolt: `full_idx = base_idx | x_val | r_idx` (bitwise OR)

These should be mathematically equivalent when bit positions don't overlap, but verification is needed.

---

## Next Steps

1. Add debug logging to Zolt's `buildTPrimePoly` to print:
   - cycle indices accessed
   - Az/Bz values at each index
   - accumulated t_prime values

2. Add similar debug logging to Jolt's `fused_materialise_polynomials_round_zero`

3. Compare the two side-by-side to find the divergence point

4. The ~79.5% ratio (~4/5) suggests a systematic difference, possibly in how constraint groups are weighted

---

## Test Commands

```bash
# All tests pass
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification (fails at Stage 1)
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Key Files

### Zolt
- `src/zkvm/spartan/streaming_outer.zig` - materializeLinearPhasePolynomials, buildTPrimePoly
- `src/poly/split_eq.zig` - getWindowEqTables, computeCubicRoundPoly
- `src/poly/multiquadratic.zig` - MultiquadraticPolynomial

### Jolt (Reference)
- `jolt-core/src/zkvm/spartan/outer.rs` - fused_materialise_polynomials_round_zero
- `jolt-core/src/poly/split_eq_poly.rs` - E_out_in_for_window
