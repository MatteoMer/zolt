# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (sumcheck output_claim mismatch)
- Stage 3+: Not reached yet

### Latest Debug Output (Round 16 - when ProductVirtualRemainder starts)
```
combined_evals[0] = {8, 220, 60, ...}
combined_evals[1] = {45, 241, 181, ...}  <- Different from [0], good!
compressed[2] (c3) = {34, 60, 60, ...}    <- Non-zero cubic, good!
```

The polynomial structure at round 16 looks correct - varying evaluations and non-zero coefficients.

### Tests
- All 712 Zolt tests pass
- Jolt verification failing at Stage 2 sumcheck

## Completed Fixes

### Session Fixes
1. **tau_stage2 construction** - Uses reversed Stage 1 challenges + tau_high_stage2
2. **Gruen polynomial construction** - Uses t0/t_inf method matching Jolt
3. **Compressed coefficient handling** - Correctly reconstructs evaluations from [c0, c2, c3]
4. **Variable shadowing fix** - Removed duplicate `const half` declaration

## Current Architecture

### Stage 2 Batched Sumcheck (26 rounds total)
- **Rounds 0-15**: Only ProductVirtualRemainder contributes (as scaled constant)
- **Round 16+**: ProductVirtualRemainder becomes active (computes real polynomials)
- **Other 4 instances**: Zero claims (valid for simple programs)

### Key Computations
1. `t0 = Σ eq * left_lo * right_lo` (constant coefficient)
2. `t_inf = Σ eq * (left_hi - left_lo) * (right_hi - right_lo)` (quadratic coefficient)
3. `computeCubicRoundPoly(t0, t_inf, current_claim)` produces evaluations

### Expected Output Claim Formula (Jolt Verifier)
```
L(tau_high, r0) * eq(tau_low, r_reversed) * fused_left(claims) * fused_right(claims)
```

## Next Investigation Steps

1. **Compare t0/t_inf values** - Add logging in both Zolt and Jolt for round 16
2. **Compare eq polynomial evaluation** - Verify E_out and E_in tables match
3. **Check current_scalar progression** - The accumulated eq value should match
4. **Verify factor claims match** - The 8 MLE evaluations at r_cycle

## Commands

```bash
# Build and test
zig build test --summary all

# Generate proof with debug output
./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/fibonacci.elf

# Test with Jolt
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

## Files of Interest

- `src/zkvm/proof_converter.zig` - Stage 2 batched sumcheck generation
- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainder prover
- `src/poly/split_eq.zig` - Gruen polynomial and eq tables
