# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (sumcheck output_claim mismatch)
- Stage 3+: Not reached yet

## Current Issue
Stage 2 sumcheck output_claim doesn't match expected_output_claim:
- `output_claim`: 7092315906761387499155192332297668483214634895487230792733107015466920310794
- `expected_output_claim`: 11535186225949250426807989625067498736367003469117527002588759500768361489976

## Recent Fixes (Session 5)
1. ✅ Fixed EqPolynomial.evals to iterate over odd indices (matching Jolt's rev().step_by(2))
2. ✅ Fixed computeRoundPolynomial to use interleaved format (adjacent pairs: 2*g and 2*g+1)
3. ✅ All 712 tests pass

## Expected Output Claim Formula (Jolt Verifier)
```
expected = L(τ_high, r0) * Eq(τ_low, r_tail_reversed) * fused_left * fused_right
```

Where:
- `fused_left = w[0]*l_inst + w[1]*is_rd_not_zero + w[2]*is_rd_not_zero + w[3]*lookup_out + w[4]*j_flag`
- `fused_right = w[0]*r_inst + w[1]*wl_flag + w[2]*j_flag + w[3]*branch_flag + w[4]*(1 - next_is_noop)`
- `w` = Lagrange weights at r0

## Key Files
- `src/zkvm/proof_converter.zig` - Stage 2 batched sumcheck generation
- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainder prover
- `src/poly/split_eq.zig` - Gruen polynomial and eq tables
- `src/poly/mod.zig` - EqPolynomial and UniPoly

## Investigation Areas
1. **Sumcheck round polynomials** - Are we computing correct t0/t_inf values?
2. **E_out/E_in tables** - Do they match Jolt's tables exactly?
3. **Gruen polynomial** - Is computeCubicRoundPoly producing correct evaluations?
4. **First round handling** - Jolt uses first_round_evals pre-computed differently

## Commands
```bash
# Build and test
zig build test --summary all

# Generate proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/fibonacci.elf

# Test with Jolt
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
