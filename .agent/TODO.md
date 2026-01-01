# Zolt-Jolt Compatibility TODO

## Current Status: Session 31 - January 2, 2026

**All 702 tests pass**

### Issue: Sumcheck round polynomial VALUES are incorrect

**Root cause confirmed**: Zolt's `computeRemainingRoundPoly()` computes polynomial evaluations s(0), s(1), s(2), s(3) that:
1. ✅ Satisfy the sumcheck constraint s(0)+s(1) = previous_claim
2. ❌ Produce wrong values when evaluated at the challenge

This means the underlying computation of t_prime_0 and t_prime_inf (from Az/Bz products with eq factors) is incorrect.

### Specific Failure Point

Round 1 verification:
- Polynomial coefficients: [c0, c2, c3] are correctly serialized
- Hint (previous claim): correctly used to recover c1
- Evaluation at r_1:
  - Zolt polynomial evaluates to: `7662922099089815801980439289975920297313075874875519371417932080660811269506`
  - Jolt expects: `8918774265116757036790564405901994162252945042425107189638772988632815931918`
  - These differ because Zolt's s(0), s(1), s(2), s(3) are wrong from the start

### What Needs Fixing

The `computeRemainingRoundPoly()` function in `streaming_outer.zig` computes:
1. t_prime_0 = Σ (E_out * E_in * Az[i] * Bz[i]) evaluated at current variable = 0
2. t_prime_inf = Σ (E_out * E_in * Az[i] * Bz[i]) evaluated at current variable = ∞

Then `computeCubicRoundPoly()` in `split_eq.zig` combines these with the eq factor to get s(0), s(1), s(2), s(3).

Something in this computation chain doesn't match Jolt's prover.

### Completed Work (This Session)

1. [x] Fixed memory layout constants to match Jolt (128MB memory, 4KB stack)
2. [x] Verified transcript produces same challenges as Jolt
3. [x] Verified polynomial evaluation formula (Horner's method) is correct
4. [x] Verified interpolation formula is correct
5. [x] Confirmed the POLYNOMIAL VALUES are wrong, not the evaluation/interpolation

### Next Steps

1. [ ] Add debug output to `computeRemainingRoundPoly()` showing t_prime_0 and t_prime_inf
2. [ ] Compare these values with what Jolt's prover would compute
3. [ ] Check if Az/Bz polynomial binding is correct (linear phase vs streaming phase)
4. [ ] Verify E_out and E_in computation matches Jolt

### Blocking Issues

1. **Dory proof generation panic** - index out of bounds in openWithRowCommitments
   - `params.g2_vec` has length 64 but `current_len` is 128
   - Need to fix SRS loading or verify polynomial degree bounds

## Verified Correct
- [x] Blake2b transcript implementation format
- [x] Field serialization (Arkworks format)
- [x] Memory layout constants match Jolt
- [x] Transcript produces same challenges as Jolt
- [x] UniSkip polynomial generation logic
- [x] R1CS constraint definitions (19 constraints, 2 groups)
- [x] Split eq polynomial factorization (E_out/E_in tables)
- [x] Lagrange kernel L(tau_high, r0)
- [x] MLE evaluation for opening claims
- [x] Gruen cubic polynomial formula (verified mathematically)
- [x] r_cycle computation (big-endian, excluding r_stream)
- [x] eq polynomial factor matches verifier
- [x] ExpandingTable (r_grid) matches Jolt
- [x] DensePolynomial.bindLow() implementation
- [x] Polynomial evaluation formula (Horner's method)
- [x] Interpolation formula (Vandermonde inverse)
- [x] evalsToCompressed produces correct compressed coefficients
- [x] All 702 Zolt tests pass

## Test Commands
```bash
# Zolt tests
zig build test --summary all

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
