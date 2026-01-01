# Zolt-Jolt Compatibility TODO

## Current Status: Session 31 - January 2, 2026

**All 702 tests pass**

### Issue: Sumcheck round polynomial values don't match verifier expectations

**Root cause identified**: The round polynomial evaluation at each challenge doesn't match what the verifier computes as `expected_output_claim`.

After fixing memory layout constants, the transcript now matches between Zolt and Jolt (same challenges). However, the sumcheck verification fails because:
- Prover output_claim: `18149181199645709635565994144274301613989920934825717026812937381996718340431`
- Expected output_claim: `9784440804643023978376654613918487285551699375196948804144755605390806131527`

### Analysis

The verification shows:
1. ✅ Round 0 polynomial s(0)+s(1) = hint (constraint satisfied)
2. ❌ Round 1 evaluation at r_1: computed `7662922...` vs expected `8918774...`
3. This indicates the polynomial coefficients being generated are incorrect

The issue is in **how Zolt's prover computes the round polynomial evaluations** s(0), s(1), s(2), s(3). The computation structure differs from what the verifier expects.

### Completed Work (This Session)

1. [x] Fixed memory layout constants to match Jolt (128MB memory, 4KB stack)
2. [x] Verified transcript now produces same challenges as Jolt
3. [x] Verified polynomial evaluation formula matches Jolt's `eval_from_hint`
4. [x] Identified that round polynomial VALUES are wrong, not the evaluation formula

### Next Steps

1. [ ] Debug the streaming outer prover's `computeRemainingRoundPoly()` function
2. [ ] Compare t_prime_0 and t_prime_inf values with what Jolt would compute
3. [ ] Verify the Az/Bz polynomial binding produces correct values
4. [ ] Check the split_eq current_scalar and eq factor computation

### Blocking Issues

1. **Dory proof generation panic** - index out of bounds in openWithRowCommitments
   - `params.g2_vec` has length 64 but `current_len` is 128
   - Need to fix SRS loading or verify polynomial degree bounds

## Verified Correct
- [x] Blake2b transcript implementation format
- [x] Field serialization (Arkworks format)
- [x] Memory layout constants now match Jolt
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
- [x] All 702 Zolt tests pass

## Test Commands
```bash
# Zolt tests
zig build test --summary all

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
