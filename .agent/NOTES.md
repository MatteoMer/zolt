# Session 28 Notes - Round 129 Polynomial Investigation

## Summary

Deep investigation into why Stage 5 sumcheck Round 129 polynomial differs between Zolt and Jolt.

## Key Finding: Round 129 Polynomial Differs

Round 128 polynomial coefficients MATCH between Zolt and Jolt.
Round 128 challenge MATCHES.
Round 128 new_claim MATCHES.

But Round 129 polynomial coefficients DIFFER!

**Zolt Round 129 coeff[0] committed (BE):**
`first_8_bytes={ 15 95 34 bc 21 be 6b 10 }`

**Jolt Round 129 coeff[0] (LE):**
`[ca, 46, b3, 41, ce, 2f, 78, d3, ed, cc, 27, 13, ff, eb, 02, 1f, 4e, 99, a1, e4, d5, d7, 40, 5d, 3b, 28, 07, e2, 5c, ba, a1, 11]`

These are completely different values!

## Transcript Verification

Verified that Zolt's appendScalar correctly reverses LE to BE for EVM compatibility (matches Jolt's behavior).

Verified that Round 128 coefficients committed to transcript match Jolt's expected values.

## Analysis

The polynomial for Round 129 depends on:
1. Instance 0 (RegistersValEvaluation) bound polynomial
2. Instance 1 (RamRaClaimReduction) bound polynomial
3. Instance 2 (LookupsReadRaf) bound polynomial
4. Batch coefficients (batch0, batch1, batch2)

Since Round 128 works and Round 129 doesn't, the issue is in how Round 129's polynomial values are computed after binding Round 128's challenge.

## Hypothesis

The polynomial binding for cycle rounds might be incorrect. Specifically:
1. The eq polynomial binding might be wrong
2. The data polynomial binding might be wrong
3. The order of operations might differ from Jolt

## Next Steps

1. Add debug to print Instance 2's polynomial evaluations (sum_evals) for Round 129
2. Compare with what Jolt would compute
3. Check the eq polynomial binding logic in cycle rounds
4. Verify the prefix-suffix decomposition is working correctly

## Files to Investigate

- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig`:
  - Lines 2900-3100: Cycle round polynomial computation
  - Check how `sum_evals` is computed for each evaluation point
  - Check how `full_coeffs` is computed via `finishMlesProductSumFromEvals`

## Test Commands

```bash
# Zolt proof generation with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o logs/zolt_proof_dory.bin --export-preprocessing logs/zolt_preprocessing.bin 2>&1 | tee /tmp/zolt_debug.log

# Jolt verification
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
