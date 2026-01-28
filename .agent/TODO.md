# Zolt-Jolt Compatibility: Status Update

## Status: DEBUGGING STAGE 4 FINAL CLAIM MISMATCH

## Summary

Zolt can now:
1. Generate proofs for RISC-V programs (`./zig-out/bin/zolt prove`)
2. Verify proofs internally (`./zig-out/bin/zolt verify`) - ALL 6 STAGES PASS
3. Export proofs in Jolt-compatible format (`--jolt-format`)
4. Export preprocessing for Jolt verifier (`--export-preprocessing`)
5. Pass all 714 unit tests ✓
6. **Proof successfully deserializes in Jolt** ✓
7. **Preprocessing successfully deserializes in Jolt** ✓

## Cross-Verification Status

Using pre-built Jolt test binary:

```
✓ test_deserialize_zolt_proof - PASSED
✓ test_load_zolt_preprocessing - PASSED
✗ test_verify_zolt_proof_with_zolt_preprocessing - FAILS at Stage 4
```

## Root Cause Analysis

The sumcheck rounds pass but the final claim check fails because:

```
batched_claim (sumcheck output): { 37, 242, 198, 79, ... }
total_expected (verifier computes): { 31, 29, 72, 150, ... }
```

These values differ completely. Investigation shows:

1. **The sumcheck polynomial computation is correct** - p(0)+p(1) = claim for each round ✓
2. **The r_cycle values are being passed correctly** between stages ✓
3. **BUT: The eq_eval computation in the final claim may be wrong**

The issue is likely in how the eq polynomial is evaluated for the final expected claim:
```rust
// Jolt's expected_output_claim:
eq_eval = EqPolynomial::mle_endian(r_cycle_from_sumcheck, params.r_cycle)
```

Zolt's computation at line 2328:
```zig
const eq_val_be = poly_mod.EqPolynomial(F).mle(r_cycle_sumcheck_be, stage3_r_cycle_be);
```

Need to verify:
1. Both vectors have correct endianness
2. The mle function matches Jolt's mle_endian when endianness matches
3. The r_cycle values are correctly extracted from sumcheck challenges

## Files Generated

- `/tmp/fib_proof.bin` - Zolt native format proof
- `/tmp/zolt_proof_dory.bin` - Jolt-compatible format proof (40KB)
- `/tmp/zolt_preprocessing.bin` - Preprocessing for Jolt verifier (26KB)

## Test Commands

```bash
# Run deserialization test (works)
/home/vivado/projects/jolt/target/debug/deps/jolt_core-14bdfbb6b9004947 \
    zolt_compat_test::tests::test_deserialize_zolt_proof --ignored --nocapture

# Run full verification (fails at Stage 4)
/home/vivado/projects/jolt/target/debug/deps/jolt_core-14bdfbb6b9004947 \
    zolt_compat_test::tests::test_verify_zolt_proof_with_zolt_preprocessing --ignored --nocapture
```

## Next Steps

1. Add debug output to show exact eq_val computation comparison
2. Verify r_cycle_sumcheck_be matches normalize_opening_point output
3. Verify stage3_r_cycle_be matches params.r_cycle
4. Consider if mle vs mle_endian difference matters

## Key Insight

The sumcheck rounds all pass correctly - p(0)+p(1) = claim for each round.
This means the polynomial computation is correct.
The issue is specifically in the **expected_output_claim** computation which
uses the eq polynomial evaluated at the final r_cycle point.

SESSION_ENDING: Have identified the specific mismatch but need more investigation to find the exact cause. The eq_eval computation or r_cycle extraction may have a bug.
