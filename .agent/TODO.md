# Zolt-Jolt Compatibility: Status Update

## Status: CROSS-VERIFICATION IN PROGRESS

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

Using pre-built Jolt test binary (`jolt_core-14bdfbb6b9004947`):

```
✓ test_deserialize_zolt_proof - PASSED
  Successfully deserialized: trace_length=256, RAM K=65536, 37 commitments

✓ test_load_zolt_preprocessing - PASSED
  Memory layout and bytecode correctly parsed

✗ test_verify_zolt_proof_with_zolt_preprocessing - FAILS
  "Verification failed: Stage 4 - Sumcheck verification failed"
```

## Stage 4 Issue Analysis

The sumcheck rounds pass, but the final claim check fails:
```
output_claim != expected_output_claim
```

In Jolt's RegistersReadWriteChecking verifier:
1. `normalize_opening_point(sumcheck_challenges)` transforms challenges:
   - Phase 1 challenges (Gruen): reversed
   - Phase 2 challenges (address): reversed
   - Phase 3 challenges (remaining cycle + address): reversed
   - Output: [r_address | r_cycle] in BIG_ENDIAN

2. `expected_output_claim` computes:
   ```
   eq_eval = EqPolynomial::mle_endian(r_cycle_from_sumcheck, params.r_cycle)
   combined = rd_wa * (inc + val) + gamma * rs1_ra * val + gamma^2 * rs2_ra * val
   result = eq_eval * combined
   ```

3. The `mle_endian` function pairs elements positionally when both are same endianness.

## Hypothesis

The issue may be in how Zolt:
- Computes the final `eq_eval` value
- Orders the r_cycle variables
- Computes the `combined` claims formula

## Files Generated

- `/tmp/fib_proof.bin` - Zolt native format proof (11KB)
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

1. Compare Zolt's Stage 4 final claim computation with Jolt's
2. Check `normalize_opening_point` implementation in Zolt
3. Verify r_cycle endianness is correct
4. Check the `combined` formula matches exactly

## Technical Notes

### Jolt's RegistersRWC expected_output_claim formula
```rust
let rd_write_value_claim = rd_wa_claim * (inc_claim + val_claim);
let rs1_value_claim = rs1_ra_claim * val_claim;
let rs2_value_claim = rs2_ra_claim * val_claim;
let eq_eval = EqPolynomial::mle_endian(&r_cycle, &self.params.r_cycle);
let combined = rd_write_value_claim + gamma * (rs1_value_claim + gamma * rs2_value_claim);
result = eq_eval * combined
```

### Variable Order in normalize_opening_point
- Phase 1: First log_T/2 rounds bind cycle vars (reversed)
- Phase 2: Next LOG_K rounds bind address vars (reversed)
- Phase 3: Remaining rounds bind cycle then address vars (both reversed)
- Final: [r_address | r_cycle] concatenated
