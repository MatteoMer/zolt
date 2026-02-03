# Session 29 Notes - Instance 1 PhaseCycle1 Investigation

## Summary

Identified that Instance 1 (RamRaClaimReduction) has `eval_1 = 0` for Round 129, which is causing the Round 129 polynomial mismatch.

## Key Finding

The PhaseCycle1 loop computes:
- `eval_0 = Σ coeff_raf * P_raf[2j] * Q_raf[2j] + coeff_rw * P_rw[2j] * Q_rw[2j] + coeff_val * P_val[2j] * Q_val[2j]`
- `eval_1 = Σ coeff_raf * P_raf[2j+1] * Q_raf[2j+1] + ...`

For Round 129 (cycle_round=1):
- `eval_0` is non-zero
- `eval_1` is ZERO (suspicious!)
- `eval_2` is non-zero

This means all the products at odd indices (2j+1) sum to zero.

## Possible Causes

1. **P arrays at odd indices are zero**: After binding from Round 128, the P_raf/P_rw/P_val arrays might have zeros at odd indices.

2. **Q arrays at odd indices are zero**: The Q_raf/Q_rw/Q_val arrays might have zeros at odd indices.

3. **Binding logic issue**: The binding at the end of Round 128 might be incorrectly zeroing out values.

4. **Indexing issue**: The P and Q arrays might be indexed differently than expected.

## Debug Added

Added debug to print P and Q array values in PhaseCycle1 for cycle_round=1:
```zig
if (cycle_round == 1 and round == LOOKUPS_LOG_K + 1) {
    std.debug.print("[INST1 PQ DEBUG] Round {} (cycle_round={}): half_len={}\n", .{ round, cycle_round, half_len });
    for (0..@min(8, current_P_len)) |j| {
        std.debug.print("  P_raf[{}]={x}, Q_raf[{}]={x}\n", .{
            j, P_raf[j].toBytesBE()[24..32].*,
            j, Q_raf[j].toBytesBE()[24..32].*,
        });
    }
}
```

## Next Steps

1. Run Zolt and capture the P/Q debug output
2. Analyze which arrays have zeros
3. Check the binding logic at end of Round 128
4. Compare with Jolt's PhaseCycle implementation

## Files Modified

- `src/zkvm/spartan/stage5_prover.zig`: Added debug for Instance 0/1 coefficients and P/Q arrays
- `jolt/jolt-core/src/subprotocols/sumcheck.rs`: Added debug for individual instance polynomials

## Test Commands

```bash
# Run Zolt and capture P/Q debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o logs/zolt_proof_dory.bin --export-preprocessing logs/zolt_preprocessing.bin 2>&1 | grep "INST1 PQ DEBUG" -A 20

# Verify with Jolt
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
