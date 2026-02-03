# Zolt-Jolt Compatibility Implementation

## Status: Session 29 - FOUND Instance 1 PhaseCycle1 Issue

## CRITICAL FINDING

**Instance 1 (RamRaClaimReduction) has `eval_1 = 0` for Round 129!**

This is suspicious because:
- `eval_0` (at X=0) is non-zero
- `eval_1` (at X=1) is ZERO
- `eval_2` (at X=2) is non-zero

This suggests the PhaseCycle1 loop might have an issue with the P or Q array indexing.

## Debugging Output Analysis

From `/tmp/zolt_inst.log`:
```
[ZOLT INST1] Round 129: ram_ra_round=17, cycle_round=1
  inst1_evals (Toom) = [
    eval_0: { 110, 217, 198, ... },  # Non-zero
    eval_1: { 0, 0, 0, ... },        # ZERO!
    eval_2: { 182, 27, 123, ... },   # Non-zero
    eval_inf: { 0, 0, 0, ... }       # Zero (expected for degree-2)
  ]
```

## Root Cause Analysis

In PhaseCycle1 (lines 1823-1858 of stage5_prover.zig):
```zig
for (0..half_len) |j| {
    // eval_0 += P[2j] * Q[2j]
    // eval_1 += P[2j+1] * Q[2j+1]  <- ALL zeros?
}
```

For Round 129 (cycle_round=1):
- prefix_size = 16
- current_P_len = 16 >> 1 = 8
- half_len = 4

The loop runs 4 times, but eval_1 remains zero. Possible causes:
1. P_raf/P_rw/P_val arrays at odd indices are all zero
2. Q_raf/Q_rw/Q_val arrays at odd indices are all zero
3. The binding from Round 128 zeroed out odd indices incorrectly

## Next Steps (For Next Session)

1. **Add debug to print P and Q arrays** for cycle_round=1
2. **Check the binding logic** at end of Round 128 for Instance 1
3. **Verify P_raf/P_rw/P_val initialization** at start of PhaseCycle
4. **Compare with Jolt's PhaseCycle implementation**

## Files Modified This Session

1. `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig`:
   - Added debug for Instance 0 coefficients (lines ~1600-1620)
   - Added debug for Instance 1 coefficients (lines ~1893-1910)

2. `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs`:
   - Added debug to print individual instance polynomials for Rounds 128-129

## Test Commands

```bash
# Zolt proof generation with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o logs/zolt_proof_dory.bin --export-preprocessing logs/zolt_preprocessing.bin 2>&1 | tee /tmp/zolt_debug.log

# Search for Instance 1 output
grep -A 15 "ZOLT INST1" /tmp/zolt_debug.log

# Jolt verification
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Component Summary

**Round 129 polynomial = batch0 * Instance0 + batch1 * Instance1 + batch2 * Instance2**

- Instance 0: Appears correct (has non-zero coefficients)
- Instance 1: **SUSPECT** - eval_1 = 0 is unusual
- Instance 2: Not yet analyzed in detail

## Previous Session Tasks (Completed)

- [x] Verify mulHiBigIntU128 correctness
- [x] Verify eq_factor computation formula
- [x] Check r_reduction/r_cycle_prime pairing order
- [x] Identify Round 129 polynomial coefficients differ
- [x] Added debug to print individual instance coefficients
- [x] Found Instance 1 has eval_1 = 0

## Pending Tasks

- [ ] Debug why Instance 1 PhaseCycle1 has eval_1 = 0
- [ ] Check P/Q array values in PhaseCycle1
- [ ] Verify binding logic between rounds for Instance 1
- [ ] Fix the polynomial computation
- [ ] Run full verification test
