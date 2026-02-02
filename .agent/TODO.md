# Zolt-Jolt Compatibility Implementation

## Status: Session 15 - Stage 5 InstructionReadRaf Polynomial Mismatch

## Current Issue

Stage 5 verification fails with:
- `output_claim` (from Zolt's sumcheck): `[63, 47, 60, 9d, ...]`
- `expected_claim` (from Jolt's verifier): `[a9, 4a, 89, dd, ...]`

**Opening claims MATCH between Zolt and Jolt** - verified by debug output:
- `ra_claims[0..7]` - match
- `table_flags[0,1,9]` - match
- `raf_flag_claim` - match

**The mismatch is in the polynomial computation during sumcheck**, not the opening claims.

## Root Cause Analysis

### Jolt's expected_output_claim Formula

```rust
expected = eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
```

Where:
- `eq_eval_r_reduction = eq(r_reduction, r_cycle_prime)` - verifier computes
- `ra_claim` = product of InstructionRa(i) opening claims
- `val_claim = Σ table_flag[i] * table_mle[i](r_address_prime)` - verifier evaluates table MLEs
- `raf_claim = (1 - raf_flag) * (left + γ*right) + raf_flag * γ * identity`

### The Key Discrepancy

The sumcheck polynomial that Zolt computes during the 136 rounds (using prefix-suffix decomposition) produces a different final value than what Jolt computes using:
1. Opening claims from the proof
2. Direct evaluation of table MLEs at r_address_prime

The prefix-suffix decomposition should produce the SAME value as direct MLE evaluation, but something is causing them to differ.

### Jolt Debug Output - Table MLE Values

```
[InstructionReadRaf] Table MLE evaluations at r_address_prime:
  table_eval[0] (FULL 32) = [4d, f2, f9, 55, ...]
  table_eval[1] (FULL 32) = [11, c4, e8, 84, ...]
  table_eval[9] (FULL 32) = [29, f0, 99, fa, ...]

val_claim: [83, cd, b7, 55, ...]  (= Σ table_flag[i] * table_eval[i])
raf_claim: [13, c2, 5c, be, ...]
final_result (expected for inst2): [b6, fd, fb, 64, ...]
```

### Potential Issues

1. **Prefix-suffix decomposition**:
   - `proverMsgReadChecking` uses prefix checkpoints and suffix Q arrays
   - Must produce same value as Jolt's `table.evaluate_mle(r_address_prime)`

2. **Prefix checkpoint updates**:
   - After 128 rounds, checkpoints should contain evaluated prefix values
   - Any error cascades through all later computations

3. **Challenge binding/normalization**:
   - Sumcheck challenges are bound in little-endian order
   - r_address_prime needs big-endian for MLE evaluation
   - Verify all conversions are correct

## Files to Investigate

1. `src/zkvm/lookup_table/prefix_suffix_prover.zig`:
   - `proverMsgReadChecking` - computes val contributions via prefix-suffix
   - `tableCombine` - combines prefix/suffix for each table type
   - Compare output with Jolt's direct MLE evaluation

2. `src/zkvm/lookup_table/prefixes.zig`:
   - Prefix checkpoint state and updates
   - Each prefix type (LowerWord, Eq, LessThan, etc.)

3. `src/zkvm/spartan/stage5_prover.zig`:
   - Lines 2011-2110: Instance 2 polynomial computation
   - How challenges are passed to prefix-suffix functions

## Next Steps

1. **Add debug to compare table MLE values**:
   - After all 128 address rounds, compute final table values from prefix-suffix
   - Compare with Jolt's table_mle[i](r_address_prime)

2. **Trace RangeCheck table (table 0)**:
   - Simplest table: `LowerWord * one + lower_word`
   - Compare suffix Q values, prefix checkpoint, combined result

3. **Verify LowerWord prefix checkpoint**:
   - After 128 rounds, should equal lower 64 bits of r_address
   - Key checkpoint for multiple tables

## Test Commands

Generate proof:
```bash
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin
```

Cross-verify:
```bash
cp logs/zolt_proof_dory.bin /tmp/ && cp logs/zolt_preprocessing.bin /tmp/
cd ../jolt && cargo test -p jolt-core --lib test_verify_zolt_proof_with_zolt_preprocessing --features zolt-debug -- --ignored --nocapture
```

## Session History

- Session 1-8: Initial implementation, transcript ordering
- Session 9: MontU128Challenge multiplication fix - internal PASSED
- Session 10-11: Cross-verification debugging
- Session 12: Verified r_address_prime challenges match
- Session 13: Fixed suffix_len overflow, Stage 5 internal PASSED
- Session 14: Internal verification passes, cross-verification fails
- Session 15: Confirmed opening claims match - polynomial computation is the issue

## SESSION_ENDING

Context is running low. Key progress this session:
1. Confirmed opening claims match between Zolt and Jolt
2. Identified that the polynomial computation via prefix-suffix decomposition is the source of mismatch
3. The same opening claims + different polynomial values = prefix-suffix not producing correct table MLE evaluations
4. Next session: Add debug to compare Zolt's computed table values with Jolt's table_mle evaluations
