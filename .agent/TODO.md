# Zolt-Jolt Compatibility Implementation

## Status: Session 14 - Stage 5 Verification Failing

## Session 14 Investigation

### Current Issue

Stage 5 sumcheck verification fails:
- `output_claim: [63, 47, 60, 9d, ...]` (from Zolt's sumcheck polynomial)
- `expected_claim: [a9, 4a, 89, dd, ...]` (computed by Jolt's verifier)

### Root Cause Analysis

The sumcheck polynomial output (from Zolt's proof) doesn't match the expected claim computed by Jolt's verifier using the opening claims.

**Key insight**: Internal verification PASSES, meaning Zolt's polynomial and opening claims are internally consistent. But Jolt's expected_claim formula produces a different result.

### Debug Analysis

For Instance 2 (InstructionReadRaf), Jolt's formula:
```
expected = eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
```

Where:
- `eq_eval_r_reduction` = eq(r_reduction, r_cycle_prime) - computed from challenges
- `ra_claim` = product of ra_chunks from opening claims
- `val_claim` = Σ table_flag[i] * table_mle[i](r_address) - uses table MLE evaluations
- `raf_claim` = (1-raf_flag)*(left_op + gamma*right_op) + raf_flag*gamma*identity

Jolt's computed values:
```
left_operand_eval:  [23, bd, 14, a1, ...]
right_operand_eval: [2b, 88, fa, 0a, ...]
identity_poly_eval: [24, 1e, 72, 5d, ...]
ra_claim:           [2c, 0e, 55, 84, ...]
val_claim:          [83, cd, b7, 55, ...]
final_result:       [b6, fd, fb, 64, ...]
```

### Potential Issues

1. **Polynomial Mismatch**: Zolt's sumcheck polynomial for Instance 2 may compute different values than what Jolt expects:
   - `read_checking + raf` where read_checking uses prefix-suffix decomposition
   - Need to verify tableCombine/prefixMle implementations match Jolt

2. **Opening Claims Inconsistency**: The ra_chunks, table_flags, raf_flag values in opening_claims may not match what the polynomial actually computes

3. **Challenge Normalization**: Need to verify r_address/r_cycle normalization matches Jolt's normalize_opening_point

### Files to Check

1. `src/zkvm/lookup_table/prefix_suffix_prover.zig`:
   - `proverMsgReadChecking` - computes val (table MLE contributions)
   - `proverMsgRaf` - computes raf contributions
   - `tableCombine` - combines prefix/suffix for table evaluations

2. `src/zkvm/spartan/stage5_prover.zig`:
   - Lines 2011-2077: Instance 2 polynomial computation
   - Lines 3256-3408: Opening claims extraction

### Next Steps

1. Add debug output to compare Zolt's polynomial values with Jolt's expectations
2. Trace through a single round to verify prefix-suffix decomposition
3. Compare table MLE evaluations between Zolt and Jolt

### Commands

Generate proof:
```bash
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin
```

Cross-verify:
```bash
cp logs/zolt_proof_dory.bin /tmp/ && cp logs/zolt_preprocessing.bin /tmp/
cd ../jolt && cargo test -p jolt-core --lib test_verify_zolt_proof_with_zolt_preprocessing --features zolt-debug -- --ignored --nocapture
```

### Previous Sessions

- Session 1-8: Initial implementation, transcript ordering, MLE evaluations
- Session 9: MontU128Challenge multiplication fix - internal verification PASSED
- Session 10: Cross-verification debugging, input claims match, polynomial mismatch
- Session 11: Deep investigation - all components match but expected_claim still differs
- Session 12: Verified r_address_prime values match. Added operand eval debug
- Session 13: Fixed suffix_len overflow, fixed leftMsbUpdateCheckpoint, Stage 5 PASSED internally

## SESSION_ENDING

Context is running low. Key progress this session:
1. Identified that internal verification passes but cross-verification fails
2. Traced through Jolt's expected_claim formula for InstructionReadRaf
3. Identified that the issue is likely in how val_claim is computed via prefix-suffix decomposition
4. Next session should focus on comparing tableCombine/prefixMle with Jolt's implementations
