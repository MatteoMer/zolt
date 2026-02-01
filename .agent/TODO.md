# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Mismatch

## Session 106 Summary

### Progress Made
1. **Fixed shift overflow in rightShiftPrefixMle** (commit pending)
   - When `y_u32 == 0`, `@ctz()` returns 32 which causes overflow in `x_u32 >> 32`
   - Added bounds check: `if (trailing_zeros >= 32) 0 else x >> trailing_zeros`
   - Also fixed in `rightShiftWPrefixMle` which had the same issue

2. **Stage 5 Verification Status**
   - Stages 1-4 pass successfully
   - Stage 5 (InstructionReadRaf sumcheck) fails with claim mismatch
   - output_claim:   `[eb, 1c, 1a, 7c, 50, c5, 1b, 64, ...]`
   - expected_claim: `[76, 19, 2f, 98, 45, 38, 7b, 09, ...]`

### Key Analysis from Jolt Expert Agent

The Stage 5 sumcheck has 136 rounds:
- **Rounds 0-127 (address)**: Use prefix-suffix decomposition
  - Compute `g(X) = read_checking(X) + raf(X)`
  - Output `[eval_0, eval_2]`, derive `eval_1 = hint - eval_0`
  - Use `from_evals_and_hint(previous_claim, [eval_0, eval_2])`

- **Rounds 128-135 (cycle)**: Use materialized polynomials
  - Product of ra_polys * combined_val * eq
  - Degree 10 polynomial (product of 9 linear factors + eq)

### Files Modified This Session
- `src/zkvm/lookup_table/prefixes.zig` - Fixed shift overflow in rightShiftPrefixMle and rightShiftWPrefixMle

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof (takes several minutes)
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Next Steps
1. Commit the shift overflow fix
2. Investigate why the sumcheck output_claim doesn't match expected_claim
3. Possible issues to check:
   - Are the Q (suffix) polynomials being initialized correctly?
   - Is the phase transition logic correct (condenseUEvals)?
   - Are the expanding tables being updated correctly?
   - Is the RAF computation matching Jolt's formula?

### Jolt Expected Claim Formula (Stage 5 Instance 2)
```
expected_claim = eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
```

Where:
- `eq_eval_r_reduction` = eq(r_cycle_prime, r_reduction)
- `ra_claim` = product of ra_chunk evaluations at r_address
- `val_claim` = Σ_tables table.evaluate_mle(r_address) * table_flag_claim[table]
- `raf_claim` = (1-raf_flag)*(left_op + γ*right_op) + raf_flag*γ*identity

### Debug Output Needed
To debug further, need to print:
1. The actual polynomial coefficients Zolt sends for each address round
2. Compare against what Jolt's verifier expects (can add debug prints in Jolt)
3. Check if challenges are matching (transcript consistency)
