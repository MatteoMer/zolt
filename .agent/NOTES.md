# Session 95 Notes - RAF Decomposition Mismatch

## Summary

Fixed UpperWord prefix formula (was using 2*XLEN-j instead of XLEN-j). Identified that the RAF (Read-Address-Flag) decomposition produces different results than brute-force computation.

## Key Findings

### UpperWord Fix
- Bug: `x_shift = 2 * XLEN - j` caused overflow at j=0 (128-bit shift)
- Fix: Use `XLEN - j` matching Jolt's upper_word.rs
- Also fixed suffix handling to extract only upper word bits

### RAF Decomposition Mismatch

At round 0:
- `explicit_raf_0 = 8d6b9084...` (from proverMsgRaf)
- `bf_raf_eval_0 = 9bac6bba...` (brute-force)

The brute-force computes:
```
bf_raf_eval_0 = Σ_{j: bit127==0} u[j] * (combined[j] - output[j])
             = Σ_{j: bit127==0} u[j] * (γ*left + γ²*right)
```

The prefix-suffix RAF computes:
```
proverMsgRaf = γ*left_sum + γ²*(identity_sum + right_sum)
```

Where each sum uses the RafDecomposition::prefixEvals() and Q polynomials.

### Read-Checking Works Correctly

Per-table values match between brute-force and prefix-suffix:
- `bf_val_per_table[0] = 821c547e...` = `eval_0_per_table[0]`

This confirms tableCombine is correct. The issue is isolated to RAF.

## Possible Issues with RAF

1. **Q polynomial initialization in initQRaf**:
   - Left/Right/Identity Q arrays might have wrong values
   - The shift coefficients might be computed incorrectly

2. **operandPrefixEvals in RafDecomposition**:
   - At round 0 with bound_value = 0, returns (0, 256) for LeftOperand
   - This might not match Jolt's OperandPolynomial::sumcheck_evals()

3. **identityPrefixEvals**:
   - Similarly needs to match Jolt's IdentityPolynomial

## Comparison Points

Jolt's prover_msg_raf at round 0:
- Uses PrefixSuffixDecomposition for left_operand_ps, right_operand_ps, identity_ps
- Each decomposition has 2 Q arrays: Q[0] for shift, Q[1] for operand/identity suffix

Zolt's proverMsgRaf at round 0:
- Uses RafDecomposition for left_raf, right_raf, identity_raf
- Each decomposition has Q[0] for shift, Q[1] for suffix

## Next Steps

1. Add debug output in initQRaf to verify Q array values
2. Compare Jolt's PrefixSuffixDecomposition::sumcheck_evals with Zolt's prefixEvals
3. Check if the shift coefficients (ShiftHalfSuffix, ShiftFullSuffix) are computed correctly

## Test Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64 2>&1 | grep -E "RAF|raf|explicit" | head -30
```
