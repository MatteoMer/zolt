# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Formula Mismatch

## Session 116 Summary

### Fixed
1. **File path issue**: Test was reading `/tmp/zolt_proof_dory.bin` (old file) instead of `logs/` (current file). Copied correct files to /tmp.
2. **Table flag claims**: Now deserialize correctly - tables 0, 1, 9 are non-zero for Fibonacci.

### Verified Correct
All three tables ARE correctly used in Fibonacci execution:
- Table 0 (RangeCheck): Used by ADD, ADDI, LUI, JAL, etc.
- Table 1 (RangeCheckAligned): Used by JALR at cycle 52
- Table 9 (NotEqual): Used by BNE branches

```
52 | 0x80000064 | 0x00008067 | JALR x0, 0(x1)  <- Uses table 1 (RangeCheckAligned)
```

### Current Issue
Stage 5 sumcheck verification fails:
```
output_claim:   [9d, 3d, dd, d4...]
expected_claim: [1b, 43, f0, ba...]
```

### Table Evaluations (from Jolt debug)
```
table_eval[0] (RangeCheck)        = [fb, 9d, 83, 09...]
table_eval[1] (RangeCheckAligned) = [9b, 3c, 87, 4b...]
table_eval[9] (NotEqual)          = [f9, b0, 80, b7...]

table_flag[0] = [34, 2b, b2, 5c...]
table_flag[1] = [aa, b9, 54, f9...]
table_flag[9] = [dd, 0d, 07, c1...]
```

### Formula Analysis

The verifier computes:
```
val_claim = Σ table_MLE[i](r_address) * table_flag[i]
          = table_eval[0]*table_flag[0] + table_eval[1]*table_flag[1] + table_eval[9]*table_flag[9]

raf_claim = (1 - raf_flag) * (left_op + gamma*right_op) + raf_flag * gamma * identity

expected = eq_eval * ra_claim * (val_claim + gamma * raf_claim)
```

### Likely Root Cause

Zolt computes `combined_vals[j]` differently from what Jolt expects:

**Zolt (stage5_prover.zig:811)**:
```zig
combined_vals[j] = lookup_output + gamma*left + gamma^2*right
```

**Jolt expects**:
```
combined = val + gamma * raf
where raf = (1-raf_flag)*(left + gamma*right) + raf_flag*gamma*identity
```

Key differences:
1. Zolt uses `gamma^2 * right`, but Jolt uses `gamma * (left + gamma*right)` in raf_claim
2. Identity path handling may differ for AddOperands instructions

### Implementation Status
- Added `evaluateTableMLE(table_index, r)` function in lookup_table/mod.zig
- Table flag computation is correct (matches which tables are used)

### Files Modified This Session
- `/home/vivado/projects/zolt/src/zkvm/lookup_table/mod.zig`: Added `evaluateTableMLE()`
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`: Added table_eval debug
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/proof_serialization.rs`: Added byte offset tracking

### Test Commands
```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy to /tmp for Jolt test
cp logs/zolt_*.bin /tmp/

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Next Steps
1. **Fix combined_vals formula** in stage5_prover.zig to match Jolt's expected formula
2. **Check identity path handling** - AddOperands instructions (ADD, ADDI, etc.) use identity path
3. **Verify gamma usage** - ensure gamma powers match between prover and verifier
4. Add more debug to trace exact computation differences

SESSION_ENDING - Progress saved to TODO.md
