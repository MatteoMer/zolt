# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ❌ FAIL | RWC Phase 2 needs AddressMajor representation |
| 3 | ⏳ Blocked | Waiting for Stage 2 |
| 4 | ⏳ Blocked | Waiting for Stage 3 |
| 5-7 | ⏳ Blocked | Waiting for Stage 4 |

## Session 38 Summary (2026-01-17)

### Completed
1. ✅ Fixed inc binding to use LowToHigh order
2. ✅ Fixed eq_evals binding to use LowToHigh order
3. ✅ Added val_init binding in Phase 2
4. ✅ Added checkpoint-aware Phase 2 polynomial computation

### Remaining Issue
The fundamental issue is that **Jolt's Phase 2 uses AddressMajor iteration** while Zolt uses cycle-major iteration:

**Jolt's Phase 2:**
- Entries sorted by column (address), then by row (cycle)
- Iterates: for each column pair (2k, 2k+1), merge entries by cycle
- Uses `even_checkpoint` and `odd_checkpoint` that track state across cycles
- Checkpoints come from val_init and are updated as entries are processed

**Zolt's Phase 2 (current):**
- Entries remain in cycle-major order
- Iterates directly over entries
- Uses bound val_init as static checkpoints
- Missing the cycle-by-cycle state tracking

### Key Difference
In Jolt, when processing a column pair, if only one column has an entry at a given cycle:
- The implicit entry gets val from the checkpoint (which is the previous val at that address)
- The checkpoint is updated after processing each cycle

This is fundamentally different from what Zolt does.

### Options to Fix

**Option A: Implement AddressMajor representation (proper fix)**
1. After Phase 1, convert entries to AddressMajor order (sort by address)
2. Implement seq_prover_message_contribution logic:
   - For each column pair, iterate by cycle
   - Track even_checkpoint and odd_checkpoint per column
   - Use checkpoints for implicit entries
3. Bind entries by address (column halving)

**Option B: Use Phase 3 approach (simpler but different)**
1. After Phase 1, materialize ra and val as dense polynomials
2. Use dense sumcheck for Phase 2 (address binding)
3. Avoids sparse matrix complexity but changes proof structure

**Option C: Skip Phase 2 sparse sumcheck**
1. If fibonacci only has 1 RAM entry, sparse computation might not matter
2. But this doesn't solve the general case

### Recommended Next Steps
1. Implement AddressMajor sparse matrix representation
2. Convert CycleMajor to AddressMajor at end of Phase 1
3. Implement proper seq_prover_message_contribution with checkpoint tracking

## Files Modified This Session
- `src/zkvm/ram/read_write_checking.zig`
  - Added LowToHigh binding for inc and eq_evals
  - Added val_init binding in Phase 2
  - Updated Phase 2 polynomial computation with checkpoint support

## Testing Commands

```bash
# Build and run prover
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verifier
cd /Users/matteo/projects/jolt && ZOLT_LOGS_DIR=/Users/matteo/projects/zolt/logs cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
