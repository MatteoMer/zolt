# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ❌ FAIL | RamReadWriteChecking Phase 2 formula needs checkpoint tracking |
| 3 | ⏳ Blocked | Waiting for Stage 2 |
| 4 | ⏳ Blocked | Waiting for Stage 3 |
| 5-7 | ⏳ Blocked | Waiting for Stage 4 |

## Session 38 Progress - RWC Debugging (2026-01-17)

### What's Working
1. ✅ Phase 1 binding with LowToHigh order for inc, eq_evals, and entries
2. ✅ Phase 1 round polynomial computation with GruenSplitEq
3. ✅ Entry transformation during Phase 1 (cycle halving)

### What's Not Working
1. ❌ Phase 2 polynomial computation - formula doesn't match Jolt exactly

### Root Cause Analysis

Jolt's Phase 2 uses an **AddressMajor sparse matrix** representation that:
1. Sorts entries by column (address) instead of row (cycle)
2. Tracks `val_init` - initial values for each address
3. Tracks `prev_val` and `next_val` for each entry to handle state transitions
4. Uses `even_checkpoint` and `odd_checkpoint` when merging column pairs

The formula in Jolt's `seq_prover_message_contribution`:
```rust
// For each pair of columns (even_col, odd_col):
// Iterate entries by row (cycle), merge even and odd entries
// Use checkpoints for implicit entries (missing even or odd)
eq.get_bound_coeff(row) * ra(X) * (val(X) + γ*(inc + val(X)))
```

Where:
- `ra(X)` and `val(X)` are linear in the address variable X
- For odd-only entry: ra(0)=0, ra(1)=ra_coeff, val(0)=checkpoint, val(1)=val_coeff
- For even-only entry: ra(0)=ra_coeff, ra(1)=0, val(0)=val_coeff, val(1)=checkpoint

### Required Changes for Phase 2

1. **Track initial RAM values** (val_init) for computing checkpoints
2. **Store prev_val/next_val** in entries for state tracking
3. **Implement proper column pair iteration** with checkpoint handling
4. **Compute s(2) correctly** using the checkpoint-aware formula:
   ```
   ra(2) = 2*ra(1) - ra(0)  // linear extrapolation
   val(2) = 2*val(1) - val(0)  // using checkpoint for missing entry
   s(2) = eq * ra(2) * (val(2) + γ*(inc + val(2)))
   ```

### Alternative Approach

Instead of implementing AddressMajor sparse matrix, could potentially:
1. Use Jolt's Phase 3 approach (dense polynomials) for all address variables
2. This would require materializing ra and val polynomials after Phase 1
3. Simpler but potentially more expensive for large RAM

## Testing Commands

```bash
# Build and run prover
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verifier
cd /Users/matteo/projects/jolt && ZOLT_LOGS_DIR=/Users/matteo/projects/zolt/logs cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Files to Modify

- `src/zkvm/ram/read_write_checking.zig` - Main RWC implementation
- Need to add val_init tracking and checkpoint computation
