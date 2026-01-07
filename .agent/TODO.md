# Zolt-Jolt Compatibility - Session Progress

## Current Status

### ✅ COMPLETED
- Stage 1 passes Jolt verification completely
- All 712 internal tests pass
- R1CS witness generation fixed:
  - RamReadValue for stores now uses pre-value (not zero)
- RWC prover improvements:
  - r_cycle uses correct slice: tau[0..n_cycle_vars]
  - eq computation uses BIG_ENDIAN order (MSB first)
  - inc = new_value - prev_value (signed difference)
  - val_coeff = pre-value for writes, value for reads
  - inc polynomial folds during Phase 1 binding
  - Phase 2 uses eq_evals[0] and inc[0] as scalars

### ❌ IN PROGRESS
Stage 2 fails because the RWC sumcheck total_sum doesn't match current_claim.

**The Issue:**
- `total_sum = Σ eq(r_cycle, j) * ra(k,j) * (val + γ*(inc + val))` over all entries
- `current_claim = ram_read_value_claim + γ * ram_write_value_claim`
- These don't match even though both are computed from the same data

**Latest Debug Output:**
```
total_sum = { 41, 42, 209, ... }
current_claim = { 36, 80, 231, ... }
```

**Possible Root Causes:**
1. Entry values (val, inc) may not match R1CS values exactly
2. Memory trace entries may not correspond 1:1 with R1CS cycles
3. r_cycle ordering between MLE evaluation and RWC sumcheck may differ

## Next Steps

1. Add debug to print first few entry contributions in detail
2. Compare entry val_coeff values with R1CS witness RamReadValue for same cycles
3. Verify memory trace is built consistently with R1CS witness
4. Check if there are cycles with RAM access in R1CS but not in memory trace

## Verification Commands

```bash
# Build and test Zolt
zig build test --summary all

# Generate Jolt-compatible proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Verify with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```

## Code Locations
- RWC prover: `/Users/matteo/projects/zolt/src/zkvm/ram/read_write_checking.zig`
- Jolt RWC: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/ram/read_write_checking.rs`
- R1CS witness: `/Users/matteo/projects/zolt/src/zkvm/r1cs/constraints.zig`
