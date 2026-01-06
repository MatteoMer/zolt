# Zolt-Jolt Compatibility TODO

## Current Status: January 6, 2026 - Session 61

**STATUS: Transcript and serialization working, inner_sum_prod mismatch persists**

### Summary

All unit tests pass (712/712). The transcript, challenges, and serialization are all correct. The issue is that Jolt's `inner_sum_prod` (computed from R1CS constraints and opening claims) differs from what Zolt's sumcheck produces.

### Key Insight

The verification formula is:
```
expected = tau_factor * inner_sum_prod
        = tau_factor * (Sum(w[i] * Az_i(z)) * Sum(w[i] * Bz_i(z)))
```

Where:
- `tau_factor` MATCHES between Zolt and Jolt ✓
- `z` (opening claims) are correct in proof ✓
- But `inner_sum_prod` differs

This means either:
1. The constraint definitions differ subtly
2. The dot product computation differs
3. There's a witness value difference in the trace

### Verified Working

1. **Transcript compatibility** - All states match
2. **Tau values** - `lagrange_tau_r0` matches exactly
3. **Opening claims** - Bytes match what Jolt reads
4. **R1CS constraint order** - FIRST/SECOND_GROUP_INDICES match

### Investigation Needed

1. **Compare constraint dot products**: Print the `lc.dot_product(z)` values from both sides
2. **Compare witness values**: For the same trace, compare Zolt's and Jolt's R1CSCycleInputs
3. **Instruction input computation**: Verify `to_instruction_inputs` logic matches
4. **Signed integer handling**: Check Product computation for signed operands

### Key Files

- `/Users/matteo/projects/zolt/src/zkvm/r1cs/constraints.zig` - Zolt's witness generation
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/r1cs/inputs.rs` - Jolt's witness generation
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/instruction/add.rs` - Example instruction inputs

### Technical Details

**Jolt's inner_sum_prod computation (from key.rs):**
```rust
for i in 0..R1CS_CONSTRAINTS_FIRST_GROUP.len() {
    let lc_a = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.a;
    let lc_b = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.b;
    az_g0 += w[i] * lc_a.dot_product::<F>(&z, z_const_col);
    bz_g0 += w[i] * lc_b.dot_product::<F>(&z, z_const_col);
}
// Final: (az_g0 * (1-r_stream) + az_g1 * r_stream) * (bz_g0 * (1-r_stream) + bz_g1 * r_stream)
```

**Note:** `z_const_col = 36` means z has 37 elements (36 inputs + 1 constant column).

### Test Commands

```bash
# All tests pass
zig build test

# Generate real proof
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 -o /tmp/zolt_proof_dory.bin

# Jolt verification (fails)
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

### Next Session Actions

1. Add debug in Jolt's `evaluate_inner_sum_product_at_point` to print per-constraint Az, Bz values
2. Add debug in Zolt's sumcheck to print per-constraint Az, Bz values for first cycle
3. Compare the values - find which constraint differs first
4. Trace back to the witness generation difference
