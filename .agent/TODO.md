# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 LookupsReadRaf Cycle Rounds Fix

## Current Session Progress (Session 94)

### Key Discovery: Stage 5 Cycle Rounds Bug

The Stage 5 sumcheck for LookupsReadRaf is computing the wrong polynomial during cycle rounds (rounds 128-135).

**Root Cause:**
- Zolt uses raw per-cycle values `lookups_combined_vals[j] = output[j] + gamma*left[j] + gamma^2*right[j]` throughout all rounds
- Jolt materializes `combined_val_polynomial` AFTER the address rounds complete (in `init_log_t_rounds()`)
- The materialized combined_val contains table MLE evaluations at the bound address point, not raw per-cycle values

**Evidence:**
- Individual sums match: `output match = true`, `left match = true`, `right match = true`
- `computed_sum = lookups_input` ✓ (verification loop matches)
- But sumcheck final claim doesn't match expected: `output_claim ≠ expected_claim`
- Round 128 polynomial coefficients are completely different between Zolt and Jolt

**Jolt's Approach (read_raf_checking.rs lines 720-774):**
1. Address rounds (0-127): Bind prefix/suffix polynomials
2. After address rounds: Call `init_log_t_rounds()` which:
   - Evaluates lookup tables at bound r_address: `table_values_at_r_addr[t_idx]`
   - Computes `raf_interleaved = gamma * left_prefix + gamma_sqr * right_prefix`
   - Computes `raf_identity = gamma_sqr * identity_prefix`
   - Creates `combined_val_poly[j] = table_values_at_r_addr[table_idx] + raf_val`
3. Cycle rounds (128-135): Use `combined_val.get_bound_coeff(2*j)` not raw values

### Required Fix

Must rewrite Stage 5 to:
1. Track prefix evaluations during address rounds
2. After round 127, materialize `combined_val_polynomial` similar to Jolt
3. Use materialized values in cycle rounds

### Previous Session Fixes (Session 93)

1. **R1CS deriveImmediate Bug Fixed**:
   - Added 0x1b (ADDIW) to I-type case at line 1231

2. **Stage 5 Witness Computation Fixed**:
   - Fixed 0x1b, 0x03, 0x23, 0x3b handling to match R1CS

### Verification Results

- Individual sums now match between Stage 2 claims and Stage 5 recomputation
- `computed_sum = lookups_input` ✓
- But sumcheck polynomial coefficients don't match Jolt

### Test Commands

```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Key Files

- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig`: Stage 5 prover
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`: Jolt reference

### Comparison: Round 128 Coefficients

**Jolt first coeff (LE):** `[e2, ee, 6f, c7, e9, ff, ea, e2, ...]`
**Zolt first coeff (BE):** `{ 30, 94, f1, 94, 6b, a0, 75, f5, ... }`

These are completely different, confirming the polynomial computation is wrong.

### Next Steps

1. Implement lookup table MLE evaluation at r_address
2. Compute prefix evaluations (left, right, identity) at r_address
3. Materialize combined_val_polynomial after address rounds
4. Update cycle round computation to use materialized values
