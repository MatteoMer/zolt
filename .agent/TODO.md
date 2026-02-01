# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Prefix-Suffix Decomposition Required

## Current Session Progress (Session 95)

### Root Cause Analysis: Stage 5 Sumcheck Failure

The Stage 5 sumcheck verification fails because Zolt's LookupsReadRaf implementation uses a fundamentally different approach than Jolt.

**The Core Equation:**
```
Σ_j Σ_k eq(j, r_reduction) * ra(k, j) * (Val_j(k) + γ·RafVal_j(k)) = lookups_input
```

**Jolt's Approach:**
1. Address rounds (0-127): Uses **prefix-suffix decomposition** to compute degree-2 polynomials
   - Each lookup table has a prefix-suffix structure
   - The polynomial is computed via `from_evals_and_hint(previous_claim, [p(0), p(2)])`
   - This produces degree-2 polynomials

2. After address rounds: Calls `init_log_t_rounds()` which:
   - Materializes `ra_polys` as polynomials over cycles
   - Materializes `combined_val_polynomial[j] = table_values_at_r_addr[table(j)] + raf_val`
   - Where `table_values_at_r_addr[t]` = **MLE of table t evaluated at r_address**
   - And `raf_val` = gamma * left_prefix + gamma^2 * right_prefix

3. Cycle rounds (128-135): Uses the **materialized** polynomials with standard sumcheck

**Zolt's Current Approach:**
1. Address rounds: Uses **bit-splitting** with degree-1 polynomials
   - Splits cycles by whether lookup_index bit = 0 or 1
   - Produces degree-1 linear polynomials (NOT degree-2!)

2. Never materializes `combined_val_polynomial`
   - Uses raw per-cycle values `lookups_combined_vals[j] = output[j] + gamma*left[j] + gamma^2*right[j]`
   - This is the **concrete lookup result**, not the table MLE at r_address

3. Cycle rounds: Uses raw values instead of table MLE evaluations

### Why This Fails

1. **Polynomial Degree Mismatch**: Jolt expects degree-2 polynomials in address rounds, Zolt outputs degree-1

2. **Value Mismatch**: After address rounds:
   - Jolt has `combined_val[j] = table_mle(r_address) + raf_prefix_eval`
   - Zolt has `combined_val[j] = lookup_output[j] + gamma*left[j] + gamma^2*right[j]`
   - These are fundamentally different values!

3. **MLE vs Raw Values**: The lookup output `f(x, y)` at concrete inputs is NOT the same as `table_mle(r)` at random point `r`

### What Needs to be Fixed

To make Zolt compatible with Jolt's verifier, we need to implement:

1. **Prefix-Suffix Decomposition for Address Rounds**
   - Implement the prefix/suffix polynomial structures
   - Compute degree-2 polynomials like Jolt does
   - Track prefix checkpoints during address rounds

2. **Table MLE Evaluation**
   - After address rounds, compute `table_values_at_r_addr` for each of 42 tables
   - Each table value is the MLE evaluated at the bound 128-bit random point
   - This requires implementing `table.combine(prefixes, suffixes)` for each table

3. **RAF Prefix Evaluation**
   - Compute `left_prefix`, `right_prefix`, `identity_prefix` at r_address
   - Create `raf_interleaved = gamma * left_prefix + gamma^2 * right_prefix`
   - Create `raf_identity = gamma^2 * identity_prefix`

4. **Materialize combined_val_polynomial**
   - For each cycle j: `combined_val[j] = table_values_at_r_addr[table(j)] + raf_val`
   - Use the appropriate raf_val based on cycle's interleave flag

### Required Files to Implement

1. `src/zkvm/lookup_tables/mod.zig` - Lookup table MLE definitions
2. `src/zkvm/lookup_tables/prefixes.zig` - Prefix structures (LeftOperand, RightOperand, Identity)
3. `src/zkvm/lookup_tables/suffixes.zig` - Suffix structures
4. `src/zkvm/spartan/prefix_suffix.zig` - PrefixSuffixDecomposition implementation

### Verification Results

```
Sumcheck verification failed!
  output_claim:   [d9, 50, 6a, 6e, 69, 84, 32, f8, ...]
  expected_claim: [bb, 2a, d3, 8c, 2c, 8c, 44, d3, ...]
```

### Test Commands

```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Key Jolt Files for Reference

- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Main LookupsReadRaf implementation
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/lookup_table/mod.rs` - Lookup table trait definitions
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/lookup_table/prefixes/*.rs` - Prefix implementations
- `/home/vivado/projects/jolt/jolt-core/src/poly/prefix_suffix.rs` - PrefixSuffixDecomposition

### Next Steps

1. Study Jolt's lookup table MLE implementations in detail
2. Understand the prefix-suffix decomposition structure
3. Implement minimal set of tables needed for Fibonacci (ADD, ADDI, BNE, etc.)
4. Test with simpler programs first
