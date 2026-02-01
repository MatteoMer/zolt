# Stage 5 Investigation Notes (Session 94)

## Summary

The Stage 5 sumcheck verification fails because the polynomial coefficients don't match between Zolt and Jolt. Despite individual sums matching (`output_sum = rv_claim`, `left_sum = left_claim`, `right_sum = right_claim`), the round polynomial coefficients are completely different.

## Key Observations

### 1. Address Round Computation Difference

**Jolt uses prefix-suffix decomposition:**
- `prover_msg_read_checking()`: Evaluates lookup table polynomials using MLE prefix/suffix structure
- `prover_msg_raf()`: Evaluates RAF (left, right, identity) polynomials
- Returns degree-2 polynomial evaluations at X∈{0, 2}

**Zolt uses address bit splitting:**
- Splits cycles by address bit: `p0 = Σ (eq * ra * combined) for bit=0`
- Creates degree-1 polynomial

### 2. Cycle Round Computation Difference

**Jolt:**
- After address rounds, materializes `combined_val_polynomial` in `init_log_t_rounds()`
- Uses `combined_val.get_bound_coeff(2*j)` in cycle rounds
- The materialized values use table MLE evaluations at r_address

**Zolt:**
- Uses raw per-cycle values `lookups_combined_vals[j]` throughout
- Never materializes the bound combined value

### 3. Round 128 Coefficient Comparison

- **Jolt coeff[0] (LE):** `[e2, ee, 6f, c7, e9, ff, ea, e2, ...]`
- **Zolt coeff[0] (BE):** `{ 30, 94, f1, 94, 6b, a0, 75, f5, ... }`

Completely different!

## Open Questions

1. **Is our address round approach valid?**
   - We use `combined(lookup_index(j), j)` as a constant per cycle
   - This relies on `ra(k, j) = 1` only when `k = lookup_index(j)`
   - Should this produce the same result as Jolt's prefix-suffix approach?

2. **Why do individual sums match but polynomial coefficients don't?**
   - The sums are computed over all cycles
   - The polynomial coefficients depend on how we split/evaluate during rounds
   - Different splitting strategies could give same sum but different polynomials

3. **Do we need to implement prefix-suffix decomposition?**
   - This is a complex optimization in Jolt
   - It might be necessary for correct polynomial computation

## Possible Fixes

### Option A: Implement Jolt's approach
- Implement `PrefixSuffixDecomposition` for address rounds
- Materialize `combined_val_polynomial` after address rounds
- Use bound coefficients in cycle rounds

### Option B: Debug current approach
- Verify that our polynomial computation is mathematically equivalent
- Check if there's a simpler bug (endianness, indexing, etc.)
- Compare intermediate values during address rounds

## Files

- Jolt reference: `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`
- Zolt Stage 5: `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig`
