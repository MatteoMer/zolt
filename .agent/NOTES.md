# Stage 5 Investigation Notes (Session 95)

## Summary

The Stage 5 sumcheck verification fails because Zolt uses a fundamentally different approach for the LookupsReadRaf sumcheck than Jolt. Jolt uses prefix-suffix decomposition with 45+ prefix types, while Zolt uses a simpler bit-splitting approach.

## Key Findings

### 1. Polynomial Degree Difference

**Jolt Address Rounds (0-127):**
- Uses prefix-suffix decomposition to compute degree-2 polynomials
- The polynomial is computed via `from_evals_and_hint(previous_claim, [p(0), p(2)])`
- This produces degree-2 polynomials from evaluations at X∈{0, 2}

**Zolt Address Rounds (0-127):**
- Uses bit-splitting: splits cycles by whether lookup_index bit = 0 or 1
- Produces degree-1 linear polynomials: p(X) = p0 + X*(p1 - p0)

### 2. Value Materialization Difference

**Jolt after address rounds:**
- Calls `init_log_t_rounds()` which materializes:
  - `ra_polys[i]`: Per-cycle values = product of expanding tables evaluated at r_address
  - `combined_val_polynomial[j]` = table_values_at_r_addr[table(j)] + raf_val
- Where `table_values_at_r_addr[t]` is the **MLE of table t evaluated at r_address**
- And `raf_val` = gamma * left_prefix + gamma^2 * right_prefix (or identity version)

**Zolt after address rounds:**
- Uses raw per-cycle values `lookups_combined_vals[j] = output[j] + gamma*left[j] + gamma^2*right[j]`
- This is the **concrete lookup result**, NOT the table MLE at r_address

### 3. The Core Issue

The lookup output `f(x, y)` at concrete inputs `(x, y)` is **not the same** as `table_mle(r)` at random point `r`.

Example: AND table
- `f(5, 3) = 5 & 3 = 1` (concrete evaluation)
- `AND_mle(r_0, r_1, ..., r_127) = Σ_i 2^i * r_{2i} * r_{2i+1}` (MLE at random point)

These are completely different values unless `r` happens to be exactly the binary encoding of `(5, 3)` (probability negligible).

### 4. Required Implementation

To achieve Jolt compatibility, Zolt needs to implement:

1. **All 45+ Prefix Types** from Jolt:
   - LowerWord, LowerHalfWord, UpperWord, Eq, And, Andn, Or, Xor, LessThan, etc.
   - Each with `prefix_mle()` and `update_prefix_checkpoint()` methods

2. **All Suffix Types** from Jolt:
   - One, And, Or, Xor, LeftMsb, RightMsb, LessThan, etc.

3. **PrefixSuffixDecomposition** for all 42 lookup tables:
   - Each table defines which suffixes it uses and how to combine prefixes/suffixes

4. **Expanding Table Accumulator (v vector)**:
   - Used to compute `ra_polys` after address rounds

5. **RAF Operand Polynomials**:
   - LeftOperand, RightOperand, Identity prefix-suffix decompositions
   - Used to compute raf_interleaved and raf_identity

### 5. Jolt Files to Reference

Main implementation:
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`

Lookup table trait:
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/lookup_table/mod.rs`

Prefixes (45+ types):
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/lookup_table/prefixes/*.rs`

Suffixes:
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/lookup_table/suffixes/*.rs`

Prefix-Suffix Decomposition:
- `/home/vivado/projects/jolt/jolt-core/src/poly/prefix_suffix.rs`

### 6. Alternative Approaches Considered

**Generalized-Lasso approach:**
The Jolt paper mentions Generalized-Lasso which doesn't require prefix-suffix decomposition, instead using direct MLE evaluation. However, this would require modifications to both prover AND verifier - defeating the "verify by Jolt" goal.

**Direct MLE evaluation after address rounds:**
Even if we compute correct `table_values_at_r_addr` after address rounds, the address round polynomials themselves must match Jolt's for the transcript to stay in sync.

### 7. Verification Output Analysis

```
Sumcheck verification failed!
  output_claim:   [d9, 50, 6a, 6e, 69, 84, 32, f8, ...]
  expected_claim: [bb, 2a, d3, 8c, 2c, 8c, 44, d3, ...]
```

The mismatch occurs because:
1. Zolt's polynomial coefficients differ from Jolt's
2. This causes different challenge values to be derived
3. Which causes different intermediate claims
4. Resulting in completely different final output claim

### 8. Next Steps

This is a substantial implementation effort requiring:
1. Porting all 45+ prefix implementations from Jolt
2. Porting all suffix implementations
3. Implementing PrefixSuffixDecomposition for all tables
4. Implementing the RAF operand decompositions
5. Testing each component against Jolt's reference implementation

Estimated effort: Several days of focused implementation work.

### 9. Files Modified

- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover (current approach)
- `.agent/TODO.md` - Updated with findings
- `.agent/NOTES.md` - This file

### 10. Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Verify with Jolt (should fail currently)
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
