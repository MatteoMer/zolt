# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Mismatch

## Session 116 Summary

### Fixed
1. **File path issue**: Test was reading `/tmp/zolt_proof_dory.bin` (old file) instead of `logs/` (current file). Copied correct files to /tmp.
2. **Table flag claims**: Now deserialize correctly - tables 0, 1, 9 are non-zero as expected for Fibonacci.

### Current Issue
Stage 5 sumcheck verification fails:
```
output_claim:   [9d, 3d, dd, d4...]
expected_claim: [1b, 43, f0, ba...]
```

### Deep Analysis of the Formula

#### Jolt Verifier Formula (read_raf_checking.rs:1285-1321)
```rust
// For each table i, get table_flag[i] from opening claims
let table_flag_claims: Vec<F> = (0..42).map(|i|
    accumulator.get_virtual_polynomial_opening(LookupTableFlag(i), InstructionReadRaf)
).collect();

// Evaluate each table's MLE at the address point
let val_evals: Vec<_> = LookupTables::iter()
    .map(|table| table.evaluate_mle(&r_address_prime.r))
    .collect();

// val_claim = Σ (table_MLE[i](r_address) * table_flag[i])
let val_claim = val_evals.into_iter()
    .zip(table_flag_claims)
    .map(|(eval, flag)| eval * flag)
    .sum();

// raf_claim formula
let raf_claim = (1 - raf_flag) * (left_op + gamma*right_op)
              + raf_flag * gamma * identity;

// Expected output
expected = eq_eval * ra_claim * (val_claim + gamma * raf_claim)
```

#### Mathematical Interpretation
The sumcheck is over:
```
Σ_j Σ_k eq(j, r_reduction) * ra(k, j) * (table_func[t(j)](k) + gamma * raf(k, j))
```

After all rounds:
- `table_flag[i]` = Σ_{j: t(j)=i} eq(r_cycle_prime, j) - this is what prover provides
- `val_claim` = Σ_i table_MLE[i](r_address) * table_flag[i]
  - This reconstructs the weighted sum of table evaluations

The KEY insight: The verifier doesn't know which table was used at each cycle. Instead:
1. Prover provides `table_flag[i]` for each table i
2. Verifier evaluates `table_MLE[i](r_address)` for each table
3. Verifier computes `val_claim = Σ_i table_MLE[i](r_address) * table_flag[i]`

This works because if cycle j uses table t(j), then:
- `table_flag[t(j)]` includes `eq(r_cycle_prime, j)`
- The contribution becomes `table_MLE[t(j)](r_address) * eq(r_cycle_prime, j)`

### Implementation Status

Added `evaluateTableMLE(table_index, r)` function in lookup_table/mod.zig to evaluate any of the 42 tables by index.

### Next Steps

1. The prover sumcheck polynomials must be computed consistently with this formula
2. Check if Zolt's combined_vals computation matches what the verifier expects
3. The prover should compute:
   - For each round: contribution = eq(j) * ra(j) * (table_output(j) + gamma * raf(j))
4. After all rounds, output_claim should equal verifier's expected_claim

### Possible Issue

Zolt computes `combined_vals[j] = lookup_output + gamma*left + gamma^2*right` where:
- `lookup_output = step.rd_value` (actual instruction output)

But the formula should be:
- `val = table_func[t(j)](address(j))` (table MLE evaluated at cycle's address)
- `raf = (1-is_identity)*(left + gamma*right) + is_identity*gamma*identity`
- `combined = val + gamma * raf`

The `lookup_output` should match `table_func[t(j)](address(j))` for correct execution, but there may be a formula mismatch in how `raf` is computed.

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy to /tmp for Jolt test
cp logs/zolt_*.bin /tmp/

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
