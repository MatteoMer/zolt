# Zolt-Jolt Compatibility TODO

## Current Status: Session 38 - January 2, 2026

**Progress: Deep analysis of t_prime_poly construction and indexing**

---

## Key Findings

### 1. EqPolynomial is Correct
- Partition of unity test passes
- Sum of eq evaluations equals 1

### 2. Individual Az/Bz MLEs Match
- `Az MLE match: true, Bz MLE match: true`

### 3. Inner Product Mismatch
- Prover's implicit inner product is ~79.5% of expected
- This is the core issue to resolve

---

## Root Cause Analysis

### Jolt's Two Materialization Paths

Jolt has two different functions for materializing Az/Bz:

1. **`fused_materialise_polynomials_round_zero`** (first linear round)
   - Uses `lagrange_evals_r0` directly (NO r_grid scaling)
   - Iterates: `full_idx = grid_size * i + j`, `time_step = full_idx >> 1`
   - When grid_size >= 2: fills two positions (az0, az1) per iteration

2. **`fused_materialise_polynomials_general_with_multiquadratic`** (subsequent rounds)
   - Uses `scaled_w = lagrange_evals_r0 * r_grid[r_idx]`
   - More complex indexing with base_idx, x_val_shifted, r_idx

### Zolt's Single Path

Zolt only has one materialization function that ALWAYS uses r_grid:
```zig
scaled_w[r_idx][i] = lagrange_evals_r0[i] * r_grid[r_idx]
```

For first round, r_grid = [1.0], so this should be equivalent. But the indexing structure may differ.

### Indexing Formula Difference

**Jolt round_zero**:
```rust
let full_idx = grid_size * i + j;  // Multiplication
let time_step_idx = full_idx >> 1;
```

**Zolt**:
```zig
const base_idx = (x_out << bits) | (x_in << bits);  // Bitwise OR
const full_idx = base_idx | x_val | r_idx;
const step_idx = full_idx >> 1;
```

These are mathematically equivalent when the bit positions don't overlap, but the structure is different.

---

## Investigation Needed

1. **Trace both implementations** with the same inputs and compare cycle indices accessed
2. **Verify head_in_bits/head_out_bits** calculation matches Jolt
3. **Check the `grid_size >= 2` special case** - Jolt fills two positions per iteration

---

## Test Commands

```bash
# Run Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Next Steps

1. Add detailed debug logging to both Zolt's materialization and Jolt's round_zero
2. Compare cycle indices, group selections, and accumulated values
3. Identify the exact point of divergence
