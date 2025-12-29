# Zolt-Jolt Compatibility TODO

## Current Status: Investigating Jolt's Index Structure (Session 24)

### Root Cause Identified

The fundamental issue is that:
- **Prover computes**: `Σ (Az * Bz)` - sum of products
- **Verifier expects**: `(Σ Az) * (Σ Bz)` - product of sums

These are mathematically different due to non-linearity of multiplication.

### Initial Fix Attempt

Modified streaming_outer.zig to:
1. Accumulate separate Az and Bz sums
2. Expand to {0, 1, ∞} grid
3. Multiply after expansion

**Result**: Output_claim changed but still doesn't match expected.

### Current Test Values

```
output_claim:    5514162482559916804432512270777756912898528478061111678566043859523325463688
expected_claim: 11278176879827447390261533528479013530892542028574834464976023912397580784757
Match: false
```

### Understanding Jolt's Structure

Jolt's streaming prover in `extrapolate_from_binary_grid_to_tertiary_grid`:

1. **Per (out_idx, in_idx) pair**:
   - Iterates over `j` (window position) and `k` (r_grid index)
   - Computes `full_idx = offset + j * klen + k`
   - Derives `step_idx = full_idx >> 1` and `selector = full_idx & 1`
   - Accumulates `grid_a[j]` and `grid_b[j]` weighted by `scaled_w[k]`

2. **Key insight**: The k-loop mixes:
   - Cycle index (`step_idx`)
   - Constraint group (`selector`)
   - Bound challenge weights (`r_grid[k]`)

3. **Multiquadratic expansion**:
   - `grid_a` and `grid_b` are expanded to {0, 1, ∞}
   - Products computed: `buff_a[i] * buff_b[i]`

4. **Accumulation**:
   - Weighted by `e_in * e_out` and summed

### What's Different in Zolt

Zolt's approach:
- Iterates directly over cycles
- Computes Az/Bz per-cycle with Lagrange weights
- Doesn't handle the mixed (step, selector, k) index space

### Key Insight

The `scaled_w[k] = lagrange_evals[constraint] * r_grid[k]` weighting means:
- Each k value corresponds to a different (step, selector) pair
- The sum over k is already a "local product-of-sums" over the bound challenges
- This structure is crucial for the math to work out

### Next Steps

1. **Study the index mapping more carefully**:
   - How does Jolt's `full_idx = offset + j * klen + k` work?
   - What's the relationship between k, step_idx, and selector?

2. **Replicate Jolt's exact structure**:
   - Build per-(out,in) grids with correct k-loop
   - Use scaled_w weighting
   - Expand and multiply on multiquadratic grid

3. **Key functions to port**:
   - `extrapolate_from_binary_grid_to_tertiary_grid` (outer.rs:593-635)
   - The index decomposition logic

### Reference Files

Jolt:
- `outer.rs:593-635` - grid building with k-loop
- `outer.rs:704-725` - multiquadratic expansion and product
- `multiquadratic_poly.rs` - {0,1,∞} expansion

## Test Commands

```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --max-cycles 1024

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
