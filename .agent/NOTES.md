# Zolt-Jolt Compatibility Notes

## Current Status (Session 65 - January 7, 2026)

### Summary

**Stage 1 PASSES, Stage 2 FAILS at sumcheck verification**

The Stage 2 batched sumcheck produces `output_claim` that doesn't match the verifier's `expected_output_claim`.

### Recent Fixes Applied

1. **tau_stage2 construction** - Now uses reversed Stage 1 challenges + tau_high_stage2
2. **Gruen polynomial construction** - Changed from direct evaluation to t0/t_inf method
3. **Compressed coefficient handling** - Correctly reconstruct evaluations from [c0, c2, c3]
4. **Index cleanup** - Using standard MLE indexing conventions

### Current Values

Latest test output:
- `output_claim`: 6712305349781773213915836621366914034919475189156389350844584049700714314367
- `expected_output_claim`: 8321209767613183988201183581193448103173376364360119728613327078546122551176

### Analysis

The `expected_output_claim` in Jolt is computed as:
```
L(tau_high, r0) * eq(tau_low, r_reversed) * fused_left(claims) * fused_right(claims)
```

Where:
- `L(tau_high, r0)` = Lagrange kernel
- `eq(tau_low, r_reversed)` = eq polynomial with reversed challenges
- `fused_left/right` = combinations of 8 factor polynomial claims

The `output_claim` is the final claim from sumcheck evaluation.

### Potential Issues

1. **Round polynomial computation** - t0 and t_inf may not match Jolt's calculation
2. **Split eq tables** - E_out/E_in indexing may differ
3. **Binding order interaction** - LowToHigh vs how variables are actually bound
4. **Initial polynomial layout** - How fused left/right are stored

### Key Insight: Jolt's Interleaved Storage

Jolt's `compute_first_quadratic_evals_and_bound_polys` stores left/right in interleaved format:
```rust
let off = 2 * x_in_val;
left_chunk[off] = left0;      // lo value (even trace idx)
left_chunk[off + 1] = left1;  // hi value (odd trace idx)
```

My Zolt code stores sequentially:
```zig
left_evals[idx] = fusedLeft(witness[idx]);
```

For standard MLE representation, this sequential storage IS correct because:
- Index i has binary decomposition where LSB = i mod 2
- Indices 2k and 2k+1 ARE the lo/hi pair for the same upper bits

### Next Steps

1. Add detailed debug output comparing:
   - t0 and t_inf values between Zolt and Jolt
   - Split eq E_out and E_in tables
   - Current scalar progression

2. Check if the issue is in:
   - How `getWindowEqTables` returns tables
   - How Gruen polynomial combines t0/t_inf
   - The claim update logic

3. Consider creating a minimal test case that runs the same computation in both Zolt and Jolt

---

## Previous Sessions

See git history for earlier notes on Stage 1 fixes.
