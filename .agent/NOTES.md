# Zolt-Jolt Cross-Verification Progress

## Session 18 Summary

### Major Progress
1. **All opening claims verified to match exactly**
   - l_inst, r_inst, is_rd_not_zero, next_is_noop ✅
   - fused_left, fused_right ✅
   - ra_claim, val_claim, inc_claim ✅

2. **Instance 0 (ProductVirtual) final claim matches** - Sumcheck produces correct result

3. **Stage 1 passes completely** - All 712 internal tests pass

### Remaining Issue: Instance 2 (RWC)

Despite all opening claims matching, the RWC sumcheck produces a different final claim.

**Numbers:**
- Our RWC final claim: 17925181248966282971112807010799772681208014801023116248823233609842789352688
- Jolt expected claim: 11216823976254905917561036500968546773134980482196871908475958474138871482864
- Ratio: ~1.6x (ours is larger)

### Root Cause Analysis

The expected formula is:
```
final_claim = eq(r_cycle_params, r_cycle_sumcheck) * ra(opening_point) * (val + γ*(val + inc))
```

Our sumcheck produces a different value because our eq polynomial handling is incorrect.

### Jolt's RWC Implementation (from expert analysis)

1. **Phase 1**: Uses `GruenSplitEqPolynomial` with E_out/E_in tables
2. **Phase 2**: Uses `merged_eq` after Phase 1 completes
3. `current_scalar` accumulates eq(w[i], r) as variables are bound

Key: The Gruen structure progressively reduces tables as challenges bind. Our simple `eq_evals[]` array doesn't properly account for binding.

### What We Tried

1. Recompute eq on-the-fly using bound challenges + params.r_cycle - Still wrong
2. Simplified to use precomputed eq_evals directly - Same result
3. Updated Phase 2 to compute eq_cycle properly - No change

### Technical Insight

When binding challenge r at round i, the eq polynomial folds:
- Original: eq(w, x) over all x ∈ {0,1}^n
- After bind: eq(w', x') over x' ∈ {0,1}^{n-1}
where the contribution from variable i is absorbed into a scalar.

Our implementation doesn't properly track this folding.

### Next Steps

1. Study Jolt's GruenSplitEqPolynomial::bind() more carefully
2. Implement proper eq folding in our RWC prover
3. Consider adding per-round debugging to compare with Jolt

## Technical References

- Jolt RAM checking: `jolt-core/src/zkvm/ram/read_write_checking.rs`
- Gruen optimization: `jolt-core/src/poly/split_eq_poly.rs`
- Our RWC prover: `src/zkvm/ram/read_write_checking.zig`
