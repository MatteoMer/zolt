# Zolt-Jolt Compatibility: Stage 2 Debug

## Status: Instance 0 Expected Claim Investigation ⏳

## Session Summary (2026-01-28 evening)

### Major Breakthrough
**Factor evaluations ARE correct!** The opening_claims stored by Zolt match what Jolt reads:
- Zolt factor[0] (LeftInstructionInput) LE: `fd 52 a8 83 5d 65 a5 6f ...`
- Jolt reads l_inst: `[fd, 52, a8, 83, 5d, 65, a5, 6f, ...]`

**r_cycle (opening point) is also correct:**
- Zolt r_cycle[0] LE: `60 93 28 51 48 90 bf 6d ...`
- Jolt opening_point[0]: `[60, 93, 28, 51, 48, 90, bf, 6d, ...]`

### Current Issue

The factor evaluations match, but Jolt's `expected_output_claim` calculation still produces a different result. This suggests the issue is in:

1. **tau parameters** - used in tau_high_bound_r0 and tau_bound_r_tail_rev
2. **r0 (univariate skip challenge)** - used in Lagrange weights and Lagrange kernel
3. **Lagrange polynomial computation** - weights w[i] for fusing

### Jolt's Expected Output Claim Formula

For Instance 0 (ProductVirtualRemainder):
```
fused_left = w[0]*l_inst + w[1]*is_rd_not_zero + w[2]*is_rd_not_zero + w[3]*lookup_out + w[4]*j_flag
fused_right = w[0]*r_inst + w[1]*wl_flag + w[2]*j_flag + w[3]*branch + w[4]*(1-next_is_noop)
tau_high_bound_r0 = L(tau_high, r0)  // Lagrange kernel
tau_bound_r_tail_rev = eq(tau_low, r_cycle^rev)

expected = tau_high_bound_r0 * tau_bound_r_tail_rev * fused_left * fused_right
```

### Jolt's Intermediate Values

From debug:
- fused_left: `[0d, 8b, ee, 53, ...]`
- fused_right: `[b5, 12, 81, 28, ...]`
- tau_high_bound_r0: `[5c, 00, 3f, 64, ...]`
- tau_bound_r_tail_rev: `[48, b6, bc, 4e, ...]`
- expected_output_claim: `[18, f9, 1f, 65, ...]`

### Next Steps

1. Add debug to Zolt to compute the same intermediate values
2. Compare tau_high, tau_low, r0 between Zolt and Jolt
3. Compare Lagrange weights w[i]
4. Compare fused_left, fused_right
5. Identify which calculation diverges

### Technical Insight

The sumcheck passes because:
- Round polynomials satisfy p(0)+p(1)=claim at each round
- Final output_claim matches between prover and verifier

But expected_output_claim differs because:
- The formula computes what the sumcheck SHOULD have produced
- If the polynomial being proved differs from what the formula expects, they won't match

### Files Modified

- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - debug output
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/spartan/product.rs` - debug output
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - debug output
