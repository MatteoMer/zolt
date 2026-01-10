# Zolt-Jolt Compatibility TODO

## Current Status: Stage 3 Expected Output Claim Investigation

### Issues Fixed
1. **Binding order** - All provers now use LowToHigh binding consistently
2. **Degree-2 extrapolation** - Properly extrapolate shift/reg to x=3 using `p(3) = p0 - 3*p1 + 3*p2`
3. **Sumcheck invariant** - `p(0)+p(1) = claim` verified at each round
4. **Batched polynomial consistency** - `batched_sum == combined_claim` verified

### Current Issue
Sumcheck rounds pass but final output_claim doesn't match expected_output_claim:
```
output_claim:          14479714286458141723416680681829317356766515671419249179721686829661484481208
expected_output_claim: 19829957079326482140855077841713818330258980683739449213840778147525912490313
```

### Understanding the Verifier
From Jolt expert analysis, verifier computes:
```rust
expected_output_claim = sumcheck_instances
    .iter()
    .zip(batching_coeffs.iter())
    .map(|(sumcheck, coeff)| {
        sumcheck.cache_openings(accumulator, transcript, r_slice);
        sumcheck.expected_output_claim(accumulator, r_slice) * coeff
    })
    .sum();
```

For Shift specifically:
```rust
expected = eq+1(r_outer, r_final) * [upc(r_final) + γ*pc(r_final) + γ²*virt(r_final) + γ³*first(r_final)]
         + γ⁴ * (1 - noop(r_final)) * eq+1(r_product, r_final)
```

### Likely Root Cause
The prover's shift polynomial should match this formula exactly. Need to verify:
1. eq+1 polynomial evaluations at r_final match between prover and verifier
2. Opening claims (upc, pc, etc.) match the MLE evaluations at r_final
3. Phase 2 transition correctly materializes eq+1 polynomials

### Next Steps
1. Add debug to print eq+1 evaluations at final challenge point
2. Verify opening claims match expected MLE values
3. Check Phase 2 eq+1 materialization in ShiftPrefixSuffixProver

## Overall Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✓ PASSES | MLE consistency fixed |
| 2 | ✓ PASSES | Product virtualization works |
| 3 | IN PROGRESS | Sumcheck rounds pass, output claim mismatch |
| 4-7 | Blocked | Waiting on Stage 3 |

## Files Modified
- `src/zkvm/spartan/stage3_prover.zig` - binding order, extrapolation, debug output

## Testing
```bash
bash scripts/build_verify.sh
```
