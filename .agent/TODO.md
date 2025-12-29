# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck - Implicit Az*Bz Mismatch (Session 26)

### Key Finding
The implicit Az*Bz (computed as output_claim / eq_factor) does NOT match the expected inner_sum_prod:

```
inner_sum_prod (expected Az*Bz) = 12743996023445103930025687297173833157935883282725550257061179867498976368827
implicit Az*Bz (output/eq_factor) = 6845670145302814045138444113000749599157896909649021689277739372381215505241
```

The ratio is NOT a simple integer, suggesting a non-trivial computational difference.

### Verified Correct
- [x] All 656 Zolt tests pass
- [x] UniSkip verification passes
- [x] All individual round equations pass (p(0)+p(1)=claim)
- [x] Lagrange kernel computation and domain matches Jolt's {-4..5}
- [x] Split eq polynomial structure matches Jolt's GruenSplitEqPolynomial
- [x] Sum-of-products computation (not product-of-sums)
- [x] ExpandingTable (r_grid) logic matches Jolt
- [x] Constraint group indexing
- [x] Eq factor accumulation (tau * challenges ordering)
- [x] r_cycle endianness for opening claims

### Root Cause Hypothesis
The Az*Bz values computed during the sumcheck differ from what the opening claims produce.
This could be due to:
1. Witness access ordering in the sumcheck
2. Constraint evaluation differences
3. Some subtle accumulation issue in the cycle rounds

### Next Steps
1. Add debug output to Zolt's streaming round to print first few Az/Bz values
2. Add debug output to Jolt test to print per-cycle Az/Bz at the opening point
3. Compare these intermediate values to find the divergence

### Test Commands
```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin --max-cycles 1024

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```

## Completed Milestones
- [x] Blake2b transcript implementation
- [x] Field serialization (Arkworks format)
- [x] UniSkip polynomial generation
- [x] Stage 1 remaining rounds sumcheck
- [x] R1CS constraint definitions
- [x] Split eq polynomial factorization
- [x] Lagrange kernel computation
- [x] Opening claims with MLE evaluation
- [x] Sum-of-products computation
