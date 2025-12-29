# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck Output Claim Mismatch (Session 26)

### The Problem
All 11 sumcheck round equations pass (p(0) + p(1) = claim), but the final output_claim doesn't match the expected_output_claim from R1CS evaluation.

**Values from test:**
```
output_claim = 7120341815860535077792666425421583012196152296139946730075156877231654137396
expected     = 2000541294615117218219795634222435854478303422072963760833200542270573423153
Ratio ≈ 3.56 (not a simple power-of-2 discrepancy)
```

### Verified Correct
- [x] All 656 Zolt tests pass
- [x] UniSkip verification passes
- [x] All individual round equations pass (p(0)+p(1)=claim)
- [x] Lagrange kernel computation matches Jolt
- [x] Split eq polynomial structure matches Jolt's GruenSplitEqPolynomial
- [x] Sum-of-products (not product-of-sums) computation
- [x] ExpandingTable (r_grid) logic matches Jolt
- [x] Constraint group indexing (FIRST_GROUP_INDICES, SECOND_GROUP_INDICES)

### Remaining Investigation Areas
1. **Eq factor accumulation** - Though structure seems correct, actual values may differ
2. **Cycle witness ordering** - Endianness of cycle indices in witness access
3. **MLE evaluation point** - The r_cycle used for opening claims vs sumcheck point

### Key Formulas
Expected claim formula:
```
expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
         = L(tau_high, r0) * eq(tau_low, r_challenges) * az_final * bz_final
```

Sumcheck output:
```
output_claim = f(r0, r_stream, r_cycle)
             = eq(tau, r) * Az(r) * Bz(r)
```

These SHOULD be equal since:
- eq(tau, r) = L(tau_high, r0) * eq(tau_low, r_challenges)
- Az(r) * Bz(r) = az_final * bz_final (MLE evaluations at the random point)

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

## Next Session
Need to add diagnostic output to pinpoint the exact point of divergence:
1. Print Zolt's final eq factor (`current_scalar` after all binds)
2. Print Zolt's implicit Az*Bz (output_claim / eq_factor)
3. Compare with Jolt's `inner_sum_prod`
