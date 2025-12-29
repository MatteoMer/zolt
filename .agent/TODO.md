# Zolt-Jolt Compatibility TODO

## Current Status: Investigating Outer Sumcheck Output Claim Mismatch (Session 26)

### Current Issue

All 11 sumcheck rounds pass individually (p(0) + p(1) = claim), but the final output_claim doesn't match the expected_output_claim computed from R1CS evaluation.

Test output:
```
output_claim (from sumcheck):    7120341815860535077792666425421583012196152296139946730075156877231654137396
expected_output_claim (from R1CS): 2000541294615117218219795634222435854478303422072963760833200542270573423153
Match: false
```

Ratio is ~3.56, not a simple power-of-2 discrepancy.

### Analysis

#### What Should Match:
```
output_claim = L(tau_high, r0) * eq(tau_low, r_challenges) * Az_final * Bz_final
expected     = L(tau_high, r0) * eq(tau_low, r_tail_reversed) * inner_sum_prod
```

#### Eq Factors Analysis:
Zolt accumulates via split_eq:
- `current_scalar = L(tau_high, r0) * eq(tau[10], r_stream) * eq(tau[9], r_1) * ... * eq(tau[0], r_10)`

Jolt computes:
- `tau_high_bound_r0 * eq(tau[0..11], [r_10, ..., r_1, r_stream])`

These should be identical (same factors, same values).

#### Likely Issue: Az * Bz Mismatch

Since eq factors match, the issue is probably:
1. **Witness data mismatch** - Zolt's cycle_witnesses may have different values than expected
2. **Constraint evaluation** - Az/Bz calculation may differ from Jolt's expectations
3. **Cycle indexing** - There may be an endianness issue in how cycles are indexed
4. **Opening claims** - The R1CS input evaluations may be computed at wrong point

### Debugging Plan

1. Add debug output to Zolt prover to print:
   - First few cycle's Az/Bz values for both groups
   - The sum over all cycles
   - The computed vs expected inner_sum_prod

2. Add debug output to Jolt test to print:
   - The z vector values
   - Az/Bz computations at the opening point

3. Compare these intermediate values to find the discrepancy

### Test Commands

```bash
# Zolt tests (all 656+ should pass)
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --max-cycles 1024

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```

## Completed

- [x] Blake2b transcript implementation
- [x] Field serialization (Arkworks format)
- [x] UniSkip polynomial generation
- [x] Stage 1 remaining rounds sumcheck
- [x] R1CS constraint definitions
- [x] Split eq polynomial factorization
- [x] Lagrange kernel computation
- [x] Opening claims with MLE evaluation
- [x] Sum-of-products computation (not product-of-sums)
