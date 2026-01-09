# Zolt-Jolt Compatibility TODO

## Current Status: Stage 3 ShiftSumcheck Debugging

### Key Finding (2025-01-09)
The prefix-suffix decomposition is mathematically correct:
- `grand_sum(P*Q)` = `{ 51, 156, 139, 8, ... }`
- `direct_sum` = `{ 51, 156, 139, 8, ... }` (matches grand_sum!)
- **BUT** `input_claim` = `{ 254, 135, 82, 218, ... }` (completely different)

The sumcheck would work if grand_sum == input_claim, but they don't match!

### Hypotheses to Test
1. **Opening claim point mismatch**: The input claim might be computed at a different point (r_outer) than where we're evaluating the sum.

2. **Wrong polynomial structure**: The "Next" virtual polynomials might not have the relationship I expect:
   - I assume: `NextUnexpandedPC[j] = UnexpandedPC[j+1]`
   - Reality might be different

3. **Witness index mismatch**: The indices used in shift sumcheck might differ from what Jolt uses.

### Next Steps
1. Add debug output to compare `r_outer` point used in input claim vs direct sum
2. Verify the relationship between "Next" polynomials and current cycle values
3. Check if the outer sumcheck opening claims are at the right point

## Implementation Done
- [x] Stage 3 prover structure
- [x] ShiftPrefixSuffixProver with 4 (P,Q) pairs
- [x] InstructionInputProver (direct computation)
- [x] RegistersPrefixSuffixProver with 1 (P,Q) pair
- [x] Phase 2 transition with eq+1 materialization

## Pending
- [ ] Debug why grand_sum != input_claim
- [ ] Test Stage 3 against Jolt verifier
- [ ] Implement Stages 4-7 sumcheck provers
- [ ] Full proof verification test

## Key Formulas

### EqPlusOnePrefixSuffixPoly Decomposition
```
eq+1((r_hi, r_lo), (y_hi, y_lo)) =
    eq+1(r_lo, y_lo) * eq(r_hi, y_hi) +
    is_max(r_lo) * is_min(y_lo) * eq+1(r_hi, y_hi)
```

Where:
- r is split at mid: r_hi = r[0..mid], r_lo = r[mid..n]
- prefix_0[j] = eq+1(r_lo, j)
- suffix_0[j] = eq(r_hi, j)
- prefix_1[0] = is_max(r_lo), all other indices = 0
- suffix_1[j] = eq+1(r_hi, j)
- is_max(x) = product of x[i] for all i (= eq((1)^n, x))

### Q Buffer Construction
```
Q[x_lo] = Σ_{x_hi} witness(x) * suffix[x_hi]
where x = x_lo + (x_hi << prefix_n_vars)
```

### Shift Sumcheck Relationship
```
input_claim = NextUnexpandedPC(r) + γ*NextPC(r) + γ²*NextIsVirtual(r) + γ³*NextIsFirst(r) + γ⁴*(1-NextIsNoop(r))
sumcheck = Σ_j eq+1(r,j) * [UPC(j) + γ*PC(j) + γ²*Virt(j) + γ³*First(j)] + γ⁴*Σ_j eq+1(r_prod,j)*(1-Noop(j))

These should be equal because:
- eq+1(r, j) = 1 when j = r + 1
- So Σ_j eq+1(r,j) * f(j) = f(r+1) = Next(r)
```

## Overall Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✓ PASSES | Outer Spartan sumcheck works |
| 2 | ✓ PASSES | Product virtualization works |
| 3 | DEBUGGING | grand_sum != input_claim |
| 4-7 | Blocked | Waiting on Stage 3 |

## Testing Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
  --export-preprocessing /tmp/zolt_preprocessing.bin \
  -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Verify with Jolt
cd /Users/matteo/projects/jolt && cargo test -p jolt-core \
  test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# Run Zolt tests
zig build test
```
