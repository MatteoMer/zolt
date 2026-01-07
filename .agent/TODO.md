# Zolt-Jolt Compatibility - Status Update

## Current Status: Session 11

### Stage 2 Sumcheck Failure

- **Stage 1: PASSING ✅** - Sumcheck output_claim matches expected
- **Stage 2: FAILING ❌** - output_claim != expected_output_claim

```
output_claim:          15813746968267243297450630513407769417288591023625754132394390395019523654383
expected_output_claim: 21370344117342657988577911810586668133317596586881852281711504041258248730449
```

### What Matches (Verified)
1. ✅ tau_high matches between Zolt and Jolt
2. ✅ Initial batched claim matches
3. ✅ All 26 Stage 2 challenges match byte-for-byte
4. ✅ Round polynomial coefficients match (c0, c2, c3 for rounds 0, 25)
5. ✅ Virtual polynomial factor evaluations match (l_inst, r_inst, etc.)
6. ✅ Opening claims serialize correctly

### Root Cause Hypothesis

Despite matching coefficients, the final polynomial evaluation differs. The sumcheck
polynomial evaluation uses `eval_from_hint` which reconstructs the polynomial and
evaluates. The discrepancy could be in:

1. **eq polynomial in the polynomial being proved** - Zolt's ProductVirtualRemainderProver
   computes `L(τ_high, r0) * Eq(τ_low, x) * fused_left(x) * fused_right(x)`

2. **The split_eq polynomial binding** - How `current_scalar` accumulates during rounds

3. **Round start offset** - ProductVirtualRemainder starts at round `log_ram_k` (16),
   but the expected_output_claim uses only the last `n_cycle_vars` (10) challenges

### Key Files

- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainderProver
- `src/poly/split_eq.zig` - GruenSplitEqPolynomial
- `src/zkvm/proof_converter.zig` - Stage 2 batched sumcheck (generateStage2BatchedSumcheckProof)

### Next Investigation Steps

1. Add detailed trace in `computeCubicRoundPoly` showing:
   - q_constant (t0_sum)
   - q_quadratic_coeff (t_inf_sum)
   - current_scalar (eq contribution)

2. Compare eq polynomial values between Zolt prover and Jolt verifier at each round

3. Check if ProductVirtualRemainder's `r_cycle` calculation matches what Jolt expects

## Previous Sessions

### Session 10
- Fixed output-sumcheck r_address_prime reversal
- Stage 1 started passing

### Session 9
- Fixed transcript challenge sampling
- Aligned MontU128Challenge representation
