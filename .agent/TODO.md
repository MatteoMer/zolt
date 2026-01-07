# Zolt-Jolt Compatibility - Status Update

## Current Status: Session 12 (Continued)

### Stage 2 Sumcheck Failure - Root Cause Found!

- **Stage 1: PASSING ✅** - Sumcheck output_claim matches expected
- **Stage 2: FAILING ❌** - output_claim != expected_output_claim

```
output_claim:          15813746968267243297450630513407769417288591023625754132394390395019523654383
expected_output_claim: 21370344117342657988577911810586668133317596586881852281711504041258248730449
```

### MAJOR BREAKTHROUGH: Final Values Match!

After extensive debugging, I found that **ALL** of the following match exactly between Zolt and Jolt:

1. ✅ **left[0]** - Prover's final left polynomial value
2. ✅ **right[0]** - Prover's final right polynomial value
3. ✅ **split_eq.current_scalar** - Equals tau_high_bound_r0 * tau_bound_r_tail_reversed
4. ✅ **fused_left** - Computed from factor claims matches left[0]
5. ✅ **fused_right** - Computed from factor claims matches right[0]
6. ✅ **left * right * eq** - ProductVirtualRemainder's expected claim = 19131931169602397700325994909403612608099043325475370665412093729674884104319
7. ✅ **Initial batched claim** - 712179532459811457325748625852270562576712947602781856366289191649144450642

### The Problem

Since the final polynomial evaluations are correct, but the sumcheck output_claim is wrong, the issue is in **how the round polynomials accumulate** during the sumcheck:

The sumcheck verifier iterates:
1. Start with batched_initial_claim (matches ✅)
2. For each round: claim = round_poly(challenge)
3. After 26 rounds: output_claim (wrong!)

The round polynomial evaluations at each challenge must be producing a different trajectory than expected, even though the final values are correct.

### Possible Issues

1. **Round polynomial construction** - The compressed [c0, c2, c3] being written might be computed correctly for the claim at that round, but the hint recovery in the verifier uses a different claim
2. **Batching coefficient application** - The way instances are combined in each round might differ
3. **Claim update timing** - When the claim is updated vs when it's used for next round

### Next Steps

1. Add detailed per-round debug to compare:
   - Zolt's claim before each round
   - Zolt's round polynomial evaluations s(0), s(1), s(2), s(3)
   - Zolt's claim after evaluating s(challenge)
2. Compare with Jolt's verifier step-by-step trace
3. Find the first round where claims diverge

### Key Finding: Batching Matters!

The expected_output_claim is **not** just the ProductVirtualRemainder's claim, but:
```
expected = coeff[0] * product_claim + coeff[1] * 0 + coeff[2] * 0 + coeff[3] * 0 + coeff[4] * 0
        = coeff[0] * product_claim
        = 338733333185391054954706473760189339532 * 19131931169602397700325994909403612608099043325475370665412093729674884104319
        = 21370344117342657988577911810586668133317596586881852281711504041258248730449 (mod p)
```

The sumcheck output_claim should equal this batched value, but it equals 15813746... instead.

### Key Files

- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainderProver
- `src/poly/split_eq.zig` - GruenSplitEqPolynomial
- `src/zkvm/proof_converter.zig` - Stage 2 batched sumcheck
- `jolt-core/src/subprotocols/sumcheck.rs` - Jolt's sumcheck verifier

## Previous Sessions

### Session 11
- Verified extensive formula matching between Zolt and Jolt
- Identified that polynomial indexing/storage must match exactly

### Session 10
- Fixed output-sumcheck r_address_prime reversal
- Stage 1 started passing

### Session 9
- Fixed transcript challenge sampling
- Aligned MontU128Challenge representation
