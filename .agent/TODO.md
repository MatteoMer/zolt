# Zolt-Jolt Compatibility TODO

## Current Status: Session 35 - January 2, 2026

**All 710 Zolt tests pass, but Jolt verification still fails**

### Completed This Session

1. **MultiquadraticPolynomial.bind()** - Added Jolt-compatible quadratic interpolation
   - Formula: `f(r) = f(0)*(1-r) + f(1)*r + f(∞)*r(r-1)`

2. **GruenSplitEqPolynomial.getEActiveForWindow()** - Added E_active projection

3. **t_prime_poly integration** - Added to StreamingOuterProver
   - Build, rebuild, bind, and project t_prime_poly

4. **LinearOnlySchedule** - Fixed round handling
   - Round 1 now initializes linear stage (matches Jolt's round 0)
   - All rounds use linear phase (no streaming rounds)

---

## Current Issue: Sumcheck Output Claim Mismatch

The sumcheck verification fails with:
```
output_claim:          8206907536993754864973510285637683658139731930814938521485939885759521476392
expected_output_claim: 5887936957248500858334092112703331331673171118046881060635640978343116912473
```

### The Expected Output Claim Formula

From `outer.rs:expected_output_claim`:
```rust
let rx_constr = &[sumcheck_challenges[0], self.params.r0];  // [r_stream, r0]
let inner_sum_prod = self.key.evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);

let tau_high_bound_r0 = LagrangePolynomial::lagrange_kernel(&tau_high, &r0);
let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);

result = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
```

### Key Variables
- `r0`: UniSkip challenge (used in Lagrange kernel)
- `r_stream = sumcheck_challenges[0]`: First remaining sumcheck challenge (constraintグループ selector)
- `r_tail_reversed`: All remaining sumcheck challenges in reverse order

### Suspected Issues

1. **E_active calculation may be wrong** for the projection
2. **t_prime_poly structure** might not match Jolt's base-3 indexing
3. **The expandGrid function** may not be implementing correctly

---

## Debug Strategy

Need to add debug output to compare:
1. t_prime_poly values at each round
2. (t_zero, t_infinity) projections at each round
3. Round polynomial evaluations [s(0), s(1), s(2), s(3)]
4. Compare with Jolt's values at each step

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/poly/multiquadratic.zig` | Added bind(), isBound(), finalSumcheckClaim() |
| `src/poly/split_eq.zig` | Added getEActiveForWindow() |
| `src/zkvm/spartan/streaming_outer.zig` | Major restructure for LinearOnlySchedule |

---

## Test Commands

```bash
# Run Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification tests
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
