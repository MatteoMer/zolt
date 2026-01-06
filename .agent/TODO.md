# Zolt-Jolt Compatibility TODO

## Current Status: January 6, 2026 - Session 61

**STATUS: ROOT CAUSE IDENTIFIED - Sumcheck output_claim is wrong**

### BREAKTHROUGH FINDING

Everything computes correctly EXCEPT the final output_claim:

| Value | Zolt | Jolt | Match? |
|-------|------|------|--------|
| eq_factor (tau_factor) | 15055487594231608754301087582065770346029795683588058855016053184153343680259 | 15055487594231608754301087582065770346029795683588058855016053184153343680259 | ✅ |
| az_final * bz_final | 1221027240042985780108460212824162278077143256096887971142513640043566180374 | 1221027240042985780108460212824162278077143256096887971142513640043566180374 | ✅ |
| expected_output_claim | 2434835346226335308248403617389563027378536009995292689449902135365447362920 | 2434835346226335308248403617389563027378536009995292689449902135365447362920 | ✅ |
| **actual output_claim** | **13737337490099890393959518040416259194985808737524904718040235576315533938309** | 2434835346226335308248403617389563027378536009995292689449902135365447362920 | ❌ |

### The Bug

The sumcheck polynomial evaluations produce the WRONG accumulated claim. The final Az/Bz polynomial values are correct, but:
```
expected = eq_factor * (az * bz)
actual = (something wrong) * (az * bz)
```

The ratio `actual / (az * bz)` gives a different value than `eq_factor`.

### Root Cause Hypothesis

The SplitEq polynomial's `current_scalar` is correct at the end, but the claim propagation during rounds doesn't properly incorporate the eq scaling into the polynomial coefficients.

In sumcheck:
- Each round polynomial s(X) should evaluate such that s(0) + s(1) = previous_claim
- The claim updates as: new_claim = s(challenge)
- After all rounds: final_claim = eq_factor * inner_sum_prod

The issue is likely that the round polynomial construction doesn't properly scale by eq values.

### Files to Investigate

1. `/Users/matteo/projects/zolt/src/zkvm/spartan/streaming_outer.zig`
   - `computeRoundPoly` - How round polynomials are constructed
   - `updateClaim` - How claim is updated after each challenge

2. `/Users/matteo/projects/zolt/src/zkvm/spartan/split_eq.zig`
   - How eq values are accumulated and bound

### Test Commands

```bash
# All tests pass (unit tests use different accumulation path)
zig build test

# Generate proof (has wrong output_claim)
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 -o /tmp/zolt_proof_dory.bin 2>&1 | grep STAGE1_FINAL

# Verify the numbers
# eq_factor * inner_sum_prod should equal output_claim but doesn't!
```

### Next Session: Debugging Steps

1. Print `split_eq.current_scalar` at each round start/end
2. Print the contribution of each cycle to the round polynomial
3. Verify that cycle contributions are weighted by correct eq values
4. Check if the "streaming round" handles eq differently
