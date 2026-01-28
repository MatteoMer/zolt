# Zolt-Jolt Compatibility: Stage 4 Bug Found!

## Bug Identified

The prover's GruenSplitEqPolynomial binding produces a DIFFERENT eq value than the MLE computation:

- `merged_eq[0]` (prover after binding) = `{ 79, 186, 5, 90, ... }`
- `eq_val_be` (MLE via EqPolynomial.mle) = `{ 12, 14, 181, 194, ... }`

The MLE value MATCHES Jolt's eq_eval, so the MLE computation is correct. **The bug is in the prover's GruenSplitEqPolynomial binding.**

## Key Observations

1. **Stage 3 challenges match** - Both Zolt and Jolt have the same params.r_cycle
2. **Stage 4 challenges match** - Both use the same transcript state
3. **Combined values match** - Both compute the same `combined = rd_write_value + gamma * (...)`
4. **MLE eq_eval matches** - Zolt's `EqPolynomial.mle()` gives same result as Jolt

## What Should Happen

The GruenSplitEqPolynomial accumulates:
```
current_scalar = ∏_{i in phase1} eq(w[n-1-i], challenge[i])
```

Then merge() creates:
```
merged_eq[j] = current_scalar * eq(w[0..remaining], j)
```

And Phase 3 binding should reduce merged_eq to:
```
merged_eq[0] = ∏_{all i} eq(w[n-1-i], challenge[i])
             = eq(w_reversed, challenges)
             = eq(challenges_normalized, w)
```

This should equal `eq_val_be` computed via MLE!

## Possible Root Causes

1. **Table indexing in evalsCached** - The E_out/E_in tables might use different bit ordering than expected
2. **Merge function** - The index layout `i = i_out * E_in.len + i_in` might not match the data polynomial layout
3. **Phase 3 binding order** - LowToHigh binding of merged_eq might not correctly correspond to the challenges

## Investigation Plan

1. Compare E_out/E_in table values with direct eq evaluations at specific indices
2. Verify merged_eq immediately after merge() matches expected values
3. Trace Phase 3 binding step by step vs direct eq binding

## Files to Fix

- `/home/vivado/projects/zolt/src/zkvm/spartan/gruen_eq.zig` - GruenSplitEqPolynomial
  - Check: evalsCached, merge, bind
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage4_gruen_prover.zig` - Phase 3 binding
  - Check: bindPolynomials for Phase 3

## Commands
```bash
# Generate proof
zig build -Doptimize=ReleaseFast run -- prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Test verification
cd /home/vivado/projects/jolt && cargo test --package jolt-core --no-default-features --features "minimal,zolt-debug" test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
