# Zolt-Jolt Compatibility Notes

## Current Status (December 29, 2024)

### Session 15 - Investigating 1.23x Discrepancy

**Status**: UniSkip passes. Stage 1 output_claim is ~1.23x off from expected.

**Current Values:**
- output_claim = 15155108253109715956971809974428807981154511443156768969051245367813784134214
- expected = 18643585735450861043207165215350408775243828862234148101070816349947522058550
- Ratio: 0.813 (about 1.23x)

**Analysis:**
The expected output claim formula is:
```
expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
```

Where:
- `tau_high_bound_r0 = L(tau_high, r0)` - Lagrange kernel from UniSkip
- `tau_bound_r_tail = eq(tau_low, r_tail_reversed)` - eq polynomial at bound point
- `inner_sum_prod = az_final * bz_final` - R1CS product at evaluation point

**Key Insights from Investigation:**

1. **Round Count**: `numRounds() = 1 + num_cycle_vars = 11` for 1024 cycles
   - The proof_converter loop runs 11 iterations, which is correct
   - HalfSplitSchedule switch-over at round 5 (streaming: 0-4, linear: 5-10)

2. **r_grid Updates**: Only during streaming phase (first 5 rounds)
   - Streaming rounds update r_grid via `r_grid.update(r_j)`
   - Linear rounds do NOT update r_grid

3. **Eq Polynomial Binding**: Matches Jolt's LowToHigh order
   - `current_scalar *= eq(tau[current_index-1], r)` for each challenge
   - Final value equals `EqPolynomial::mle(tau_low, r_tail_reversed)`

4. **The 0.813 ratio** doesn't correspond to a simple missing factor
   - Not tau_high_bound_r0, tau_bound_r_tail, or inner_sum_prod alone
   - Suggests the error is in how the sumcheck accumulates values

**Hypothesis:**
The issue might be in how we compute `t_zero` and `t_infinity` during cycle rounds.
Specifically, the index structure `full_idx >> 1` for step_idx might be off.

**Next Steps:**
1. Add debug output to compare per-round claims between Zolt and Jolt
2. Verify the r_grid values match at each round
3. Check if the streaming round computation is correct

---

### Session 14 - Stage 1 Remaining Rounds - Current_scalar Fix

**Status**: UniSkip passes. Stage 1 output_claim still ~10x off from expected.

**Key Fixes Made This Session:**
1. **Multiquadratic t'(infinity) fix**: Changed from `(sum slope_Az) * (sum slope_Bz)` to `sum (slope_Az * slope_Bz)` (sum of slope PRODUCTS, not product of slope sums)
2. **current_scalar fix**: Removed `current_scalar` multiplication from t' computation. It should ONLY be applied in `computeCubicRoundPoly` when computing the linear l(X) polynomial. This matches Jolt's approach where `E_active_for_window` excludes `current_scalar`.
3. **r_grid scope**: Investigated whether r_grid should be updated for streaming round. Currently updating for all rounds.

---

## Working Components

| Component | Status | Notes |
|-----------|--------|-------|
| Blake2b Transcript | Working | State and n_rounds match |
| Challenge Derivation | Working | MontU128Challenge-compatible |
| Dory Commitment | Working | GT elements match, MSM correct |
| Proof Structure | Working | 7 stages, claims, all parse |
| Serialization | Working | Byte-level compatible |
| UniSkip Algorithm | Working | Domain sum = 0, claims match |
| Preprocessing Export | Working | Full JoltVerifierPreprocessing |
| DoryVerifierSetup | Working | Precomputed pairings |

---

## Commands

```bash
# Test Zolt (all 632 tests)
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Run Jolt debug test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
