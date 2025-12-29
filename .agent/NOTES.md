# Zolt-Jolt Compatibility Notes

## Current Status (December 29, 2024)

### Session 16 - Deep Investigation of 0.8129 Ratio

**Status**: UniSkip passes. Stage 1 output_claim is ~0.8129 of expected (close to 13/16 = 0.8125).

**Current Values:**
- output_claim = 15155108253109715956971809974428807981154511443156768969051245367813784134214
- expected = 18643585735450861043207165215350408775243828862234148101070816349947522058550
- Ratio: 0.8129 (close to 13/16 = 0.8125)

**Verified Components (All Match Jolt):**

1. **Constraint Group Ordering**
   - First group: indices {1,2,3,4,5,6,11,14,17,18} (10 constraints)
   - Second group: indices {0,7,8,9,10,12,13,15,16} (9 constraints)
   - Both use same Lagrange weight array `w[0..N]`

2. **Streaming Round Index Structure**
   - Jolt: for each (out_idx, in_idx), for j in 0..2:
     - full_idx = i * 2 + j
     - step_idx = full_idx >> 1 = i (cycle index)
     - selector = full_idx & 1 = j (group selector)
   - Zolt: same logic in streaming iteration

3. **Eq Table Factorization**
   - E_out: 5 bits (32 entries)
   - E_in: 5 bits (32 entries)
   - Total: 32 × 32 = 1024 for 1024 cycles

4. **r_cycle Computation**
   - Skip challenges[0] (r_stream)
   - Take challenges[1..]
   - Reverse for BIG_ENDIAN

5. **Gruen Polynomial Construction**
   - l(X) = eq(0) + (eq(1) - eq(0)) * X
   - q(0) = t'(0), q(∞) = t'(∞)
   - q(1) = (claim - l(0)*q(0)) / l(1)
   - s(X) = l(X) * q(X)

6. **ExpandingTable Update**
   - `values[i] = (1-r)*old`, `values[i+len] = r*old`

**Remaining Investigation Areas:**

The discrepancy must be in t'(0) and t'(∞) computation:

1. **Az/Bz Evaluation**
   - Verify `condition` and `magnitude` match Jolt's `a·z` and `b·z`

2. **Lagrange Weight Application**
   - Verify `lagrange_evals_r0[i]` matches Jolt's `w[i]`

3. **Product of Slopes**
   - t'(∞) = Σ eq * (Az_g1 - Az_g0) * (Bz_g1 - Bz_g0)

4. **current_scalar Handling**
   - Ensure not double-applied in streaming round

**Key Files:**

Jolt:
- `outer.rs::extrapolate_from_binary_grid_to_tertiary_grid` (563-635)
- `outer.rs::compute_evaluation_grid_from_trace` (641-751)
- `evaluation.rs::fmadd_*_group_at_r`

Zolt:
- `streaming_outer.zig::computeRemainingRoundPoly`
- `streaming_outer.zig::computeCycleAzBzForMultiquadratic`
- `split_eq.zig::computeCubicRoundPoly`

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
