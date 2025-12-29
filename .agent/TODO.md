# Zolt-Jolt Compatibility TODO

## Current Status: FUNDAMENTAL BUG FOUND (Session 24)

### ROOT CAUSE IDENTIFIED

**The Problem**: Zolt computes `Σ (Az * Bz)` (sum of products) but Jolt expects `(Σ Az) * (Σ Bz)` (product of sums).

These are mathematically DIFFERENT due to the non-linearity of multiplication!

**Example**:
- Prover (Zolt): `Σ_cycle eq * Az(cycle) * Bz(cycle)` = sum of diagonal products
- Verifier (Jolt): `(Σ_cycle eq * Az(cycle)) * (Σ_cycle eq * Bz(cycle))` = cross-product

### The Fix Required

Zolt must change from computing:
```zig
// WRONG (current Zolt approach):
const prod_0 = az_g0.mul(bz_g0);  // Product BEFORE summing
t_zero = t_zero.add(eq_val.mul(prod_0));
```

To computing:
```zig
// CORRECT (Jolt approach):
// 1. Keep Az and Bz separate throughout sumcheck
// 2. Build separate Az_expanded and Bz_expanded grids on {0,1,∞}^d
// 3. Multiply AFTER expansion: t_prime[idx] = Az_expanded[idx] * Bz_expanded[idx]
// 4. Project t_prime to get round polynomial
```

### How Jolt Solves This

1. **Maintain separate polynomials**: `OuterLinearStage` has separate `az: DensePolynomial` and `bz: DensePolynomial`

2. **Expand to multiquadratic grid**: Each polynomial is expanded to {0, 1, ∞}^d grid where ∞ stores the slope

3. **Multiply on expanded grid**: `t'[idx] = Az_expanded[idx] * Bz_expanded[idx]`

4. **Project for round poly**: Use `t'(0)` and `t'(∞)` with Gruen interpolation

The ∞ encodings capture all cross-terms that arise from `(Σ Az) * (Σ Bz)`.

### Implementation Plan

1. **Modify `computeCycleAzBzForMultiquadratic`**:
   - Return separate (Az_g0, Az_g1, Bz_g0, Bz_g1) instead of products
   - Or return the multiquadratic expanded values

2. **Modify streaming round computation**:
   - Accumulate separate Az_sum and Bz_sum grids over cycles
   - Expand each to {0, 1, ∞} grid
   - THEN compute product

3. **Update `computeCubicRoundPoly`**:
   - Accept multiquadratic t' values (after Az*Bz expansion)
   - Project to get q_constant and q_quadratic_coeff

### Key Files to Modify

1. `src/zkvm/spartan/streaming_outer.zig`:
   - `computeRemainingRoundPoly()` - streaming round
   - `computeCycleAzBzForMultiquadratic()` - cycle values

2. May need new `MultiquadraticPolynomial` module to handle {0,1,∞} expansion

### Reference Implementation

See Jolt's:
- `outer.rs:704-725` - multiquadratic expansion and product
- `multiquadratic_poly.rs:40-142` - {0,1,∞} grid expansion
- `multiquadratic_poly.rs:326-351` - projection to first variable

### Verification

After fix:
- `output_claim` should equal `expected_output_claim`
- `inner_sum_prod_prover` should equal `inner_sum_prod_verifier`

Current values:
- Prover Az*Bz: `6845670145302814045138444113000749599157896909649021689277739372381215505241`
- Verifier Az*Bz: `12743996023445103930025687297173833157935883282725550257061179867498976368827`

## Test Commands

```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --max-cycles 1024

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
