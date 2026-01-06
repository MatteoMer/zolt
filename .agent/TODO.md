# Zolt-Jolt Compatibility TODO

## Current Status: Session 58 - January 6, 2026

**STATUS: UniSkip polynomial verification failing - domain sum check**

### Changes Completed

1. ✅ Added `evaluateAzBzAtDomainPointForGroup` with group parameter
2. ✅ Added `buildEqTable` helper for factored E_out * E_in computation
3. ✅ Updated `computeFirstRoundPoly` to iterate over both groups
4. ✅ Added `full_tau` field to store tau for UniSkip
5. ✅ Fixed eq_table structure to use factored E_out * E_in (dropping tau_high)
6. ✅ Updated `proof_converter.zig` to use `StreamingOuterProver`

### Current Debug State

The UniSkip verification now fails at a different point - the **domain sum check**:
```
check_sum_evals: Σ_j a_j · S_j == claim
```

Where:
- `claim = F::zero()` (input_claim for outer UniSkip is always 0)
- `S_j` = power sums over symmetric domain {-4, ..., 5}
- The polynomial `s(Y) = L(τ_high, Y) · t₁(Y)` should sum to zero over the domain

Debug output shows correct structure:
- full_tau.len = 12
- m = 6, E_out.len = 64, E_in.len = 32
- extended_evals at base points (Y=0) are zero (satisfied constraints)
- Non-zero values at extended points (Y=-9, Y=9)

### ROOT CAUSE IDENTIFIED

**Zolt computes extended_evals at ALL 19 domain points, but Jolt only computes 9 extended_evals at INTERLEAVED targets.**

Jolt's `build_uniskip_first_round_poly` expects:
- `extended_evals`: DEGREE=9 evaluations at interleaved targets `{-5, 6, -6, 7, -7, 8, -8, 9, -9}`
- `base_evals`: Optional, defaults to zeros for the 10 base points `{-4,...,5}`
- `t1_vals`: 19 values constructed by placing extended_evals at targets, zeros elsewhere

The fix requires:
1. Modify `computeFirstRoundPoly` to compute evals at ONLY the 9 target points
2. Build `t1_vals` array correctly (zeros at base, extended_evals at targets)
3. Interpolate t1 from these 19 points to get 19 coefficients
4. Multiply with 10-coefficient Lagrange kernel to get 28 coefficients

### Detailed Jolt Algorithm

From `jolt-core/src/subprotocols/univariate_skip.rs:77-125`:

```rust
// 1. Build t1_vals[EXTENDED_SIZE=19] array
let mut t1_vals: [F; EXTENDED_SIZE] = [F::zero(); EXTENDED_SIZE];
// Base evals (if provided) go at positions 5-14 (for base_left=-4)
// Extended evals go at target positions

// 2. Fill extended evals at target positions
for (idx, &val) in extended_evals.iter().enumerate() {
    let z = targets[idx];  // e.g., -5, 6, -6, 7, ...
    let pos = (z + DEGREE) as usize;  // maps -9..9 to 0..18
    t1_vals[pos] = val;
}

// 3. Interpolate t1 (degree-18) from 19 evaluations
let t1_coeffs = LagrangePolynomial::interpolate_coeffs::<EXTENDED_SIZE>(&t1_vals);

// 4. Compute Lagrange kernel L(tau_high, Y) coefficients (degree-9)
let lagrange_values = LagrangePolynomial::evals::<Challenge, DOMAIN_SIZE>(&tau_high);
let lagrange_coeffs = LagrangePolynomial::interpolate_coeffs::<DOMAIN_SIZE>(&lagrange_values);

// 5. Multiply polynomials: (deg-9) * (deg-18) = deg-27 → 28 coefficients
for (i, &a) in lagrange_coeffs.iter().enumerate() {
    for (j, &b) in t1_coeffs.iter().enumerate() {
        s1_coeffs[i + j] += a * b;
    }
}
```

The targets for outer sumcheck:
- DOMAIN_SIZE=10, DEGREE=9
- base_left=-4, base_right=5
- targets = [-5, 6, -6, 7, -7, 8, -8, 9, -9]

---

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Key Jolt Reference for UniSkip Verification

From `jolt-core/src/poly/unipoly.rs:327-338`:
```rust
pub fn check_sum_evals<const N: usize, const OUT_LEN: usize>(&self, claim: F) -> bool {
    debug_assert_eq!(self.degree() + 1, OUT_LEN);
    let power_sums = LagrangeHelper::power_sums::<N, OUT_LEN>();

    // Check domain sum Σ_j a_j * S_j == claim
    let mut sum = F::zero();
    for (j, coeff) in self.coeffs.iter().enumerate() {
        sum += coeff.mul_i128(power_sums[j]);
    }
    sum == claim
}
```

For Stage 1 outer:
- N = OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE = 10
- OUT_LEN = OUTER_FIRST_ROUND_POLY_NUM_COEFFS = 28
- claim = F::zero()

The power_sums[j] = Σ_i (base_left + i)^j for i ∈ {0, ..., N-1} where base_left = -4

---

## Previous Sessions

### Session 57 - SECOND_GROUP Fix Identified

Identified that `computeFirstRoundPoly` only evaluated FIRST_GROUP, missing SECOND_GROUP constraints.

### Sessions 51-56 - Various Fixes

- Batching coefficient Montgomery form fix
- Round offset fix
- Transcript flow matching
