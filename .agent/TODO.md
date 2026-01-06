# Zolt-Jolt Compatibility TODO

## Current Status: Session 58 - January 6, 2026

**STATUS: UniSkip polynomial passes domain sum check! Now debugging remaining sumcheck rounds.**

### Changes Completed

1. ✅ Added `evaluateAzBzAtDomainPointForGroup` with group parameter
2. ✅ Added `buildEqTable` helper for factored E_out * E_in computation
3. ✅ Updated `computeFirstRoundPoly` to iterate over both groups
4. ✅ Added `full_tau` field to store tau for UniSkip
5. ✅ Fixed eq_table structure to use factored E_out * E_in (dropping tau_high)
6. ✅ Updated `proof_converter.zig` to use `StreamingOuterProver`
7. ✅ **Restructured UniSkip to compute 9 extended_evals at target points only**
8. ✅ **UniSkip polynomial now passes domain sum check!**

### Current Debug State

**UniSkip PASSES!** The domain sum check now passes. The remaining sumcheck rounds fail:

```
output_claim:          12640140186890871835397399826551813568033071071342265234934089331959908111894
expected_output_claim: 1227477921082322617298641147962849807051365419641170255596401662086293721633
```

The issue is now in the remaining sumcheck rounds after UniSkip (rounds 1-11).

### Previous Issue (NOW FIXED)

UniSkip was computing extended_evals at ALL 19 domain points instead of 9 target points.
This has been fixed. The UniSkip polynomial now passes domain sum verification.

### Next Steps

1. Debug why the remaining sumcheck rounds fail after UniSkip
2. Check if uni_skip_claim is computed correctly from polynomial evaluation at r0
3. Verify the streaming rounds are using correct eq and Az/Bz values

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
