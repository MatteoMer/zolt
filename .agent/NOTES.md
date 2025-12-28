# Zolt-Jolt Compatibility Notes

## Current Status (December 28, 2024, Session 4)

### Working Components
- ✅ Blake2b Transcript (matches Jolt - all test vectors pass)
- ✅ Dory Commitment Scheme (GT elements verified matching)
- ✅ Proof Structure (JoltProof with 7 stages)
- ✅ Serialization Format (arkworks-compatible, Jolt can deserialize)
- ✅ LagrangeHelper with shift_coeffs_i32 (matches Jolt)
- ✅ COEFFS_PER_J precomputed Lagrange weights
- ✅ **UniSkip Cross-Product Algorithm (FIXED!)**
- ✅ All 618 unit tests pass
- ✅ **Zolt can run Jolt's compiled ELF files**
- ✅ **Jolt can deserialize Zolt proofs (test_deserialize_zolt_proof PASSES)**

### Cross-Verification Status

**Format Compatibility: VERIFIED ✅**
- Proof serialization matches arkworks format
- Opening claims, commitments, stage proofs all parse correctly
- GT elements (Dory commitments) serialize/deserialize correctly

**Full Verification: BLOCKED ⚠️**
- Verification fails at "Stage 1 univariate skip first round"
- Root cause: Preprocessing mismatch
  - Jolt preprocessing is for fib(50) trace
  - Zolt proof has different trace length
- The verification math is correct, but the inputs differ

### What's Needed for End-to-End Verification

Option 1: **Same-Program Verification**
- Generate Jolt preprocessing for the SAME trace length as Zolt
- Requires running `cargo run --example fibonacci` with matching parameters
- Then Jolt would generate preprocessing for the same execution

Option 2: **Zolt Exports Preprocessing**
- Implement Jolt's verifier preprocessing serialization format
- Generate preprocessing in Zolt that Jolt can use
- More complex but provides full independence

### Stage 1 Sumcheck - FIXED

**Problem Solved:**
The UniSkip polynomial was producing all-zero coefficients because we were using the wrong algorithm for extended evaluations.

**Root Cause:**
The old algorithm computed Az(y_j) and Bz(y_j) separately via Lagrange interpolation, then multiplied. For satisfied constraints, Bz[i] = 0, so this always gave zero.

**Solution - Jolt's Cross-Product Approach:**
Jolt doesn't interpolate Az and Bz separately. Instead, it partitions constraints:

```
az_eval = Σ_i (where Az[i] is TRUE): coeffs[j][i]
bz_eval = Σ_i (where Az[i] is FALSE): coeffs[j][i] * Bz[i]
Product = az_eval * bz_eval
```

This works because:
1. When guard is TRUE (Az=1): Bz=0 (constraint satisfied), contributes to az_eval
2. When guard is FALSE (Az=0): Bz can be non-zero, contributes to bz_eval
3. The cross-product of these sums gives non-zero at extended points!

**Example:**
- Constraint 0: Guard=TRUE, Bz=0 → contributes coeffs[0] to az_eval
- Constraint 1: Guard=FALSE, Bz=42 → contributes coeffs[1]*42 to bz_eval
- Product = coeffs[0] * coeffs[1] * 42 ≠ 0

## Architecture Overview

### UniSkip Extended Evaluation

From Jolt's `evaluation.rs`:
```rust
pub fn extended_azbz_product_first_group(&self, j: usize) -> S192 {
    let coeffs_i32 = &COEFFS_PER_J[j];
    let az = self.eval_az_first_group();
    let bz = self.eval_bz_first_group();

    let mut az_eval_i32: i32 = 0;
    let mut bz_eval_s128: S128Sum = S128Sum::zero();

    // For each constraint i:
    // - If Az[i] is TRUE: add coeffs[i] to az_eval
    // - If Az[i] is FALSE: add coeffs[i] * Bz[i] to bz_eval

    let c0_i32 = coeffs_i32[0];
    if az.not_load_store {
        az_eval_i32 += c0_i32;
    } else {
        bz_eval_s128.fmadd(&c0_i32, &bz.ram_addr);
    }
    // ... (same pattern for all 10 constraints)

    az_eval_s64.mul_trunc(&bz_eval_s128.sum)
}
```

### Key Files
- `src/zkvm/r1cs/univariate_skip.zig` - LagrangeHelper, COEFFS_PER_J, UniPoly
- `src/zkvm/spartan/outer.zig` - SpartanOuterProver with cross-product algorithm
- `src/zkvm/r1cs/evaluators.zig` - Az/Bz evaluation structures
- `src/zkvm/r1cs/constraints.zig` - R1CS constraint definitions

### Constraint Groups

**First Group (10 constraints, indices 0-9):**
- Boolean Az guards (0 or 1)
- Bz fits in ~64 bits
- Domain: {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

**Second Group (9 constraints, indices 10-18):**
- Mixed Az types
- Bz can be ~128-160 bits

## Next Steps

1. **Integration Testing**: Generate a full proof with the fixed algorithm and verify with Jolt

2. **Remaining Stages**: Stages 2-7 may need similar attention for their univariate skip implementations

3. **Performance**: The cross-product loop could be optimized with SIMD

## Test Commands

```bash
# Run all Zolt tests
cd /Users/matteo/projects/zolt
zig build test --summary all

# Run specific outer.zig tests
zig build test -- "uniskip polynomial with satisfied constraints"

# Generate proof for testing
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci/fibonacci.elf -o proof.jolt
```
