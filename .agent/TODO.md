# Zolt-Jolt Compatibility Implementation

## Status: Session 73 - Montgomery R^2 Scaling Applied

## Current Issue: Stage 1 Sumcheck Output Claim Mismatch (After R^2 Fix)

### Problem Summary

The Jolt verifier still fails to verify the Stage 1 sumcheck proof:

```
Sumcheck verification failed!
  output_claim:   [99, 9c, b9, b6, 17, 1c, d1, c6, ...]
  expected_claim: [b9, 9d, d3, 21, 0b, 82, 30, e4, ...]
```

### Changes Made This Session

1. **Added `F.rSquared()` constant to BN254Scalar** (src/field/mod.zig):
   - Returns R^2 as a field element in Montgomery form
   - Computed as: raw R^2 bytes converted to Montgomery form

2. **Applied R^2 scaling to Stage 1 UniSkip extended_evals** (streaming_outer.zig):
   - After computing sum = Σ eq_val * Az*Bz, multiply by R^2
   - This matches Jolt's `* outer_scale` at line 226 of outer.rs

3. **Applied R^2 scaling to Stage 2 ProductVirtual extended_evals** (univariate_skip.zig):
   - Same R^2 multiplication applied to extended_evals

### Debug Output Comparison

**Zolt BEFORE R^2 scaling:**
```
extended_evals[0] (target_y=-5) = { 4, 136, 82, 238, 142, 127, 178, 244, ... }
```

**Zolt AFTER R^2 scaling:**
```
extended_evals[0] (target_y=-5) = { 26, 148, 234, 222, 178, 10, 191, 9, ... }
```

### Root Cause Analysis (Ongoing)

The R^2 scaling is applied but the sumcheck still fails. Possible remaining issues:

1. **Eq Table Computation Difference**:
   - Jolt uses `GruenSplitEqPolynomial` which builds E_out and E_in in a specific way
   - Zolt uses `buildEqTable` which might index differently

2. **Constraint Evaluation Difference**:
   - Jolt's `extended_azbz_product_first_group` uses i32/S128/S192 accumulation
   - Zolt evaluates Az and Bz as field elements and multiplies
   - These SHOULD be equivalent for correct witnesses but might have precision differences

3. **Lagrange Coefficient Order**:
   - COEFFS_PER_J might be computed differently
   - TARGET_SHIFTS might differ

4. **Group Interleaving**:
   - Jolt interleaves FIRST_GROUP and SECOND_GROUP via x_in LSB
   - Zolt does the same but might have different ordering

### Next Steps to Fix

1. **Add debug to Jolt prover to print extended_evals**:
   - Need to run Jolt's prover (not just verifier) with debug
   - Compare byte-for-byte with Zolt's extended_evals

2. **Verify E_out and E_in tables match**:
   - Print E_out[0] and E_in[0] from both implementations
   - Check tau splits are identical

3. **Verify Lagrange coefficient calculation**:
   - Print COEFFS_PER_J[0] from both
   - Verify TARGET_SHIFTS match

4. **Check constraint evaluation at cycle 0**:
   - For cycle 0, print individual Az and Bz values for each constraint
   - Compare with Jolt's R1CSCycleInputs

### Files Modified This Session

- `src/field/mod.zig`: Added `rSquared()` method
- `src/zkvm/spartan/streaming_outer.zig`: Added R^2 scaling to extended_evals
- `src/zkvm/r1cs/univariate_skip.zig`: Added R^2 scaling to ProductVirtual extended_evals
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/spartan/outer.rs`: Added debug output for extended_evals

### Debug Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64 2>&1 | grep -E "(UNISKIP|extended_evals|BEFORE|AFTER)" | head -30

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Investigation History

### Session 71
- Discovered challenge value mismatch between Zolt prover and Jolt verifier
- r_stream and r0 values completely different

### Session 72
- Traced issue to UniSkip extended_evals differing
- Found Montgomery R^2 scaling difference
- Found constraint evaluation approach difference
- Verified split parameters match

### Session 73
- Implemented R^2 scaling for Stage 1 and Stage 2 UniSkip
- Verification still fails - need deeper investigation

## Verified Working Components

- ✅ Blake2b transcript matches between Zolt and Jolt
- ✅ Initial transcript state matches
- ✅ Scalar serialization format (32-byte LE)
- ✅ Tau challenges generated correctly
- ✅ Split eq parameters (m, num_x_out_bits, num_x_in_bits) match
- ✅ R^2 scaling now applied to extended_evals
