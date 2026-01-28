# Zolt-Jolt Compatibility: Stage 4 Progress

## Completed Fix: merge() Function

Fixed the `merge()` function in GruenSplitEqPolynomial to match Jolt's implementation:

**Before (broken):**
```zig
// Used cached E_out/E_in tables with incorrect bit decomposition
merged[i] = self.current_scalar.mul(E_out[i_out]).mul(E_in[i_in]);
```

**After (correct):**
```zig
// Use EqPolynomial to directly compute eq(w[0..current_index], x)
// This matches Jolt's: EqPolynomial::evals_parallel(&self.w[..self.current_index], Some(self.current_scalar))
const EqPoly = poly.EqPolynomial(F);
const merged = try EqPoly.evalsSliceWithScaling(F, allocator, self.w[0..remaining_vars], self.current_scalar);
```

**Verification:** The eq_eval values now match between Zolt prover and Jolt verifier:
- Zolt's `merged_eq[0]` = `d8 d0 62 f3 fe 93 6a 58 ...`
- Jolt's `eq_eval` = `[d8, d0, 62, f3, fe, 93, 6a, 58, ...]`

## Current Issue: Opening Claims Mismatch

The merge() fix is correct, but Stage 4 verification still fails because:

1. **Sumcheck is internally consistent:**
   - Zolt's batched_claim = `73 57 fa 78 ...`
   - Jolt's output_claim = `73 57 fa 78 ...` ✓ MATCH

2. **Expected claim doesn't match output:**
   - Expected = coeff[0] * (eq * combined) = `b9 89 d9 c7 ...`
   - Output = `73 57 fa 78 ...`
   - These DON'T match!

3. **Root cause:** The opening claims stored in the accumulator don't match the actual polynomial evaluations at the sumcheck point.

## Next Steps

1. Investigate how claims are stored in the opening accumulator
2. Compare Zolt's `cache_openings` implementation with Jolt's
3. Verify that the polynomial evaluations (val, ra, wa, inc) after binding equal the claimed openings

## Files Modified
- `/home/vivado/projects/zolt/src/zkvm/spartan/gruen_eq.zig` - Fixed merge() function
- `/home/vivado/projects/zolt/src/poly/mod.zig` - Added evalsSliceWithScaling()

## Debug Commands
```bash
# Generate proof
zig build -Doptimize=ReleaseFast run -- prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Test verification
cd /home/vivado/projects/jolt && cargo test --package jolt-core --no-default-features --features "minimal,zolt-debug" test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
