# Zolt-Jolt Compatibility Implementation

## Status: Session 75 - Tau Vector Length Fix + R^2 Scaling Removal

## Current Issue: Stage 1 Sumcheck Output/Expected Claim Mismatch

### Root Causes Found & Fixed This Session

1. **FIXED: Synthetic termination step constraint violation**
   - Changed `recordTerminationWrite` in tracer/mod.zig to mark termination step as NoOp
   - Result: 0 constraint violations across 55 cycles × 19 constraints ✅

2. **FIXED: R^2 scaling was mathematically wrong for Zolt**
   - Removed R^2 scaling from both streaming_outer.zig and univariate_skip.zig
   - Jolt's R^2 compensates for its integer/Montgomery mixed pipeline; Zolt's pure field arithmetic already correct
   - Result: extended_evals now computed without extra R^2 multiplication ✅

3. **FIXED: num_cycle_vars computed from actual cycle count (55) instead of trace_length (64)**
   - Zolt: `log2_int(55) = 5`, giving `num_rows_bits = 7`, tau.len = 7
   - Jolt: `log2(64) = 6`, giving `num_rows_bits = 8`, tau.len = 8
   - Fix: Use `trace_length` instead of `cycle_witnesses.len` for computing num_cycle_vars
   - Fixed in all 3 places in mod.zig (lines 692, 1031, 1265)
   - Result: Transcript state now matches at UncompressedUniPoly_begin ✅
   - Result: r0 values match between Zolt prover and Jolt verifier ✅

### Remaining Issue

Stage 1 sumcheck still fails with output_claim ≠ expected_claim:
- output_claim matches Zolt's prover computation (sumcheck proof is internally consistent)
- expected_claim is computed by Jolt verifier from R1CS input evaluations (opening claims)
- The opening claims in the proof may not correctly represent the polynomial evaluations

### Theory for Remaining Issue

The committed polynomials (R1CS inputs like PC, RS1, RS2, etc.) might not be padded to 64 cycles
in Zolt, while Jolt expects them to be. When the verifier evaluates these polynomials at the
sumcheck challenge point, it gets different values than what the prover used.

### Next Steps

1. **Check committed polynomial padding**: Verify that R1CS input polynomials are padded to
   `trace_length` (64 cycles) with NoOp witness values for cycles 55-63.

2. **Compare opening claims**: Check the opening claims in the proof against what the verifier
   expects.

3. **Verify inner_sum_product computation**: The verifier's `evaluate_inner_sum_product_at_point`
   uses the opening claims. Trace through this computation.

### Files Modified This Session

- `src/tracer/mod.zig`: Changed `recordTerminationWrite` to mark termination step as NoOp
- `src/zkvm/mod.zig`: Fixed num_cycle_vars to use trace_length (3 places)
- `src/zkvm/spartan/streaming_outer.zig`: Removed R^2 scaling from extended_evals
- `src/zkvm/r1cs/univariate_skip.zig`: Removed R^2 scaling from ProductVirtual extended_evals
- `src/zkvm/proof_converter.zig`: Added comprehensive constraint violation checker

### Jolt Files Modified (Debug Only)

- `jolt-core/src/subprotocols/univariate_skip.rs`: Added UniSkip verification debug
- `jolt-core/src/transcripts/blake2b.rs`: Extended transcript debug to UncompressedUniPoly

### Debug Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Verified Working Components

- ✅ Blake2b transcript matches between Zolt and Jolt
- ✅ Initial transcript state matches
- ✅ Scalar serialization format (32-byte LE)
- ✅ Tau challenges generated correctly (now with correct count)
- ✅ Split eq parameters match
- ✅ Guard-routing pattern applied to evaluateAzBzAtTargetY
- ✅ Zero constraint violations (after NoOp termination fix)
- ✅ Zero base domain violations
- ✅ UncompressedUniPoly_begin transcript state matches
- ✅ r0 values match between Zolt prover and Jolt verifier
- ✅ UniSkip check_sum_evals passes
- ✅ No R^2 scaling (correctly removed)
- ❌ Stage 1 remaining sumcheck output_claim ≠ expected_claim
