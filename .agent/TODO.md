# Zolt-Jolt Compatibility - Current Status

## Summary (Session 30 - Updated)

**Stage 1**: PASSES with Zolt preprocessing ✓
**Stage 2**: PASSES with Zolt preprocessing ✓
**Stage 3**: FAILS - Transcript state diverges before gamma derivation
**Stage 4-7**: Untested (blocked on Stage 3)

### Latest Progress: Transcript Debugging

**Finding:** All claim VALUES match between Zolt and Jolt:
- 8 Stage 2 factor claims ✓
- 3 RWC claims (0, 0, 0) ✓
- 1 RAF claim (0) ✓
- 2 Output claims (RamValFinal, 0) ✓
- 3 InstructionLookups claims (LookupOutput, LeftOperand, RightOperand) ✓

**Finding:** Stage 2 round polynomials match between Zolt and Jolt ✓

**Finding:** Stage 2 input_claims now being appended before batching coeffs ✓

**Issue:** Transcript state still diverges after Stage 2 cache_openings:
- **Zolt state:** `25 57 93 8e d9 16 ec 86`
- **Jolt state:** `a5 8c 14 ac b6 53 de 71`

### Current Hypothesis

Even though all claim VALUES match, the transcript states are different. This could be because:
1. Claim order differs in some subtle way
2. A hidden claim is being appended that we're not tracking
3. Serialization format difference (unlikely since we verified appendScalar matches)

### Next Steps

1. [ ] Add side-by-side transcript state tracing to find exact divergence point
2. [ ] Check if any claims are missing in Stage 2 cache_openings
3. [ ] Verify appendBytes produces identical hashes for same input

### Key Debug Output

Stage 3 start states:
```
Zolt: { 37, 87, 147, 142, 217, 22, 236, 134 } = 25 57 93 8e d9 16 ec 86
Jolt: [a5, 8c, 14, ac, b6, 53, de, 71]
```

### Testing Commands

```bash
# Generate proof with preprocessing
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
  --export-preprocessing /tmp/zolt_preprocessing.bin \
  -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Test with Zolt preprocessing
cd /Users/matteo/projects/jolt/jolt-core && \
  cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```

### Files Modified
- `/Users/matteo/projects/zolt/src/zkvm/proof_converter.zig` - Main proof conversion logic
- `/Users/matteo/projects/zolt/src/zkvm/spartan/stage3_prover.zig` - Stage 3 prover
