# Zolt-Jolt Compatibility - Current Status

## Summary (Session 29)

**Stage 1**: PASSES with Zolt preprocessing ✓
**Stage 2**: PASSES with Zolt preprocessing ✓
**Stage 3**: FAILS - Transcript divergence causes gamma mismatch
**Stage 4-7**: Untested (blocked on Stage 3)

### Current Issue: Transcript Divergence After Stage 2

The gamma values for Stage 3 are derived from the transcript AFTER Stage 2's cache_openings.
Even though all claim VALUES match between Zolt and Jolt, the transcript states diverge.

**Zolt transcript state after cache_openings:** `{ 37, 87, 147, 142, 217, 22, 236, 134 }`
**Jolt gamma_powers[1]:** 167342415292111346589945515279189495473

These produce completely different gamma values, which causes Stage 3 to fail.

### Verified Claim Values Match

All 17 Stage 2 cache_openings claims match between Zolt and Jolt:
- 8 PRODUCT_UNIQUE_FACTOR_VIRTUALS claims ✓
- 1 RamRa (RAF) claim = 0 ✓
- 3 RamReadWriteChecking claims (RamVal=0, RamRa=0, RamInc=0) ✓
- 2 OutputSumcheck claims (RamValFinal=16094983..., RamValInit=0) ✓
- 3 InstructionLookupsClaimReduction claims ✓

### Investigation Needed

Since claim VALUES match but gamma values differ, the issue must be:
1. **Byte format**: How claims are serialized to bytes before hashing
2. **Order**: Order in which claims are appended to transcript
3. **Additional data**: Extra data being appended that Zolt is missing

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

## Next Steps

1. [ ] Add byte-level debug to see exactly what bytes each claim appends
2. [ ] Compare transcript states after EACH claim is appended
3. [ ] Identify where the divergence first occurs

## Previous Session Notes

### Stage 3 Architecture (for reference)

Stage 3 is a batched sumcheck with 3 instances (all n_cycle_vars rounds):

1. **ShiftSumcheck** (degree 2)
   - Expected output: `Σ γ[i] * claim[i] * eq+1(r_outer, r) + γ[4] * (1-noop) * eq+1(r_prod, r)`

2. **InstructionInputSumcheck** (degree 3)
   - Expected output: `(eq(r, r_stage1) + γ² * eq(r, r_stage2)) * (right + γ * left)`

3. **RegistersClaimReduction** (degree 2)
   - Expected output: `eq(r, r_spartan) * (rd + γ * rs1 + γ² * rs2)`
