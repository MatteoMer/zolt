# Zolt-Jolt Compatibility: Status Update

## Status: DEBUGGING STAGE 4 BATCHED SUMCHECK MISMATCH

## Summary

Zolt can now:
1. Generate proofs for RISC-V programs (`./zig-out/bin/zolt prove`)
2. Verify proofs internally (`./zig-out/bin/zolt verify`) - ALL 6 STAGES PASS
3. Export proofs in Jolt-compatible format (`--jolt-format`)
4. Export preprocessing for Jolt verifier (`--export-preprocessing`)
5. Pass all 714 unit tests ✓
6. **Proof successfully deserializes in Jolt** ✓
7. **Preprocessing successfully deserializes in Jolt** ✓

## Cross-Verification Status

Verification fails at Stage 4: `batched_claim ≠ total_expected`

## Current Issue Analysis (Session 71 - Updated)

### Recent Progress (This Session)

1. **Fixed termination write issue**:
   - The synthetic termination write IS now being recorded
   - `[TRACE] Recorded synthetic termination write: addr=0x7fffc008, cycle=54`

2. **Fixed start_address in proof_converter**:
   - Now uses `memory_layout.getLowestAddress() = 0x7FFF8000`
   - IncPolynomial correctly includes the termination write: `Write at idx=2049, timestamp=54, old_val=0, new_val=1, inc=1`

3. **ValEval and ValFinal now produce NONZERO expected outputs**:
   - Instance 1 (ValEval): `inc*wa*lt={ 18, 125, 12, 131, ... }` (was all zeros before)
   - Instance 2 (ValFinal): `inc*wa={ 12, 200, 234, 237, ... }` (was all zeros before)

### Current Mismatch

After all 15 rounds of Stage 4:
- `batched_claim (sumcheck output)` = `{ 13, 174, 120, 9, 233, 120, 62, 18, ... }`
- `total_expected (weighted sum)` = `{ 18, 61, 142, 143, 28, 54, 66, 104, ... }`

These still don't match!

### Debug Details

```
[ZOLT STAGE4 VERIFY CHECK]
  batched_claim (sumcheck output) = { 13, 174, 120, 9, 233, 120, 62, 18, ... }
  Instance 0: expected={ 0, 223, 38, 219, ... }
  Instance 1 (ValEval): inc*wa*lt={ 18, 125, 12, 131, ... }  <- NONZERO NOW!
  Instance 2 (ValFinal): inc*wa={ 12, 200, 234, 237, ... }  <- NONZERO NOW!
  total_expected = { 18, 61, 142, 143, ... }
  Do they match? false
```

### Remaining Issues to Debug

1. **LT polynomial computation**:
   - The LT polynomial uses r_cycle ordering
   - Need to verify r_cycle_LE vs r_cycle_BE ordering matches Jolt

2. **eq polynomial for expected output**:
   - RegistersRWC expected output uses `eq(r_cycle_sumcheck, stage3_r_cycle)`
   - Need to verify the endianness of both points

3. **Sumcheck polynomial composition**:
   - The batched sumcheck combines RegistersRWC, ValEval, and ValFinal
   - Need to verify the polynomial contribution at each round

### Key Files

- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 prover
  - Line 1786-1793: start_address now uses memory_layout.getLowestAddress() ✓
  - Line 2336: eq_val_be computation for RegistersRWC

- `/home/vivado/projects/zolt/src/zkvm/ram/val_evaluation.zig` - ValEvaluation prover
  - LtPolynomial implementation

- `/home/vivado/projects/jolt/jolt-core/src/zkvm/ram/val_evaluation.rs` - Reference

### Test Commands

```bash
# Generate Jolt-compatible proof with debug output
./zig-out/bin/zolt prove examples/fibonacci.elf -o /tmp/proof.bin \
    --jolt-format /tmp/jolt_proof.bin 2>&1 | tee /tmp/stage4_debug.log

# Check specific debug info
grep "IncPolynomial" /tmp/stage4_debug.log
grep "VERIFY CHECK" /tmp/stage4_debug.log
```

## Files Generated

- `/tmp/test_proof.bin` - Zolt native format proof
- `/tmp/test_jolt.bin` - Jolt-compatible format proof (40531 bytes)
- `/tmp/jolt_prove.log` - Full debug output
