# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (OutputSumcheck polynomial coefficients don't match expected)
- Stage 3+: Not reached yet
- All Zolt tests pass

## Session 9 Progress Summary

### What Was Implemented
1. **OutputSumcheckProver** (`src/zkvm/ram/output_check.zig`)
   - Created prover structure with val_init, val_final, val_io, io_mask, eq_r_address
   - Implemented computeRoundPolynomial() and bindChallenge()
   - Fixed EQ polynomial to use LowToHigh (LSB-first) binding order
   - Fixed address mapping (indexToAddress uses lowest + k * 8)
   - Fixed termination handling (set BOTH val_final AND val_io to 1)

2. **Integration into Stage 2** (`src/zkvm/proof_converter.zig`)
   - Added config parameter with initial_ram and final_ram pointers
   - Sample r_address challenges in correct order
   - Initialize OutputSumcheckProver when RAM state data is available
   - Call computeRoundPolynomial() for instance 3

3. **Data Flow** (`src/zkvm/mod.zig`)
   - Pass emulator.ram.memory as final_ram
   - Pass empty hash map as initial_ram

### Current Issue

The Stage 2 sumcheck verification fails:
```
output_claim:          10555406300081192179452048418528136201389824333451681887399411041092911249053
expected_output_claim: 12558447015227526731091241411293250621525229972846007269528435424240713158110
```

**Root Cause Analysis:**
1. OutputSumcheck produces all-zero polynomials because:
   - For a correctly executing program, val_final = val_io everywhere in IO region
   - This means (val_final - val_io) = 0 everywhere
   - So the sumcheck polynomial Σ eq * io_mask * 0 = 0

2. BUT Jolt's verifier computes a NON-ZERO expected output:
   - val_final_claim from proof = 9598091504631331533319367171027748529696916355377122795493041621516413382685
   - val_io_eval computed = 5891669863525244341570117655318363701041112705294068835900114259741218367302
   - These don't match! (difference = 3706421641106087191749249515709384828655803650083053959592927361775195015383)

3. The mismatch between val_final_claim and val_io_eval means:
   - Either our val_final_claim serialization is wrong
   - Or Jolt's val_io_eval computation differs from ours
   - Most likely: bit ordering/endianness issue in the eq(idx, r) computation

### Next Steps

1. **Debug val_final_claim vs val_io_eval mismatch**
   - Compare the exact bytes being serialized for RamValFinal claim
   - Verify the termination_index computation matches Jolt
   - Check if LowToHigh vs BIG_ENDIAN convention is correct

2. **Ensure OutputSumcheck polynomials are computed correctly**
   - For a correctly executing program, the polynomial SHOULD be zero
   - This is correct behavior when val_final = val_io

3. **Verify the expected_output_claim computation in Jolt matches**
   - The batched sumcheck output should match the individual instance outputs

### Files Modified This Session
- src/zkvm/ram/output_check.zig - NEW: OutputSumcheckProver
- src/zkvm/ram/mod.zig - Export OutputSumcheck module
- src/zkvm/proof_converter.zig - Integrate OutputSumcheck, pass config
- src/zkvm/mod.zig - Pass RAM state data to converter

### Commits This Session
- `cc1985e`: feat(output-sumcheck): Add OutputSumcheckProver infrastructure

## Previous Session Summary
All Stage 2 internal values match Jolt:
- Input claims (all 5)
- Gamma values (rwc, instr)
- Batching coefficients
- ProductVirtualRemainder polynomial coefficients
- Sumcheck challenges (all 26 rounds)
