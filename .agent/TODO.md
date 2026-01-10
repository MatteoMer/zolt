# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 PASSES, Stage 3 FAILS

### Issues Fixed (This Session)
1. **Test configuration** - Now uses ZOLT preprocessing instead of JOLT preprocessing
2. **Tau length** - Uses correct `num_rows_bits = num_cycle_vars + 2`
3. **Stage 1 verification** - Now passes completely

### Current Issue: Stage 3 Expected Output Claim Mismatch

Error:
```
output_claim:          14748667221680201172814846157814975891768271694788348475862484542619931435291
expected_output_claim: 21057875695859459713163748271204739962975184771552325082098103420465800885546
```

Stage 3 consists of 3 sumcheck instances:
1. ShiftSumcheckVerifier
2. InstructionInputSumcheckVerifier
3. RegistersClaimReductionSumcheckVerifier

The output_claim from ZOLT's sumcheck matches what JOLT receives.
The expected_output_claim computed by JOLT's verifier doesn't match.

This means the witness opening claims sent by ZOLT for Stage 3 polynomials are incorrect
or being computed at the wrong evaluation point.

### Next Steps
1. Investigate Stage 3's expected_output_claim computation in Jolt
2. Check what witness openings are being sent for Stage 3
3. Verify the evaluation point (r_cycle) matches between ZOLT and JOLT
4. Fix the Stage 3 opening claims

### Overall Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASSES | Using ZOLT preprocessing, tau length fixed |
| 2 | Unknown | Not reached yet |
| 3 | ❌ FAILS | output_claim correct, expected_output_claim wrong |
| 4-7 | Blocked | Waiting on Stage 3 |

### Key Files
- `/Users/matteo/projects/zolt/src/zkvm/proof_converter.zig` - Stage 3 proof generation
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/shift.rs` - Shift sumcheck
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/instruction_input.rs` - InstrInput sumcheck
- `/Users/matteo/projects/jolt/jolt-core/src/registers/claim_reduction.rs` - Registers sumcheck

## Testing
```bash
bash scripts/build_verify.sh
```
