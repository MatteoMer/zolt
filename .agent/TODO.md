# Zolt-Jolt Compatibility - Current Status

## Summary

**Stage 1**: PASSES (outer sumcheck) ✓
**Stage 2**: PASSES (product virtualization + RAM RAF) ✓
**Stage 3**: FAILS - Zero polynomials produce wrong output claim
**Stage 4-7**: Untested (blocked on Stage 3)

## Progress This Session

### Stage 3 Analysis Complete

The Stage 3 batched sumcheck has 3 instances:
1. **ShiftSumcheck** (degree 2, n_cycle_vars rounds)
2. **InstructionInputSumcheck** (degree 3, n_cycle_vars rounds)
3. **RegistersClaimReduction** (degree 2, n_cycle_vars rounds)

### Expected Output Claim Formulas

1. **ShiftSumcheck**:
   ```
   Σ gamma[i] * claim[i] * eq_plus_one(r_outer, r_sumcheck)
   + gamma[4] * (1 - is_noop_claim) * eq_plus_one(r_product, r_sumcheck)
   ```
   Where claims are: [unexpanded_pc, pc, is_virtual, is_first_in_sequence]

2. **InstructionInputSumcheck**:
   ```
   (eq(r_sumcheck, r_stage1) + gamma² * eq(r_sumcheck, r_stage2))
   * (right_input + gamma * left_input)
   ```
   Where:
   - left_input = left_is_rs1 * rs1_value + left_is_pc * unexpanded_pc
   - right_input = right_is_rs2 * rs2_value + right_is_imm * imm

3. **RegistersClaimReduction**:
   ```
   eq(r_sumcheck, r_spartan) * (rd_write_value + gamma * rs1_value + gamma² * rs2_value)
   ```

### Files Created

- `src/zkvm/spartan/stage3_prover.zig` - Stage 3 prover framework (incomplete)

### Current Issue

The Stage 3 prover needs:
1. **EqPlusOnePolynomial** implementation (for ShiftSumcheck)
2. Proper MLE construction from cycle witnesses
3. Correct round polynomial computation for batched sumcheck
4. Proper binding/folding of MLEs during rounds

## Implementation Status

### Stage 3 Prover Components Needed

1. [ ] EqPlusOnePolynomial - evaluates eq(r, x+1) for shift relations
2. [x] ShiftMLEs struct - holds the 5 MLE arrays for shift sumcheck
3. [x] InstructionInputMLEs struct - holds the 8 MLE arrays
4. [x] RegistersMLEs struct - holds the 3 MLE arrays
5. [ ] computeShiftRoundPoly - needs EqPlusOne integration
6. [ ] computeInstructionInputRoundPoly - needs proper eq evaluation
7. [ ] computeRegistersRoundPoly - needs proper eq evaluation
8. [ ] Proper transcript flow matching Jolt's verifier

### Transcript Flow for Stage 3

1. Derive 5 gamma powers (ShiftSumcheckParams)
2. Derive 1 gamma (InstructionInputParams)
3. Derive 1 gamma (RegistersClaimReductionSumcheckParams)
4. Append 3 input claims
5. Derive 3 batching coefficients
6. For each round: append compressed poly, derive challenge
7. Append 16 opening claims (5 + 8 + 3)

## Testing Commands

```bash
# Build Zolt
zig build

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```

## Next Steps

1. Implement EqPlusOnePolynomial in poly/mod.zig
2. Fix Stage 3 prover to use R1CSInputIndex correctly
3. Implement proper round polynomial computation
4. Test with simple trace to verify output claims match
