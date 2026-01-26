# Session 64 Progress - Stage 4 Output Claim Investigation

## Key Findings

### 1. The Issue (after Session 63 fixes)
- Instance 1 (RamValEvaluation) expected_claim = 0 ✅
- Instance 2 (RamValFinalEvaluation) expected_claim = 0 ✅
- Stage 4 output_claim still doesn't match expected_output_claim

Current numbers:
- Jolt reads from proof: `2794768927403232170685203001712134750206965869554042859404932801547924672323`
- Jolt expects: `19036722498929976088547735251378923562016308482664214076291639064331774676064`

### 2. Verified Components

**EqPolynomial formula** - CONFIRMED CORRECT:
- Zolt uses: `∏ (r_i * x_i + (1 - r_i) * (1 - x_i))` ✅
- Same as Jolt's formula
- Positional pairing when same endianness

**Sumcheck binding** - CONFIRMED CORRECT:
- `new_val = val_lo * (1 - c) + val_hi * c` ✅
- Same challenge used for:
  - evalFromHint (batched_claim update)
  - regs_prover.bindChallenge()
  - stage4_r_sumcheck storage

**Opening claims for Instance 0**:
- val_claim, rs1_ra_claim, rs2_ra_claim, rd_wa_claim, inc_claim
- Stored at `jolt_proof.opening_claims` with `sumcheck_id = .RegistersReadWriteChecking`

### 3. The Batched Sumcheck Equation

```
expected_output_claim = Σ coeff[i] × instance[i].expected_output_claim
                     = coeff[0] × eq(r_cycle, stage3_r_cycle) × combined + 0 + 0

where combined = rd_wa × (inc + val) + γ × (rs1_ra × val) + γ² × (rs2_ra × val)
```

### 4. Debug Output Added

Added verify check at proof_converter.zig:2117-2132:
- Shows batched_claim (sumcheck output)
- Shows expected_output for Instance 0
- Shows coeff[0] × expected_output (weighted)
- Shows Instance 1&2 expected (should be 0)
- Compares if they match

### 5. Waiting For

Prover output to see exact values:
- Does batched_claim equal coeff[0] × expected_output?
- Are Instance 1&2 expected values actually 0?

### 6. Potential Root Causes to Investigate

1. **Transcript divergence** - batching coefficients computed differently
2. **Opening claims mismatch** - stored values don't match polynomial evaluations
3. **eq polynomial endianness** - r_cycle_sumcheck vs stage3_r_cycle ordering

### 7. Files Modified

- `src/zkvm/proof_converter.zig` - Added VERIFY CHECK debug block
- `.agent/TODO.md` - Updated with Session 64 analysis
- `.agent/SESSION_64_ANALYSIS.md` - Created analysis document

### 8. Next Steps

1. Capture prover verify check output
2. Compare individual polynomial openings with verifier expectations
3. Check if eq polynomial computation uses correct endianness for both vectors
4. Verify batching coefficients match between Zolt and Jolt
