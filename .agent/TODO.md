# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ✅ PASS | Batched sumcheck (5 instances) |
| 3 | ✅ PASS | Challenges match between Zolt and Jolt |
| 4 | ❌ FAIL | output_claim ≠ expected_output_claim |
| 5-6 | ⏳ Blocked | Waiting for Stage 4 |

## Stage 4 Root Cause: FOUND!

### The Bug: Missing Batching Coefficient

**The claims (val, rd_wa, rs1_ra, etc.) all MATCH between Zolt and Jolt!**

The eq_val and combined values also MATCH:
- eq_val bytes: [94, 4a, 2b, 5f, ...] ✅
- combined bytes: [80, 5d, c7, 14, ...] ✅

**The problem is the BATCHING COEFFICIENT!**

Stage 4 is a batched sumcheck with 3 instances:
1. Instance 0: RegistersReadWriteChecking (non-zero polynomial)
2. Instance 1: RamValEvaluation (zero polynomial for Zolt)
3. Instance 2: RamValFinalEvaluation (zero polynomial for Zolt)

Jolt's BatchedSumcheck::verify does:
```rust
// 1. Append input_claim for each instance to transcript
sumcheck_instances.iter().for_each(|sumcheck| {
    let input_claim = sumcheck.input_claim(accumulator);
    transcript.append_scalar(&input_claim);
});

// 2. Derive batching coefficients from transcript
let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

// 3. Verify: output_claim == sum_i(coeff_i * instance_i_expected_claim)
```

The weighted expected is: `coeff_0 * expected_0 + coeff_1 * 0 + coeff_2 * 0 = coeff_0 * expected_0`

But Zolt uses `F.one()` (coefficient = 1) instead of `coeff_0`!

### Fix Required

Before running Stage 4 sumcheck:
1. Compute input_claim for all 3 instances
2. Append them to transcript IN ORDER
3. Get `batching_coeffs = transcript.challenge_vector(3)`
4. Pass `batching_coeffs[0]` to Stage 4 prover to scale round polynomials

The round polynomials should be: `round_poly * batching_coeffs[0]`

### Debug Output Analysis

The eq values are being used correctly:
```
[STAGE4 INIT] eq_cycle_evals[0] = { 59, 244, 75, 167, 205, 155, 74, 77 }
[ZOLT STAGE4 EQ_USE] eq_cycle_evals[0] = { 59, 244, 75, 167, 205, 155, 74, 77 }
```

The round polynomial evaluations are non-zero:
```
[PROOF_CONV STAGE4] Round 0 regs_evals:
  regs_evals[0] = { 91, 171, 190, 37, ... }
  regs_evals[1] = { 67, 79, 33, 77, ... }
```

### Next Investigation Steps

1. Compare the polynomial values (val, rd_wa, rs1_ra, rs2_ra, inc) between Zolt and Jolt
2. Check if GruenSplitEqPolynomial has special handling that Zolt is missing
3. Verify MLE evaluation point matches what Jolt expects
4. Compare exact round polynomial formulas

## Testing Commands

```bash
# Build and run prover
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verifier
cd /Users/matteo/projects/jolt && ZOLT_LOGS_DIR=/Users/matteo/projects/zolt/logs cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
