# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ✅ PASS | Batched sumcheck (5 instances) |
| 3 | ✅ PASS | Challenges match between Zolt and Jolt |
| 4 | ❌ FAIL | output_claim ≠ expected_output_claim |
| 5-6 | ⏳ Blocked | Waiting for Stage 4 |

## Stage 4 Deep Dive

### Key Finding: Eq Polynomial Ordering

Investigated the eq polynomial ordering between LE and BE. The key insight:

**The eq polynomial evaluation is mathematically the same regardless of LE/BE ordering!**

This is because:
```
eq(r, j) = Π_i [(1-r_i)*(1-j_i) + r_i*j_i]
```

The product is commutative, so reversing `r` doesn't change the final value.

### Current Verification Failure

```
output_claim = 17388657012463501289028458753211654741331023344380174987495804360843521599428
expected_output_claim = 13087017565662880187225932472750198262574881263786061171449896105154105298773
```

The mismatch suggests the round polynomial computation doesn't match Jolt's expectation.

### Instance Analysis

- Instance 0 (RegistersReadWriteChecking): expected_claim = 14832... (non-zero, from MLE)
- Instance 1 (RamValEvaluation): expected_claim = 0 (zeros being stored)
- Instance 2 (RamValFinal): expected_claim = 0 (zeros being stored)

### Possible Root Causes (Remaining)

1. **Polynomial Construction Difference**
   - Jolt uses sparse matrix (ReadWriteMatrixCycleMajor) with GruenSplitEqPolynomial
   - Zolt uses dense polynomial representation
   - The round polynomial coefficients might differ due to different computation methods

2. **Variable Binding Order Mismatch**
   - Both use LowToHigh (LSB first) binding
   - But the internal handling might differ

3. **MLE Claim Values**
   - The opening claims Zolt stores might be at the wrong evaluation point

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
