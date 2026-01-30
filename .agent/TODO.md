# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 4 Sumcheck Mismatch

## Current Issue (2026-01-30)

### Root Cause Analysis

Stage 4 verification fails because the sumcheck proof produces a different `output_claim` than expected:
- `output_claim = 2794768927403232170685203001712134750206965869554042859404932801547924672323`
- `expected_output_claim = 19036722498929976088547735251378923562016308482664214076291639064331774676064`

### Understanding the Mismatch

1. **Stage 4 Structure:**
   - Instance 0: RegistersReadWriteChecking (15 rounds: LOG_K=7 + log_T=8)
   - Instance 1: ValEvaluation (0 weighted - zero claim)
   - Instance 2: ValFinal (0 weighted - zero claim)

2. **Expected Output Claim Computation:**
   ```
   expected_output = eq_val * combined
   eq_val = EqPolynomial::mle_endian(r_cycle_stage4, params.r_cycle_from_stage3)
   combined = rd_write_value_claim + gamma * (rs1_value_claim + gamma * rs2_value_claim)
   ```

3. **Key Observation:**
   - `r_cycle (from Stage 4 sumcheck)` != `params.r_cycle (from Stage 3)`
   - This is **expected** - they come from different sumchecks
   - The eq_val evaluates the "equality" between these two points

4. **The Problem:**
   - Zolt's Stage 4 round polynomials produce wrong challenges via Fiat-Shamir
   - These wrong challenges lead to wrong `output_claim`
   - The verifier's `expected_output_claim` is based on the claims (correct) but output_claim (from proof) is wrong

### Likely Causes

1. **Round Polynomial Computation:**
   - Zolt's `computeRoundPolynomialGruen()` may be computing wrong coefficients
   - The eq polynomial factorization (Gruen optimization) may have issues

2. **Polynomial Ordering/Indexing:**
   - Stage 4 uses a 3-phase structure (cycle vars, address vars, remaining cycle)
   - The variable ordering in Zolt may not match Jolt

3. **eq Polynomial Setup:**
   - The GruenSplitEqPolynomial is initialized with `r_cycle` from Stage 3
   - The r_cycle ordering (LE vs BE) may be wrong

### Verification Results
- Stages 1-3: PASSED
- Stage 4: FAILED (sumcheck output mismatch)
- Instance 0 expected_claim calculation: CORRECT
- Instance 1, 2: Zero contributions (correct for fibonacci example)

## Test Commands

### Jolt Cross-verification
```bash
cd /home/vivado/projects/zolt/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Generate Fresh Proof
```bash
cd /home/vivado/projects/zolt
zig build run -- prove examples/fibonacci.elf \
  --export-preprocessing logs/zolt_preprocessing.bin \
  -o logs/zolt_proof_dory.bin
```

## Key Files

### Zolt Stage 4 Implementation
- `src/zkvm/spartan/stage4_gruen_prover.zig` - Main Gruen-optimized prover
  - `phase1ComputeMessage()` - First cycle vars via Gruen
  - `phase2ComputeMessage()` - Address vars (no eq binding)
  - `phase3ComputeMessage()` - Remaining cycle vars via dense eq
  - `gruenPolyDeg3()` - Converts [q(0), q_X2] to cubic coefficients

- `src/zkvm/proof_converter.zig` - Batched sumcheck coordination
  - Lines 2135-2300: Stage 4 round loop
  - Combines RegistersRWC, ValEval, ValFinal round polynomials

### Jolt Stage 4 Implementation
- `jolt-core/src/zkvm/registers/read_write_checking.rs`
  - `RegistersReadWriteCheckingVerifier::expected_output_claim()` - Lines 815-921
  - `normalize_opening_point()` - Lines 151-200 (phase 1/2/3 splitting)

## Next Steps

1. **Debug Round 0 of Stage 4:**
   - Print Zolt's round polynomial coefficients [c0, c1, c2, c3]
   - Compare with what Jolt verifier expects
   - Check if p(0)+p(1) = input_claim holds

2. **Check Gruen eq Polynomial Setup:**
   - Verify r_cycle ordering passed to GruenSplitEqPolynomial
   - Compare E_in, E_out arrays with Jolt's prover

3. **Trace Polynomial Values:**
   - Print val_poly, ra_poly, wa_poly, inc_poly values for first few (k,j) pairs
   - Compare with Jolt's polynomial evaluations

4. **Test Individual Round Verification:**
   - Check if each round's compressed polynomial [c0, c2, c3] matches
   - Verify challenge derivation is identical
