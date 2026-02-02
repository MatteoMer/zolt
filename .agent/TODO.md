# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Expected Output Claim Mismatch

## Session 127 Summary

### Progress Made

1. **Fixed polynomial degree mismatch for address rounds:**
   - Instance 2's `eval_1` was not being added to `combined_poly`
   - Now computes `eval_1_inst2 = lookups_claim - eval_0_inst2` from sumcheck property
   - Uses `toomCookToCompressed()` instead of `fromEvalsAndHint()` for proper degree-3 encoding

2. **Fixed claim update formula for degree-3 polynomials:**
   - Was: `c1 = claim - c0 - c2 - c3` (WRONG)
   - Now: `c1 = claim - 2*c0 - c2 - c3` (correct, from p(0)+p(1)=claim property)

### Current Status

**What's Working:**
- Stage 5 transcript progresses through all 136 rounds correctly
- Polynomial coefficients match between Zolt and Jolt for round 0
- Challenges match between prover and verifier
- Sumcheck output_claim matches after all rounds complete

**What's Failing:**
- Expected output claim doesn't match the sumcheck output claim
- The verifier computes expected_claim from instance evaluations:
  - Instance 0 (RegistersValEvaluation): `expected_claim_0 * batch0`
  - Instance 1 (RamRaClaimReduction): `expected_claim_1 * batch1`
  - Instance 2 (InstructionReadRaf): `expected_claim_2 * batch2`
  - Sum should equal the sumcheck output_claim

**Error Details:**
```
output_claim:   [46, 98, ab, 10, 06, ee, 18, b1, 5d, ba, c5, b6, 19, fc, e0, 15, ...]
expected_claim: [fb, cc, 05, 3a, fa, 11, 16, ac, bb, a3, f1, ed, 45, 66, a0, 3b, ...]
```

### Root Cause Analysis

The expected_claim is computed by evaluating the bound polynomials at the sumcheck challenges. This means one of the following is incorrect in Zolt's prover:

1. **Instance 0 (RegistersValEvaluation) polynomial binding:**
   - The `inc_evals`, `wa_evals`, `lt_evals` arrays are bound each round
   - Final claim should be `inc[0] * wa[0] * lt[0]` at the reduced point

2. **Instance 1 (RamRaClaimReduction) polynomial binding:**
   - B_1, B_2 arrays are bound during address rounds
   - eq_raf/rw/val_bound evolve during cycle rounds
   - Final claim should be sparse sum over RAM accesses

3. **Instance 2 (InstructionReadRaf) polynomial binding:**
   - `lookups_eq_evals`, `ra_chunk_weights`, `lookups_combined_vals` are bound
   - Final claim should match `eq_eval * ra_eval * combined_val`

### Next Steps

1. **Debug Instance 0 binding:**
   - Add debug output for `inc_evals[0]`, `wa_evals[0]`, `lt_evals[0]` after each round
   - Compare final values with Jolt's expected values

2. **Debug Instance 1 binding:**
   - Verify B_1, B_2 binding produces correct final values
   - Check eq_raf_bound, eq_rw_bound, eq_val_bound computation

3. **Debug Instance 2 binding:**
   - Verify ra_chunk_weights binding produces correct final ra_claim
   - Check lookups_eq_evals final evaluation

### Key Files

**Zolt:**
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck prover

**Jolt:**
- `jolt-core/src/subprotocols/sumcheck.rs` - Batched sumcheck verify (expected_output_claim computation)
- `jolt-core/src/zkvm/registers/val_evaluation.rs` - Instance 0 expected claim
- `jolt-core/src/zkvm/claim_reductions/ram_ra.rs` - Instance 1 expected claim
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Instance 2 expected claim

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin 2>&1 | grep -E "STAGE5"

# Copy and verify
cp logs/zolt_*.bin /tmp/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
