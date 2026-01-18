# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ✅ PASS | Fixed MontU128Challenge |
| 2 | ✅ PASS | ✅ PASS | - |
| 3 | ✅ PASS | ✅ PASS | - |
| 4 | ✅ PASS | ❌ FAIL | Input claim mismatch for val_eval/val_final |
| 5 | ✅ PASS | - | Blocked by Stage 4 |
| 6 | ✅ PASS | - | Blocked by Stage 4 |

## Session 45 Progress (2026-01-18)

### Fixed Issues
1. ✅ **Stage 4 actual polynomial sums**: Now use actual polynomial sums from provers (0 for Fibonacci)
   - val_eval_prover.computeInitialClaim() = 0
   - val_final_prover.computeInitialClaim() = 0

### Root Cause Analysis

**The Problem:**
Stage 4's batched sumcheck has 3 instances. For each instance, Jolt appends `input_claim` to transcript before sampling batching coefficients.

**Jolt's input_claim computation:**
1. RegistersReadWriteChecking: `rd_wv_claim + gamma * (rs1_rv_claim + gamma * rs2_rv_claim)`
2. RamValEvaluation: `rwc_val_claim - init_eval` where init_eval = initial_ram.evaluate(r_address)
3. RamValFinal: `val_final_claim - val_init_eval` where val_init_eval computed from initial_ram

**For Fibonacci (no RAM ops):**
- Instances 2 and 3 should have input_claim = 0
- But Jolt computes non-zero values because the opening claims in the proof differ from Jolt's computed init_eval

**Claim Sources:**
- `rwc_val_claim`: From Zolt's RWC prover's getOpeningClaims()
- `init_eval`: Jolt computes from initial_ram_state.evaluate(r_address)

For no-RAM programs, RamVal polynomial = initial_ram everywhere, so:
- rwc_val_claim SHOULD equal init_eval
- But they differ due to r_address endianness or polynomial representation differences

### Investigation Needed

1. **r_address Endianness**: Zolt's RWC prover reverses challenges when building r_address
   - Line 1158-1159: `r_address[log_k - 1 - i] = r_sumcheck[log_t + i]`
   - This creates BE r_address from LE challenges
   - Need to verify this matches what Jolt expects

2. **val_init Population**: Check how Zolt's RWC prover's val_init is populated
   - Is it the same as Jolt's initial_ram_state?

3. **Opening Point Storage**: Verify the opening point stored in proof matches r_address used in computation

### Files Involved
- `src/zkvm/ram/read_write_checking.zig`: RWC prover and getOpeningClaims()
- `src/zkvm/proof_converter.zig`: Claims storage and Stage 4 batching

## Commands

```bash
# Generate proof with Jolt's SRS
zig build run -- prove examples/fibonacci.elf --jolt-format --srs /tmp/jolt_dory_srs.bin --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Test cross-verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --nocapture --ignored

# Run Zolt tests
zig build test
```

## Success Criteria
- All 578+ Zolt tests pass
- Zolt proof verifies with Jolt verifier for Fibonacci example
- No modifications needed on Jolt side

## Previous Sessions Summary
- **Session 45**: Fixed Stage 4 polynomial sums, identified input_claim mismatch for val_eval/val_final
- **Session 44**: Fixed eq polynomial analysis, identified index reversal issue
- **Session 43**: Fixed Stage 1 MontU128Challenge conversion
