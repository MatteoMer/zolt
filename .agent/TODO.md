# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ✅ PASS | Batched sumcheck (5 instances) |
| 3 | ✅ PASS | Challenges match between Zolt and Jolt |
| 4 | ❌ FAIL | input_claim_val_final mismatch |
| 5-6 | ⏳ Blocked | Waiting for Stage 4 |

## Stage 4 Root Cause Analysis

### Fixed: Batching Coefficient ✅
Zolt now uses `batch0` from transcript instead of `F.one()`.
- Commit: `fix(stage4): use correct batching coefficient from transcript`
- The batching coefficients match between Zolt and Jolt.

### Verified Matching Values ✅
- Claims (val, rd_wa, rs1_ra, etc.) - MATCH
- eq_val bytes - MATCH
- combined bytes - MATCH
- batching_coeffs[0] = 93683484670461660293859922911333626135 - MATCH

### Remaining Issue: input_claim_val_final ❌

Stage 4 has 3 instances:
1. Instance 0: RegistersReadWriteChecking - ✅ Working
2. Instance 1: RamValEvaluation - input_claim = 0 ✅
3. Instance 2: RamValFinalEvaluation - **input_claim ≠ 0 BUT Jolt expects 0**

**The Problem:**

Jolt's verifier computes input_claim for Instance 2:
```rust
// From proof:
let val_final_claim = accumulator.get(RamValFinal@RamOutputCheck);

// Computed from actual initial RAM:
let val_init_eval = initial_ram_state.evaluate(&r_address_final);

// Input claim:
input_claim = val_final_claim - val_init_eval
```

For input_claim = 0:
- `val_final_claim` must equal `val_init_eval`
- Zolt must store RamValFinal@RamOutputCheck = evaluation of initial RAM

**Current State:**
```
Zolt: output_val_final_claim ≠ output_val_init_claim (from Stage 2 output sumcheck)
Jolt: expects RamValFinal = val_init_eval (so input_claim = 0)
```

### Root Cause

Zolt's Stage 2 output sumcheck produces `output_val_final_claim` that doesn't match
the initial RAM evaluation. This could be because:

1. The output sumcheck polynomial is computing the wrong values
2. The binding point is incorrect
3. The relationship between val_final and val_init is misunderstood

### Next Steps to Fix Stage 4

1. **Investigate Stage 2 Output Sumcheck**
   - Understand what val_final and val_init represent
   - For a correct execution, val_final at output addresses should equal expected output
   - The output sumcheck should verify this relationship

2. **Check Jolt's Expected Behavior**
   - For fibonacci with input (0,1) and output 55
   - What should val_final_claim evaluate to?
   - Why does Jolt expect input_claim = 0?

3. **Possible Fix**
   - Store RamValFinal@RamOutputCheck = initial RAM evaluation (same as RamValInit)
   - This would make input_claim = 0
   - But need to verify this doesn't break Stage 2 verification

## Testing Commands

```bash
# Build and run prover
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verifier
cd /Users/matteo/projects/jolt && ZOLT_LOGS_DIR=/Users/matteo/projects/zolt/logs cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
