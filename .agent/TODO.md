# Zolt-Jolt Compatibility TODO

## Current Status: January 6, 2026 - Session 63

**STATUS: STAGE 1 PASSES - Stage 2 fails at univariate skip (input_claim mismatch)**

### Session 63 Progress

**Investigated Stage 2 UniSkip input_claim mismatch**

Key findings:
1. **All 5 base_evals match** between Zolt and Jolt:
   - Product ✅
   - WriteLookupOutputToRD ✅
   - WritePCtoRD (both 0) ✅
   - ShouldBranch (both 0) ✅
   - ShouldJump (both 0) ✅

2. **Product virtualization witness values fixed**:
   - Added `computeLookupOutput` for branch instructions (returns 0/1)
   - Fixed `WriteLookupOutputToRD = IsRdNotZero * FlagWriteLookupOutputToRD`
   - Fixed `WritePCtoRD = IsRdNotZero * FlagJump`
   - Fixed `ShouldBranch = LookupOutput * BranchFlag`
   - Fixed `ShouldJump = FlagJump * (1 - NextIsNoop)`

3. **tau_high sampling**:
   - Jolt samples a NEW tau_high for Stage 2 from transcript (not reuse Stage 1's)
   - Stage 2 tau = [r_cycle_outer, tau_high] where tau_high is freshly sampled
   - The transcript state before sampling must match for tau_high to match

**Current issue:**
- Transcript state diverges before sampling Stage 2's tau_high
- This causes tau_high mismatch → input_claim mismatch
- Need to investigate what's being appended to transcript between Stage 1 and Stage 2

### What Now Matches

| Component | Status |
|-----------|--------|
| Preamble (all fields) | ✅ |
| Commitments (all 5) | ✅ |
| Stage 1 sumcheck (all 46 values) | ✅ |
| Stage 2 base_evals (all 5) | ✅ |
| Stage 2 tau_high | ❌ (transcript divergence) |
| Stage 2 input_claim | ❌ |

### Next Steps

1. Debug transcript state divergence before Stage 2 tau_high sampling
2. Check what claims/values Jolt appends to transcript between Stage 1 and Stage 2
3. Verify Zolt is appending the same claims in the same order

### Test Commands

```bash
# Generate Zolt proof (with empty I/O)
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf 2>&1 | tee /tmp/zolt.log

# Run Jolt verification
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture 2>&1 | tee /tmp/jolt.log

# Run comparison script
python3 scripts/compare_sumcheck.py /tmp/zolt.log /tmp/jolt.log --verbose
```

### Key Code Changes (Session 63)

**`src/zkvm/r1cs/constraints.zig`:**
- Added branch condition evaluation in `computeLookupOutput` for opcode 0x63
- Added product virtualization computation block in `fromTraceStep`:
  - `WriteLookupOutputToRD = IsRdNotZero * FlagWriteLookupOutputToRD`
  - `WritePCtoRD = IsRdNotZero * FlagJump`
  - `ShouldBranch = LookupOutput * BranchFlag`
  - `ShouldJump = FlagJump * (1 - NextIsNoop)`

**`src/zkvm/proof_converter.zig`:**
- Stage 2 tau_high: Reverted to sampling fresh challenge (matching Jolt)
