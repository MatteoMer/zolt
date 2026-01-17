# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ✅ PASS | Fixed MontU128Challenge |
| 2 | ✅ PASS | ✅ PASS | - |
| 3 | ✅ PASS | ✅ PASS | - |
| 4 | ✅ PASS | ❌ FAIL | r_cycle mismatch |
| 5 | ✅ PASS | - | - |
| 6 | ✅ PASS | - | - |

## Session 43 Progress (2026-01-17)

### Fixed: Stage 1 MontU128Challenge Conversion

**Root Cause:**
The `challengeScalar128Bits` function was calling `toMontgomery()` on the result, but Jolt's `from_bigint_unchecked` does NOT do Montgomery conversion - it stores the limbs directly.

**Fix:**
Changed from:
```zig
const standard_form = F{ .limbs = .{ 0, 0, masked_low, masked_high } };
const result = standard_form.toMontgomery();
```
To:
```zig
const result = F{ .limbs = .{ 0, 0, masked_low, masked_high } };
```

This matches Jolt's behavior where `MontU128Challenge::into()` calls `from_bigint_unchecked` which wraps the BigInt directly without conversion.

### Current Status: Stage 4 Failing

**Error:**
```
output_claim:          19271728596168755243423895321875251085487803860811927729070795448153376555895
expected_output_claim: 5465056395000139767713092380206826725893519464559027111920075372240160609265
```

**Evidence:**
The r_cycle values don't match:
- `r_cycle (from sumcheck)` - 8 elements from Zolt's proof
- `params.r_cycle (stored)` - 8 elements expected by Jolt

These are completely different values, causing the eq polynomial evaluation to fail.

### Next Steps

1. [ ] Debug Stage 4 r_cycle derivation in proof_converter.zig
2. [ ] Check how r_cycle is stored in preprocessing
3. [ ] Verify Stage 3 sumcheck challenges are correct
4. [ ] Fix eq polynomial computation

### Commands

```bash
# Generate proof with Jolt's SRS
zig build run -- prove examples/fibonacci.elf --jolt-format --srs /tmp/jolt_dory_srs.bin --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Test cross-verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --nocapture --ignored
```

## Previous Sessions Summary

- **Session 42**: Identified streaming outer prover Az*Bz mismatch (was actually MontU128 issue)
- **Session 41**: Fixed Stage 4 Montgomery conversion, fixed proof serialization
- **Session 40**: Fixed Stage 2 synthetic termination write
- **Earlier**: Fixed Stage 3 prefix-suffix, Stage 1 NextPC=0 for NoOp
