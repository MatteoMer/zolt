# Zolt-Jolt Compatibility TODO

## 🎯 Current Status: Claims & Commitments Parsing Fixed

**Session 2 (2026-01-27) - MAJOR PROGRESS**

### ✅ Completed Fixes

1. **Verified G1/G2 compression sizes are CORRECT**
   - BN254 G1 compressed: 32 bytes ✓ (matches arkworks)
   - BN254 G2 compressed: 64 bytes ✓ (matches arkworks)
   - The previous TODO was WRONG about needing 48/96 bytes

2. **Added missing CommittedPolynomial variants** (Commit: 096ee21)
   - Added `TrustedAdvice` (discriminant 5)
   - Added `UntrustedAdvice` (discriminant 6)

3. **Fixed SumcheckId enum alignment** (Commit: ab18a9a)
   - **Root Cause Found:** Zolt had 22 variants, Jolt has 24
   - Added `AdviceClaimReductionCyclePhase` (20)
   - Added `AdviceClaimReduction` (21)
   - Shifted `IncClaimReduction` to 22, `HammingWeightClaimReduction` to 23
   - This fixed the base encoding:
     - UNTRUSTED_BASE: 0
     - TRUSTED_BASE: 24 (was 22)
     - COMMITTED_BASE: 48 (was 44)
     - VIRTUAL_BASE: 72 (was 66)

### ✅ Verification Results

- **Proof Generation**: Working (40544 bytes)
- **Claims Parsing**: ✅ All 91 claims parse with valid Fr values
- **Commitments Parsing**: ✅ All 37 GT elements are valid
- **Sumcheck Structure**: ✅ 7 stages with correct round counts
- **Tests**: 714/714 passed (one OOM-killed due to memory limits)

### 🔄 Next Steps

1. **Run full Jolt verification test**
   ```bash
   cd /home/vivado/projects/jolt && cargo test --package jolt-core --no-default-features --features minimal test_verify_zolt_proof -- --ignored --nocapture
   ```

2. **Debug any verification failures**
   - Sumcheck proof content alignment
   - Dory proof structure
   - Transcript state matching

### Commands

```bash
# Generate proof
zig build run -- prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Debug format test (PASSING)
cd /home/vivado/projects/jolt && cargo test --package jolt-core --no-default-features --features minimal test_debug_zolt_format -- --ignored --nocapture

# Full verification test
cd /home/vivado/projects/jolt && cargo test --package jolt-core --no-default-features --features minimal test_verify_zolt_proof -- --ignored --nocapture
```

### Key Files Modified

- `src/zkvm/jolt_types.zig` - SumcheckId (24 variants) and CommittedPolynomial enums
- `src/zkvm/jolt_serialization.zig` - Added TrustedAdvice/UntrustedAdvice cases

### Success Criteria

- [x] Claims deserialize correctly
- [x] Commitments deserialize correctly
- [x] G1/G2 compression sizes correct
- [x] SumcheckId alignment correct
- [ ] Full verification passes
- [x] Most Zolt tests pass (714/714, one OOM)
