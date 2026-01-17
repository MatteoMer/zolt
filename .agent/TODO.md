# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ❌ FAIL | Sumcheck claim mismatch |
| 2 | ✅ PASS | - | - |
| 3 | ✅ PASS | - | - |
| 4 | ✅ PASS | Montgomery fix applied |
| 5 | ✅ PASS | - | - |
| 6 | ✅ PASS | - | - |

## Session 42 Progress (2026-01-17)

### Current Blocker: Stage 1 Cross-Verification Failure

**Error Details:**
```
=== SUMCHECK VERIFICATION FAILED ===
output_claim:          10634556229438437044377436930505224242122955378672273598842115985622605632100
expected_output_claim: 17179895292250284685319038554266025066782411088335892517994655937988141579529
```

**Key Observations:**
1. **Proof serialization works** - Jolt can deserialize the proof successfully
2. **Preprocessing exports** - But may have issues (many zeros in hex dump)
3. **Transcript might diverge** - Seeing zeros being appended: `first 8 bytes=[00, 00, 00, 00, 00, 00, 00, 00]`

**Possible Root Causes:**
1. Transcript appending different data (commitments may not match)
2. Sumcheck coefficients computed with different challenges
3. Polynomial evaluations differ due to witness mismatch
4. Preprocessing data format mismatch

### Test Results
- **714/714 unit tests pass** ✅
- One integration test killed (signal 9, likely OOM)

### Commands Used
```bash
# Generate proof with Jolt's SRS
zig build run -- prove examples/fibonacci.elf --jolt-format --srs /tmp/jolt_dory_srs.bin --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Test cross-verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --nocapture --ignored
```

## Next Steps

1. **Debug transcript states** - Add logging to compare Zolt prover vs Jolt verifier transcript at each step
2. **Verify preprocessing format** - Ensure bytecode/RAM preprocessing matches Jolt's expected format
3. **Check commitment values** - The zeros in transcript suggest missing/wrong commitment data
4. **Compare sumcheck coefficients** - Verify coefficients match between proof and verifier expectations

## Previous Sessions

### Session 41
- ✅ Fixed Stage 4 Montgomery conversion
- ✅ Fixed proof serialization format
- ✅ Identified SRS loading approach

### Session 40
- ✅ Fixed Stage 2 - removed synthetic termination write
- ✅ Deep investigation of Stage 4

### Earlier
- ✅ Fixed Stage 3 - prefix-suffix decomposition
- ✅ Fixed Stage 1 - NextPC = 0 for NoOp padding
