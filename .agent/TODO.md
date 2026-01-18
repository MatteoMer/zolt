# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ✅ PASS | Fixed MontU128Challenge |
| 2 | ✅ PASS | ✅ PASS | - |
| 3 | ✅ PASS | ✅ PASS | - |
| 4 | ✅ PASS | ❌ FAIL | Sumcheck output_claim mismatch |
| 5 | ✅ PASS | - | Blocked by Stage 4 |
| 6 | ✅ PASS | - | Blocked by Stage 4 |

## Session 45 Progress (2026-01-18)

### Fixed Issues
1. ✅ **val_eval and val_final input claims**: Now use actual polynomial sums (0 for Fibonacci)
   - Before: Used derived claims from Stage 2 (non-zero)
   - After: Use actual polynomial sums computed from provers (0 for no-RAM programs)
   - This fixed Jolt's Instance 1 and Instance 2 expected_claim to be 0

### Remaining Issue
The sumcheck polynomial rounds produce a different `output_claim` than `expected_output_claim`:
```
output_claim:          3159763944798181886722852590115930947586131532755679042258164540994444897089
expected_output_claim: 4857024169349606329580068783301423991985019660972366542411131427015650777104
```

This indicates the sumcheck round polynomial coefficients don't evaluate correctly at the challenges.

### Debug Values from Last Run
Jolt Stage 4:
- Instance 0 expected_claim = 17177621565056141430840966297488157406781932625004861742899573144370995217735
- coeff = 116658923796321962861521498013863413240
- eq_val = 11120493948656633218659538332974463697959928087456130038626595260978605918283
- combined = 6240241591786886181707067559668125496115098553042097044753792988593967656359

### Possible Causes
1. **Batching coefficient mismatch**: Different transcript states leading to different batch coefficients
2. **Round polynomial computation**: Zolt's round polynomials may not match Jolt's expected format
3. **Eq polynomial index binding**: The eq polynomial may still have incorrect index pairing

### Next Steps
1. [ ] Compare Zolt's batching_coeff[0] with Jolt's coeff to verify they match
2. [ ] If batching coeffs differ, trace transcript divergence after Stage 3
3. [ ] If batching coeffs match, the issue is in round polynomial generation
4. [ ] Add debug output for round 0 polynomial coefficients comparison

## Previous Session Analysis (Session 44)

**Root Cause Found:**
The eq polynomial binding order doesn't match between Zolt's prover and Jolt's verifier.

**Jolt's verifier expects:**
```
eq_val = Π_i eq_term(r_cycle_BE[i], params.r_cycle_BE[i])
       = Π_i eq_term(challenges[log_T-1-i], stage3[log_T-1-i])
```
This pairs challenges[j] with stage3[j] (same index).

**Zolt's prover computes:**
```
eq_val = Π_i eq_term(r_cycle_be[i], c[i])
       = Π_i eq_term(stage3[log_T-1-i], c[i])
```
This pairs stage3[log_T-1-i] with c[i] (reversed index pairing).

## Files Modified This Session
- `src/zkvm/proof_converter.zig`: Fixed Stage 4 input claims for val_eval and val_final

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
- **Session 44**: Fixed eq polynomial analysis, identified index reversal issue
- **Session 43**: Fixed Stage 1 MontU128Challenge conversion
- **Session 42**: Identified streaming outer prover Az*Bz mismatch
- **Session 41**: Fixed Stage 4 Montgomery conversion, proof serialization
- **Session 40**: Fixed Stage 2 synthetic termination write
