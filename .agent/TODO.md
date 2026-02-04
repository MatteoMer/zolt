# Zolt-Jolt Compatibility Implementation

## Status: Session 47 - Stage 4 RegistersRWC Mismatch

## Progress Summary

### MILESTONE: Stage 2 Now Passes!
Commit `971f5c8` fixed Stage 2 by re-enabling termination bit in val_final and val_io.

### Fixes Applied This Session
1. **Disabled synthetic termination writes** - Causing R1CS/RAF mismatch
2. **Removed Stage 5 "correction" hack** - Fixed
3. **Fixed ValFinal input_claim calculation** - Using prover's computeInitialClaim()
4. **Re-enabled termination bit in val_final and val_io** - Required for OutputSumcheck

### Current Issue: Stage 4 RegistersRWC

Stage 4 has 3 instances:
1. **RegistersRWC** - FAILING with claim mismatch
2. **ValEvaluation** - PASSING (zero claims expected)
3. **ValFinal** - PASSING (zero claims expected)

**Debug Output Analysis**:
```
Stage 4 Instance 0 expected_output_claim:
  claim: [d7, a6, 2d, d3, ...]
  expected_claim (coeff*claim): [6b, f3, 99, f5, ...]

output_claim:   [b2, ce, 1c, 8b, ...]  <- from sumcheck
expected_claim: [6b, f3, 99, f5, ...]  <- from polynomial openings
```

**Key Insight**: The r_cycle mismatch is INTENTIONAL - they're connected via eq polynomial:
```rust
eq_eval = EqPolynomial::mle_endian(&r_cycle_sumcheck, &params.r_cycle_stage3)
expected_output_claim = eq_eval * combined
```

The real issue is in the polynomial claims used for `combined`:
- RegistersVal, Rs1Ra, Rs2Ra, RdWa (virtual)
- RdInc (committed)

### Root Cause Investigation Needed

1. **Polynomial Claims**: Check if Zolt stores correct claims for RegistersRWC
2. **Sumcheck Prover**: Verify Stage 4 RegistersRWC sumcheck computation
3. **Opening Point**: Ensure opening point from challenges is correct

### Files to Investigate
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 prover, claims storage
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage4_gruen_prover.zig` - Stage 4 sumcheck
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/registers/read_write_checking.rs` - Jolt verifier

### Test Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Session Commits
1. `39d8386` - Fix Stage 5 claim mismatch: disable synthetic termination writes
2. `971f5c8` - Fix Stage 2 OutputSumcheck by re-enabling termination bit

### SESSION_ENDING
- **Stage 2 is FIXED** - termination bit re-enabled in val_final and val_io
- **Stage 4 is the blocker** - RegistersRWC expected_output_claim doesn't match
- The r_cycle difference is intentional (connected via eq polynomial)
- Need to investigate polynomial claim storage for RegistersRWC
