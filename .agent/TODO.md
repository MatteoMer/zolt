# Zolt-Jolt Compatibility Implementation

## Status: Session 47 - Stage 4 Verification Failure (After Stage 5 Fix)

## Progress Summary

### Key Fix Applied in This Session
1. **Disabled synthetic termination writes** - The tracer was injecting fake memory writes for program termination that weren't reflected in R1CS witnesses. This caused a mismatch between Stage 1 (RamAddress=0) and Stage 5's RAM trace (had the synthetic write).

2. **Removed the "correction" hack in Stage 5** - Stage 5 was overriding `current_batched_claim` with a "corrected" value computed from the trace. This broke verification because the verifier uses the original claims.

### Current State
- Stage 5 now passes the sumcheck round checks (p(0)+p(1)=claim)
- BUT Stage 4 now fails verification!
- The Corrected batched claim = Original batched claim (both match now)

### Stage 4 Failure Details
```
Sumcheck verification failed!
  output_claim:   [14, 98, cf, e7, ...]
  expected_claim: [2a, 83, f2, 6d, ...]
```

The Stage 4 sumcheck produces a different output_claim than what the verifier expects.

## Next Steps

1. Debug Stage 4's sumcheck computation vs verifier's expected_output_claim
2. Check if the RegistersRWC opening claims are correct
3. Verify that the Stage 4 polynomial coefficients match

## Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Changes Made This Session

1. `/home/vivado/projects/zolt/src/tracer/mod.zig` - Disabled `recordTerminationWrite()`
2. `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Disabled claim correction at line 1411
