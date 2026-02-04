# Zolt-Jolt Compatibility Implementation

## Status: Session 47 - Stage 4 r_cycle Mismatch

## Progress Summary

### Fixes Applied in This Session
1. **Disabled synthetic termination writes** - The tracer was injecting fake memory writes for program termination that weren't reflected in R1CS witnesses.

2. **Removed the "correction" hack in Stage 5** - Stage 5 was overriding `current_batched_claim` with a computed value.

### Stage 5 Status: FIXED
- Stage 5 now has matching Corrected/Original batched claims
- The sumcheck internally passes (p(0)+p(1)=claim)
- Stage 5 no longer causes verification failure

### Current Issue: Stage 4 r_cycle Mismatch

The Stage 4 verification fails because the RegistersRWC expected_output_claim uses the wrong `r_cycle`:

From Jolt's debug output:
```
r_cycle (from sumcheck): 8 elements
  r_cycle[0]: [..., d3, f4, a1, 90, cb, 66, 87, c8, ...]
params.r_cycle (from Stage 3): 8 elements
  params.r_cycle[0]: [..., b8, 79, 98, ad, 83, 13, f0, 03, ...]
```

These should be the SAME! The `params.r_cycle` from Stage 3 should match what Stage 4's sumcheck produces via `normalize_opening_point`.

The issue is either:
1. Zolt's Stage 4 sumcheck challenges are wrong
2. Zolt's Stage 3 `r_cycle` passed to Stage 4 is wrong
3. The mapping between sumcheck challenges and opening point is wrong

### Next Steps
1. Compare Stage 3's r_cycle with Stage 4's sumcheck challenges
2. Check if `normalize_opening_point` is being applied correctly
3. Verify the phase ordering in Stage 4 sumcheck

## Key Files
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 prover initialization
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage4_gruen_prover.zig` - Stage 4 polynomial computation

## Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Session Commits
1. `39d8386` - Fix Stage 5 claim mismatch: disable synthetic termination writes

## SESSION_ENDING
Context is getting long. Key state:
- Stage 5 is fixed (claims are consistent now)
- Stage 4 fails due to r_cycle mismatch between Stage 3 and Stage 4 sumcheck
- Need to trace the r_cycle flow from Stage 3 -> Stage 4
