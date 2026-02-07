# Zolt-Jolt Compatibility Implementation

## Status: Stage 6 sumcheck verification fails, bytecode entries now match

### Current Issue: Stage 6 Sumcheck Mismatch (output_claim != expected_claim)

Bytecode entries now match perfectly between Zolt prover and Jolt verifier (including entry k=0 termination store and all static ELF instructions).

The sumcheck still fails because the prover's round polynomials produce a different output_claim than what the verifier expects. Possible causes:
1. BytecodeRa polynomial commitments/evaluations mismatch
2. r_cycle values from opening accumulator don't match
3. Other instances (1,3,4,5) have incorrect polynomial evaluations
4. Batching coefficients mismatch

### What's Fixed
- Static ELF bytecode population (buildBytecodeEntries now uses static ELF, not just trace)
- k=0 termination store flag accumulation
- NoOp/padding entries
- All bytecode entries match between Zolt and Jolt

### What's Working
- Stages 1-5 PASS
- Stage 6 bytecode entries match perfectly (verified via debug comparison)
- Stage 6 round polynomial generation runs
- Stage 7 not yet implemented

### What Needs Fixing
1. Stage 6 round polynomial values (output_claim from proof != expected_claim)
   - Need to add prover-side debug to dump per-instance contributions
   - Need to compare Val polynomial evaluations at the opening point
2. Stage 7 (HammingWeightClaimReduction) not yet implemented

### Test Commands
```bash
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram
```
