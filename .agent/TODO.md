# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Internal (Zolt) | Notes |
|-------|-----------------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ✅ PASS | RWC - removed synthetic termination write |
| 3 | ✅ PASS | RegistersClaimReduction |
| 4 | ✅ PASS | **FIXED** - Montgomery conversion fix |
| 5 | ✅ PASS | RegistersValEvaluation |
| 6 | ✅ PASS | RAM evaluation |

## Session 41 Progress (2026-01-17)

### Completed
1. ✅ **Fixed Stage 4 Montgomery Conversion**
   - Root cause: Jolt's MontU128Challenge stores [0, 0, L, H] as BigInt, representing 2^128 * v
   - FIX: Store [0, 0, L, H] as standard form, then call toMontgomery()
   - Commit: 54200fa

2. ✅ **Fixed Proof Serialization Format**
   - Use `--jolt-format` flag (not `--jolt`)
   - Jolt can deserialize Zolt proofs successfully

3. ✅ **Identified SRS Loading**
   - Use `--srs /tmp/jolt_dory_srs.bin` to load Jolt's exported SRS
   - Ensures same G1/G2 points for commitment computation

### Current Blocker: Execution Trace Mismatch

Cross-verification fails because Zolt and Jolt produce **different execution traces**:
- Zolt executes `examples/fibonacci.elf` (Zolt's binary)
- Jolt executes `fibonacci-guest` (Jolt's guest binary)

Even with the same SRS, different polynomial evaluations → different commitments → different transcript state → different challenges.

**For true cross-verification, we need:**
1. **Option A**: Use EXACTLY the same binary
   - Export Jolt's compiled guest binary
   - Have Zolt execute and prove it
2. **Option B**: Export polynomial evaluations
   - Have Jolt export its execution trace
   - Zolt uses the same trace for proving
3. **Option C**: Match execution semantics
   - Ensure Zolt's emulator produces identical traces for identical programs
   - Requires byte-level compatibility of RISC-V implementation

## Next Steps
1. Export Jolt's fibonacci guest binary and use it in Zolt
2. Compare execution traces byte-by-byte
3. Identify any emulator differences that cause trace divergence

## Test Commands

```bash
# Generate Jolt reference files
cd /path/to/jolt/examples/fibonacci && cargo run --release -- --save

# Generate Zolt proof with Jolt's SRS
zig build run -- prove examples/fibonacci.elf --jolt-format --srs /tmp/jolt_dory_srs.bin -o /tmp/zolt_proof_dory.bin

# Run cross-verification test
cd /path/to/jolt/jolt-core && cargo test test_verify_zolt_proof --release -- --nocapture --ignored
```

## Commit History
- d328b37: docs: update notes with cross-verification findings
- 54200fa: fix: convert challenge scalar to proper Montgomery form (Stage 4 fix)
- 5cec222: fix: remove synthetic termination write from memory trace (Stage 2 fix)
