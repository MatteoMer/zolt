# Zolt-Jolt Compatibility: Current Status

## Status: Stage 4 Fails (Stages 1-3 PASS!) 🟡

## Session 78 Summary (2026-01-29)

### MAJOR PROGRESS: Stages 1-3 now PASS!

**Fixed Stage 2 issue:**
- Root cause: Synthetic termination write was included in memory trace for RAF/RWC provers
- When `input_claim = 0`, the RAF/RWC provers should produce zero polynomials
- Fix: Skip prover initialization when input_claim is zero for RAF (Instance 1) and RWC (Instance 2)
- Code changes in `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig`:
  - Added `use_raf_prover` check at line 3115
  - Added `use_rwc_prover` check at line 2957

### Current Failure: Stage 4

**Error:**
```
Sumcheck verification failed!
  output_claim:   [fc, f5, b5, 80, ...]
  expected_claim: [2d, 76, 8b, f4, ...]
Verification failed: Stage 4
```

**Debug observation from Jolt:**
```
r_cycle (from sumcheck): 8 elements - [7b, be, 40, f8, ...]
params.r_cycle (from Stage 3): 8 elements - [d7, 9b, 60, 5e, ...]
```

The `r_cycle` used in sumcheck differs from `params.r_cycle` used to compute expected claims.
This could be:
1. An ordering/endianness issue in how challenges are extracted
2. Wrong phase configuration
3. Mismatch between prover's challenge extraction and verifier's

### Next Steps
1. [ ] Debug Stage 4 r_cycle mismatch
2. [ ] Verify phase configuration (phase1, phase2, phase3) matches Jolt
3. [ ] Check if normalize_opening_point is implemented correctly
4. [ ] Run verification test after fix

### How to Run Tests

```bash
# Zolt proof generation
cd /home/vivado/projects/zolt
zig build -Doptimize=ReleaseSafe && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Jolt verification test
cd /home/vivado/projects/jolt
ZOLT_LOGS_DIR=/home/vivado/projects/zolt/logs cargo test --features "minimal,zolt-debug" --no-default-features -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --nocapture --ignored
```

---

## Previous Sessions

### Session 77
- Fixed config serialization format
- Stage 1 passes
- Stage 2 fails: output_claim != expected_claim

### Session 76
- Fixed ZOLT header issue using `--jolt-format`
