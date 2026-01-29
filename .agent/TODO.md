# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Serialization Fixed, Need to Regenerate Proof

Two serialization bugs were fixed in this session. Need to regenerate proof and retest.

## Current Verification Results (2026-01-29)

```
Old proofs (40544 bytes): Stages 1-3 PASSED, Stage 4 FAILED
New proofs (40531 bytes): Can't deserialize (serialization bugs)
```

## Fixes Applied This Session

### Fix 1: SumcheckId COUNT (commit b04bc93)
- Zolt had COUNT=24 but Jolt expects COUNT=22
- Removed `AdviceClaimReductionCyclePhase` and `AdviceClaimReduction` from SumcheckId enum
- This was accidentally reverted by commit 0baedb0

### Fix 2: JoltProof Config Serialization (commit cfd4441)
- Jolt expects 5 usizes at end: trace_length, ram_K, bytecode_K, log_k_chunk, lookups_ra_virtual_log_k_chunk
- Zolt was writing: 3 usizes + ReadWriteConfig (4 u8s) + OneHotConfig (2 u8s) + DoryLayout (1 u8)
- Fixed to write 5 usizes (40 bytes total, matching Jolt)

## Test Commands

### Jolt Cross-verification
```bash
cd /home/vivado/projects/zolt/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Generate Jolt-compatible Proof
```bash
zig build run -- prove examples/fibonacci.elf \
  --jolt-format \
  --export-preprocessing logs/zolt_preprocessing.bin \
  -o logs/zolt_proof_dory.bin \
  --srs /tmp/jolt_dory_srs.bin
```

### Internal Pipeline (Uses Internal Verifier)
```bash
zig build example-pipeline
```

## Unit Tests
- **714/714 tests pass** (all actual tests succeed)

## Working Proof Files (for reference)
- `/tmp/zolt_proof_dory2.bin` (40544 bytes, Jan 26) - old format, can test Stage 4 failure

## Key Fixes Applied (All Sessions)

### Session 82 (Current): Serialization Fixes
1. SumcheckId COUNT: 24 → 22 (match Jolt)
2. Config serialization: Use 5 usizes instead of mixed u8s

### Session 80: Stage 4 Fix
1. `rwc_val_claim` computation for null RWC prover
2. `val_final_prover` r_address endianness (LE, not BE)
3. Synthetic termination writes filtered from IncPolynomial

### Session 78: Stage 2 Fix
- Skip RAF/RWC prover initialization when input_claim is zero

### Session 77: Stage 1 Fix
- Correct config serialization format (trace_length, ram_K, bytecode_K, configs)

## Next Steps

1. **Regenerate proof with fixed serialization** - need ELF file or find alternative test
2. Test new proof against Jolt verifier
3. If deserializes correctly, debug Stage 4 sumcheck failure
4. The Stage 4 `r_cycle` mismatch suggests preprocessing exports wrong values
