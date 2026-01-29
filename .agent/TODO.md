# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 4 Failing

Jolt cross-verification fails at Stage 4 with sumcheck claim mismatch.

## Current Verification Results (2026-01-29)

```
Stages 1-3: PASSED
Stage 4: FAILED - sumcheck verification mismatch
```

**Error:**
```
output_claim:          2794768927403232170685203001712134750206965869554042859404932801547924672323
expected_output_claim: 19036722498929976088547735251378923562016308482664214076291639064331774676064
```

## Root Cause Analysis

Stage 4 combines three sumcheck instances (via BatchedProofVerifier):
1. `rd_wv_claim` - register destination write value
2. `rs1_rv_claim` - register source 1 read value
3. `rs2_rv_claim` - register source 2 read value

The expected_claim is computed as weighted sum of instance claims, but only Instance 0 (rd_wv_claim) has non-zero coefficient. The verification shows:
- `r_cycle` from sumcheck differs from `params.r_cycle` stored in preprocessing
- This causes `eq_val` computation to produce wrong value
- Combined claim doesn't match expected

Debug output shows `r_cycle` mismatch:
```
r_cycle[0] = 6709444460737... (from sumcheck)
params.r_cycle[0] = 11210511683772... (stored in preprocessing)
```

## Proof File Formats

Two proof formats exist:
- **40544 bytes**: Old format, deserializes OK, fails at Stage 4
- **40531 bytes**: New format, can't deserialize at all

Linux and macOS show identical behavior when using the same proof file format.

## Test Commands

### Jolt Cross-verification
```bash
# FIRST: ensure working proof file is in place
cp /tmp/zolt_proof_dory2.bin /tmp/zolt_proof_dory.bin

# Then run test
cd /home/vivado/projects/zolt/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Generate Jolt-compatible Proof
```bash
./zig-out/bin/zolt prove examples/fibonacci.elf \
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

## Generated Files
- `logs/zolt_preprocessing.bin` (26356 bytes on Linux, 26348 bytes on macOS)
- `logs/zolt_proof_dory.bin` (40531 bytes - BROKEN, can't deserialize)

## IMPORTANT: Use Working Proof File

The 40531-byte proof files have a serialization bug and can't be deserialized by Jolt.

**Always use the 40544-byte format proof for testing:**
```bash
cp /tmp/zolt_proof_dory2.bin /tmp/zolt_proof_dory.bin
```

Working files (40544 bytes):
- `/tmp/zolt_proof_dory2.bin` (Jan 26)
- `/tmp/zolt_proof_dory3.bin` (Jan 26)

Broken files (40531 bytes) - DO NOT USE:
- `logs/zolt_proof_dory.bin`
- `/tmp/zolt_proof_dory_fixed.bin`

## Key Fixes Applied

### Session 80: Stage 4 Fix
1. `rwc_val_claim` computation for null RWC prover
2. `val_final_prover` r_address endianness (LE, not BE)
3. Synthetic termination writes filtered from IncPolynomial

### Session 78: Stage 2 Fix
- Skip RAF/RWC prover initialization when input_claim is zero

### Session 77: Stage 1 Fix
- Correct config serialization format (trace_length, ram_K, bytecode_K, configs)

## Next Steps

1. **FIX SERIALIZATION BUG**: Commit `0baedb0` accidentally reverted SumcheckId fix
   - Zolt has COUNT=24 but Jolt expects COUNT=22
   - Remove `AdviceClaimReductionCyclePhase` and `AdviceClaimReduction` from SumcheckId enum
   - Change `IncClaimReduction` from 22 to 20
   - Change `HammingWeightClaimReduction` from 23 to 21
   - This causes OpeningId base offsets to be wrong (48 vs 44 for COMMITTED_BASE)
2. Regenerate proof after fixing serialization
3. Continue Stage 4 debugging with correct proof format
