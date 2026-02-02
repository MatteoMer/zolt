# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Challenges Mismatch Discovered

## Session 128 Summary

### Progress Made

1. **Fixed bit-reversal issue in ra_polys materialization:**
   - Removed unnecessary bit-reversal since expanding tables use HighToLow binding
   - Verified that the expanding table values are computed correctly

2. **Added detailed debug output:**
   - Print exact lookup indices for each cycle
   - Print expanding table values at round 128
   - Trace ra_chunk computation step by step

### KEY FINDING: Challenges Mismatch

**The address round challenges are DIFFERENT between Zolt and Jolt!**

Zolt's Round 0 challenge (in hex, lower 16 bytes):
```
7c a2 5a 17 c1 29 02 cb 92 c0 d5 87 8c 3b 73 da
```

Jolt's r_address_prime r[0] (serialized):
```
[5a, d0, bd, aa, 13, 59, 1f, 8e, 48, 0d, ca, d7, ce, eb, b2, 15]
```

**These are completely different values!**

This means the transcript diverged BEFORE the address rounds (round 0), which causes:
1. Different challenges during sumcheck
2. Different expanding table values
3. Different ra_polys materialization
4. Different final ra_claims
5. Different expected_output_claim

### Root Cause Analysis

The transcript state must diverge in one of these places:
1. Initial transcript state
2. Polynomial commitments appended to transcript
3. Batch coefficients appended to transcript
4. Previous stage challenges

### Next Steps

1. **Debug transcript state at Stage 5 start:**
   - Compare transcript state between Zolt and Jolt at the beginning of Stage 5
   - Check what was appended to transcript in Stage 4

2. **Verify input claims match:**
   - The three input claims (regs_val, ram_ra, lookups) are appended at Stage 5 start
   - Verify these match between Zolt and Jolt

3. **Check batch coefficients:**
   - batch0, batch1, batch2 are derived from transcript
   - These affect how the three instances are combined

### Key Files

**Zolt:**
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck prover

**Jolt:**
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf implementation

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin 2>&1 | grep -E "STAGE5"

# Copy and verify
cp logs/zolt_*.bin /tmp/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
