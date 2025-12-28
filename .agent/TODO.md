# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: FORMAT COMPATIBLE ✅, EXECUTION INTEGRATION IN PROGRESS 🔄**

### Latest Progress (2024-12-28, Agent Session 5 Continued)

#### Completed This Session

1. **Format Compatibility Verified** ✅
   - All Jolt deserialization tests pass
   - Field/GT element serialization matches arkworks
   - Blake2b transcript produces identical challenges

2. **I/O Region Support** ✅
   - Added `isIOAddress()`, `readByteWithIO()`, `writeByteWithIO()` functions
   - Modified LOAD instructions (LB, LBU, LH, LHU, LW, LWU, LD) to read from I/O region
   - Modified STORE instructions (SB, SH, SW, SD) to write to I/O region
   - Tests pass: 622/622 (4 new I/O tests added)

3. **CLI Input Support** ✅
   - Added `--input FILE` option to load input from file
   - Added `--input-hex HEX` option for hex-encoded input
   - Example: `zolt run --input-hex 32 program.elf` (input = 50)

4. **I/O Read Works** ✅
   - Confirmed I/O reads happen correctly
   - Address 0x7fffa000 returns the correct input byte
   - Debug output: `[IO READ] addr=0x000000007fffa000 -> 0x32`

#### In Progress

**Jolt Guest Execution Investigation** 🔄

The fibonacci-guest still exits early (21 cycles) even with correct I/O:
- The LB instruction reads `0x32` (50) from input region ✅
- But program branches to early exit path
- May be a postcard encoding issue (Jolt uses varint encoding)

### Next Steps

1. **Investigate postcard encoding**: Jolt SDK uses postcard crate for serialization
   - Small numbers like 50 are encoded as single varint byte
   - But Jolt might read a length prefix first

2. **Try memory-ops-guest**: This Jolt guest has no inputs
   - Should execute the same way in both Jolt and Zolt
   - Can test execution compatibility without I/O complexity

3. **Debug execution divergence**: Add more trace output to understand why branches differ

---

## Test Results Summary

### Zolt: 622/622 tests PASS ✅

### Jolt Cross-Verification Tests

| Test | Status | Details |
|------|--------|---------|
| `test_serialization_vectors` | ✅ PASS | Field/GT serialization matches |
| `test_zolt_compatibility_vectors` | ✅ PASS | Blake2b transcript compatible |
| `test_debug_zolt_format` | ✅ PASS | Proof structure parseable |
| `test_deserialize_zolt_proof` | ✅ PASS | Full proof deserializes correctly |
| `test_gt_serialization_size` | ✅ PASS | GT element size (384 bytes) correct |
| `test_verify_zolt_proof` | ⚠️ BLOCKED | Different programs |

---

## Session 5 File Changes

### Modified Files
1. **src/tracer/mod.zig**
   - Added I/O-aware memory access (readByteWithIO, writeByteWithIO, etc.)
   - Updated all LOAD/STORE variants to use I/O-aware functions
   - Added I/O region unit tests

2. **src/main.zig**
   - Added `--input FILE` option
   - Added `--input-hex HEX` option
   - Updated runEmulator to pass input bytes to emulator

### Commits
- `docs: update compatibility status - format fully verified`
- `feat(tracer): add I/O region support for Jolt guest programs`
- `feat(cli): add --input and --input-hex options for guest programs`

---

## Commands Reference

```bash
# Build and test Zolt
cd /Users/matteo/projects/zolt
zig build test --summary all

# Generate Jolt-format proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Run with input
./zig-out/bin/zolt run --input-hex 32 /path/to/guest.elf

# Run Jolt tests
cd /Users/matteo/projects/jolt
cargo test -p jolt-core test_deserialize_zolt_proof -- --ignored --nocapture
```

---

## Architecture Summary

### What Works
1. **Proof Format**: Zolt proofs deserialize correctly in Jolt
2. **Transcript**: Blake2b challenges match exactly
3. **Dory Commitments**: GT elements serialize correctly
4. **I/O Memory**: Reads from 0x7fffa000 region work

### What Needs Work
1. **Guest Program Execution**: Programs don't run identically yet
2. **Postcard Encoding**: May need to match Jolt's input serialization
3. **Full Verification**: Blocked by execution mismatch

### Verification Equation

For verification to pass:
```
Zolt_Proof(ELF, Input) verified_by Jolt_Verifier(Preprocessing(ELF))
```

Currently we have:
- Zolt_Proof(fibonacci.c) ≠ Jolt_Preprocessing(fibonacci-guest)

Need:
- Same ELF in both
- Same input in both
- Same execution trace
