# Zolt-Jolt Compatibility TODO

## Current Status: Session 46 - January 3, 2026

**FIXED: memory_size mismatch**

Session 46 fixed:
1. Fixed `memory_size` mismatch - Zolt was using 128MB (DEFAULT_MEMORY_SIZE), Jolt fibonacci uses 32KB
2. Added `memory_size` parameter to `JoltDevice.fromEmulator()`
3. Updated all prover functions to use `memory_size = 32768`

**CRITICAL FINDING: GT bytes are IDENTICAL, but transcript states still diverge**

After fixing memory_size, the preamble VALUES match perfectly:
- Zolt: `memory_size=32768, inputs=[32], outputs=[e1,f2,cc,f1,2e], panic=0`
- Jolt: `memory_size=32768, inputs=[32], outputs=[e1,f2,cc,f1,2e], panic=0`

The GT commitment bytes (384 bytes each) are also **IDENTICAL**:
- Raw bytes match: `3e c9 a6 2a a1 6c 11 ca...`
- Reversed bytes match: `18 6e 82 cd 51 66 6b ca...`

**BUT the transcript states DIVERGE before GT commitments are appended:**
- Jolt state before GT[0]: `67 4b fe 13 8c 09 ea f3`
- Zolt state before GT[0]: `52 3c fb ce 6e ee a2 60`

This means the hash function is producing different results for the same preamble values!

---

## Session 46 Debugging

### Issues Fixed This Session

1. **memory_size Mismatch** (FIXED)
   - Zolt was hardcoding `DEFAULT_MEMORY_SIZE = 128 MB`
   - Jolt fibonacci uses `memory_size = 32768` (32 KB)
   - Fixed in `jolt_device.zig` and `zkvm/mod.zig`

### Key Discovery: Transcript State Divergence

Added aggressive debug output to both Zolt and Jolt transcripts. Findings:

**Preamble values are NOW identical:**
```
[ZOLT PREAMBLE] appendU64: max_input_size=4096
[ZOLT PREAMBLE] appendU64: max_output_size=4096
[ZOLT PREAMBLE] appendU64: memory_size=32768
[ZOLT PREAMBLE] appendBytes: inputs.len=1, inputs={ 32 }
[ZOLT PREAMBLE] appendBytes: outputs.len=5, outputs={ e1 f2 cc f1 2e }
[ZOLT PREAMBLE] appendU64: panic=0
[ZOLT PREAMBLE] appendU64: ram_K=65536
[ZOLT PREAMBLE] appendU64: trace_length=1024

[JOLT PREAMBLE] appendU64: max_input_size=4096
[JOLT PREAMBLE] appendU64: max_output_size=4096
[JOLT PREAMBLE] appendU64: memory_size=32768
[JOLT PREAMBLE] appendBytes: inputs.len=1, inputs=[32]
[JOLT PREAMBLE] appendBytes: outputs.len=5, outputs=[e1, f2, cc, f1, 2e]
[JOLT PREAMBLE] appendU64: panic=0
[JOLT PREAMBLE] appendU64: ram_K=65536
[JOLT PREAMBLE] appendU64: trace_length=1024
```

**GT commitment bytes are identical (verified):**
- First commitment raw: `[3e, c9, a6, 2a, a1, 6c, 11, ca, ...]`
- First commitment reversed: `[18, 6e, 82, cd, 51, 66, 6b, ca, ...]`

### Root Cause: append_u64 implementation difference

The states diverge DURING the preamble. Both implementations:
1. Pack u64 into 32 bytes (24 zeros + 8 BE bytes)
2. Hash state + packed bytes
3. Update state

Need to compare exact byte-by-byte what each `append_u64` call produces.

---

## Next Steps

1. **Debug append_u64 step-by-step**
   - Add debug output to Jolt's `append_u64` showing packed bytes and state transitions
   - Compare byte-by-byte with Zolt's implementation
   - Find exact point where states first diverge

2. **Possible differences to check:**
   - Byte packing format (BE vs LE, padding position)
   - Hash function initialization (round number handling)
   - State update mechanism

---

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Previous Sessions Summary

- **Session 45**: Fixed RV64 word operations, fib(50) now works
- **Session 44**: Added --input-hex to prove command
- **Session 43**: Fixed hardcoded empty outputs
- **Session 42**: Fixed challenge limb ordering and endianness
- **Session 41**: Fixed 125-bit challenge masking
