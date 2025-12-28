# Zolt-Jolt Compatibility Notes

## Current Status (December 28, 2024, Session 6)

### Session 6 Progress

**Preprocessing Export Implemented!**

1. Created `src/zkvm/preprocessing.zig`:
   - `JoltInstruction` with JSON serialization (matches Jolt's serde format)
   - `BytecodePreprocessing` with PC mapper
   - `RAMPreprocessing` for initial memory state
   - `JoltSharedPreprocessing` combining all components

2. Added `--export-preprocessing` CLI option:
   ```bash
   ./zolt prove --jolt-format -o proof.jolt --export-preprocessing prep.dat program.elf
   ```

3. Test output:
   - Proof: 27.8 KB (`/tmp/zolt_proof.jolt`)
   - Preprocessing: 554 KB (`/tmp/zolt_preprocessing.dat`)

### Remaining Work

The exported preprocessing only contains `JoltSharedPreprocessing`, not the full
`JoltVerifierPreprocessing` which also needs `PCS::VerifierSetup` (Dory generators).

To complete cross-verification:
1. Also export Dory verifier setup (generators) from Zolt
2. OR: Create Jolt test that builds generators from scratch

---

## Previous Status (December 28, 2024, Session 5)

### Summary

**Zolt proof format is FULLY COMPATIBLE with Jolt.**

All serialization tests pass. The only remaining blocker for full verification is that:
- Zolt generates proofs for bare-metal C programs (`fibonacci.c`)
- Jolt generates preprocessing for SDK-compiled Rust programs (`fibonacci-guest`)

These are fundamentally different programs, so verification fails as expected.

### Working Components ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Blake2b Transcript | ✅ Working | All test vectors match |
| Dory Commitment | ✅ Working | GT elements match, MSM correct |
| Proof Structure | ✅ Working | 7 stages, claims, all parse |
| Serialization | ✅ Working | Byte-level compatible |
| UniSkip Algorithm | ✅ Working | Cross-product approach |

### Jolt Tests Results

```
test_serialization_vectors       ✅ PASS
test_zolt_compatibility_vectors  ✅ PASS
test_debug_zolt_format           ✅ PASS
test_deserialize_zolt_proof      ✅ PASS
test_gt_serialization_size       ✅ PASS
test_jolt_proof_roundtrip        ✅ PASS
test_verify_zolt_proof           ⚠️ BLOCKED (different programs)
```

### Why Verification Fails

When `test_verify_zolt_proof` runs, it loads:
1. `/tmp/jolt_verifier_preprocessing.dat` - Generated for Jolt's fibonacci-guest
2. `/tmp/zolt_proof_dory.bin` - Generated for Zolt's fibonacci.elf
3. `/tmp/fib_io_device.bin` - Jolt's I/O device (input = 50)

The error "Stage 1 univariate skip first round" occurs because:
- The preprocessing encodes the bytecode polynomial from `fibonacci-guest`
- The proof was generated for different bytecode (`fibonacci.elf`)
- The R1CS constraints don't match

**This is NOT a format issue - it's a program mismatch.**

## Architecture Details

### Jolt's I/O Memory Layout

Jolt SDK guest programs read input from a specific memory region:
- Input address: ~0x7fffa000 (varies based on MemoryLayout)
- Format: postcard-serialized bytes
- The guest reads this via standard RISC-V load instructions

When Zolt runs the Jolt ELF without setting up the input:
- Load from 0x7fffa000 returns 0
- Program takes different execution path
- Trace differs from Jolt's expected trace

### Preprocessing Structure

Jolt's `JoltVerifierPreprocessing` contains:
```rust
pub struct JoltVerifierPreprocessing {
    pub generators: PCS::VerifierSetup,  // Dory generators
    pub shared: JoltSharedPreprocessing,
}

pub struct JoltSharedPreprocessing {
    pub bytecode: BytecodePreprocessing,  // Program bytecode
    pub ram: RAMPreprocessing,            // Initial memory state
    pub memory_layout: MemoryLayout,      // Memory region addresses
}
```

For verification to work, the proof must match the preprocessing's bytecode.

### Options to Achieve Full Verification

**Option A: Make Zolt understand Jolt's I/O**
1. Parse MemoryLayout from preprocessing
2. Set up input bytes at correct address
3. Run exact same execution as Jolt

**Option B: Create matching bare-metal program**
1. Write C program that works without I/O
2. Compile for both RISC-V targets
3. Generate matching proofs

**Option C: Export preprocessing from Zolt**
1. Implement BytecodePreprocessing in Zolt
2. Implement RAMPreprocessing in Zolt
3. Serialize in arkworks format
4. Use Zolt preprocessing with Zolt proof in Jolt verifier

## Key Files

### Zolt
- `src/transcripts/blake2b.zig` - Blake2b transcript (matches Jolt)
- `src/zkvm/serialization.zig` - Arkworks-compatible serialization
- `src/zkvm/prover.zig` - 7-stage proof generation
- `src/poly/commitment/dory.zig` - Dory commitment scheme
- `src/zkvm/spartan/outer.zig` - UniSkip cross-product algorithm

### Jolt (Reference)
- `jolt-core/src/transcripts/blake2b.rs` - Reference transcript
- `jolt-core/src/zkvm/proof_serialization.rs` - Proof format spec
- `jolt-core/src/zkvm/verifier.rs` - Verifier and preprocessing
- `jolt-core/src/zolt_compat_test.rs` - Cross-verification tests

## Commands

```bash
# Test Zolt (all 618 tests)
cd /Users/matteo/projects/zolt
zig build test --summary all

# Generate Jolt-format proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Generate Jolt preprocessing (for different program)
cd /Users/matteo/projects/jolt/examples/fibonacci
cargo run --release -- --save

# Run deserialization test (PASSES)
cd /Users/matteo/projects/jolt
cargo test -p jolt-core test_deserialize_zolt_proof -- --ignored --nocapture

# Run verification test (FAILS due to program mismatch)
cargo test -p jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

## Session History

### Session 4
- Fixed UniSkip cross-product algorithm
- All 618 tests passing
- Proof deserialization working

### Session 5
- Investigated verification failure
- Confirmed format compatibility is complete
- Identified program mismatch as the blocker
- Documented three options for full verification
- Updated TODO.md with comprehensive status
