# Zolt zkVM Implementation TODO

## Current Status

**Project Status: PREPROCESSING EXPORT COMPLETE 🎉**

### Session 6 Summary

#### Completed ✅
1. **Preprocessing Module** - `src/zkvm/preprocessing.zig`
   - JoltInstruction with JSON serialization (matching Jolt's serde format)
   - BytecodePreprocessing with PC mapper
   - RAMPreprocessing for initial memory state
   - JoltSharedPreprocessing combining all components
   - All tests passing (630/630)

2. **CLI Export** - `--export-preprocessing` flag
   - Exports bytecode, RAM, and memory layout
   - Serialized in arkworks-compatible format
   - Works alongside proof generation

#### Test Results
- Zolt tests: 630/630 PASS ✅
- Proof generation: WORKING (27.8 KB)
- Preprocessing export: WORKING (554 KB)
- Jolt deserialization: PASS ✅

---

## Cross-Verification Status

### What Works
- Proof serialization matches Jolt format
- Preprocessing shared components exported
- All cryptographic primitives verified compatible

### Remaining Work

The exported preprocessing only contains `JoltSharedPreprocessing`, not the full
`JoltVerifierPreprocessing` which also requires:
- `PCS::VerifierSetup` - Dory generators (G1, G2 points from SRS)

**Options to complete:**
1. Export Dory verifier setup from Zolt alongside preprocessing
2. Create Jolt test that builds generators from SRS
3. Use shared SRS files between both implementations

---

## Usage

```bash
# Generate Jolt-compatible proof with preprocessing
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove --jolt-format \
    -o /tmp/zolt_proof.jolt \
    --export-preprocessing /tmp/zolt_preprocessing.dat \
    examples/fibonacci.elf

# Files created:
#   /tmp/zolt_proof.jolt (27 KB)
#   /tmp/zolt_preprocessing.dat (554 KB)
```

---

## Files Modified This Session

- `src/zkvm/preprocessing.zig` - NEW: Preprocessing module
- `src/zkvm/mod.zig` - Added preprocessing export
- `src/main.zig` - Added --export-preprocessing CLI option
- `.agent/NOTES.md` - Updated status

---

## Commits This Session

1. `feat: add Jolt-compatible preprocessing serialization`
2. `feat: add --export-preprocessing CLI option`

---

## Previous Sessions

### Session 5
- Format compatibility verified
- ECALL handling implemented
- Guest execution debugging (loops indefinitely)

### Session 4
- UniSkip cross-product algorithm fixed
- All 618 tests passing
- Proof deserialization working
