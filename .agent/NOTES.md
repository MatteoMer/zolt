# Zolt-Jolt Compatibility Notes

## Current Status (December 28, 2024, Session 7)

### Session 7 Progress

**DoryVerifierSetup Complete!**

1. Implemented `DoryVerifierSetup` in `src/zkvm/preprocessing.zig`:
   - Precomputed pairing values: delta_1l, delta_1r, delta_2l, delta_2r, chi
   - Full GT/G1/G2 serialization in arkworks format
   - `fromSRS()` creates verifier setup from prover SRS

2. Updated `--export-preprocessing` to export full JoltVerifierPreprocessing:
   - DoryVerifierSetup (generators)
   - JoltSharedPreprocessing (bytecode, RAM, memory layout)

3. All 632 tests passing

### Test Output
```
Exporting preprocessing to: test_preprocessing.bin
  Preprocessing exported successfully! (311347 bytes)
```

The exported file now contains the complete verifier preprocessing needed by Jolt.

### Architecture of DoryVerifierSetup

```zig
pub const DoryVerifierSetup = struct {
    delta_1l: std.ArrayListUnmanaged(GT),  // Δ₁L[k] = e(Γ₁[..2^(k-1)], Γ₂[..2^(k-1)])
    delta_1r: std.ArrayListUnmanaged(GT),  // Δ₁R[k] = e(Γ₁[2^(k-1)..2^k], Γ₂[..2^(k-1)])
    delta_2l: std.ArrayListUnmanaged(GT),  // Same as Δ₁L
    delta_2r: std.ArrayListUnmanaged(GT),  // Δ₂R[k] = e(Γ₁[..2^(k-1)], Γ₂[2^(k-1)..2^k])
    chi: std.ArrayListUnmanaged(GT),       // χ[k] = e(Γ₁[..2^k], Γ₂[..2^k])
    g1_0: G1Point,                         // First G1 generator
    g2_0: G2Point,                         // First G2 generator
    h1: G1Point,                           // Blinding generator in G1
    h2: G2Point,                           // Blinding generator in G2
    ht: GT,                                // h_t = e(h₁, h₂)
    max_log_n: usize,                      // Maximum log₂ of polynomial size
};
```

### Serialization Format
- GT elements: 384 bytes (12 × 32 byte Fp elements)
- G1 points: 64 bytes (2 × 32 byte Fp elements, uncompressed)
- G2 points: 128 bytes (4 × 32 byte Fp elements, uncompressed)
- All in arkworks little-endian format

---

## Previous Status (December 28, 2024, Session 6)

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

---

## Working Components ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Blake2b Transcript | ✅ Working | All test vectors match |
| Dory Commitment | ✅ Working | GT elements match, MSM correct |
| Proof Structure | ✅ Working | 7 stages, claims, all parse |
| Serialization | ✅ Working | Byte-level compatible |
| UniSkip Algorithm | ✅ Working | Cross-product approach |
| Preprocessing Export | ✅ Working | Full JoltVerifierPreprocessing |
| DoryVerifierSetup | ✅ Working | Precomputed pairings |

---

## Key Files

### Zolt
- `src/transcripts/blake2b.zig` - Blake2b transcript (matches Jolt)
- `src/zkvm/serialization.zig` - Arkworks-compatible serialization
- `src/zkvm/preprocessing.zig` - Full preprocessing export
- `src/zkvm/prover.zig` - 7-stage proof generation
- `src/poly/commitment/dory.zig` - Dory commitment scheme
- `src/zkvm/spartan/outer.zig` - UniSkip cross-product algorithm

### Jolt (Reference)
- `jolt-core/src/transcripts/blake2b.rs` - Reference transcript
- `jolt-core/src/zkvm/proof_serialization.rs` - Proof format spec
- `jolt-core/src/zkvm/verifier.rs` - Verifier and preprocessing

---

## Commands

```bash
# Test Zolt (all 632 tests)
cd /Users/matteo/projects/zolt
zig build test --summary all

# Generate proof with full preprocessing export
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf \
    --export-preprocessing prep.bin \
    -o proof.bin

# With custom SRS
./zig-out/bin/zolt prove examples/fibonacci.elf \
    --srs jolt_srs.bin \
    --export-preprocessing prep.bin \
    -o proof.bin
```

---

## Session History

### Session 7
- Implemented DoryVerifierSetup
- Full JoltVerifierPreprocessing export
- All 632 tests passing

### Session 6
- Preprocessing module complete
- CLI export working
- 630/630 tests passing

### Session 5
- Format compatibility verified
- ECALL handling implemented

### Session 4
- UniSkip cross-product algorithm fixed
- All 618 tests passing
