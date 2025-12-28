# Zolt-Jolt Compatibility TODO

## Phase 1: Transcript Compatibility ✅ COMPLETE
- [x] Create Blake2bTranscript in Zolt
- [x] Port Blake2b-256 hash function
- [x] Implement 32-byte state with round counter
- [x] Match Jolt's append/challenge methods exactly
- [x] Test vector validation - same inputs produce same challenges

## Phase 2: Proof Structure Refactoring ✅ COMPLETE
- [x] Restructure JoltProof in zkvm/mod.zig
- [x] Add 7 explicit stage proof fields
- [x] Match stage ordering with Jolt
- [x] Opening claims structure for batched verification

## Phase 3: Serialization Alignment ✅ COMPLETE
- [x] Implement arkworks-compatible field element serialization
- [x] Remove ZOLT magic header (pure arkworks format)
- [x] Match usize encoding (u64 little-endian)
- [x] GT/G1/G2 point serialization in arkworks format
- [x] Dory commitment serialization

## Phase 4: Commitment Scheme ✅ COMPLETE
- [x] Complete Dory implementation with Jolt-compatible SRS
- [x] SRS loading from Jolt-exported files
- [x] MSM with same point format as arkworks
- [x] Pairing operations matching arkworks

## Phase 5: Verifier Preprocessing Export ✅ COMPLETE (Session 7)
- [x] DoryVerifierSetup structure with precomputed pairings
- [x] delta_1l, delta_1r, delta_2l, delta_2r, chi computation
- [x] Full GT element serialization (Fp12 -> 12 * 32 bytes)
- [x] G1/G2 point serialization with flags
- [x] JoltVerifierPreprocessing (generators + shared)
- [x] CLI --export-preprocessing includes verifier setup

## Phase 6: Integration Testing 🔄 IN PROGRESS

### Unit Tests ✅
- [x] Transcript: Same inputs -> same challenges
- [x] Field: Same element -> same bytes
- [x] Commitment: Same polynomial -> same commitment
- [x] Pairing: e(G1,G2) matches Jolt output

### Integration Tests 🔄 NEXT
- [ ] Generate proof in Zolt for fibonacci.elf
- [ ] Export proof + preprocessing in Jolt-compatible format
- [ ] Create Rust test harness to verify with Jolt
- [ ] End-to-end cross-verification test

---

## Session 7 Summary

### Completed ✅
1. **DoryVerifierSetup** - `src/zkvm/preprocessing.zig`
   - Precomputed pairing values for verification
   - delta_1l, delta_1r, delta_2l, delta_2r, chi arrays
   - Full GT/G1/G2 serialization in arkworks format
   - fromSRS() creates setup from prover SRS

2. **Updated CLI Export**
   - --export-preprocessing now includes verifier setup
   - Full JoltVerifierPreprocessing format

3. **All 632 tests passing**

### Test Results
- Zolt tests: 632/632 PASS ✅
- Proof generation: WORKING (12.3 KB for sum.elf)
- Preprocessing export: WORKING (304 KB for sum.elf)

---

## Current Status

**632 tests passing** - All cryptographic components working correctly.

### What's Working
1. **Blake2bTranscript** - Exact match with Jolt's transcript
2. **Field arithmetic** - BN254 scalar/base field matching arkworks
3. **Dory SRS loading** - Load Jolt-exported SRS files
4. **Dory commitments** - Same polynomial -> same GT commitment
5. **Pairings** - Miller loop + final exponentiation matching arkworks
6. **Preprocessing export** - Full JoltVerifierPreprocessing format

### What's Next
1. Create Rust test harness in Jolt for verification
2. Generate test vectors with known inputs/outputs
3. Debug any byte-level mismatches
4. Full end-to-end proof verification

---

## Commands

```bash
# Run all tests
zig build test --summary all

# Build release
zig build -Doptimize=ReleaseFast

# Generate proof with preprocessing export
./zig-out/bin/zolt prove examples/fibonacci.elf \
    --export-preprocessing prep.bin \
    -o proof.bin

# With Jolt-exported SRS
./zig-out/bin/zolt prove examples/fibonacci.elf \
    --srs jolt_srs.bin \
    --export-preprocessing prep.bin \
    -o proof.bin
```

## File Sizes (Example: sum.elf)
- Proof: 12.3 KB (12,641 bytes)
- Preprocessing: 304 KB (311,347 bytes)
- SRS (20 vars): ~2.6 MB

---

## Architecture Notes

### Preprocessing Export Format
```
[DoryVerifierSetup]
  - delta_1l: Vec<GT>   (k+1 elements, 384 bytes each)
  - delta_1r: Vec<GT>
  - delta_2l: Vec<GT>
  - delta_2r: Vec<GT>
  - chi: Vec<GT>
  - g1_0: G1            (64 bytes)
  - g2_0: G2            (128 bytes)
  - h1: G1
  - h2: G2
  - ht: GT              (384 bytes)
  - max_log_n: u64

[SharedPreprocessing]
  - BytecodePreprocessing
  - RAMPreprocessing
  - MemoryLayout
```

---

## Previous Sessions

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
