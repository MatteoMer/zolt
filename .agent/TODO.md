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

## Phase 5: Verifier Preprocessing Export ✅ COMPLETE
- [x] DoryVerifierSetup structure with precomputed pairings
- [x] delta_1l, delta_1r, delta_2l, delta_2r, chi computation
- [x] Full GT element serialization (Fp12 -> 12 * 32 bytes)
- [x] G1/G2 point serialization with flags
- [x] JoltVerifierPreprocessing (generators + shared)
- [x] CLI --export-preprocessing includes verifier setup

## Phase 6: Integration Testing ✅ MOSTLY COMPLETE

### Proof Deserialization ✅
- [x] Jolt can deserialize Zolt proof in --jolt-format
- [x] Opening claims: 48 entries, all valid
- [x] Commitments: 5 GT elements, all valid
- [x] Sumcheck proofs: structure matches

### Preprocessing Deserialization ✅ COMPLETE
- [x] DoryVerifierSetup parses correctly
- [x] BytecodePreprocessing parses correctly
  - Fixed NoOp/UNIMPL to serialize as unit variants ("NoOp", "UNIMPL")
  - Fixed immediate types (u64 for FormatI/U/J, i128 for B, i64 for S)
  - Fixed FENCE/ECALL to use FormatI operands
- [x] RAMPreprocessing parses correctly
- [x] MemoryLayout parses correctly
- [x] Full JoltVerifierPreprocessing::deserialize_uncompressed works!

### End-to-End Verification 🚧
- [ ] Run full verification with Zolt proof + Zolt preprocessing

---

## Session 8 Summary

### Completed ✅
1. **Proof serialization working**
   - 30.9 KB proof with --jolt-format
   - 48 opening claims all valid Fr
   - 5 Dory commitments (GT elements)
   - Jolt `test_deserialize_zolt_proof` passes

2. **Preprocessing fully working**
   - DoryVerifierSetup parses correctly
   - BytecodePreprocessing parses correctly (after JSON format fixes)
   - RAMPreprocessing parses correctly
   - MemoryLayout parses correctly
   - Full `JoltVerifierPreprocessing::deserialize_uncompressed` works!

### Key Fixes Applied
1. **NoOp/UNIMPL serialization**: Changed from `{"NoOp":{...}}` to `"NoOp"`
2. **Immediate types**:
   - FormatI/U/J: use u64 (sign-extended from i32)
   - FormatS: use i64
   - FormatB: use i128
3. **FENCE/ECALL**: Use FormatI operands instead of None

### Test Results
- Zolt tests: 632/632 PASS ✅
- Proof generation: 30.9 KB in Jolt format ✅
- Proof deserialization: PASS ✅
- Preprocessing deserialization: PASS (with deserialize_uncompressed) ✅

---

## Commands

```bash
# Run all tests
zig build test --summary all

# Build release
zig build -Doptimize=ReleaseFast

# Generate proof in Jolt format
./zig-out/bin/zolt prove examples/sum.elf \
    --jolt-format \
    --export-preprocessing /tmp/zolt_preprocessing.bin \
    -o /tmp/zolt_proof_dory.bin

# Run Jolt preprocessing test
cd /path/to/jolt
cargo test --package jolt-core test_load_zolt_preprocessing -- --ignored --nocapture

# Run Jolt proof deserialization test
cargo test --package jolt-core test_deserialize_zolt_proof -- --ignored --nocapture
```

## File Sizes
- Proof (Jolt format): 30.9 KB (30,926 bytes)
- Preprocessing: 62.2 KB (62,223 bytes)

---

## Architecture Notes

### Proof Format (Jolt-compatible)
```
[Opening Claims] - BTreeMap<OpeningId, Fr>
  - Length: u64
  - For each claim: OpeningId (1-2 bytes) + Fr (32 bytes)

[Commitments] - Vec<GT>
  - Length: u64
  - For each: GT (384 bytes = 12 * 32)

[Stage 1] - UniSkipFirstRoundProof + SumcheckInstanceProof
[Stage 2] - UniSkipFirstRoundProof + SumcheckInstanceProof
[Stages 3-7] - SumcheckInstanceProof each

[Advice Proofs] - Optional<DoryProof>
[Config] - trace_length, ram_K, bytecode_K, log_k_chunk, lookups_ra_virtual_log_k_chunk
```

### Preprocessing Format
```
[DoryVerifierSetup]
  - delta_1l: Vec<GT>
  - delta_1r: Vec<GT>
  - delta_2l: Vec<GT>
  - delta_2r: Vec<GT>
  - chi: Vec<GT>
  - g1_0, g2_0, h1, h2, ht, max_log_n

[JoltSharedPreprocessing]
  - BytecodePreprocessing
    - code_size: usize
    - bytecode: Vec<Instruction> (JSON format, length-prefixed)
    - pc_map: BytecodePCMapper
  - RAMPreprocessing
    - min_bytecode_address: u64
    - bytecode_words: Vec<u64>
  - MemoryLayout (20 x u64 fields)
```
