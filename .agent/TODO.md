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

## Phase 6: Integration Testing ✅ COMPLETE

### Proof Deserialization ✅
- [x] Jolt can deserialize Zolt proof in --jolt-format
- [x] Opening claims: 48 entries, all valid
- [x] Commitments: 5 GT elements, all valid
- [x] Sumcheck proofs: structure matches

### Preprocessing Deserialization ✅
- [x] DoryVerifierSetup parses correctly
- [x] BytecodePreprocessing parses correctly
- [x] RAMPreprocessing parses correctly
- [x] MemoryLayout parses correctly
- [x] Full JoltVerifierPreprocessing::deserialize_uncompressed works!

### End-to-End Verification 🚧 PARTIAL
- [x] Proof and preprocessing load correctly
- [ ] Verification fails at "Stage 1 univariate skip first round"
- [ ] Need to debug Fiat-Shamir transcript alignment
- [ ] Need to debug sumcheck round polynomial format

---

## Current Status: VERIFICATION FAILS

The end-to-end test runs successfully but verification fails:
```
Loaded all files:
  Preprocessing: 62223 bytes
  Proof: 30926 bytes
  Trace length: 1024
  Commitments: 5
Verifier created, running verification...
Verification failed: Stage 1 univariate skip first round
```

### Root Causes to Investigate
1. **Fiat-Shamir transcript** - Challenges may differ between Zolt and Jolt
2. **Sumcheck round polynomials** - Format or values may not match
3. **UniSkipFirstRoundProof** - Missing or incorrect implementation in Zolt

---

## Session 8 Summary

### Completed ✅
1. **Proof serialization working**
   - 30.9 KB proof with --jolt-format
   - 48 opening claims all valid Fr
   - 5 Dory commitments (GT elements)

2. **Preprocessing fully working**
   - DoryVerifierSetup parses correctly
   - BytecodePreprocessing parses correctly
   - RAMPreprocessing parses correctly
   - MemoryLayout parses correctly

3. **End-to-end test runs**
   - Both proof and preprocessing deserialize successfully
   - Verification runs but fails at Stage 1

### Key Fixes Applied
1. **NoOp/UNIMPL serialization**: Changed from `{"NoOp":{...}}` to `"NoOp"`
2. **Immediate types**: Fixed u64/i64/i128 types for different formats
3. **FENCE/ECALL**: Use FormatI operands

### Test Results
- Zolt tests: 632/632 PASS ✅
- Proof generation: 30.9 KB in Jolt format ✅
- Proof deserialization: PASS ✅
- Preprocessing deserialization: PASS ✅
- End-to-end verification: RUNS (but fails at Stage 1)

---

## Next Steps

1. Debug the Fiat-Shamir transcript to ensure Zolt produces same challenges as Jolt
2. Verify sumcheck round polynomial format matches Jolt
3. Check UniSkipFirstRoundProof implementation
4. Add more debug output to identify exact mismatch

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

# Run Jolt e2e test
cd /path/to/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## File Sizes
- Proof (Jolt format): 30.9 KB (30,926 bytes)
- Preprocessing: 62.2 KB (62,223 bytes)
