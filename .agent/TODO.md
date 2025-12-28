# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: SERIALIZATION FULLY COMPATIBLE ✅**

### Latest Progress (2024-12-28, Agent Session 5)

**✅ VERIFIED: Complete Format Compatibility with Jolt**

The Zolt proof serialization is now fully compatible with Jolt's arkworks-based format. All deserialization tests pass in Jolt:

1. **Proof Deserialization**: `test_deserialize_zolt_proof` PASSES
   - 48 opening claims parsed correctly
   - 5 Dory commitments (GT elements) parsed correctly
   - All 7 stage proofs (UniSkip + Sumcheck) parsed correctly

2. **Field/GT Serialization**: `test_serialization_vectors` PASSES
   - BN254 field elements serialize identically
   - Fq12 (GT) elements serialize identically
   - All bytes match expected arkworks format

3. **Blake2b Transcript**: `test_zolt_compatibility_vectors` PASSES
   - Challenge generation matches Jolt exactly
   - State/round counter implementation correct

### Verification Status

**Full Verification: Requires Same-Program Execution**

The `test_verify_zolt_proof` test fails at Stage 1 univariate skip because:
- Jolt's preprocessing is for `fib(50)` using Jolt SDK guest program
- Zolt's proof is for a different program (bare-metal fibonacci.c)

**This is expected behavior** - the format is correct, but verification requires:
1. Same bytecode (ELF file)
2. Same execution trace
3. Same R1CS constraints

### Next Steps for Full Cross-Verification

**Option A: Make Zolt run Jolt guest programs**
- Implement Jolt's I/O memory layout in Zolt emulator
- Set up input at correct memory address (0x7fffa000 region)
- Parse and apply postcard-serialized inputs
- Run same execution as Jolt

**Option B: Create no-I/O test program**
- Write a Jolt guest that requires no input
- Compile and run in both systems
- Generate matching proofs

**Option C: Export preprocessing from Zolt**
- Implement Jolt's preprocessing format in Zolt
- Export BytecodePreprocessing, RAMPreprocessing, MemoryLayout
- Allow Jolt verifier to use Zolt-generated preprocessing

---

## Completed Items ✅

### Phase 1: Blake2b Transcript
1. ✅ Implement Blake2b-256 hash function
2. ✅ 32-byte state with round counter
3. ✅ EVM-compatible scalar serialization (LE serialize → reverse to BE)
4. ✅ Label padding (right-pad to 32 bytes)
5. ✅ Vector append format matching Jolt

### Phase 2: Proof Structure
6. ✅ Restructure JoltProof to 7-stage layout
7. ✅ Add UniSkipFirstRoundProof for stages 1-2
8. ✅ Implement opening_claims BTreeMap-like structure
9. ✅ Add configuration parameters (trace_length, ram_K, etc.)

### Phase 3: Serialization
10. ✅ Arkworks-compatible field element serialization
11. ✅ Arkworks-compatible GT (Fq12) element serialization
12. ✅ OpeningId encoding (Virtual/Committed/Advice)
13. ✅ VirtualPoly type encoding
14. ✅ Compressed polynomial format for sumcheck rounds

### Phase 4: Dory Commitment Scheme
15. ✅ BN254 elliptic curve implementation
16. ✅ G1/G2 point serialization matching arkworks
17. ✅ GT (Fq12) element serialization
18. ✅ MSM (multi-scalar multiplication)
19. ✅ Pairing operation (miller loop + final exp)
20. ✅ SRS generation from "Jolt Dory URS seed"
21. ✅ Row commitments via MSM
22. ✅ Final commitment via multi-pairing

### Phase 5: R1CS/Spartan
23. ✅ COEFFS_PER_J precomputed Lagrange weights
24. ✅ LagrangeHelper with shift_coeffs_i32
25. ✅ Cross-product algorithm for UniSkip extended evaluation
26. ✅ Stage proofs matching Jolt structure

---

## Test Results Summary

### Zolt: All tests passing ✅ (618/618)
```bash
zig build test --summary all
# 618/618 tests passed
```

### Jolt Cross-Verification Tests

| Test | Status | Details |
|------|--------|---------|
| `test_serialization_vectors` | ✅ PASS | Field/GT serialization matches |
| `test_zolt_compatibility_vectors` | ✅ PASS | Blake2b transcript compatible |
| `test_debug_zolt_format` | ✅ PASS | Proof structure parseable |
| `test_deserialize_zolt_proof` | ✅ PASS | Full proof deserializes correctly |
| `test_gt_serialization_size` | ✅ PASS | GT element size (384 bytes) correct |
| `test_jolt_proof_roundtrip` | ✅ PASS | Field element roundtrip works |
| `test_verify_zolt_proof` | ⚠️ BLOCKED | Different programs (preprocessing mismatch) |

---

## Key Technical Achievements

### 1. Blake2b Transcript Match
- Implemented `Blake2bTranscript` in `src/transcripts/blake2b.zig`
- Matches Jolt's `jolt-core/src/transcripts/blake2b.rs` exactly
- Same challenge derivation for same inputs

### 2. UniSkip Cross-Product Algorithm
Fixed the univariate skip polynomial computation to match Jolt:
```
az_eval = Σ_i (where Az[i] is TRUE): coeffs[j][i]
bz_eval = Σ_i (where Az[i] is FALSE): coeffs[j][i] * Bz[i]
Product = az_eval * bz_eval
```

### 3. Dory Commitment Implementation
- Full BN254 pairing implementation
- MSM for row commitments
- Multi-pairing for final commitment
- GT element matches Jolt's arkworks serialization

### 4. Proof Serialization
- Exact byte-level compatibility with arkworks CanonicalSerialize
- OpeningId encoding matches Jolt's scheme
- VirtualPoly enum encoding correct

---

## What "Format Compatible" Means

When we say format compatible, this means:
1. **Byte-level**: Zolt proofs deserialize correctly in Jolt
2. **Structure**: All fields (claims, commitments, stages) parse correctly
3. **Cryptographic**: Field elements and GT elements are valid
4. **Protocol**: Transcript, sumcheck, and opening structures match

The only missing piece for full verification is **same-program execution**, which requires either:
- Running Jolt SDK guest programs in Zolt, or
- Creating a minimal test program that works in both

---

## Commands Reference

```bash
# Build and test Zolt
cd /Users/matteo/projects/zolt
zig build test --summary all

# Generate Jolt-format proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Generate Jolt preprocessing
cd /Users/matteo/projects/jolt/examples/fibonacci
cargo run --release -- --save
# Creates: /tmp/jolt_verifier_preprocessing.dat, /tmp/fib_proof.bin, /tmp/fib_io_device.bin

# Run Jolt cross-verification tests
cd /Users/matteo/projects/jolt
cargo test -p jolt-core test_deserialize_zolt_proof -- --ignored --nocapture
cargo test -p jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
