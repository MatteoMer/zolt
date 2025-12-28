# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: JOLT COMPATIBILITY - Core Implementation Complete**

The following Jolt-compatibility components are now working:
- Blake2b transcript with identical Fiat-Shamir challenges
- Jolt-compatible proof types (JoltProof, SumcheckInstanceProof, etc.)
- Arkworks-compatible serialization with byte-perfect output
- Proof converter (6-stage Zolt → 7-stage Jolt)
- JoltProver with `proveJoltCompatible()` method
- Jolt proof serialization with `serializeJoltProof()`
- End-to-end serialization tests verifying format compatibility

---

## Phase 2: Jolt Compatibility

### 1. Transcript Alignment ✅ COMPLETE

- [x] **Create Blake2bTranscript** (`src/transcripts/blake2b.zig`)
  - [x] Blake2b-256 using Zig stdlib
  - [x] 32-byte state with round counter
  - [x] Message padding to 32 bytes
  - [x] Scalar serialization (LE then reverse to BE)
  - [x] Vector append with begin/end markers
  - [x] 128-bit challenges

- [x] **Test Vector Validation** - 7 test vectors verified

### 2. Proof Types ✅ COMPLETE

- [x] **JoltProof Structure** (`src/zkvm/jolt_types.zig`)
  - [x] SumcheckId enum (22 variants matching Jolt)
  - [x] CommittedPolynomial and VirtualPolynomial enums
  - [x] OpeningId with compact encoding
  - [x] CompressedUniPoly for round polynomials
  - [x] SumcheckInstanceProof with compressed_polys
  - [x] UniSkipFirstRoundProof for stages 1-2
  - [x] OpeningClaims as sorted map
  - [x] JoltProof with 7 explicit stages

### 3. Serialization ✅ COMPLETE

- [x] **Arkworks-Compatible Format** (`src/zkvm/jolt_serialization.zig`)
  - [x] Field elements as 32 bytes LE (from Montgomery form)
  - [x] usize as u64 little-endian
  - [x] OpeningId compact encoding
  - [x] Roundtrip deserialization

- [x] **Test Vectors Verified**
  - [x] Fr(42) → `[2a, 00, ...]`
  - [x] Fr(0) → `[00, 00, ...]`
  - [x] Fr(1) → `[01, 00, ...]`
  - [x] Fr(0xDEADBEEF) → `[ef, be, ad, de, ...]`
  - [x] usize(1234567890) → `[d2, 02, 96, 49, ...]`

### 4. Prover Wiring ✅ COMPLETE

- [x] **Proof Converter** (`src/zkvm/proof_converter.zig`)
  - [x] Map Zolt 6-stage to Jolt 7-stage format
  - [x] Create UniSkipFirstRoundProof for stages 1-2
  - [x] Populate OpeningClaims with SumcheckId mappings
  - [x] Configuration for bytecode_K, log_k_chunk, etc.

- [x] **JoltProver Integration** (`src/zkvm/mod.zig`)
  - [x] `proveJoltCompatible()` - Generate Jolt-compatible proof
  - [x] `serializeJoltProof()` - Serialize to arkworks format
  - [x] Wire up proof converter with prover

### 5. Integration Testing ✅ COMPLETE

- [x] **End-to-End Serialization Tests** (`src/zkvm/jolt_serialization.zig`)
  - [x] Test JoltProof with all 7 stages
  - [x] Verify opening claims serialization
  - [x] Verify commitments serialization
  - [x] Test empty proof serialization
  - [x] Validate config parameters serialization

### 6. Remaining Work (Future Phase)

- [ ] **Commitment Scheme Alignment**
  - [ ] Wire up Dory commitments (Zolt uses HyperKZG)
  - [ ] Serialize GT elements in arkworks format
  - [ ] Match Jolt's SRS generation (SHA3-256 seed: `"Jolt Dory URS seed"`)

- [ ] **Cross-Verification Test** (Requires Jolt-side changes)
  - [ ] Create Rust test in Jolt that loads Zolt proof file
  - [ ] Generate proof in Zolt for fibonacci
  - [ ] Verify proof loads and parses correctly in Jolt
  - [ ] Attempt verification (may fail without Dory)

---

## Cross-Verification Notes

To complete cross-verification, the following steps are needed:

1. **In Zolt**: Generate a proof file in Jolt format
   ```zig
   var prover = JoltProver(F).init(allocator);
   var jolt_proof = try prover.proveJoltCompatible(bytecode, inputs);
   const bytes = try prover.serializeJoltProof(&jolt_proof);
   // Write bytes to "proof.jolt" file
   ```

2. **In Jolt** (requires adding test): Load and verify
   ```rust
   #[test]
   fn verify_zolt_proof() {
       let proof: JoltProof<_, Dory, _> =
           JoltProof::from_file("proof.jolt").unwrap();
       // Would need matching preprocessing, memory layout, etc.
   }
   ```

The challenge is that Jolt verification requires:
- Preprocessing data (bytecode, memory layout)
- Program IO (inputs, outputs)
- Matching commitment scheme (Dory vs HyperKZG)

Without Dory commitment alignment, the proof will fail at Stage 8 (batch opening).

---

## Phase 1 Complete: Core zkVM

### Verified Test Status

| Component | Status | Tests |
|-----------|--------|-------|
| Field Arithmetic | Working | Pass |
| Extension Fields | Working | Pass |
| Sumcheck Protocol | Working | Pass |
| RISC-V Emulator | Working | Pass |
| ELF Loader | Working | Pass |
| MSM | Working | Pass |
| HyperKZG | Working | Pass |
| Dory | Partial | Pass |
| Spartan | Working | Pass |
| Lasso | Working | Pass |
| Multi-stage Prover | Working | Pass |
| Multi-stage Verifier | Working | Pass |
| **Blake2b Transcript** | **Working** | **Pass** |
| **Jolt Types** | **Working** | **Pass** |
| **Jolt Serialization** | **Working** | **Pass** |
| **Proof Converter** | **Working** | **Pass** |

### All Tests Passing

```
zig build test --summary all
Build Summary: 5/5 steps succeeded; 578/578 tests passed
```

### Verified C Examples (All 9 Working)

| Program | Result | Cycles | Description |
|---------|--------|--------|-------------|
| fibonacci.elf | 55 | 52 | Fibonacci(10) |
| sum.elf | 5050 | 6 | Sum 1-100 |
| factorial.elf | 3628800 | 34 | 10! |
| gcd.elf | 63 | 50 | GCD via Euclidean |
| collatz.elf | 111 | 825 | Collatz n=27 |
| primes.elf | 25 | 8000+ | Primes < 100 |
| signed.elf | -39 | 5 | Signed arithmetic |
| bitwise.elf | 209 | 169 | AND/OR/XOR/shifts |
| array.elf | 1465 | - | Array load/store |

---

## Key Reference Files

### Zolt (Modified)
| File | Status | Purpose |
|------|--------|---------|
| `src/transcripts/blake2b.zig` | ✅ Done | Blake2bTranscript |
| `src/zkvm/jolt_types.zig` | ✅ Done | Jolt proof types |
| `src/zkvm/jolt_serialization.zig` | ✅ Done | Arkworks serialization + e2e tests |
| `src/zkvm/proof_converter.zig` | ✅ Done | 6→7 stage converter |
| `src/zkvm/mod.zig` | ✅ Done | JoltProver with Jolt export |

### Jolt (Reference Only)
| File | Purpose |
|------|---------|
| `jolt-core/src/transcripts/blake2b.rs` | ✅ Verified |
| `jolt-core/src/zkvm/proof_serialization.rs` | ✅ Analyzed |
| `jolt-core/src/subprotocols/sumcheck.rs` | Reference |
| `jolt-core/src/poly/opening_proof.rs` | Reference |

---

## Success Criteria

1. ✅ `zig build test` passes all 578 tests
2. ✅ Zolt can generate a proof in Jolt format (`proveJoltCompatible`)
3. ✅ Zolt can serialize proofs in arkworks format (`serializeJoltProof`)
4. ✅ E2E serialization tests verify format compatibility
5. ⏳ The proof can be loaded and verified by Jolt (requires Dory + Jolt test)
6. ⏳ No modifications needed on the Jolt side (requires Dory alignment)

## Priority Order

1. ✅ **Transcript** - Matching Fiat-Shamir complete
2. ✅ **Proof Types** - JoltProof structure defined
3. ✅ **Serialization** - Byte-perfect compatibility verified
4. ✅ **Prover Wiring** - Connect types to prover
5. ✅ **Integration Tests** - E2E serialization verified
6. ⏳ **Dory Commitment** - Required for full verification
7. ⏳ **Cross-Verification** - Jolt-side test needed
