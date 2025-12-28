# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: JOLT COMPATIBILITY - Serialization Complete**

The following Jolt-compatibility components are now working:
- Blake2b transcript with identical Fiat-Shamir challenges
- Jolt-compatible proof types (JoltProof, SumcheckInstanceProof, etc.)
- Arkworks-compatible serialization with byte-perfect output

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

### 4. Prover Wiring (NEXT)

- [ ] **Connect JoltProof to Prover**
  - [ ] Modify prover to output JoltProof instead of current format
  - [ ] Populate all 7 stage proofs
  - [ ] Add UniSkipFirstRoundProof for stages 1-2
  - [ ] Populate opening_claims correctly

- [ ] **Commitment Scheme**
  - [ ] Wire up Dory commitments
  - [ ] Serialize GT elements in arkworks format

### 5. Integration Testing

- [ ] **Cross-Verification Tests**
  - [ ] Generate proof in Zolt for fibonacci.elf
  - [ ] Serialize in Jolt-compatible format
  - [ ] Load in Jolt and verify

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
| `src/zkvm/jolt_serialization.zig` | ✅ Done | Arkworks serialization |
| `src/zkvm/prover.zig` | Pending | Wire up Jolt format |

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
2. ⏳ Zolt can generate a proof in Jolt format
3. ⏳ The proof can be loaded and verified by Jolt
4. ⏳ No modifications needed on the Jolt side

## Priority Order

1. ✅ **Transcript** - Matching Fiat-Shamir complete
2. ✅ **Proof Types** - JoltProof structure defined
3. ✅ **Serialization** - Byte-perfect compatibility verified
4. ⏳ **Prover Wiring** - Connect types to prover
5. ⏳ **Integration** - End-to-end verification
