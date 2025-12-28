# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: JOLT COMPATIBILITY - Transcript Complete**

The Blake2b transcript is now byte-compatible with Jolt. Next: Proof structure alignment.

---

## Phase 2: Jolt Compatibility

### 1. Transcript Alignment ✅ COMPLETE

- [x] **Create Blake2bTranscript** (`src/transcripts/blake2b.zig`)
  - [x] Port Blake2b-256 hash function (use std.crypto.hash.blake2.Blake2b256)
  - [x] Implement 32-byte state with round counter
  - [x] Match Jolt's hasher(): state || [0u8; 28] || round.to_be_bytes()
  - [x] Implement `append_message()` with 32-byte right-padding
  - [x] Implement `append_scalar()` with LE serialize, then reverse to BE
  - [x] Implement `append_scalars()` with begin/end vector markers
  - [x] Implement `challenge_scalar()` returning 128-bit challenges
  - [x] Add round counter increment on each state update

- [x] **Test Vector Validation**
  - [x] Create test in Jolt that outputs challenge for known inputs
  - [x] Verify Zolt produces identical challenges (7 test vectors)
  - [x] Test: same transcript state after identical operations

**Reference**: `jolt-core/src/transcripts/blake2b.rs`

### 2. Proof Structure Refactoring (HIGH - NEXT)

- [ ] **Restructure JoltProof** (`src/zkvm/mod.zig`)
  - [ ] Add `opening_claims: OpeningClaims(F)` (BTreeMap-like)
  - [ ] Add `commitments: []PolyCommitment`
  - [ ] Add `stage1_uni_skip_first_round_proof: UniSkipFirstRoundProof(F)`
  - [ ] Add `stage1_sumcheck_proof: SumcheckInstanceProof(F)`
  - [ ] Add `stage2_uni_skip_first_round_proof: UniSkipFirstRoundProof(F)`
  - [ ] Add `stage2_sumcheck_proof` through `stage7_sumcheck_proof`
  - [ ] Add `joint_opening_proof: DoryProof`
  - [ ] Add advice proof fields (trusted/untrusted)
  - [ ] Add config: `trace_length`, `ram_K`, `bytecode_K`, `log_k_chunk`

- [ ] **Add UniSkipFirstRoundProof** (`src/subprotocols/mod.zig`)
  - [ ] Create struct with `uni_poly: UniPoly(F)` field
  - [ ] Implement for stages 1 and 2

- [ ] **Add SumcheckInstanceProof** structure
  - [ ] Match Jolt's `compressed_polys: Vec<CompressedUniPoly<F>>`

- [ ] **Refactor Prover** (`src/zkvm/prover.zig`)
  - [ ] Generate 7 explicit stage proofs instead of 6 grouped
  - [ ] Generate UniSkipFirstRoundProof for stages 1-2
  - [ ] Populate opening_claims correctly
  - [ ] Match Jolt's stage ordering

**Reference**: `jolt-core/src/zkvm/proof_serialization.rs`

### 3. Serialization Alignment (HIGH)

- [ ] **Implement Arkworks Format** (`src/zkvm/serialization.zig`)
  - [ ] Remove "ZOLT" magic header
  - [ ] Implement arkworks field element format (32 bytes LE)
  - [ ] Implement arkworks G1Affine point format
  - [ ] Implement usize as u64 little-endian
  - [ ] Implement BTreeMap serialization (length + sorted pairs)

- [ ] **Implement OpeningId Encoding**
  - [ ] `NUM_SUMCHECKS = 11`
  - [ ] `UntrustedAdvice(id)` -> `id`
  - [ ] `TrustedAdvice(id)` -> `NUM_SUMCHECKS + id`
  - [ ] `Committed(poly, id)` -> `BASE + id + poly_index * NUM_SUMCHECKS`
  - [ ] `Virtual(poly, id)` -> compact encoding

- [ ] **Implement SumcheckInstanceProof Serialization**
  - [ ] `compressed_polys: Vec<CompressedUniPoly>` format

**Reference**: `jolt-core/src/zkvm/proof_serialization.rs`

### 4. Dory Commitment Completion (HIGH)

- [ ] **Complete Dory Implementation** (`src/poly/commitment/dory.zig`)
  - [ ] Implement streaming commitment from Jolt
  - [ ] Use SRS seed: `"Jolt Dory URS seed"` (SHA3-256)
  - [ ] Match GT element serialization (Fp12 arkworks format)
  - [ ] Implement `DoryProof` structure
  - [ ] Implement batch opening

- [ ] **SRS Generation**
  - [ ] Match Jolt's ChaCha20 deterministic randomness
  - [ ] Two-tier structure (row/column commitments)

**Reference**: `jolt-core/src/poly/commitment/dory/`

### 5. Integration Testing

- [ ] **Cross-Verification Tests**
  - [ ] Generate proof in Zolt for `fibonacci.elf`
  - [ ] Save in Jolt-compatible format
  - [ ] Load and verify in Jolt (add test in Jolt repo)
  - [ ] Test with all 9 example programs

- [ ] **Round-Trip Tests**
  - [ ] Serialize proof in Zolt
  - [ ] Deserialize in Jolt
  - [ ] Verify structure matches

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
| Serialization | Working | Pass |
| **Blake2b Transcript** | **Working** | **Pass** |

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

### CLI Commands (All Working)

```
zolt help              # Show help message
zolt version           # Show version
zolt info              # Show zkVM capabilities
zolt run [opts] <elf>  # Run RISC-V ELF binary
zolt trace <elf>       # Show execution trace
zolt prove [opts] <elf> # Generate ZK proof
zolt verify <proof>    # Verify a saved proof
zolt stats <proof>     # Show proof statistics
zolt decode <hex>      # Decode instruction
zolt srs <ptau>        # Inspect PTAU file
zolt bench             # Run benchmarks
```

---

## Key Reference Files

### Zolt (Modify)
| File | Purpose |
|------|---------|
| `src/transcripts/blake2b.zig` | ✅ Blake2bTranscript (complete) |
| `src/zkvm/mod.zig` | Restructure JoltProof |
| `src/zkvm/prover.zig` | 7-stage prover |
| `src/zkvm/serialization.zig` | Arkworks format |
| `src/poly/commitment/mod.zig` | Complete Dory |
| `src/subprotocols/mod.zig` | Add UniSkipFirstRoundProof |

### Jolt (Reference Only)
| File | Purpose |
|------|---------|
| `jolt-core/src/transcripts/blake2b.rs` | ✅ Transcript impl (verified) |
| `jolt-core/src/zkvm/proof_serialization.rs` | Proof structure |
| `jolt-core/src/zkvm/verifier.rs` | 8-stage verification |
| `jolt-core/src/subprotocols/sumcheck.rs` | Sumcheck format |
| `jolt-core/src/poly/commitment/dory/` | Dory commitment |

---

## Success Criteria

The implementation is complete when:
1. `zig build test` passes all tests ✅ (578/578 pass)
2. Zolt can generate a proof for any example program
3. The proof can be loaded and verified by Jolt's verifier
4. No modifications needed on the Jolt side

## Priority Order

1. **Transcript** ✅ - Matching Fiat-Shamir complete
2. **Proof Structure** - Must match Jolt's 7-stage expectations (NEXT)
3. **Serialization** - Byte-perfect compatibility required
4. **Dory Commitment** - Complete with same SRS seed
5. **Integration Tests** - Validate end-to-end flow
