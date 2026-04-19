# Zolt Native Verifier — Implementation Progress

## Cross-Verification Baseline ✅ (Mar 7 2026)
All 8 test programs produce valid proofs verified by upstream a16z/jolt Rust verifier:
fibonacci, factorial, bitwise, collatz, primes, sum, gcd, signed

---

## Phase 1: Foundation

### Formally Verified Field Arithmetic (fiat-crypto)
- [ ] Generate fiat-crypto C code for BN254 Fr (scalar field)
- [ ] Generate fiat-crypto C code for BN254 Fp (base field)
- [ ] Integrate generated C via `@cImport` into Zig build
- [ ] Validate against existing differential test fixtures (`testdata/zolt-arith-diff/`)
- [ ] Add algebraic property tests (commutativity, associativity, distributivity, inverse)

### Extension Fields on FV Base
- [ ] Implement Fp2 = Fp[u]/(u²+1) using simple algebraic formulas (no optimizations)
- [ ] Implement Fp6 = Fp2[v]/(v³−ξ) via Karatsuba
- [ ] Implement Fp12 = Fp6[w]/(w²−v) via Karatsuba
- [ ] Validate all against existing diff fixtures (`fp2_ops.txt`, `fp6_ops.txt`, `fp12_ops.txt`)

### Pairing on FV Fields
- [ ] Implement standard optimal ate pairing (no shortcuts/optimizations)
- [ ] Implement inversion via Fermat's little theorem (a^(p-2)) on fiat-crypto mul/square
- [ ] Validate against existing pairing test vectors

---

## Phase 2: Core Verification Logic

### Sumcheck Verifier
- [x] Generic round verification (compressed polynomial recovery + claim check)
- [ ] Enforce degree bounds per stage/instance (hardcoded degree table)
- [ ] Batched sumcheck support (multiple instances with staggered rounds)
- [x] Unit tests with known-good fixtures

### Per-Stage Verification (Stages 1–7)
- [ ] Stage 1: Outer Spartan + UniSkip
- [ ] Stage 2: Batched (5 instances — ProductVirtualRemainder, RamRafEvaluation, RamReadWriteChecking, OutputSumcheck, InstructionLookupsClaimReduction)
- [ ] Stage 3: Batched (3 instances — ShiftSumcheck, InstructionInputSumcheck, RegistersClaimReduction)
- [ ] Stage 4: Registers RW + RAM val evaluation + RAM val final
- [ ] Stage 5: Registers val eval + RAM RA claim reduction + lookups read RAF
- [ ] Stage 6: Batched (6 instances — BytecodeRAF, Hamming, Booleanity, RA virtual, Inc claim reduction)
- [ ] Stage 7: Hamming weight claim reduction
- [ ] Transcript checkpoint assertions at each stage boundary

### Dory Verifier
- [ ] VMV message verification
- [ ] Reduce-and-fold rounds
- [ ] Final scalar product check
- [ ] Pairing equation verification

### Opening Claims
- [ ] Opening claim accumulation and checking
- [ ] Eliminate silent zero-fallback (`orelse F.zero()` → explicit errors)

---

## Phase 3: Integration & Testing

### Top-Level Verify
- [x] Wire all stages into `verify()` entry point
- [x] Proof deserialization (reuse existing `jolt_serialization.zig`)
- [x] Preprocessing deserialization (lightweight `VerifierPreprocessingData`)
- [ ] Config consistency checks (proof config vs preprocessing-derived config)

### Deserialization Hardening
- [ ] Bounded allocation (MAX_COMMITMENTS, MAX_SUMCHECK_ROUNDS, MAX_OPENING_CLAIMS)
- [ ] Field element range validation (scalars in [0, r))
- [ ] GT subgroup checks on Dory commitments

### Transcript Comparison Infrastructure
- [ ] `--dump-transcript` mode for Zig verifier (stage|operation|label|state_hex)
- [ ] Instrumentation patch for Rust verifier
- [ ] Comparison tool (diff snapshots line-by-line, report first divergence)
- [ ] Per-stage transcript checkpoints (golden state after preamble, commitments, tau, each stage)

### E2E Test Suite
- [ ] All 8+ guest programs: prove → verify (Rust) → verify (Zig) → transcript match
- [ ] Negative tests: bit-flip mutations on commitments, sumcheck polys, opening claims, Dory proof, config, preprocessing, I/O
- [ ] Canary proofs (15 targeted mutations, one per verification step — see RESEARCH.md §8.2)
- [ ] Cross-platform validation (x86-64 + ARM64)

### Known Answer Tests (KAT)
- [ ] Field arithmetic KAT at verifier init
- [ ] Transcript KAT at verifier init
- [ ] Pairing KAT at verifier init (optional)

---

## Phase 4: Audit Preparation

- [ ] Write mathematical specification document (transcript protocol, sumcheck, stages 1–7, Dory)
- [ ] Run mutation testing, target >95% kill rate
- [ ] CI pipeline (unit, diff, property, negative, E2E smoke, nightly full suite)
- [ ] Prepare audit checklist and TCB inventory (~3,300 lines: 24% FV, 48% needs audit, 28% well-tested)
- [ ] Fuzz testing: deserialization, sumcheck round, Dory verifier, field arithmetic

---

## Phase 5: Hardening (Ongoing)

- [ ] Adversarial input programs (noop, max_memory, branch_heavy, all_instructions, etc.)
- [ ] Third-party cross-check (SageMath reference verifier)
- [ ] Proof forgery attempts (null proof, replayed stage, commitment-claim mismatch, transcript grinding)
- [ ] Verifier state machine invariant checking (monotonic transcript rounds, no duplicate claims, etc.)
- [ ] Study cross-audit intelligence from Lasso/Jolt, Plonky2, Halo2, SP1, Risc0 audit reports

---

## Housekeeping
- [ ] Remove the jolt/ fork directory (replace fully with jolt-verifier/)
- [x] Create verifier module at `src/zkvm/verifier/` (avoids circular dep; separate package deferred)
- [x] Add `--native` CLI flag for native Zig verification
