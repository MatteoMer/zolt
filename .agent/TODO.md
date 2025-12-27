# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 9)

### Preprocessing Implementation
- [x] Implemented `SharedPreprocessing(F)` with bytecode size, padding, memory layout
- [x] Computed initial memory hash for public verification
- [x] Implemented `Preprocessing.preprocess()` to generate ProvingKey and VerifyingKey
- [x] ProvingKey contains full SRS for polynomial commitment
- [x] VerifyingKey contains G1, G2, tau_G2 for pairing checks
- [x] Fixed SharedPreprocessing.deinit() const pointer issue

### Zig 0.15 API Compatibility Fixes
- [x] Fixed bytecode BytecodeTable to use ArrayListUnmanaged
- [x] Fixed tracer ExecutionTrace to use ArrayListUnmanaged
- [x] Fixed tracer setInputs to pass allocator to appendSlice
- [x] Fixed SplitEqPolynomial WPair type to be a named struct
- [x] Fixed all `transcript.challengeScalar()` calls to use `try`
- [x] Fixed all `Transcript` type parameters from `*Transcript` to `*Transcript(F)`
- [x] Fixed all `.inverse()` calls to unwrap optional with `.?`
- [x] Fixed evaluatePolynomialAtChallenge calls to pass F type
- [x] Fixed prover to extract rd/rs1/rs2 from instruction encoding

### End-to-End Testing
- [x] Added `e2e: simple addi program execution trace` test
- [x] Added `e2e: preprocessing generates usable keys` test
- [x] Added `e2e: multi-instruction program emulation` test
- [x] Added `e2e: execute and trace multiple instructions` test
- [x] All 324 tests pass

## Completed (Previous Sessions - Iterations 1-8)

### R1CS-Spartan Integration (Iteration 8)
- [x] Created `src/zkvm/r1cs/jolt_r1cs.zig` with JoltR1CS type
- [x] Implemented witness generation from execution trace
- [x] Implemented Az, Bz, Cz computation for Spartan
- [x] Created JoltSpartanInterface for sumcheck integration
- [x] Updated Stage 1 prover to use actual sumcheck
- [x] Proper round polynomial computation and challenge binding
- [x] Evaluation claims for Az, Bz, Cz at final point
- [x] Fixed constraints.zig to use TraceStep (not ExecutionStep)
- [x] Derive immediate values from instruction encoding
- [x] Set circuit flags from instruction opcode
- [x] Fixed EqPolynomial.evals shift for Zig 0.15 compatibility
- [x] Added R1CS-Spartan integration test

### Batch Polynomial Commitment Verification
- [x] Created `src/poly/commitment/batch.zig` with batch verification
- [x] Implemented `BatchOpeningAccumulator` for collecting opening claims
- [x] Added `OpeningClaim` type for individual claims
- [x] Implemented batched pairing check verification
- [x] Added `OpeningClaimConverter` for stage proof integration

### R1CS Constraint Generation
- [x] Created `src/zkvm/r1cs/constraints.zig` with uniform constraints
- [x] Defined 36 witness input variables per execution cycle
- [x] Implemented 19 uniform R1CS constraints (equality-conditional form)
- [x] Created `R1CSCycleInputs` for per-cycle witness extraction
- [x] Created `R1CSWitnessGenerator` for full trace witness generation
- [x] Added constraint satisfaction verification

### Multi-Stage Verifier Implementation
- [x] Created `src/zkvm/verifier.zig` with full sumcheck verification
- [x] Implemented `MultiStageVerifier` for all 6 stages
- [x] Added Lagrange interpolation for polynomial evaluation at challenge points
- [x] Implemented `OpeningClaimAccumulator` for batch verification
- [x] Connected stage proofs to JoltVerifier
- [x] Updated JoltProof to include stage proofs

### Phase 1: Lookup Arguments
- [x] Lookup table infrastructure (14 tables)
- [x] LookupBits utility for bit manipulation
- [x] MLE evaluation with MSB-first ordering

### Lasso Infrastructure
- [x] ExpandingTable for EQ polynomial accumulation
- [x] SplitEqPolynomial (Gruen's optimization)
- [x] PrefixSuffixDecomposition
- [x] LassoProver and LassoVerifier

### Phase 2: Instruction Proving
- [x] CircuitFlags enum (13 flags)
- [x] InstructionFlags enum (7 flags)
- [x] LookupTables(XLEN) enum
- [x] Instruction lookups: Add, Sub, And, Or, Xor, Slt, Sltu, Beq, Bne

### Phase 3: Memory Checking
- [x] RAF checking infrastructure
- [x] Val Evaluation sumcheck
- [x] Lookup trace integration with Emulator

### Phase 4: Multi-Stage Prover
- [x] 6-stage sumcheck orchestration
- [x] StageProof and OpeningAccumulator
- [x] BatchedSumcheckProver interface
- [x] Full implementations for all 6 stages
- [x] Stage 1 now runs actual Spartan sumcheck

### Phase 5: Commitment Schemes
- [x] BN254 G1/G2 generators with real coordinates
- [x] HyperKZG SRS generation
- [x] setupFromSRS() for importing trusted setup

### Phase 6: Integration
- [x] host.execute() implementation
- [x] JoltProver.prove() implementation
- [x] JoltVerifier.verify() framework

## Summary

**Iteration 9 completed:**

1. **Preprocessing Implementation**: Full `Preprocessing.preprocess()` that generates
   ProvingKey (with SRS) and VerifyingKey (with pairing parameters)

2. **Zig 0.15 Compatibility**: Fixed numerous API issues including ArrayList patterns,
   transcript type parameters, optional unwrapping, and register field extraction

3. **End-to-End Tests**: Added 4 new e2e tests that verify the emulator runs programs
   correctly and generates execution traces

All 324 tests pass.

## Next Steps (Future Iterations)

### Lasso Integration Bug
- [ ] Fix SplitEqPolynomial assertion failure when called from LassoProver
  - The split_eq.init() receives mismatched w.len vs num_vars
  - Need to trace through LassoProver.init() parameter passing

### Production Readiness
- [ ] G2 scalar multiplication for proper [τ]_2
- [ ] Full pairing verification with real trusted setup
- [ ] Import production SRS from Ethereum ceremony
- [ ] Performance optimization with SIMD

### Full Prove/Verify Integration
- [ ] Fix Lasso prover parameter mismatch
- [ ] Add full prove/verify e2e test
- [ ] Benchmark against reference Rust implementation
