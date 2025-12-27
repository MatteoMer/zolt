# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 8)

### R1CS-Spartan Integration
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

## Completed (Previous Sessions - Iterations 1-7)

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

### Proof Structure Updates
- [x] Added `stage_proofs: ?JoltStageProofs(F)` to JoltProof
- [x] Fixed ownership transfer of stage proofs from prover
- [x] Added `proofs_transferred` flag to MultiStageProver

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

**Iteration 8 completed R1CS-Spartan integration:**

Stage 1 (Outer Spartan) now properly runs sumcheck to prove:
  sum_{x} eq(tau, x) * [Az(x) * Bz(x) - Cz(x)] = 0

The implementation:
1. Builds JoltR1CS from execution trace (uniform constraints x cycles)
2. Generates witness vector from trace steps
3. Computes Az, Bz, Cz evaluations
4. Runs sumcheck with proper round polynomials
5. Records evaluation claims for verification

Also fixed:
- Constraint generation to use proper TraceStep type
- Immediate value derivation from instruction encoding
- EqPolynomial shift operation for Zig 0.15

All 312 tests pass.

## Next Steps (Future Iterations)

### Production Readiness
- [ ] G2 scalar multiplication for proper [τ]_2
- [ ] Full pairing verification with real trusted setup
- [ ] Import production SRS from Ethereum ceremony
- [ ] End-to-end tests with full emulator execution
- [ ] Fix bytecode module ArrayList API for Zig 0.15
