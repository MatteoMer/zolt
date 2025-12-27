# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 7)

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

## Completed (Previous Sessions - Iterations 1-6)

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

### Phase 5: Commitment Schemes
- [x] BN254 G1/G2 generators with real coordinates
- [x] HyperKZG SRS generation
- [x] setupFromSRS() for importing trusted setup

### Phase 6: Integration
- [x] host.execute() implementation
- [x] JoltProver.prove() implementation
- [x] JoltVerifier.verify() framework

## Summary

**Iteration 7 completed multi-stage verifier implementation:**

The verifier now properly verifies all 6 sumcheck stages:
1. Stage 1: Outer Spartan - R1CS verification
2. Stage 2: RAM RAF - Memory read-after-final checking
3. Stage 3: Lasso - Instruction lookup verification
4. Stage 4: Value Evaluation - Memory consistency
5. Stage 5: Register Evaluation - Register consistency
6. Stage 6: Booleanity - Flag constraint verification

Each stage:
- Verifies round polynomial sum checks (p(0) + p(1) = claim)
- Updates claims using Lagrange interpolation
- Accumulates opening claims for batch verification
- Uses Fiat-Shamir transcript for challenge derivation

All tests pass.

## Next Steps (Future Iterations)

### Wire R1CS to Spartan
- [ ] Connect R1CS witness to Spartan prover in Stage 1
- [ ] Compute Az, Bz, Cz witness polynomials
- [ ] Integrate with multi-stage prover

### Production Readiness
- [ ] G2 scalar multiplication for proper [τ]_2
- [ ] Full pairing verification with real trusted setup
- [ ] Import production SRS from Ethereum ceremony
- [ ] End-to-end integration tests with real RISC-V programs
