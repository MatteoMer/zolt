# Zolt zkVM Implementation Plan

## Current Status (December 2024 - Iteration 8)

Major progress on R1CS-Spartan integration:
1. R1CS witness generation is COMPLETE with JoltR1CS type
2. Spartan sumcheck is COMPLETE with proper Az, Bz, Cz computation
3. Stage 1 prover now runs actual sumcheck (not placeholder)

The prover now properly runs sumcheck for Stage 1:
  sum_{x} eq(tau, x) * [Az(x) * Bz(x) - Cz(x)] = 0

## Phase 1: Lookup Arguments COMPLETED

### Step 1.1: Lookup Table Infrastructure
- [x] Created `src/zkvm/lookup_table/mod.zig`
- [x] Defined `LookupTable` generic interface
- [x] Implemented bit interleaving/uninterleaving (Jolt-compatible)
- [x] Added LookupBits utility in `src/utils/mod.zig`

### Step 1.2: Basic Tables (14 tables implemented)
All tables with materializeEntry() and evaluateMLE() implementations:
1. RangeCheckTable, AndTable, OrTable, XorTable
2. EqualTable, NotEqualTable
3. UnsignedLessThan, SignedLessThan
4. UnsignedGreaterThanEqual, UnsignedLessThanEqual
5. SignedGreaterThanEqual, Movsign, Sub, Andn

### Step 1.3: Lasso Prover/Verifier
- [x] ExpandingTable: Incrementally builds EQ polynomial evaluations
- [x] SplitEqPolynomial: Gruen's optimization for EQ evaluation
- [x] PrefixSuffixDecomposition: Table decomposition into prefix/suffix products
- [x] PrefixPolynomial, SuffixPolynomial: Component polynomials
- [x] PrefixRegistry: Caching for prefix polynomials
- [x] LassoParams, LassoProver, LassoProof: Prover infrastructure
- [x] LassoVerifier, verifyLassoProof: Verifier infrastructure

## Phase 2: Instruction Proving COMPLETED

### Step 2.1: Instruction Flags
- [x] CircuitFlags enum (13 flags for R1CS constraints)
- [x] InstructionFlags enum (7 flags for instruction metadata)
- [x] CircuitFlagSet and InstructionFlagSet types

### Step 2.2: Instruction Lookup Interfaces
- [x] LookupTables(XLEN) enum with materializeEntry()
- [x] InstructionLookup(XLEN) interface
- [x] Flags interface
- [x] LookupQuery(XLEN) interface

### Step 2.3: Lookup Trace Integration
- [x] LookupEntry: Stores lookup operation data
- [x] LookupTraceCollector: Records lookups during execution
- [x] Integration with Emulator.step()
- [x] Statistics collection for lookup analysis

## Phase 3: Memory Checking COMPLETED

### Step 3.1: RAF (Read-After-Final) Checking
- [x] RafEvaluationParams: Sumcheck parameters
- [x] RaPolynomial: ra(k) computation
- [x] UnmapPolynomial: Address remapping
- [x] RafEvaluationProver and RafEvaluationVerifier

### Step 3.2: Value Consistency
- [x] ValEvaluationParams: Parameters for value checking
- [x] IncPolynomial, WaPolynomial, LtPolynomial
- [x] ValEvaluationProver and ValEvaluationVerifier

## Phase 4: Multi-Stage Sumcheck COMPLETED

### Step 4.1: Prover Infrastructure
- [x] SumcheckInstance: Trait interface for batched proving
- [x] StageProof: Round polynomials and challenges
- [x] JoltStageProofs: All 6 stage proofs combined
- [x] OpeningAccumulator: Polynomial opening claims
- [x] MultiStageProver: 6-stage orchestration
- [x] BatchedSumcheckProver: Parallel instance execution

### Step 4.2: Stage Implementations (ALL COMPLETE)
- [x] **Stage 1: Outer Spartan** - R1CS instruction correctness (NOW WITH SUMCHECK)
- [x] **Stage 2: RAM RAF** - Memory read-after-final checking
- [x] **Stage 3: Lasso Lookup** - Instruction lookup reduction
- [x] **Stage 4: Value Evaluation** - Memory value consistency
- [x] **Stage 5: Register Evaluation** - Register value consistency
- [x] **Stage 6: Booleanity** - Flag constraint verification

### Step 4.3: Verifier Infrastructure
- [x] MultiStageVerifier: Verifies all 6 sumcheck stages
- [x] Stage-specific verification for degrees 2 and 3
- [x] Lagrange interpolation for polynomial evaluation
- [x] OpeningClaimAccumulator for batch verification
- [x] Integration with JoltVerifier

### Step 4.4: R1CS-Spartan Integration (NEW - Iteration 8)
- [x] JoltR1CS: Builds full R1CS from uniform constraints
- [x] JoltSpartanInterface: Provides sumcheck polynomial for Stage 1
- [x] Witness generation from execution trace
- [x] Az, Bz, Cz computation for Spartan
- [x] Evaluation claims for Az, Bz, Cz at final point

## Phase 5: Complete Commitment Schemes COMPLETED

### Step 5.1: BN254 Curve Constants
- [x] G1 generator: (1, 2) added to AffinePoint.generator()
- [x] G2 generator: Real Ethereum/EIP-197 coordinates

### Step 5.2: HyperKZG Improvements
- [x] Proper SRS generation with scalar multiplication
- [x] setupFromSRS() for importing trusted setup data
- [x] Documentation of security implications

### Step 5.3: Batch Verification
- [x] BatchOpeningAccumulator for collecting claims
- [x] OpeningClaim type for individual claims
- [x] Batched pairing check verification
- [x] OpeningClaimConverter for stage proof integration

## Phase 6: Integration COMPLETED

### Step 6.1: Implement execute()
- [x] Creates emulator with memory config
- [x] Loads program and sets entry point
- [x] Runs until completion or max cycles
- [x] Returns ExecutionTrace

### Step 6.2: Implement JoltProver.prove()
- [x] Initializes emulator and runs program
- [x] Generates execution and lookup traces
- [x] Runs multi-stage prover (all 6 stages)
- [x] Returns JoltProof with stage proofs

### Step 6.3: Implement JoltVerifier.verify()
- [x] Initializes Fiat-Shamir transcript
- [x] Verifies bytecode proof
- [x] Verifies memory proof
- [x] Verifies register proof
- [x] Verifies R1CS proof
- [x] Verifies multi-stage sumcheck proofs

## Key Architecture Decisions

### Zig-specific Patterns
- `comptime` generics instead of Rust traits
- `std.ArrayListUnmanaged(T)` for dynamic arrays
- `E!T` error unions
- Explicit allocators everywhere

### Following Existing Conventions
- Field elements use `BN254Scalar` type
- Tests in same file as implementation

### Multi-Stage Prover/Verifier Architecture
- 6 sequential sumcheck stages
- Each stage records round polynomials and challenges
- Opening accumulator collects claims for batch verification
- Fiat-Shamir transcript for challenge generation
- Lagrange interpolation for polynomial evaluation at arbitrary points

### R1CS-Spartan Integration
- JoltR1CS expands uniform constraints across cycles
- Witness layout: [1, cycle_0_inputs..., cycle_1_inputs..., ...]
- Equality-conditional form: condition * (left - right) = 0
- Az = condition evaluations, Bz = (left - right) evaluations, Cz = 0

### Lasso Protocol Architecture
- Two-phase sumcheck: address binding + cycle binding
- Prefix-suffix decomposition for efficiency
- Gruen split EQ optimization

## Files Modified (Iteration 8)

### New Files
- `src/zkvm/r1cs/jolt_r1cs.zig` - R1CS-Spartan integration
  - JoltR1CS: Full R1CS builder from uniform constraints
  - JoltSpartanInterface: Sumcheck polynomial provider
  - Witness generation from trace
  - Az, Bz, Cz computation

### Updated Files
- `src/zkvm/r1cs/mod.zig`:
  - Export JoltR1CS and JoltSpartanInterface
- `src/zkvm/prover.zig`:
  - Stage 1 now uses JoltSpartanInterface
  - Proper sumcheck with round polynomials
  - Evaluation claims for Az, Bz, Cz

## Next Steps for Future Iterations

1. **End-to-End Integration Test**: Test with actual RISC-V programs
2. **G2 Scalar Multiplication**: For proper [τ]_2 computation
3. **Production SRS**: Import from Ethereum ceremony
4. **Performance Optimization**: Parallelize sumcheck rounds

All tests pass.
