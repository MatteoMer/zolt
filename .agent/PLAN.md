# Zolt zkVM Implementation Plan

## Current Status (December 2024)
Completed Phase 1-3 (lookup arguments, instruction proving, memory checking),
and Phase 4 multi-stage prover skeleton. Ready for integration work.

## Phase 1: Lookup Arguments ✅ COMPLETED

### Step 1.1: Lookup Table Infrastructure ✅
- [x] Created `src/zkvm/lookup_table/mod.zig`
- [x] Defined `LookupTable` generic interface
- [x] Implemented bit interleaving/uninterleaving (Jolt-compatible)
- [x] Added LookupBits utility in `src/utils/mod.zig`

### Step 1.2: Basic Tables ✅ (14 tables implemented)
All tables with materializeEntry() and evaluateMLE() implementations:
1. RangeCheckTable, AndTable, OrTable, XorTable
2. EqualTable, NotEqualTable
3. UnsignedLessThan, SignedLessThan
4. UnsignedGreaterThanEqual, UnsignedLessThanEqual
5. SignedGreaterThanEqual, Movsign, Sub, Andn

### Step 1.3: Lasso Prover/Verifier ✅
- [x] ExpandingTable: Incrementally builds EQ polynomial evaluations
- [x] SplitEqPolynomial: Gruen's optimization for EQ evaluation
- [x] PrefixSuffixDecomposition: Table decomposition into prefix/suffix products
- [x] PrefixPolynomial, SuffixPolynomial: Component polynomials
- [x] PrefixRegistry: Caching for prefix polynomials
- [x] LassoParams, LassoProver, LassoProof: Prover infrastructure
- [x] LassoVerifier, verifyLassoProof: Verifier infrastructure

## Phase 2: Instruction Proving ✅ COMPLETED

### Step 2.1: Instruction Flags ✅
- [x] CircuitFlags enum (13 flags for R1CS constraints)
- [x] InstructionFlags enum (7 flags for instruction metadata)
- [x] CircuitFlagSet and InstructionFlagSet types

### Step 2.2: Instruction Lookup Interfaces ✅
- [x] LookupTables(XLEN) enum with materializeEntry()
- [x] InstructionLookup(XLEN) interface
- [x] Flags interface
- [x] LookupQuery(XLEN) interface

### Step 2.3: Lookup Trace Integration ✅
- [x] LookupEntry: Stores lookup operation data (cycle, pc, table, index, result)
- [x] LookupTraceCollector: Records lookups during execution
- [x] Integration with Emulator.step()
- [x] Statistics collection for lookup analysis

## Phase 3: Memory Checking ✅ COMPLETED

### Step 3.1: RAF (Read-After-Final) Checking ✅
- [x] RafEvaluationParams: Sumcheck parameters (log_k, start_address, r_cycle)
- [x] RaPolynomial: Computes ra(k) = Σ_j eq(r_cycle, j) · 1[address(j) = k]
- [x] UnmapPolynomial: Converts remapped address k to original address
- [x] RafEvaluationProver: Round polynomial computation and binding
- [x] RafEvaluationVerifier: Verification with challenge generation
- [x] Helper functions: computeEqEvals, computeEqAtPoint

### Step 3.2: Value Consistency ✅
- [x] ValEvaluationParams: Parameters for value consistency checking
- [x] IncPolynomial: Value increments at writes (val_new - val_old)
- [x] WaPolynomial: Write-address indicator
- [x] LtPolynomial: Less-than MLE for timestamp ordering
- [x] ValEvaluationProver: Computes initial claim and round polynomials
- [x] ValEvaluationVerifier: Round verification

## Phase 4: Multi-Stage Sumcheck ✅ SKELETON COMPLETED

### Step 4.1: Prover Infrastructure ✅
- [x] SumcheckInstance: Trait interface for batched proving
- [x] StageProof: Round polynomials and challenges per stage
- [x] JoltStageProofs: All 6 stage proofs combined
- [x] OpeningAccumulator: Polynomial opening claims
- [x] MultiStageProver: 6-stage orchestration skeleton
- [x] BatchedSumcheckProver: Parallel instance execution

### Step 4.2: Stage Implementation (Future Work)
- [ ] Stage 1: Outer Spartan (R1CS instruction correctness)
- [ ] Stage 2: RAM RAF & read-write checking
- [ ] Stage 3: Instruction lookup reduction (Lasso)
- [ ] Stage 4: Memory value evaluation
- [ ] Stage 5: Register evaluation & RA reduction
- [ ] Stage 6: Booleanity and Hamming weight checks

## Phase 5: Complete Commitment Schemes (Pending)
- [ ] Fix HyperKZG verification (current stub returns true)
- [ ] Add real BN254 curve constants (replace placeholder G2 generator)

## Phase 6: Integration (Pending)
- [ ] Implement host.execute() (connects to tracer)
- [ ] Implement JoltProver.prove() (wire up MultiStageProver)
- [ ] Implement JoltVerifier.verify() (implement verification logic)

## Key Design Decisions

### Zig-specific Patterns
- Use `comptime` generics instead of Rust traits
- Use `std.ArrayListUnmanaged(T)` for dynamic arrays
- Use `E!T` error unions
- Explicit allocators everywhere

### Following Existing Conventions
- Field elements use `BN254Scalar` type
- Tests in same file as implementation

### Bit Interleaving Convention (Matches Jolt)
- interleaveBits(x, y) puts y bits at even positions, x bits at odd positions
- Result = (spread(x) << 1) | spread(y)
- MLE evaluation uses MSB-first ordering in r array

### Lasso Protocol Architecture
- Two-phase sumcheck:
  1. Address binding (LOG_K rounds): Uses prefix-suffix decomposition
  2. Cycle binding (log_T rounds): Uses Gruen split EQ
- Proves: rv(r) + γ·left_op(r) + γ²·right_op(r) = Σ eq(j;r) · ra(k,j) · (Val(k) + γ·RafVal(k))

### RAF Protocol Architecture
- Proves: Σ_{k=0}^{K-1} ra(k) · unmap(k) = raf_claim
- ra(k) = Σ_j eq(r_cycle, j) · 1[address(j) = k]
- unmap(k) = start_address + k * 8

### Val Evaluation Protocol
- Proves: Val(r) - Val_init(r_address) = Σ_{j=0}^{T-1} inc(j) · wa(r_address, j) · LT(j, r_cycle)
- Ensures memory values are consistent with writes

## Files Added This Session (Iteration 4)

### Lookup Trace Integration
- `src/zkvm/instruction/lookup_trace.zig` - LookupEntry, LookupTraceCollector

### RAF Checking
- `src/zkvm/ram/raf_checking.zig` - Full RAF sumcheck infrastructure

### Val Evaluation
- `src/zkvm/ram/val_evaluation.zig` - Memory value consistency checking

### Multi-Stage Prover
- `src/zkvm/prover.zig` - 6-stage sumcheck orchestration

All 264 tests pass.
