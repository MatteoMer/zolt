# Zolt zkVM Implementation Plan

## Current Status (December 2024)
Completed Phase 1 lookup table infrastructure and Phase 2.1 instruction flags.
Phase 2.2 instruction lookup interfaces also complete.

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

### Step 1.3: Lasso Prover/Verifier (Future)
- [ ] Implement Lasso lookup argument infrastructure
- [ ] RAF (Read-After-Final) checking
- [ ] Integrate with existing sumcheck protocol

## Phase 2: Instruction Proving (IN PROGRESS)

### Step 2.1: Instruction Flags ✅
- [x] CircuitFlags enum (13 flags for R1CS constraints)
- [x] InstructionFlags enum (7 flags for instruction metadata)
- [x] CircuitFlagSet and InstructionFlagSet types

### Step 2.2: Instruction Lookup Interfaces ✅
- [x] LookupTables(XLEN) enum with materializeEntry()
- [x] InstructionLookup(XLEN) interface
- [x] Flags interface
- [x] LookupQuery(XLEN) interface

### Step 2.3: Per-Instruction Implementation (Next)
- [ ] Implement lookup queries for ADD, SUB, AND, etc.
- [ ] Connect to lookup table evaluation
- [ ] Generate R1CS constraints per instruction

## Phase 3: Memory Checking
- [ ] RAF checking
- [ ] Value consistency
- [ ] One-hot addressing

## Phase 4: Multi-Stage Sumcheck
- [ ] 7-stage orchestration

## Phase 5: Complete Commitment Schemes
- [ ] Fix HyperKZG verification
- [ ] Add real BN254 curve constants

## Phase 6: Integration
- [ ] Implement execute()
- [ ] Implement JoltProver.prove()
- [ ] Implement JoltVerifier.verify()

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
