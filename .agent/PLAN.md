# Zolt zkVM Implementation Plan

## Current Status (December 2024)
Starting implementation of lookup table infrastructure - the core Jolt technique.

## Phase 1: Lookup Arguments (CRITICAL - In Progress)

### Step 1.1: Lookup Table Infrastructure
- [x] Analyze Jolt Rust reference implementation
- [ ] Create `src/zkvm/lookup_table/mod.zig`
- [ ] Define `JoltLookupTable` interface
- [ ] Define `PrefixSuffixDecomposition` trait
- [ ] Implement bit manipulation utilities

### Step 1.2: Basic Tables (Priority Order)
1. [ ] RangeCheckTable - Verify values in range [0, 2^k)
2. [ ] AndTable - Bitwise AND lookup
3. [ ] OrTable - Bitwise OR lookup
4. [ ] XorTable - Bitwise XOR lookup
5. [ ] EqualTable - Equality check
6. [ ] UnsignedLessThanTable - Unsigned comparison
7. [ ] SignedLessThanTable - Signed comparison

### Step 1.3: Lasso Prover/Verifier
- [ ] Implement Lasso lookup argument
- [ ] RAF (Read-After-Final) checking

## Phase 2: Instruction Proving (Next)
- [ ] Define CircuitFlags and InstructionFlags
- [ ] Generate constraints per instruction

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
- [ ] implement execute()
- [ ] implement JoltProver.prove()
- [ ] implement JoltVerifier.verify()

## Key Design Decisions

### Zig-specific Patterns
- Use `comptime` generics instead of Rust traits
- Use `std.ArrayListUnmanaged(T)` for dynamic arrays
- Use `E!T` error unions
- Explicit allocators everywhere

### Following Existing Conventions
- Field elements use `BN254Scalar` type
- Tests in same file as implementation
