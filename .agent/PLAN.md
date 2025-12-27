# Zolt zkVM Implementation Plan

## Current Status (December 2024)
Completed Phase 1 lookup table infrastructure with 14 tables implemented.

## Phase 1: Lookup Arguments (COMPLETED)

### Step 1.1: Lookup Table Infrastructure ✅
- [x] Created `src/zkvm/lookup_table/mod.zig`
- [x] Defined `LookupTable` generic interface
- [x] Implemented bit interleaving/uninterleaving utilities

### Step 1.2: Basic Tables ✅ (14 tables implemented)
1. [x] RangeCheckTable - Verify values in range [0, 2^k)
2. [x] AndTable - Bitwise AND lookup
3. [x] OrTable - Bitwise OR lookup
4. [x] XorTable - Bitwise XOR lookup
5. [x] EqualTable - Equality check
6. [x] NotEqualTable - Inequality check
7. [x] UnsignedLessThanTable - Unsigned less-than comparison
8. [x] SignedLessThanTable - Signed less-than comparison
9. [x] UnsignedGreaterThanEqualTable - Unsigned >= comparison
10. [x] UnsignedLessThanEqualTable - Unsigned <= comparison
11. [x] SignedGreaterThanEqualTable - Signed >= comparison
12. [x] MovsignTable - Extract sign bit
13. [x] SubTable - Subtraction (wrapping)
14. [x] AndnTable - Bitwise AND-NOT (x & ~y)

All tables implement:
- `materializeEntry(index)` - Direct table lookup
- `evaluateMLE(r)` - Multilinear extension evaluation

### Step 1.3: Lasso Prover/Verifier (NEXT)
- [ ] Implement Lasso lookup argument infrastructure
- [ ] RAF (Read-After-Final) checking
- [ ] Integrate with existing sumcheck protocol

## Phase 2: Instruction Proving (Upcoming)
- [ ] Define CircuitFlags and InstructionFlags enums
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

## Notes on Jolt MLE Formulas

From analyzing the Jolt codebase:

1. **UnsignedLessThan MLE**:
   ```
   Σ_i (1 - x_i) * y_i * Π_{j<i} eq(x_j, y_j)
   ```

2. **SignedLessThan MLE** (elegant):
   ```
   x_sign - y_sign + unsigned_lt(x, y)
   ```
   This works because the sign bit contribution handles cross-sign comparisons.

3. **Prefix-Suffix Decomposition**:
   Jolt uses a clever decomposition where tables are split into prefix and suffix components,
   allowing efficient incremental evaluation during sumcheck.
