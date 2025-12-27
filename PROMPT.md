# Zolt zkVM Implementation Guide

You are helping complete the Zolt zkVM, a Zig port of a16z's Jolt zkVM. This document outlines the current state and what needs to be implemented.

## Current State

The following is ACTUALLY IMPLEMENTED and working:

### Working Components
- **BN254 Scalar Field** (`src/field/mod.zig`): Montgomery form CIOS multiplication, all field ops
- **Extension Fields** (`src/field/pairing.zig`): Fp2, Fp6, Fp12 tower with pairing infrastructure
- **Sumcheck Protocol** (`src/subprotocols/mod.zig`): Complete prover/verifier
- **RISC-V Emulator** (`src/tracer/mod.zig`): Full RV64IMC execution with tracing
- **ELF Loader** (`src/host/elf.zig`): Complete ELF32/ELF64 parsing
- **Basic MSM** (`src/msm/mod.zig`): Affine/projective point arithmetic
- **Transcripts** (`src/transcripts/mod.zig`): Keccak-based Fiat-Shamir

Other parts MIGHT BE not working or fully implemented at the time of your iteration:

- **HyperKZG** (`src/poly/commitment/mod.zig`): commit() works, verify() is stub (returns true)
- **Dory** (`src/poly/commitment/mod.zig`): commit() works, open() is placeholder
- **Spartan** (`src/zkvm/spartan/mod.zig`): Matrix operations work, verifier incomplete
- **Pairing** (`src/field/pairing.zig`): Miller loop exists but uses placeholder generators

- **Lookup Arguments / Lasso**: ZERO implementation (the core Jolt technique)
- **JoltProver.prove()**: Panics at `src/zkvm/mod.zig:149`
- **JoltVerifier.verify()**: Panics at `src/zkvm/mod.zig:174`
- **host.execute()**: Panics at `src/host/mod.zig:173`
- **Preprocessing.preprocess()**: Panics at `src/host/mod.zig:243`
- **Multi-stage sumcheck**: Only basic sumcheck, not 7-stage orchestration
- **Virtual instructions**: None of the 80+ decomposition helpers
- **Memory RAF checking**: No read-after-final verification
- **Instruction constraint generation**: Emulation works but no ZK constraints

## Reference Implementation

The production Rust implementation is at `~/projects/jolt`. Key directories:
- `jolt-core/src/zkvm/lookup_table/` - 50+ lookup tables
- `jolt-core/src/zkvm/instruction_lookups/` - Instruction lookup handling
- `jolt-core/src/zkvm/prover.rs` - JoltCpuProver (1,848 lines)
- `jolt-core/src/zkvm/verifier.rs` - JoltVerifier (792 lines)
- `jolt-core/src/poly/commitment/dory/` - Streaming Dory commitment

## Implementation Priority

### Phase 1: Lookup Arguments (CRITICAL - Start Here)

Lookup arguments are THE core technique that makes Jolt efficient. Without them, nothing else works.

**Step 1.1: Create lookup table infrastructure**
Create `src/zkvm/lookup_table/mod.zig`:
```zig
pub fn LookupTable(comptime F: type) type {
    return struct {
        pub fn materializeEntry(index: u128) u64;
        pub fn evaluateMLE(r: []const F) F;
    };
}
```

**Step 1.2: Implement basic tables**
Start with these essential tables:
- `RangeCheckTable` - Verify values are in range [0, 2^k)
- `AndTable` - Bitwise AND lookup
- `OrTable` - Bitwise OR lookup
- `XorTable` - Bitwise XOR lookup
- `LessThanTable` - Unsigned comparison
- `SignedLessThanTable` - Signed comparison
- `EqualTable` - Equality check

**Step 1.3: Implement Lasso prover/verifier**
The Lasso lookup argument proves correct table lookups. See the Lasso paper for details.
Reference: `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`

### Phase 2: Instruction Proving

**Step 2.1: Define instruction flags**
```zig
pub const CircuitFlags = enum {
    AddOperands,
    SubtractOperands,
    MultiplyOperands,
    Load,
    Store,
    Jump,
    WriteLookupOutputToRD,
    // ... 13 total
};

pub const InstructionFlags = enum {
    LeftOperandIsPC,
    RightOperandIsImm,
    LeftOperandIsRs1Value,
    RightOperandIsRs2Value,
    // ... 8 total
};
```

**Step 2.2: Generate constraints per instruction**
Each RISC-V instruction must generate R1CS constraints plus lookup queries.
Reference: `jolt-core/src/zkvm/instruction/` (one file per instruction family)

### Phase 3: Memory Checking

**Step 3.1: Implement RAF (Read-After-Final) checking**
Ensures all memory reads return the most recently written value.
Reference: `jolt-core/src/zkvm/ram/raf_evaluation.rs`

**Step 3.2: Add value consistency**
Verify memory values match across read/write operations.
Reference: `jolt-core/src/zkvm/ram/val_evaluation.rs`

**Step 3.3: Implement one-hot addressing**
Efficient sparse memory access representation.
Reference: `jolt-core/src/zkvm/ram/mod.rs` (OneHotParams)

### Phase 4: Multi-Stage Sumcheck

Jolt uses 7 sumcheck stages:
1. Outer Spartan (instruction correctness)
2. RAM/bytecode RAF checking
3. Instruction lookup reduction
4. RAM value evaluation
5. Register increment checking
6. RAM increment checking
7. Hamming weight & booleanity

Create `src/zkvm/prover.zig` to orchestrate these stages.
Reference: `jolt-core/src/zkvm/prover.rs:526-1132`

### Phase 5: Complete Commitment Schemes

**Step 5.1: Fix HyperKZG verification**
The current verify() is a stub. Implement actual pairing check:
```zig
// In src/poly/commitment/mod.zig:verifyWithPairing
// Actually call pairing.pairingCheck() with computed values
```

**Step 5.2: Add real BN254 curve constants**
Replace placeholder G2 generator in `src/field/pairing.zig:367-374`

### Phase 6: Integration

**Step 6.1: Implement execute()**
Connect `src/host/mod.zig:execute()` to the tracer.

**Step 6.2: Implement JoltProver.prove()**
Remove panic at `src/zkvm/mod.zig:149`, wire up all components.

**Step 6.3: Implement JoltVerifier.verify()**
Remove panic at `src/zkvm/mod.zig:174`, implement verification logic.

## Key Architectural Decisions

### Use Zig Patterns
- Use `comptime` generics instead of Rust traits
- Use `std.ArrayListUnmanaged(T)` instead of `Vec<T>`
- Use `E!T` error unions instead of `Result<T,E>`
- Use explicit allocators everywhere

### Follow Existing Conventions
- Field elements use the existing `BN254Scalar` type
- Polynomials use the `DensePolynomial` pattern
- Tests go in the same file as implementation

### Performance Considerations
- Batch inversions using Montgomery's trick (already in field/mod.zig)
- Use SIMD where possible (Zig's @Vector)
- Consider parallel sumcheck round computation

Make a commit and push your changes after every single small advancement.

Use the .agent/ directory as a scratchpad for your work:
- Store your porting plan in .agent/PLAN.md. Update it all the time with the situation you encounter.
- Track progress in .agent/TODO.md
