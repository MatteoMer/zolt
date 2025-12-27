# Zolt: Jolt zkVM Port to Zig

## Overview
Porting the Jolt zkVM from Rust to Zig. The source code is at `/Users/matteo/projects/jolt`.

## Project Structure

```
zolt/
├── build.zig          # Build system
├── build.zig.zon      # Dependencies
├── src/
│   ├── root.zig       # Main library entry point
│   ├── common/        # Common types and constants
│   ├── field/         # Finite field arithmetic
│   ├── poly/          # Polynomial operations
│   │   └── commitment/  # Commitment schemes (Dory, HyperKZG)
│   ├── subprotocols/  # Sumcheck, GKR
│   ├── utils/         # Utility functions
│   ├── zkvm/          # RISC-V zkVM
│   │   ├── bytecode/
│   │   ├── instruction/
│   │   ├── r1cs/
│   │   ├── ram/
│   │   ├── registers/
│   │   └── spartan/
│   ├── msm/           # Multi-scalar multiplication
│   ├── host/          # Prover/verifier host interface
│   ├── transcripts/   # Fiat-Shamir transcripts
│   └── tracer/        # RISC-V execution tracer
└── tests/             # Integration tests
```

## Porting Status: ✅ COMPLETE (Core Implementation)

All core phases have been completed. The port includes:

### Phase 1: Foundation ✅
1. [x] Create build.zig and project structure
2. [x] Port common/constants.rs → src/common/constants.zig
3. [x] Port common/attributes.rs → src/common/attributes.zig
4. [x] Port common/jolt_device.rs → src/common/jolt_device.zig

### Phase 2: Field Arithmetic ✅
5. [x] Define JoltField interface
6. [x] Port BN254 scalar field with full Montgomery form
7. [x] Port extension fields (Fp2, Fp6, Fp12 for pairings)
8. [x] Port Montgomery CIOS multiplication

### Phase 3: Polynomials ✅
9. [x] Port dense_mlpoly.rs (dense multilinear polynomials)
10. [x] Port eq_poly.rs (equality polynomial)
11. [x] Port unipoly.rs (univariate polynomials)
12. [x] Port multilinear polynomial evaluation

### Phase 4: Commitment Schemes ✅
13. [x] Port commitment_scheme.rs (interface)
14. [x] Port HyperKZG with G2 points
15. [x] Port Dory (transparent setup)
16. [x] Add pairing infrastructure

### Phase 5: Subprotocols ✅
17. [x] Port sumcheck.rs with full prover
18. [x] Port round generation

### Phase 6: Utils ✅
19. [x] Port math utilities
20. [x] Port errors
21. [x] Port serialization stubs

### Phase 7: MSM ✅
22. [x] Port curve points (affine + projective)
23. [x] Port point operations (add, double)
24. [x] Implement Pippenger's algorithm

### Phase 8: Transcripts ✅
25. [x] Port Fiat-Shamir transcript with Keccak-f[1600]

### Phase 9: ZKVM Core ✅
26. [x] Port bytecode handling
27. [x] Port full RISC-V instruction set (RV64IMC)
28. [x] Port RAM/memory checking
29. [x] Port register handling
30. [x] Port R1CS constraints with sparse matrices
31. [x] Port Spartan prover/verifier

### Phase 10: Host Interface ✅
32. [x] Port ELF loader with full ELF32/64 parsing
33. [x] Port Program struct

### Phase 11: Tracer ✅
34. [x] Port complete RISC-V emulator
35. [x] Port instruction tracing with cycle counting

### Phase 12: Testing ✅
36. [x] Add 155 unit and integration tests

## Type Mapping Reference

| Rust | Zig |
|------|-----|
| `struct Foo { ... }` | `pub const Foo = struct { ... };` |
| `enum Foo { A, B(T) }` | `pub const Foo = union(enum) { a, b: T };` |
| `trait Foo { fn bar(&self) }` | Interface via vtable or comptime generics |
| `Vec<T>` | `std.ArrayList(T)` or `[]T` slice |
| `Result<T, E>` | `E!T` error union |
| `Option<T>` | `?T` optional |
| `Arc<T>` / `Rc<T>` | Manual memory with allocators |
| `Box<T>` | `*T` pointer with allocator |
| `&[T]` | `[]const T` or `[]T` slice |
| `impl Trait` | Comptime generics |
| `dyn Trait` | `*const Interface` vtable |
| `#[derive(...)]` | Comptime reflection or manual impl |
| `panic!()` | `@panic()` |
| `assert!()` | `std.debug.assert()` |

## Key Architectural Decisions

1. **Memory Management**: Use allocator pattern throughout. Create arena allocators for proof generation.

2. **Generics**: Use Zig comptime generics instead of Rust traits where possible.

3. **SIMD**: Leverage Zig's `@Vector` for SIMD operations in field arithmetic.

4. **Parallelism**: Use Zig's std.Thread and thread pools instead of rayon.

5. **Serialization**: Implement custom binary serialization instead of serde.

6. **Field Arithmetic**: Consider using big integer libraries or implementing from scratch with SIMD.

## Dependencies to Consider

- Crypto primitives (Keccak/SHA3 for transcripts)
- Big integer arithmetic (for field elements)
- Optional: GPU acceleration via Zig's CUDA/OpenCL bindings

## Testing Strategy

- Unit tests for each module (inline Zig tests)
- Integration tests in tests/ directory
- Compare outputs with Rust implementation for correctness
