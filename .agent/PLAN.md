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

## Porting Strategy

### Phase 1: Foundation (Current)
1. [x] Create build.zig and project structure
2. [ ] Port common/constants.rs → src/common/constants.zig
3. [ ] Port common/attributes.rs → src/common/attributes.zig
4. [ ] Port common/jolt_device.rs → src/common/jolt_device.zig

### Phase 2: Field Arithmetic
5. [ ] Define JoltField interface (from field/mod.rs)
6. [ ] Port BN254 scalar field implementation
7. [ ] Port challenge types
8. [ ] Port Montgomery/Barrett reduction

### Phase 3: Polynomials
9. [ ] Port dense_mlpoly.rs (dense multilinear polynomials)
10. [ ] Port eq_poly.rs (equality polynomial)
11. [ ] Port unipoly.rs (univariate polynomials)
12. [ ] Port multilinear_polynomial.rs
13. [ ] Port other polynomial types

### Phase 4: Commitment Schemes
14. [ ] Port commitment_scheme.rs (interface)
15. [ ] Port KZG implementation
16. [ ] Port HyperKZG
17. [ ] Port Dory

### Phase 5: Subprotocols
18. [ ] Port sumcheck.rs
19. [ ] Port sumcheck_prover.rs
20. [ ] Port sumcheck_verifier.rs
21. [ ] Port mles_product_sum.rs
22. [ ] Port booleanity.rs

### Phase 6: Utils
23. [ ] Port math.rs
24. [ ] Port errors.rs
25. [ ] Port thread.rs
26. [ ] Port accumulation.rs

### Phase 7: MSM
27. [ ] Port MSM (multi-scalar multiplication)

### Phase 8: Transcripts
28. [ ] Port Fiat-Shamir transcript

### Phase 9: ZKVM Core
29. [ ] Port bytecode handling
30. [ ] Port instruction set (RISC-V)
31. [ ] Port RAM/memory checking
32. [ ] Port register handling
33. [ ] Port R1CS constraints
34. [ ] Port Spartan

### Phase 10: Prover/Verifier
35. [ ] Port prover.rs
36. [ ] Port verifier.rs
37. [ ] Port proof serialization
38. [ ] Port witness generation

### Phase 11: Host Interface
39. [ ] Port program.rs
40. [ ] Port toolchain.rs

### Phase 12: Tracer
41. [ ] Port RISC-V emulator
42. [ ] Port instruction tracing

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
