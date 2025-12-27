# Zolt

**Jolt zkVM ported to Zig** - A high-performance zero-knowledge virtual machine implementation.

This project is a comprehensive port of [a16z's Jolt zkVM](https://github.com/a16z/jolt) from Rust to Zig, leveraging Zig's comptime, explicit memory management, and system-level control for performance-critical cryptographic operations.

> This project is 100% AI generated with the [ralph method](https://ghuntley.com/ralph/)

## Overview

Zolt implements a zkVM (zero-knowledge virtual machine) that can prove correct execution of RISC-V programs. It uses lookup arguments (via the Lasso/Jolt technique) to achieve efficient proof generation.

### Key Features

- **BN254 Scalar Field**: Full Montgomery form arithmetic with CIOS multiplication
- **Polynomial Commitments**: HyperKZG and Dory commitment schemes
- **RISC-V Support**: Full RV64IMC instruction set (I, M, and C extensions)
- **Sumcheck Protocol**: Complete prover with efficient round generation
- **Spartan**: R1CS satisfiability prover/verifier
- **Fiat-Shamir**: Keccak-f[1600] based transcript for non-interactive proofs
- **Multi-Scalar Multiplication**: Pippenger's algorithm with parallel execution
- **ELF Loading**: Complete ELF32/ELF64 parser for loading RISC-V binaries

## Building

Requires Zig 0.15.0 or later.

```bash
# Build the project
zig build

# Run tests
zig build test

# Build optimized release
zig build -Doptimize=ReleaseFast

# Run the CLI
zig build run
```

## Project Structure

```
zolt/
├── build.zig              # Build configuration
├── build.zig.zon          # Dependencies
├── src/
│   ├── root.zig           # Library entry point
│   ├── main.zig           # CLI application
│   ├── bench.zig          # Benchmarks
│   ├── common/            # Constants, attributes
│   ├── field/             # Finite field arithmetic
│   │   ├── mod.zig        # BN254Scalar, BatchOps
│   │   └── pairing.zig    # Fp2, Fp6, Fp12, G2Point
│   ├── poly/              # Polynomial operations
│   │   ├── mod.zig        # Dense, Eq, UniPoly
│   │   └── commitment/    # HyperKZG, Dory, Mock
│   ├── subprotocols/      # Sumcheck protocol
│   ├── utils/             # Errors, math, serialization
│   ├── zkvm/              # RISC-V zkVM core
│   │   ├── bytecode/      # Bytecode handling
│   │   ├── instruction/   # RISC-V decoder (RV64IMC)
│   │   ├── r1cs/          # R1CS constraint system
│   │   ├── ram/           # Memory checking
│   │   ├── registers/     # Register file
│   │   └── spartan/       # Spartan prover/verifier
│   ├── msm/               # Multi-scalar multiplication
│   ├── host/              # ELF loader, program interface
│   ├── guest/             # Guest program I/O
│   ├── tracer/            # RISC-V emulator
│   └── transcripts/       # Fiat-Shamir transcripts
└── .agent/                # Development notes
```

## Architecture

### Type Mapping (Rust → Zig)

| Rust | Zig |
|------|-----|
| `struct Foo { ... }` | `pub const Foo = struct { ... };` |
| `enum Foo { A, B(T) }` | `pub const Foo = union(enum) { a, b: T };` |
| `trait Foo` | Comptime generics or vtable interface |
| `Vec<T>` | `std.ArrayListUnmanaged(T)` or `[]T` |
| `Result<T, E>` | `E!T` error union |
| `Option<T>` | `?T` optional |
| `Arc<T>` / `Rc<T>` | Manual memory with allocators |
| `Box<T>` | `*T` pointer with allocator |

### Core Components

1. **Field Arithmetic** (`src/field/mod.zig`)
   - BN254 scalar field in Montgomery form
   - CIOS multiplication algorithm
   - Batch operations (add, multiply, inverse)
   - Optimized squaring using Karatsuba-like technique

2. **Polynomials** (`src/poly/mod.zig`)
   - Dense multilinear polynomials
   - Equality polynomials for sumcheck
   - Univariate polynomials

3. **Commitment Schemes** (`src/poly/commitment/mod.zig`)
   - **HyperKZG**: Pairing-based polynomial commitment
   - **Dory**: Transparent setup scheme
   - Extension fields (Fp2, Fp6, Fp12) for pairings

4. **Sumcheck Protocol** (`src/subprotocols/mod.zig`)
   - Interactive proof for polynomial identity
   - Round-by-round prover with univariate reduction

5. **Spartan** (`src/zkvm/spartan/mod.zig`)
   - R1CS satisfiability prover
   - Verifier with sumcheck integration

6. **RISC-V VM** (`src/zkvm/`, `src/tracer/`)
   - Full RV64IMC instruction decoder
   - Complete emulator with tracing
   - Memory checking for RAM and registers

7. **MSM** (`src/msm/mod.zig`)
   - Affine and projective point representations
   - Pippenger's bucket method
   - Parallel execution with `std.Thread`

## Usage Example

```zig
const std = @import("std");
const zolt = @import("zolt");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Load a RISC-V ELF binary
    const program = try zolt.host.ElfLoader.loadFile("program.elf", allocator);
    defer program.deinit(allocator);

    // Create and run the emulator
    var emulator = zolt.tracer.Emulator.init(allocator);
    try emulator.loadProgram(program.code);
    try emulator.run();

    // Get the execution trace
    const trace = emulator.getTrace();

    // Generate witness for proving
    var witness_gen = zolt.tracer.WitnessGenerator(zolt.field.BN254Scalar).init();
    const witness = try witness_gen.generateFromTrace(&trace, allocator);
    defer allocator.free(witness);
}
```

## Testing

The project includes 193 tests covering:

- Field arithmetic (Montgomery operations, batch ops)
- Polynomial operations (evaluation, summation)
- Commitment schemes
- RISC-V instruction decoding and execution
- Sumcheck protocol
- Spartan prover/verifier
- MSM algorithms
- ELF parsing
- Integration tests

Run all tests:
```bash
zig build test
```

## Performance

Key optimizations implemented:

- **Montgomery Multiplication**: CIOS algorithm for field operations
- **Batch Inverse**: Montgomery's trick (O(3n) muls + 1 inv instead of n invs)
- **Pippenger MSM**: Bucket method with optimal window selection
- **Parallel MSM**: Multi-threaded execution for large inputs
- **Optimized Squaring**: Karatsuba-like technique saves ~25% multiplications

## Differences from Rust Jolt

1. **Memory Management**: Explicit allocators instead of Rust's ownership
2. **Generics**: Zig comptime instead of Rust traits
3. **Error Handling**: Error unions instead of `Result`
4. **Serialization**: Custom binary format instead of serde
5. **Parallelism**: `std.Thread` instead of rayon

## License

This project follows the same license as the original Jolt implementation.

## Acknowledgments

- [a16z/jolt](https://github.com/a16z/jolt) - Original Rust implementation
- [Lasso paper](https://eprint.iacr.org/2023/1216) - Lookup argument foundation
- [Jolt paper](https://eprint.iacr.org/2023/1217) - SNARK for virtual machines
