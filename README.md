# Zolt

A Zig zkVM prover that generates proofs verifiable by the unmodified upstream [a16z/jolt](https://github.com/a16z/jolt) verifier — no patched fork needed. Zero dependencies, zero FFI — all cryptography (field arithmetic, pairings, MSM, polynomial commitments) is implemented from scratch using only the Zig standard library.

> **WARNING:** This project is experimental and has not been audited. Do not use in production.

## Quick Start

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate a proof
./zig-out/bin/zolt prove examples/fibonacci.elf -o /tmp/proof.bin --export-preprocessing /tmp/preproc.bin

# Verify with upstream Jolt verifier
cargo run --release --manifest-path jolt-verifier/Cargo.toml -- \
  --proof /tmp/proof.bin --preprocessing /tmp/preproc.bin
```

## CLI Reference

### `zolt prove [options] <elf>`

Generate a ZK proof for a RISC-V ELF binary.

| Flag | Description |
|------|-------------|
| `-o, --output FILE` | Save proof to FILE (required) |
| `--export-preprocessing FILE` | Export Jolt-compatible preprocessing data |
| `--srs PATH` | Use Dory SRS from PATH |
| `--input-hex HEX` | Set program input as hex bytes |

### `zolt run [options] <elf>`

Run a RISC-V ELF binary in the emulator.

| Flag | Description |
|------|-------------|
| `--regs` | Show final register state |
| `--trace` | Show execution trace |
| `--max N` | Max trace entries to display (default: 100) |
| `--input FILE` | Load input bytes from FILE |
| `--input-hex HEX` | Set input as hex bytes |

### `zolt help` / `zolt version`

Show help or version information.

## Example Programs

All 8 example programs produce valid proofs verified by the upstream Jolt verifier:

| Program | Description | Expected Result |
|---------|-------------|-----------------|
| `fibonacci.elf` | Compute Fibonacci(10) | 55 |
| `factorial.elf` | Compute 10! | 3628800 |
| `bitwise.elf` | AND, OR, XOR, shifts | - |
| `collatz.elf` | Collatz sequence for n=27 | 111 steps |
| `primes.elf` | Count primes < 100 | 25 |
| `sum.elf` | Sum of 1..100 | 5050 |
| `gcd.elf` | GCD using div/rem | 63 |
| `signed.elf` | Signed arithmetic ops | -39 |

### Building Examples

Requires a RISC-V toolchain (e.g., `riscv32-unknown-elf-gcc`):

```bash
cd examples
make all
```

## Benchmarks

Compared against [Jolt](https://github.com/a16z/jolt) (Rust) on Apple M1 Pro, 16 GB RAM, macOS.

| Program | Trace | Jolt (ms) | Zolt (ms) | Ratio |
|---------|------:|----------:|----------:|------:|
| fibonacci | 128 | 700.95 | 157.25 | 0.22x |
| factorial | 64 | 686.91 | 139.28 | 0.20x |
| bitwise | 512 | 769.48 | 205.33 | 0.27x |
| collatz | 2048 | 867.76 | 281.20 | 0.32x |
| primes | 2048 | 840.17 | 303.15 | 0.36x |
| sum | 16 | 652.35 | 131.52 | 0.20x |
| gcd | 256 | 682.48 | 193.43 | 0.28x |
| signed | 16 | 672.60 | 143.37 | 0.21x |
| primes_large | 65536 | 1769.51 | 1260.96 | 0.71x |
| sha256_128 | 16384 | 1464.11 | 903.51 | 0.62x |
| sha256 | 16384 | 1822.50 | 964.19 | 0.53x |
| sha256_512 | 32768 | 1992.00 | 1268.11 | 0.64x |
| sha256_1024 | 65536 | 2712.83 | 1848.69 | 0.68x |
| sha256_2048 | 131072 | 4669.83 | 3191.04 | 0.68x |

## Upstream Compatibility

The `jolt-verifier/` and `jolt-bench/` crates are pinned to upstream commit [`b10e80ac`](https://github.com/a16z/jolt/commit/b10e80ac). All programs are verified against this version. This includes the Fiat-Shamir soundness fixes from PRs #1358 (config parameters bound to transcript) and #1408 (preprocessing digest bound to transcript).

## Project Structure

The codebase is split into 3 packages with explicit dependencies:

```
zolt/
├── packages/
│   ├── zolt-pool/                 # Thread pool + parallel primitives (0 deps)
│   │   └── src/
│   │       ├── thread_pool.zig    # Chase-Lev work-stealing deque
│   │       ├── parallel_sort.zig  # Parallel sample sort
│   │       └── helpers.zig        # parallelReduceOptional, parallelForOptional
│   │
│   └── zolt-arith/                # Arithmetic library (depends on: zolt-pool)
│       └── src/
│           ├── field/             # BN254 scalar/base fields, Montgomery mul, extensions (Fp2/6/12), pairings, G2
│           ├── poly/              # Polynomials, Dory commitment scheme, interpolation, product trees
│           ├── msm/               # Multi-scalar multiplication (Pippenger + GLV)
│           ├── gpu/               # Metal GPU acceleration (Apple Silicon, optional)
│           ├── transcripts/       # Fiat-Shamir (Blake2b, Keccak, Poseidon)
│           └── subprotocols/      # Sumcheck protocol
│
├── src/                           # zkVM package (depends on: zolt-arith, zolt-pool)
│   ├── main.zig                   # CLI dispatcher
│   ├── cli/                       # Argument parsing
│   ├── commands/                  # run + prove command implementations
│   ├── common/                    # Constants, device config
│   ├── host/                      # ELF loader
│   ├── tracer/                    # RISC-V emulator + witness generation
│   └── zkvm/                      # Core proving logic
│       ├── jolt_prover.zig        # 7-stage Jolt-compatible prover (orchestrator)
│       ├── jolt_types.zig         # Proof types, sumcheck IDs
│       ├── jolt_serialization.zig # Arkworks-compatible serialization
│       ├── preprocessing.zig      # Bytecode preprocessing
│       ├── instruction_decoder.zig # RISC-V instruction decoding
│       ├── eq_utils.zig           # Shared EQ polynomial evaluation (BE + LE)
│       ├── debug.zig              # Shared debug output flag
│       ├── spartan/               # Spartan prover stages (2-6) + sumcheck helpers
│       ├── shout/                 # Shout lookup argument
│       ├── r1cs/                  # R1CS constraints + evaluation
│       ├── instruction/           # Instruction lookups (BinaryLookup comptime generic)
│       ├── lookup_table/          # Lookup table MLEs + prefix/suffix decomposition
│       ├── ram/                   # Memory checking
│       └── registers/             # Register file checking
│
├── examples/                      # Pre-built RISC-V ELF binaries + C sources
├── jolt-verifier/                 # Standalone upstream Jolt verifier (Rust)
└── docs/research/                 # Refactoring plan + analysis
```

### Dependency graph

```
zolt-pool           (leaf — zero deps, only Zig std)
    │
    ▼
zolt-arith          (depends on: zolt-pool)
    │
    ▼
zolt                (depends on: zolt-arith, zolt-pool)
```

## Prerequisites

- **Zig 0.15.0+** — build and run Zolt
- **Rust toolchain** — build `jolt-verifier/` for proof verification
- **RISC-V toolchain** — build example C programs (pre-built ELFs included)

## License

MIT
