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

Compared against [Jolt](https://github.com/a16z/jolt) (Rust) on a Hetzner CPX41 — 8 vCPU AMD EPYC (Zen 2), 16 GB RAM, Ubuntu 22.04.

| Program | Trace | Jolt (ms) | Zolt (ms) | Ratio |
|---------|------:|----------:|----------:|------:|
| fibonacci | 256 | 1240.01 | 929.33 | 0.75x |
| factorial | 256 | 1150.95 | 1027.77 | 0.89x |
| bitwise | 512 | 1351.32 | 1475.52 | 1.09x |
| collatz | 2048 | 1567.96 | 2541.66 | 1.62x |
| primes | 2048 | 1464.57 | 3443.07 | 2.35x |
| sum | 256 | 1260.56 | 953.11 | 0.76x |
| gcd | 256 | 1271.58 | 1427.87 | 1.12x |
| signed | 256 | 1323.29 | 976.55 | 0.74x |
| primes_large | 65536 | 2541.32 | 30773.29 | 12.11x |

## Upstream Compatibility

The `jolt-verifier/` and `jolt-bench/` crates are pinned to upstream commit [`997c1543`](https://github.com/a16z/jolt/commit/997c1543). This is the version all 8 programs are verified against. Updating to a newer upstream commit may require adjustments if the verification protocol changes.

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
