# Zolt Architecture Guide

Welcome! This document walks you through the zolt codebase — a Zig zkVM prover
that generates proofs verifiable by the Rust [Jolt](https://github.com/a16z/jolt) verifier.

## The 30-Second Version

Zolt takes a RISC-V ELF binary, emulates it, and produces a zero-knowledge proof
that the execution was correct. The proof format is byte-compatible with Jolt's Rust
verifier — no patched fork needed.

```
ELF binary → Emulator → Execution Trace → 7-Stage Prover → Proof (verified by Jolt)
```

## Three Packages

The codebase is split into three packages with a clean dependency chain:

```
zolt-pool     (packages/zolt-pool/)     Zero deps — thread pool primitives
    ↓
zolt-arith    (packages/zolt-arith/)    Depends on zolt-pool — crypto math library
    ↓
zolt          (src/)                    Depends on both — the actual zkVM
```

### zolt-pool — `packages/zolt-pool/src/`

A work-stealing thread pool (Chase-Lev deque) used for parallel polynomial operations.
If you've used Rust's Rayon, this is the Zig equivalent.

- `thread_pool.zig` — The pool itself (~2K LOC)
- `parallel_sort.zig` — Parallel sample sort
- `helpers.zig` — `parallelReduceOptional()`, `parallelForOptional()` for optional-pool patterns

### zolt-arith — `packages/zolt-arith/src/`

All the cryptographic math. Think of this as Zig's arkworks:

| Directory | What it does | Key types |
|-----------|-------------|-----------|
| `field/` | BN254 finite field arithmetic | `BN254Scalar`, `Fp2`, `Fp6`, `Fp12`, `GT` |
| `field/pairing.zig` | Optimal Ate pairing (Miller loop + final exp) | `millerLoop()`, `finalExponentiation()` |
| `msm/` | Multi-scalar multiplication (Pippenger + GLV endomorphism) | `AffinePoint`, `ProjectivePoint` |
| `poly/` | Multilinear polynomials + Dory commitment scheme | `DensePolynomial`, `EqPolynomial`, `UniPoly`, `DoryCommitmentScheme` |
| `gpu/` | Metal GPU acceleration (Apple Silicon only, stubs elsewhere) | `GpuPolyOps`, `GpuMsmOps` |
| `transcripts/` | Fiat-Shamir transcript (Blake2b) | `Blake2bTranscript` |
| `subprotocols/` | Interactive proof protocols | `Sumcheck` |

**Start here if:** you want to understand the math primitives.

### zolt — `src/`

The zkVM itself. This is where the RISC-V execution and proof generation happens.

## How a Proof Gets Generated

When you run `zolt prove fibonacci.elf`, here's the call chain:

### 1. ELF Loading (`src/host/`)

`ELFLoader` parses the RISC-V ELF binary, extracts the `.text` section (code) and
`.rodata` (read-only data), and determines the memory layout.

### 2. Emulation (`src/tracer/mod.zig`)

The `Emulator` struct is a full RISC-V emulator (~5K LOC). It executes every instruction
and records an `ExecutionTrace` — a list of `TraceStep` structs capturing the CPU state
at each cycle (PC, registers, memory accesses).

### 3. Preprocessing (`src/zkvm/preprocessing.zig`)

`BytecodePreprocessing` decodes the raw bytes into Jolt's instruction format. This includes:
- Expanding "virtual" instructions (e.g., a single DIV becomes 7-14 micro-steps)
- Building the PC→cycle mapper (`bytecode_pc_mapper.zig`)
- Adding a termination sequence (LUI + ADDI + SB + JAL)

### 4. Proof Generation (`src/zkvm/jolt_prover.zig`)

The `JoltProver` is the heart of zolt. Its `proveWithTranscript()` method runs
7 sequential stages, each producing sumcheck proofs:

| Stage | What it proves | Prover file |
|-------|---------------|-------------|
| 1 | R1CS constraint satisfaction (outer Spartan) | `spartan/streaming_outer.zig` |
| 2 | Product virtualization + RAM/register consistency | `spartan/stage2_sumcheck.zig` |
| 3 | Shift, instruction input, register claim reductions | `spartan/stage3_prover.zig` |
| 4 | Register read-write + RAM value evaluation | `spartan/stage4_gruen_prover.zig` |
| 5 | Register value eval + RAM RA + lookup RAF | `spartan/stage5_prover.zig` |
| 6 | Bytecode + hamming + booleanity + RA virtual | `spartan/stage6_prover.zig` |
| 7 | Joint opening proof assembly | (inline in jolt_prover.zig) |

Each stage feeds its output (challenges, opening points) into the next.
The transcript ensures Fiat-Shamir soundness — challenge derivation is deterministic.

### 5. Serialization (`src/zkvm/jolt_serialization.zig`)

The proof is serialized in arkworks-compatible format so the Rust Jolt verifier
can deserialize and verify it without any Zig-specific code.

## Key Concepts

### Sumcheck Protocol

The core proving technique. Almost every stage runs a sumcheck — an interactive
protocol that reduces a claim about a multivariate polynomial to a claim about
a single point. The `subprotocols/mod.zig` has the generic framework, but each
stage has its own specialized prover (for performance).

Shared helpers live in `spartan/sumcheck_helpers.zig`:
- `inactiveContribution()` — handles instances that haven't started yet
- `finiteDifferencesCompress()` — compresses evaluations for the transcript
- `deriveGammaPowers()` — common gamma challenge pattern

### Lookup Arguments

Jolt's key insight: verify RISC-V instructions via lookup tables rather than
arithmetic circuits. The lookup infrastructure lives in:

- `instruction/lookups.zig` — Per-instruction lookup definitions (uses `BinaryLookup` comptime generic)
- `instruction/lookup_trace.zig` — Factory methods creating lookup entries from instructions
- `lookup_table/` — The actual table MLEs, prefix/suffix decomposition, and Shout protocol

### R1CS Constraints (`src/zkvm/r1cs/`)

19 uniform constraints encoding RISC-V execution rules (e.g., "if this is a LOAD
instruction, the RAM read value must equal the RAM write value"). Defined in
`constraints.zig`, evaluated in `evaluation.zig`.

### Memory Checking (`src/zkvm/ram/`)

Proves RAM and register file consistency using read-write checking, RAF (Read-After-Final)
checking, and value evaluation protocols. The pattern: cycle-major → address-major
two-phase sumcheck with prefix-suffix decomposition.

## Comptime Patterns

Zolt makes heavy use of Zig's comptime for zero-cost generics:

- **`BinaryLookup(XLEN, config)`** — comptime struct generates instruction lookups from a config
- **`inline for` over struct fields** — iterates heterogeneous types at compile time (used in batched sumcheck)
- **Duck-typed interfaces** — instance provers just need `computeRoundEvals()` and `bind()` methods
- **Comptime enum strategies** — compression format selection without runtime dispatch

## Testing & Verification

```bash
zig build test              # Run all unit tests
zig build -Doptimize=ReleaseFast  # Build optimized binary

# Generate and verify a proof:
./zig-out/bin/zolt prove examples/fibonacci.elf \
  -o /tmp/proof.bin --export-preprocessing /tmp/pp.bin

cd jolt-verifier && cargo run --release -- \
  --proof /tmp/proof.bin --preprocessing /tmp/pp.bin
# Should print: VERIFIED: proof is valid
```

## Where to Start Contributing

- **Small fixes:** `src/zkvm/debug.zig` controls debug output — set `verbose = true` to see proof internals
- **New instructions:** Add a lookup in `instruction/lookups.zig` using the `BinaryLookup` generic
- **Performance:** The stage provers in `spartan/` are the hot path — profile with `ZOLT_BENCH=1`
- **Arithmetic:** The field/poly/msm code in `zolt-arith` is independent and well-tested
- **Missing verifier:** There's no native Zig verifier yet — proofs are verified via the Rust `jolt-verifier/`
