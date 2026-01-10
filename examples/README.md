# Zolt Examples

This directory contains example programs and demonstrations.

## Zig Examples

These demonstrate the low-level APIs of Zolt:

| Example | Description |
|---------|-------------|
| `field_arithmetic.zig` | BN254 field operations and batch processing |
| `simple_proof.zig` | Basic polynomial commitment and Fiat-Shamir |
| `risc_v_emulation.zig` | RISC-V instruction decoding |
| `hyperkzg_commitment.zig` | HyperKZG polynomial commitment scheme |
| `sumcheck_protocol.zig` | Sumcheck protocol demonstration |
| `full_pipeline.zig` | End-to-end proving and verification |

### Running Zig Examples

```bash
# From the project root:
zig build example-field        # Field arithmetic
zig build example-proof        # Simple polynomial commitment
zig build example-riscv        # RISC-V instruction decoding
zig build example-hyperkzg     # HyperKZG commitment scheme
zig build example-sumcheck     # Sumcheck protocol
zig build example-pipeline     # Full proving pipeline
```

## C Examples

These are simple C programs that can be compiled to RISC-V ELF binaries and proven with Zolt.

| Program | Description | Expected Result |
|---------|-------------|-----------------|
| `fibonacci.c` | Compute Fibonacci(10) | 55 |
| `sum.c` | Sum of 1 to 100 | 5050 |
| `factorial.c` | Compute 10! | 3628800 |
| `gcd.c` | GCD using Euclidean algorithm | 63 |
| `collatz.c` | Collatz sequence for n=27 | 111 steps |
| `primes.c` | Count primes less than 100 | 25 |
| `signed.c` | Signed arithmetic operations | -39 |
| `bitwise.c` | AND, OR, XOR, shift operations | 209 |
| `array.c` | Array store/load with sum/max | 1465 |

### Pre-built ELF Files

Each C example has a pre-compiled `.elf` file included. These were built with the RISC-V GCC toolchain.

### Building C Examples (requires RISC-V toolchain)

If you have `riscv64-unknown-elf-gcc` installed:

```bash
cd examples
make all        # Build all examples
make clean      # Remove compiled files
make help       # Show available targets
```

Individual targets:
```bash
make fibonacci.elf
make sum.elf
make factorial.elf
make gcd.elf
make collatz.elf
make primes.elf
make signed.elf
make bitwise.elf
make array.elf
```

### Running C Examples

```bash
# From the project root:
zig build run -- run examples/fibonacci.elf          # Run program
zig build run -- run --regs examples/fibonacci.elf   # Show registers
zig build run -- trace examples/fibonacci.elf        # Show execution trace
```

### Proving C Examples

```bash
# Generate and verify a proof
zig build run -- prove examples/fibonacci.elf

# Save proof to a file
zig build run -- prove -o proof.bin examples/fibonacci.elf

# Save as JSON
zig build run -- prove --json -o proof.json examples/fibonacci.elf

# Verify a saved proof
zig build run -- verify proof.bin

# Show proof statistics
zig build run -- stats proof.bin
```

## How C Programs Work with Zolt

Each C program:

1. Has a `_start` function as the entry point
2. Calls a `main()` function that computes a result
3. Places the result in register `a0` (x10)
4. Calls `ecall` to terminate execution

The zkVM captures the execution trace and generates a proof that the program ran correctly.

### Minimal C Program Template

```c
// Entry point - no standard library
void _start(void) {
    register int result asm("a0");
    result = main();

    // Terminate with ecall
    asm volatile("ecall");
    __builtin_unreachable();
}

int main(void) {
    // Your computation here
    return 42;  // Result returned in a0
}
```

## Writing Your Own Programs

1. Write a C program following the template above
2. Compile with RISC-V toolchain:
   ```bash
   riscv64-unknown-elf-gcc -march=rv64im -mabi=lp64 \
       -nostdlib -nostartfiles -static -O2 \
       -o myprogram.elf myprogram.c
   ```
3. Run with Zolt:
   ```bash
   zig build run -- run myprogram.elf
   ```
4. Generate a proof:
   ```bash
   zig build run -- prove myprogram.elf
   ```

## Performance Notes

- **Primes example**: Takes 8000+ cycles, proving takes longer
- **Collatz example**: Takes 825 cycles
- **Simple examples**: Finish in < 100 cycles

The proving time scales with the trace length. For faster proving during development, use `--trace-length` to configure the proof system.
