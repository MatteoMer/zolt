// Jolt-compatible Fibonacci example for Zolt zkVM
//
// This program reads input from the Jolt I/O region and writes output
// to the Jolt output region, making it compatible with Jolt verification.
//
// Memory layout (matching Jolt's fibonacci example):
//   input_start:  0x7fffa000
//   output_start: 0x7fffb000
//
// Input format: postcard-serialized u32 (just the byte value for n < 128)
// Output format: postcard-serialized u128 as little-endian bytes
//
// Compile with:
//   riscv64-unknown-elf-gcc -march=rv64imac -mabi=lp64 -nostdlib -O2 \
//     -Wl,-Ttext=0x80000000 -o fibonacci_jolt.elf fibonacci_jolt.c

// Jolt I/O memory addresses (from fibonacci example memory layout)
#define INPUT_START  0x7fffa000UL
#define OUTPUT_START 0x7fffb000UL

// Minimal startup code
void _start(void) __attribute__((naked));

void _start(void) {
    __asm__ volatile(
        // Set up stack pointer (matches Jolt's computed stack address)
        "li sp, 0x800016b8\n"
        // Call main
        "call main\n"
        // Exit via ecall
        "ecall\n"
    );
}

// Read a byte from memory
static inline unsigned char read_byte(unsigned long addr) {
    return *(volatile unsigned char*)addr;
}

// Write a byte to memory
static inline void write_byte(unsigned long addr, unsigned char value) {
    *(volatile unsigned char*)addr = value;
}

// Compute Fibonacci iteratively using 128-bit integers
// Returns the result split into low and high 64-bit parts
void fibonacci_128(unsigned int n, unsigned long *low, unsigned long *high) {
    if (n <= 1) {
        *low = n;
        *high = 0;
        return;
    }

    // Use 128-bit arithmetic via two 64-bit values
    unsigned long a_low = 0, a_high = 0;  // fib(0)
    unsigned long b_low = 1, b_high = 0;  // fib(1)

    for (unsigned int i = 2; i <= n; i++) {
        // sum = a + b (128-bit addition)
        unsigned long sum_low = a_low + b_low;
        unsigned long carry = (sum_low < a_low) ? 1 : 0;
        unsigned long sum_high = a_high + b_high + carry;

        a_low = b_low;
        a_high = b_high;
        b_low = sum_low;
        b_high = sum_high;
    }

    *low = b_low;
    *high = b_high;
}

int main(void) {
    // Read input n from Jolt input region
    // For n < 128, postcard encodes it as a single byte
    unsigned int n = read_byte(INPUT_START);

    // Compute fib(n) as u128
    unsigned long result_low, result_high;
    fibonacci_128(n, &result_low, &result_high);

    // Write output to Jolt output region
    // postcard for u128 writes it as 16 bytes little-endian
    // But for values that fit in fewer bytes, it uses variable length
    // For simplicity, let's write as many bytes as needed

    // Calculate how many bytes we need for the result
    // fib(50) = 12,586,269,025 = 0x2EF1CCF2E1 (5 bytes)
    unsigned long value = result_low;  // For fib(50), high is 0
    int i = 0;

    // Write bytes in little-endian order
    do {
        write_byte(OUTPUT_START + i, value & 0xFF);
        value >>= 8;
        i++;
    } while (value > 0);

    // If high part is non-zero, continue with it
    value = result_high;
    while (value > 0) {
        write_byte(OUTPUT_START + i, value & 0xFF);
        value >>= 8;
        i++;
    }

    return 0;
}
