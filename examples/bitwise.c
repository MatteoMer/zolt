// Bitwise operations example for Zolt zkVM
// Demonstrates AND, OR, XOR, and shift operations
//
// Compile with:
//   riscv32-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -O2 \
//     -Wl,-Ttext=0x80000000 -o bitwise.elf bitwise.c

// Minimal startup code
void _start(void) __attribute__((naked));

void _start(void) {
    __asm__ volatile(
        // Set up stack pointer
        "li sp, 0x80010000\n"
        // Call main and store result in a0
        "call main\n"
        // Exit via ecall
        "ecall\n"
    );
}

// Count number of set bits (popcount)
int popcount(unsigned int n) {
    int count = 0;
    while (n) {
        count += (n & 1);  // Uses AND instruction
        n >>= 1;           // Uses SRLI instruction
    }
    return count;
}

// Bit manipulation operations
int bit_ops(unsigned int a, unsigned int b) {
    unsigned int x = a & b;   // AND
    unsigned int y = a | b;   // OR
    unsigned int z = a ^ b;   // XOR
    unsigned int w = a << 4;  // SLLI
    unsigned int v = b >> 2;  // SRLI

    // Combine results
    return (x + y + z + w + v);
}

int main(void) {
    unsigned int a = 0xF0F0F0F0;
    unsigned int b = 0x0F0F0F0F;

    // popcount(0xF0F0F0F0) = 16
    int pop = popcount(a);

    // bit_ops computes combined result
    int ops = bit_ops(a, b);

    // Return low bits of combined result
    return (pop + (ops & 0xFF));
}
