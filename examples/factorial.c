// Factorial example for Zolt zkVM
// Computes factorial of 10 = 3628800
//
// Compile with:
//   riscv32-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -O2 \
//     -Wl,-Ttext=0x80000000 -o factorial.elf factorial.c

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

// Compute factorial iteratively
int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;  // Uses MUL instruction
    }
    return result;
}

int main(void) {
    // Compute 10! = 3628800
    int result = factorial(10);
    return result;
}
