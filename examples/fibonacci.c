// Simple Fibonacci example for Zolt zkVM
// Computes the 10th Fibonacci number
//
// Compile with:
//   riscv32-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -O2 \
//     -Wl,-Ttext=0x80000000 -o fibonacci.elf fibonacci.c

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

// Compute Fibonacci iteratively
int fibonacci(int n) {
    if (n <= 1) return n;

    int a = 0;
    int b = 1;
    for (int i = 2; i <= n; i++) {
        int c = a + b;
        a = b;
        b = c;
    }
    return b;
}

int main(void) {
    // Compute fib(10) = 55
    int result = fibonacci(10);
    return result;
}
