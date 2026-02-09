// Fibonacci example that avoids ADDIW instructions
// Uses only 64-bit (long) types so the compiler uses ADDI instead of ADDIW.
//
// Compile with:
//   riscv64-unknown-elf-gcc -march=rv64im -mabi=lp64 -nostdlib -O2 \
//     -Wl,-Ttext=0x80000000 -o fibonacci_rv64.elf fibonacci_rv64.c

void _start(void) __attribute__((naked));

void _start(void) {
    __asm__ volatile(
        "li sp, 0x80010000\n"
        "call main\n"
        "1: j 1b\n"
    );
}

// Compute Fibonacci iteratively using 64-bit types only
long fibonacci(long n) {
    if (n <= 1) return n;

    long a = 0;
    long b = 1;
    long i = 2;
    while (i <= n) {
        long c = a + b;
        a = b;
        b = c;
        i = i + 1;
    }
    return b;
}

long main(void) {
    long result = fibonacci(10);
    return result;
}
