// GCD (Greatest Common Divisor) example for Zolt zkVM
// Uses Euclidean algorithm with modulo operation
//
// Compile with:
//   riscv32-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -O2 \
//     -Wl,-Ttext=0x80000000 -o gcd.elf gcd.c

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

// Euclidean GCD algorithm using modulo
// Uses REM instruction from M extension
int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;  // Uses REM instruction
        a = t;
    }
    return a;
}

// LCM using GCD: lcm(a,b) = a * b / gcd(a,b)
int lcm(int a, int b) {
    return (a * b) / gcd(a, b);  // Uses MUL and DIV instructions
}

int main(void) {
    // gcd(48, 18) = 6
    int g1 = gcd(48, 18);

    // gcd(252, 105) = 21
    int g2 = gcd(252, 105);

    // lcm(12, 18) = 36
    int l = lcm(12, 18);

    // Return combined result
    return g1 + g2 + l;  // 6 + 21 + 36 = 63
}
