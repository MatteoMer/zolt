// Signed arithmetic example for Zolt zkVM
// Demonstrates signed integer operations and comparisons

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

int main(void) {
    // Signed addition with negative numbers
    int a = -10;
    int b = 25;
    int sum = a + b;  // Should be 15

    // Signed multiplication
    int c = -7;
    int d = 6;
    int product = c * d;  // Should be -42

    // Signed division
    int e = -100;
    int f = 7;
    int quotient = e / f;  // Should be -14 (truncated toward zero)

    // Signed comparisons using SLT instruction
    int cmp1 = (a < b) ? 1 : 0;  // -10 < 25 => 1
    int cmp2 = (e < a) ? 1 : 0;  // -100 < -10 => 1
    int cmp3 = (d < c) ? 1 : 0;  // 6 < -7 => 0

    // Compute a result combining all operations
    // sum=15, product=-42, quotient=-14
    // 15 + (-42) + (-14) = -41
    int result = sum + product + quotient;

    // Add comparison results: 1 + 1 + 0 = 2
    result = result + cmp1 + cmp2 + cmp3;  // -41 + 2 = -39

    // Return the result (-39 in two's complement)
    return result;
}
