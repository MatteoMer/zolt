// Signed arithmetic example for Zolt zkVM
// Demonstrates signed integer operations and comparisons

// Declarations for libzolt runtime
void zolt_io_write_u64(unsigned long value);
unsigned long zolt_io_read_u64(void);

// Cast to unsigned for output
static inline unsigned long to_unsigned(long x) {
    return (unsigned long)x;
}

int main(void) {
    // Signed addition with negative numbers
    long a = -10;
    long b = 25;
    long sum = a + b;  // Should be 15

    // Signed subtraction resulting in negative
    long diff = a - b;  // Should be -35

    // Signed multiplication
    long c = -7;
    long d = 6;
    long product = c * d;  // Should be -42

    // Signed division
    long e = -100;
    long f = 7;
    long quotient = e / f;  // Should be -14 (truncated toward zero)
    long remainder = e % f;  // Should be -2

    // Signed comparisons using SLT instruction
    int cmp1 = (a < b) ? 1 : 0;  // -10 < 25 => 1
    int cmp2 = (e < a) ? 1 : 0;  // -100 < -10 => 1
    int cmp3 = (d < c) ? 1 : 0;  // 6 < -7 => 0

    // Compute a result combining all operations
    // sum=15, product=-42, quotient=-14
    // 15 + (-42) + (-14) = -41
    long result = sum + product + quotient;

    // Add comparison results: 1 + 1 + 0 = 2
    result = result + cmp1 + cmp2 + cmp3;  // -41 + 2 = -39

    // For zkVM output, we cast to unsigned
    // -39 in two's complement (64-bit): 0xFFFFFFFFFFFFFFD9
    zolt_io_write_u64(to_unsigned(result));

    return 0;
}
