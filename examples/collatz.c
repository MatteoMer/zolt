// Collatz sequence example for Zolt zkVM
// Computes the length of the Collatz sequence starting from n=27
// (known to require 111 steps to reach 1)
//
// The Collatz conjecture: Starting from any positive integer n:
// - If n is even: n = n / 2
// - If n is odd: n = 3*n + 1
// Eventually reaches 1

// Declarations for libzolt runtime
void zolt_io_write_u64(unsigned long value);
unsigned long zolt_io_read_u64(void);

int main(void) {
    // Start with n = 27, which has a sequence of 111 steps
    unsigned long n = 27;
    unsigned long steps = 0;
    unsigned long max_value = n;  // Track the highest value reached

    // Count steps until we reach 1
    while (n > 1) {
        if (n % 2 == 0) {
            // Even: divide by 2
            n = n / 2;
        } else {
            // Odd: 3n + 1
            n = 3 * n + 1;
        }

        steps++;

        // Track maximum value
        if (n > max_value) {
            max_value = n;
        }
    }

    // Output the number of steps
    // For n=27, this should be 111
    zolt_io_write_u64(steps);

    return 0;
}
