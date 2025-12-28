// Collatz sequence example for Zolt zkVM
// Computes the length of the Collatz sequence starting from n=27
// (known to require 111 steps to reach 1)
//
// The Collatz conjecture: Starting from any positive integer n:
// - If n is even: n = n / 2
// - If n is odd: n = 3*n + 1
// Eventually reaches 1

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
    // Start with n = 27, which has a sequence of 111 steps
    unsigned int n = 27;
    unsigned int steps = 0;

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
    }

    // Return the number of steps (should be 111)
    return (int)steps;
}
