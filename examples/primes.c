// Prime number counting example for Zolt zkVM
// Counts the number of primes less than 100 using trial division

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

// Check if n is prime using trial division
int is_prime(unsigned int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;

    // Check odd divisors up to sqrt(n)
    // Since we don't have sqrt, we check until i*i > n
    unsigned int i = 3;
    while (i * i <= n) {
        if (n % i == 0) return 0;
        i += 2;
    }
    return 1;
}

int main(void) {
    unsigned int count = 0;
    unsigned int limit = 100;

    // Count primes less than limit
    for (unsigned int n = 2; n < limit; n++) {
        if (is_prime(n)) {
            count++;
        }
    }

    // There are 25 primes less than 100:
    // 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    // 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    return (int)count;
}
