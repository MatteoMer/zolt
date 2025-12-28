// Prime number sieve example for Zolt zkVM
// Counts the number of primes less than 100 using trial division

// Declarations for libzolt runtime
void zolt_io_write_u64(unsigned long value);
unsigned long zolt_io_read_u64(void);

// Check if n is prime using trial division
int is_prime(unsigned long n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;

    // Check odd divisors up to sqrt(n)
    // Since we don't have sqrt, we check until i*i > n
    unsigned long i = 3;
    while (i * i <= n) {
        if (n % i == 0) return 0;
        i += 2;
    }
    return 1;
}

int main(void) {
    unsigned long count = 0;
    unsigned long limit = 100;

    // Count primes less than limit
    for (unsigned long n = 2; n < limit; n++) {
        if (is_prime(n)) {
            count++;
        }
    }

    // There are 25 primes less than 100:
    // 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    // 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    zolt_io_write_u64(count);

    return 0;
}
