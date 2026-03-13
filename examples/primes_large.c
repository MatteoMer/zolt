// Large prime counting for benchmarking (limit=1000, ~50k+ trace steps)
void _start(void) __attribute__((naked));

void _start(void) {
    __asm__ volatile(
        "li sp, 0x80010000\n"
        "call main\n"
        "1: j 1b\n"
    );
}

int is_prime(unsigned int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    unsigned int i = 3;
    while (i * i <= n) {
        if (n % i == 0) return 0;
        i += 2;
    }
    return 1;
}

int main(void) {
    unsigned int count = 0;
    unsigned int limit = 1000;
    for (unsigned int n = 2; n < limit; n++) {
        if (is_prime(n)) {
            count++;
        }
    }
    return (int)count;  // 168 primes below 1000
}
