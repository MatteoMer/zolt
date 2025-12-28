// Array operations example for Zolt zkVM
// Demonstrates load/store operations with arrays
//
// Compile with:
//   riscv32-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -O2 \
//     -Wl,-Ttext=0x80000000 -o array.elf array.c

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

// Global array in BSS
int arr[16];

// Initialize array with values
void init_array(int *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = i * i;  // SW instruction
    }
}

// Sum array elements
int sum_array(int *a, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];  // LW instruction
    }
    return sum;
}

// Find maximum value
int max_array(int *a, int n) {
    int max = a[0];
    for (int i = 1; i < n; i++) {
        if (a[i] > max) {
            max = a[i];  // Uses SLT for comparison
        }
    }
    return max;
}

int main(void) {
    // Initialize array with squares: 0, 1, 4, 9, 16, 25, 36, 49, ...
    init_array(arr, 16);

    // Sum = 0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 + 100 + 121 + 144 + 169 + 196 + 225
    //     = 1240
    int sum = sum_array(arr, 16);

    // Max = 15^2 = 225
    int max = max_array(arr, 16);

    return sum + max;  // 1240 + 225 = 1465
}
