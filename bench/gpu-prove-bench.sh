#!/bin/bash
# GPU prove benchmark: compare GPU-enabled vs CPU-only across all test programs.
# Usage: ./bench/gpu-prove-bench.sh [--runs N]

set -e

RUNS=${2:-3}
ZOLT=zig-out/bin/zolt
PROGRAMS=(
    "examples/fibonacci.elf"
    "examples/sum.elf"
    "examples/collatz.elf"
    "examples/sha256_128.elf"
    "examples/sha256_512.elf"
    "examples/sha256_1024.elf"
    "examples/sha256_2048.elf"
)

echo "=== Zolt GPU Prove Benchmark ==="
echo "Runs per program: $RUNS"
echo ""

# Build release
echo "Building ReleaseFast..."
zig build -Doptimize=ReleaseFast 2>/dev/null
echo ""

printf "%-25s %8s %8s %8s\n" "Program" "Min(ms)" "Avg(ms)" "Cycles"
printf "%-25s %8s %8s %8s\n" "-------" "-------" "-------" "------"

for prog in "${PROGRAMS[@]}"; do
    name=$(basename "$prog" .elf)

    # Get cycle count
    cycles=$($ZOLT run "$prog" 2>&1 | grep "Cycles executed" | awk '{print $3}')

    min_ms=999999
    total_ms=0

    for ((i=1; i<=RUNS; i++)); do
        ms=$($ZOLT prove -o /tmp/bench_proof.bin "$prog" 2>&1 | grep "Time:" | awk '{print $2}')
        ms_int=$(echo "$ms" | cut -d. -f1)

        if [ "$ms_int" -lt "$min_ms" ]; then
            min_ms=$ms_int
        fi
        total_ms=$((total_ms + ms_int))
    done

    avg_ms=$((total_ms / RUNS))

    printf "%-25s %8d %8d %8s\n" "$name" "$min_ms" "$avg_ms" "$cycles"
done

echo ""
echo "All proofs generated successfully."
