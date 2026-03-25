#!/usr/bin/env bash
#
# compare.sh — Side-by-side ThreadPool vs Rayon parallel reduce benchmark
#
# Usage: bash bench/threadpool_vs_rayon/compare.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Building Zig benchmark..."
zig build bench-tp -Doptimize=ReleaseFast 2>/dev/null &
ZIG_PID=$!

echo "Building Rust benchmark..."
cargo build --release --manifest-path bench/threadpool_vs_rayon/Cargo.toml -q 2>/dev/null &
RUST_PID=$!

wait $ZIG_PID
wait $RUST_PID

echo ""
echo "Running benchmarks (50 iterations each)..."
echo ""

ZIG_OUT=$(zig build bench-tp -Doptimize=ReleaseFast 2>&1)
RUST_OUT=$(bench/threadpool_vs_rayon/target/release/rayon-bench 2>&1)

# Parse Zig results
ZIG_THREADS=$(echo "$ZIG_OUT" | grep "Threads:" | grep -oP '[0-9]+')
readarray -t ZIG_LINES < <(echo "$ZIG_OUT" | grep -P '^\s+\d')

# Parse Rust results
RUST_THREADS=$(echo "$RUST_OUT" | grep "Threads:" | grep -oP '[0-9]+')
readarray -t RUST_LINES < <(echo "$RUST_OUT" | grep -P '^\s+\d')

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Parallel Reduce: Zig ThreadPool ($ZIG_THREADS threads) vs Rust Rayon ($RUST_THREADS threads)"
echo "  Workload: Σ a[i]*b[i] over BN254 field element pairs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "  %-8s │ %8s %8s %6s │ %8s %8s %6s │ %-18s │ %-12s\n" \
    "N" "Zig seq" "Zig par" "ratio" "Rs seq" "Rs par" "ratio" "Abs winner" "Ratio winner"
echo "  ─────────┼──────────────────────────┼──────────────────────────┼────────────────────┼─────────────"

for i in $(seq 0 $((${#ZIG_LINES[@]} - 1))); do
    # Parse Zig line
    read -r ZN ZS _ ZP _ ZR _ <<< "${ZIG_LINES[$i]}"
    # Parse Rust line
    read -r RN RS _ RP _ RR _ <<< "${RUST_LINES[$i]}"

    # Remove 'x' suffix from ratios
    ZR=${ZR%x}
    RR=${RR%x}

    # Determine winners
    if (( $(echo "$ZP < $RP" | bc -l) )); then
        ABS_WIN="Zig $(echo "scale=1; $RP / $ZP" | bc)x faster"
    elif (( $(echo "$ZP > $RP" | bc -l) )); then
        ABS_WIN="Rayon $(echo "scale=1; $ZP / $RP" | bc)x faster"
    else
        ABS_WIN="tied"
    fi

    if (( $(echo "$ZR > $RR" | bc -l) )); then
        RATIO_WIN="Zig"
    elif (( $(echo "$ZR < $RR" | bc -l) )); then
        RATIO_WIN="Rayon"
    else
        RATIO_WIN="tied"
    fi

    printf "  %-8s │ %7s %7s %5sx │ %7s %7s %5sx │ %-18s │ %-12s\n" \
        "$ZN" "$ZS" "$ZP" "$ZR" "$RS" "$RP" "$RR" "$ABS_WIN" "$RATIO_WIN"
done

echo "  ─────────┼──────────────────────────┼──────────────────────────┼────────────────────┼─────────────"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
