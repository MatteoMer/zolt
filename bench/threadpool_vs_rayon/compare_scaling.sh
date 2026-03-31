#!/usr/bin/env bash
#
# compare_scaling.sh — Side-by-side Zig ThreadPool vs Rayon scaling benchmarks
#
# Tests: parallelFor, repeated dispatch, multi-array bind, nested parallel
#
# Usage: bash bench/threadpool_vs_rayon/compare_scaling.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Building Zig scaling benchmark..."
zig build -Doptimize=ReleaseFast 2>/dev/null &
ZIG_PID=$!

echo "Building Rust scaling benchmark..."
cargo build --release --manifest-path bench/threadpool_vs_rayon/Cargo.toml --bin rayon-scaling-bench -q 2>/dev/null &
RUST_PID=$!

wait $ZIG_PID
echo "  Zig build done."
wait $RUST_PID
echo "  Rust build done."

echo ""
echo "Running benchmarks..."
echo ""

ZIG_OUT=$(./zig-out/bin/bench-scaling 2>&1)
RUST_OUT=$(bench/threadpool_vs_rayon/target/release/rayon-scaling-bench 2>&1)

ZIG_THREADS=$(echo "$ZIG_OUT" | grep "Threads:" | grep -oP '[0-9]+')
RUST_THREADS=$(echo "$RUST_OUT" | grep "Threads:" | grep -oP '[0-9]+')

# Extract numeric data lines from a section (between "--- title ---" headers)
extract_section() {
    local output="$1"
    local pattern="$2"
    # Match section start, stop at next "--- " header (but not "----------" separator)
    echo "$output" | awk "
        /--- ${pattern}/{ found=1; next }
        found && /^--- [^-]/{ found=0 }
        found
    " | grep -P '^\s+\d' || true
}

print_comparison() {
    local title="$1"
    local zig_section="$2"
    local rust_section="$3"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $title — Zig ($ZIG_THREADS threads) vs Rayon ($RUST_THREADS threads)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  %-8s │ %8s %8s %6s │ %8s %8s %6s │ %-18s │ %-12s\n" \
        "N" "Zig seq" "Zig par" "ratio" "Rs seq" "Rs par" "ratio" "Abs winner" "Ratio winner"
    echo "  ─────────┼──────────────────────────┼──────────────────────────┼────────────────────┼─────────────"

    readarray -t ZIG_LINES <<< "$zig_section"
    readarray -t RUST_LINES <<< "$rust_section"

    local count=${#ZIG_LINES[@]}
    local rcount=${#RUST_LINES[@]}
    if (( rcount < count )); then count=$rcount; fi

    for i in $(seq 0 $((count - 1))); do
        [[ -z "${ZIG_LINES[$i]}" ]] && continue
        [[ -z "${RUST_LINES[$i]}" ]] && continue

        read -r ZN ZS _ ZP _ ZR _ <<< "${ZIG_LINES[$i]}"
        read -r RN RS _ RP _ RR _ <<< "${RUST_LINES[$i]}"

        ZR=${ZR%x}
        RR=${RR%x}

        if (( $(echo "$ZP == 0 && $RP == 0" | bc -l) )); then
            ABS_WIN="tied"
        elif (( $(echo "$ZP == 0" | bc -l) )); then
            ABS_WIN="Zig (instant)"
        elif (( $(echo "$RP == 0" | bc -l) )); then
            ABS_WIN="Rayon (instant)"
        elif (( $(echo "$ZP < $RP" | bc -l) )); then
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
}

# Extract and compare each benchmark section
ZIG_FOR_LIGHT=$(extract_section "$ZIG_OUT" "parallelFor: u64 in-place")
RUST_FOR_LIGHT=$(extract_section "$RUST_OUT" "parallelFor: u64 in-place")
print_comparison "parallelFor: u64 in-place write" "$ZIG_FOR_LIGHT" "$RUST_FOR_LIGHT"

ZIG_FOR_HEAVY=$(extract_section "$ZIG_OUT" "parallelFor: BN254 field bind")
RUST_FOR_HEAVY=$(extract_section "$RUST_OUT" "parallelFor: BN254 field bind")
print_comparison "parallelFor: BN254 field bind" "$ZIG_FOR_HEAVY" "$RUST_FOR_HEAVY"

ZIG_DISPATCH=$(extract_section "$ZIG_OUT" "Repeated dispatch")
RUST_DISPATCH=$(extract_section "$RUST_OUT" "Repeated dispatch")
print_comparison "Repeated dispatch (per-call avg)" "$ZIG_DISPATCH" "$RUST_DISPATCH"

ZIG_MULTI=$(extract_section "$ZIG_OUT" "Multi-array bind")
RUST_MULTI=$(extract_section "$RUST_OUT" "Multi-array bind")
print_comparison "Multi-array bind (8 arrays)" "$ZIG_MULTI" "$RUST_MULTI"

ZIG_NESTED=$(extract_section "$ZIG_OUT" "Nested parallel")
RUST_NESTED=$(extract_section "$RUST_OUT" "Nested parallel")
print_comparison "Nested parallel (8 arrays x T)" "$ZIG_NESTED" "$RUST_NESTED"

echo ""
echo "Done."
