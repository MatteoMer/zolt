#!/usr/bin/env bash
#
# flamegraph-compare.sh — Profile Zolt vs Jolt (Rust) and generate flamegraphs
#
# Usage:
#   ./bench/flamegraph-compare.sh [program]
#
# Examples:
#   ./bench/flamegraph-compare.sh fibonacci
#   ./bench/flamegraph-compare.sh              # runs all programs
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTDIR="$ROOT/bench/results"
ZOLT_BIN="$ROOT/zig-out/bin/zolt"
JOLT_BIN="$ROOT/jolt-bench/target/release/jolt-bench"

PROGRAMS=(fibonacci factorial bitwise collatz primes sum gcd signed)
PERF_FREQ=997

# Check prerequisites
check_prereqs() {
    local missing=0
    for cmd in perf inferno-collapse-perf inferno-flamegraph; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "[ERR] Missing: $cmd"
            missing=1
        fi
    done
    [ ! -x "$ZOLT_BIN" ] && echo "[ERR] Zolt binary not found — run: zig build -Doptimize=ReleaseSafe" && missing=1
    [ ! -x "$JOLT_BIN" ] && echo "[ERR] Jolt bench not found — run: cd jolt-bench && cargo build --release" && missing=1
    [ $missing -ne 0 ] && exit 1
    return 0
}

# Profile a command and generate flamegraph
# Usage: profile_one LABEL PREFIX CMD [ARGS...]
profile_one() {
    local label="$1" prefix="$2"
    shift 2

    echo "[....] Profiling: $label"

    # Record with perf (frame-pointer unwinding for speed)
    perf record -g --call-graph fp -F "$PERF_FREQ" \
        -o "${prefix}.perf.data" -- "$@" \
        > "${prefix}.output.txt" 2>&1

    # Collapse stacks
    perf script -i "${prefix}.perf.data" 2>/dev/null \
        | inferno-collapse-perf 2>/dev/null \
        > "${prefix}.collapsed"

    # Generate SVG flamegraph
    inferno-flamegraph --title "$label" --countname samples \
        < "${prefix}.collapsed" \
        > "${prefix}.svg" 2>/dev/null

    local nstacks
    nstacks=$(wc -l < "${prefix}.collapsed")
    echo "[ OK ] ${prefix}.svg ($nstacks stacks)"

    # Cleanup large perf data
    rm -f "${prefix}.perf.data"
}

# Print top functions from collapsed stacks
print_top() {
    local collapsed="$1" label="$2"
    [ ! -s "$collapsed" ] && return 0

    echo ""
    echo "Top functions ($label):"
    awk '{
        sub(/ [0-9]+$/, "")
        n = split($0, f, ";")
        leaf = f[n]
        # Get count from original line
    }' < /dev/null  # placeholder

    # Proper parsing: each line is "stack;stack;leaf COUNT"
    while IFS= read -r line; do
        local count="${line##* }"
        local stack="${line% *}"
        local leaf="${stack##*;}"
        echo "$count $leaf"
    done < "$collapsed" \
        | awk '{a[$2]+=$1} END{for(k in a) print a[k], k}' \
        | sort -rn \
        | head -15 \
        | while IFS= read -r row; do
            printf "  %10s  %s\n" "${row%% *}" "${row#* }"
        done
}

# Benchmark one program
bench_program() {
    local prog="$1"
    local elf="$ROOT/examples/${prog}.elf"
    local d="$OUTDIR/$prog"

    [ ! -f "$elf" ] && echo "[ERR] ELF not found: $elf" && return 1

    echo ""
    echo "=========================================="
    echo "  $prog"
    echo "=========================================="
    mkdir -p "$d"

    # Zolt
    profile_one "Zolt - $prog" "$d/zolt" \
        "$ZOLT_BIN" prove "$elf" \
            --jolt-format -o "$d/zolt_proof.bin" \
            --export-preprocessing "$d/zolt_preproc.bin"

    # Jolt (Rust)
    profile_one "Jolt (Rust) - $prog" "$d/jolt" \
        "$JOLT_BIN" "$elf"

    # Top functions
    print_top "$d/zolt.collapsed" "zolt"
    print_top "$d/jolt.collapsed" "jolt"

    # Timing
    local zolt_time jolt_time
    zolt_time=$(grep -oP 'Total time: \K[0-9.]+' "$d/zolt.output.txt" 2>/dev/null || echo "N/A")
    jolt_time=$(grep -oP 'Total:\s+\K[0-9.]+' "$d/jolt.output.txt" 2>/dev/null || echo "N/A")

    echo ""
    echo "  Timing:"
    printf "    Zolt:        %10s ms\n" "$zolt_time"
    printf "    Jolt (Rust): %10s ms\n" "$jolt_time"
}

# Summary
generate_report() {
    local report="$OUTDIR/report.txt"

    echo ""
    echo "=========================================="
    echo "  Summary"
    echo "=========================================="

    (
        printf "%-12s  %12s  %12s  %10s\n" "Program" "Zolt (ms)" "Jolt (ms)" "Ratio"
        printf "%-12s  %12s  %12s  %10s\n" "--------" "---------" "---------" "------"

        for prog in "${run_programs[@]}"; do
            local d="$OUTDIR/$prog"
            local zt jt ratio

            zt=$(grep -oP 'Total time: \K[0-9.]+' "$d/zolt.output.txt" 2>/dev/null || echo "0")
            jt=$(grep -oP 'Total:\s+\K[0-9.]+' "$d/jolt.output.txt" 2>/dev/null || echo "0")

            if [ "$jt" != "0" ] && [ "$zt" != "0" ]; then
                ratio=$(awk "BEGIN{printf \"%.2fx\", $zt/$jt}")
            else
                ratio="N/A"
            fi

            printf "%-12s  %12s  %12s  %10s\n" "$prog" "$zt" "$jt" "$ratio"
        done
    ) | tee "$report"

    echo ""
    echo "Report: $report"
    echo ""
    echo "Flamegraphs:"
    for prog in "${run_programs[@]}"; do
        echo "  $OUTDIR/$prog/zolt.svg"
        echo "  $OUTDIR/$prog/jolt.svg"
    done
}

# --- Main ---
check_prereqs
mkdir -p "$OUTDIR"

declare -a run_programs
if [ $# -ge 1 ]; then
    run_programs=("$1")
else
    run_programs=("${PROGRAMS[@]}")
fi

for prog in "${run_programs[@]}"; do
    bench_program "$prog"
done

generate_report
