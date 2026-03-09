#!/usr/bin/env bash
#
# flamegraph-compare.sh — Benchmark Zolt vs Jolt (Rust) with flamegraphs
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

PROGRAMS=(fibonacci factorial bitwise collatz primes sum gcd signed primes_large)
PERF_FREQ=997

# ── Prerequisites ──────────────────────────────────────────────────────────

check_prereqs() {
    local missing=0
    for cmd in perf inferno-collapse-perf inferno-flamegraph; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "[ERR] Missing: $cmd"
            missing=1
        fi
    done
    [ ! -x "$ZOLT_BIN" ] && echo "[ERR] Zolt not found — run: zig build -Doptimize=ReleaseFast" && missing=1
    [ ! -x "$JOLT_BIN" ] && echo "[ERR] Jolt not found — run: cd jolt-bench && cargo build --release" && missing=1
    [ $missing -ne 0 ] && exit 1
}

# ── Profile + flamegraph ───────────────────────────────────────────────────

profile_one() {
    local label="$1" prefix="$2"
    shift 2

    perf record -g --call-graph fp -F "$PERF_FREQ" \
        -o "${prefix}.perf.data" -- "$@" \
        > "${prefix}.output.txt" 2>&1

    perf script -i "${prefix}.perf.data" 2>/dev/null \
        | inferno-collapse-perf 2>/dev/null \
        > "${prefix}.collapsed"

    inferno-flamegraph --title "$label" --countname samples \
        < "${prefix}.collapsed" \
        > "${prefix}.svg" 2>/dev/null

    rm -f "${prefix}.perf.data"
}

# ── Bench one program ─────────────────────────────────────────────────────

bench_program() {
    local prog="$1"
    local elf="$ROOT/examples/${prog}.elf"
    local d="$OUTDIR/$prog"

    [ ! -f "$elf" ] && echo "[ERR] ELF not found: $elf" && return 1

    mkdir -p "$d"

    printf "  %-12s  " "$prog"

    # Zolt
    profile_one "Zolt - $prog" "$d/zolt" \
        "$ZOLT_BIN" prove "$elf" \
            --jolt-format -o "$d/zolt_proof.bin" \
            --export-preprocessing "$d/zolt_preproc.bin"

    # Jolt
    profile_one "Jolt (Rust) - $prog" "$d/jolt" \
        "$JOLT_BIN" "$elf"

    # Extract timings
    local zt jt
    zt=$(grep -oP 'Total time: \K[0-9.]+' "$d/zolt.output.txt" 2>/dev/null || echo "0")
    jt=$(grep -oP 'Total:\s+\K[0-9.]+' "$d/jolt.output.txt" 2>/dev/null || echo "0")

    # Store for summary
    ZOLT_TIMES[$prog]="$zt"
    JOLT_TIMES[$prog]="$jt"

    printf "done\n"
}

# ── Results table ─────────────────────────────────────────────────────────

print_results() {
    local report="$OUTDIR/report.txt"

    local total_z=0 total_j=0

    # Header
    local hdr
    hdr=$(printf "%-12s │ %10s │ %10s │ %7s │ %s" "Program" "Zolt (ms)" "Jolt (ms)" "Ratio" "")
    local sep="─────────────┼────────────┼────────────┼─────────┼────────────"

    echo ""
    echo "  $sep"
    echo "  $(printf '%-12s │ %10s │ %10s │ %7s │ %s' 'Program' 'Zolt (ms)' 'Jolt (ms)' 'Ratio' 'Winner')"
    echo "  $sep"

    for prog in "${run_programs[@]}"; do
        local zt="${ZOLT_TIMES[$prog]}"
        local jt="${JOLT_TIMES[$prog]}"
        local ratio winner

        if [ "$jt" != "0" ] && [ "$zt" != "0" ]; then
            ratio=$(awk "BEGIN{printf \"%.2fx\", $zt/$jt}")
            if (( $(awk "BEGIN{print ($zt < $jt) ? 1 : 0}") )); then
                winner="Zolt"
            else
                winner="Jolt"
            fi
            total_z=$(awk "BEGIN{printf \"%.2f\", $total_z + $zt}")
            total_j=$(awk "BEGIN{printf \"%.2f\", $total_j + $jt}")
        else
            ratio="N/A"
            winner="—"
        fi

        printf "  %-12s │ %10s │ %10s │ %7s │ %s\n" "$prog" "$zt" "$jt" "$ratio" "$winner"
    done

    echo "  $sep"

    if [ "$total_j" != "0" ]; then
        local total_ratio
        total_ratio=$(awk "BEGIN{printf \"%.2fx\", $total_z/$total_j}")
        printf "  %-12s │ %10s │ %10s │ %7s │\n" "TOTAL" "$total_z" "$total_j" "$total_ratio"
    fi

    echo "  $sep"
    echo ""

    # Save plain-text report
    {
        printf "%-12s  %12s  %12s  %10s\n" "Program" "Zolt (ms)" "Jolt (ms)" "Ratio"
        printf "%-12s  %12s  %12s  %10s\n" "--------" "---------" "---------" "------"
        for prog in "${run_programs[@]}"; do
            local zt="${ZOLT_TIMES[$prog]}"
            local jt="${JOLT_TIMES[$prog]}"
            local ratio="N/A"
            [ "$jt" != "0" ] && [ "$zt" != "0" ] && ratio=$(awk "BEGIN{printf \"%.2fx\", $zt/$jt}")
            printf "%-12s  %12s  %12s  %10s\n" "$prog" "$zt" "$jt" "$ratio"
        done
    } > "$report"

    echo "  Report:      $report"
    echo "  Flamegraphs:"
    for prog in "${run_programs[@]}"; do
        echo "    $OUTDIR/$prog/zolt.svg"
        echo "    $OUTDIR/$prog/jolt.svg"
    done
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────

check_prereqs
mkdir -p "$OUTDIR"

declare -A ZOLT_TIMES JOLT_TIMES
declare -a run_programs

if [ $# -ge 1 ]; then
    run_programs=("$1")
else
    run_programs=("${PROGRAMS[@]}")
fi

echo ""
echo "  Benchmarking Zolt vs Jolt (${#run_programs[@]} programs)"
echo ""

for prog in "${run_programs[@]}"; do
    bench_program "$prog"
done

print_results
