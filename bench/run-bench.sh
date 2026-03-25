#!/usr/bin/env bash
#
# run-bench.sh — Quick timing benchmark Zolt vs Jolt (no flamegraphs)
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZOLT_BIN="$ROOT/zig-out/bin/zolt"
JOLT_BIN="$ROOT/jolt-bench/target/release/jolt-bench"

PROGRAMS=(fibonacci factorial bitwise collatz primes sum gcd signed primes_large sha256_128 sha256 sha256_512 sha256_1024 sha256_2048)

if [ $# -ge 1 ]; then
    PROGRAMS=("$@")
fi

declare -A ZOLT_TOTAL JOLT_TOTAL JOLT_PROVE ZOLT_PROVE CYCLES TRACE_LEN

echo ""
echo "  Benchmarking Zolt vs Jolt (${#PROGRAMS[@]} programs)"
echo ""

for prog in "${PROGRAMS[@]}"; do
    elf="$ROOT/examples/${prog}.elf"
    [ ! -f "$elf" ] && echo "  [SKIP] $elf not found" && continue

    printf "  %-14s " "$prog"

    # --- Zolt ---
    zolt_out=$("$ZOLT_BIN" prove "$elf" --jolt-format -o /tmp/zolt_bench_proof.bin --export-preprocessing /tmp/zolt_bench_preproc.bin 2>&1)
    zt=$(echo "$zolt_out" | grep -oP '^\s+Time: \K[0-9.]+' || echo "0")
    zp=$(echo "$zolt_out" | grep -oP '^\s+Time: \K[0-9.]+' || echo "0")
    ZOLT_TOTAL[$prog]="$zt"
    ZOLT_PROVE[$prog]="$zp"

    # --- Jolt ---
    jolt_out=$("$JOLT_BIN" "$elf" 2>&1)
    jt=$(echo "$jolt_out" | grep -oP 'Total:\s+\K[0-9.]+' || echo "0")
    jp=$(echo "$jolt_out" | grep -oP 'Prove:\s+\K[0-9.]+' || echo "0")
    cy=$(echo "$jolt_out" | grep -oP '\((\K[0-9]+)(?= cycles)' || echo "?")
    tl=$(echo "$jolt_out" | grep -oP 'padded to \K[0-9]+' || echo "?")
    JOLT_TOTAL[$prog]="$jt"
    JOLT_PROVE[$prog]="$jp"
    CYCLES[$prog]="$cy"
    TRACE_LEN[$prog]="$tl"

    printf "done\n"
done

# ── Results ──
sep="──────────────┼─────────┼───────────┼────────────┼────────────┼─────────┼────────────"

echo ""
echo "  $sep"
printf "  %-13s │ %7s │ %9s │ %10s │ %10s │ %7s │ %10s\n" \
    "Program" "Cycles" "Trace Len" "Zolt (ms)" "Jolt (ms)" "Ratio" "Diff (ms)"
echo "  $sep"

total_z=0
total_j=0

for prog in "${PROGRAMS[@]}"; do
    [ -z "${ZOLT_TOTAL[$prog]+x}" ] && continue
    zt="${ZOLT_TOTAL[$prog]}"
    jt="${JOLT_TOTAL[$prog]}"
    cy="${CYCLES[$prog]}"
    tl="${TRACE_LEN[$prog]}"

    if [ "$jt" != "0" ] && [ "$zt" != "0" ]; then
        ratio=$(awk "BEGIN{printf \"%.2fx\", $zt/$jt}")
        diff=$(awk "BEGIN{d=$zt-$jt; printf \"%+.0f\", d}")
        total_z=$(awk "BEGIN{printf \"%.2f\", $total_z + $zt}")
        total_j=$(awk "BEGIN{printf \"%.2f\", $total_j + $jt}")
    else
        ratio="N/A"
        diff="N/A"
    fi

    printf "  %-13s │ %7s │ %9s │ %10s │ %10s │ %7s │ %10s\n" \
        "$prog" "$cy" "$tl" "$zt" "$jt" "$ratio" "$diff"
done

echo "  $sep"

if [ "$total_j" != "0" ]; then
    total_ratio=$(awk "BEGIN{printf \"%.2fx\", $total_z/$total_j}")
    total_diff=$(awk "BEGIN{d=$total_z-$total_j; printf \"%+.0f\", d}")
    printf "  %-13s │ %7s │ %9s │ %10s │ %10s │ %7s │ %10s\n" \
        "TOTAL" "" "" "$total_z" "$total_j" "$total_ratio" "$total_diff"
fi

echo "  $sep"
echo ""
