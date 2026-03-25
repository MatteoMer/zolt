#!/usr/bin/env bash
#
# run-bench.sh — Quick timing benchmark Zolt vs Jolt (no flamegraphs)
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZOLT_BIN="$ROOT/zig-out/bin/zolt"
JOLT_BIN="$ROOT/jolt-bench/target/release/jolt-bench"
VERIFIER_BIN="$ROOT/jolt-verifier/target/release/jolt-verifier"

PROGRAMS=(fibonacci factorial bitwise collatz primes sum gcd signed primes_large sha256_128 sha256 sha256_512 sha256_1024 sha256_2048)

if [ $# -ge 1 ]; then
    PROGRAMS=("$@")
fi

# --- Build binaries if needed ---
if [ ! -f "$ZOLT_BIN" ]; then
    echo "  Building Zolt..."
    (cd "$ROOT" && zig build -Doptimize=ReleaseFast) || { echo "ERROR: Zolt build failed"; exit 1; }
fi

if [ ! -f "$JOLT_BIN" ]; then
    echo "  Building Jolt bench (release)..."
    (cd "$ROOT/jolt-bench" && cargo build --release) || { echo "ERROR: Jolt bench build failed"; exit 1; }
fi

if [ ! -f "$VERIFIER_BIN" ]; then
    echo "  Building Jolt verifier (release)..."
    (cd "$ROOT/jolt-verifier" && cargo build --release) || { echo "ERROR: Jolt verifier build failed"; exit 1; }
fi

# --- OS-portable helpers ---
# grep -oP (PCRE) is GNU-only; use sed as a portable alternative.
extract() {
    # Usage: extract "pattern with (capture group)" <<< "$text"
    # Returns the first capture group match, or the fallback.
    local pattern="$1" fallback="${2:-0}"
    sed -nE "s/.*$pattern.*/\1/p" | head -1 | grep . || echo "$fallback"
}

# Use temp files instead of associative arrays (bash 3 compat)
RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT

echo ""
echo "  Benchmarking Zolt vs Jolt (${#PROGRAMS[@]} programs)"
echo ""

for prog in "${PROGRAMS[@]}"; do
    elf="$ROOT/examples/${prog}.elf"
    [ ! -f "$elf" ] && echo "  [SKIP] $elf not found" && continue

    printf "  %-14s " "$prog"

    # --- Zolt ---
    zolt_out=$("$ZOLT_BIN" prove "$elf" --jolt-format -o /tmp/zolt_bench_proof.bin --export-preprocessing /tmp/zolt_bench_preproc.bin 2>&1)
    zt=$(echo "$zolt_out" | extract 'Time: ([0-9.]+)')
    zp="$zt"
    echo "$zt" > "$RESULTS_DIR/${prog}_zt"
    echo "$zp" > "$RESULTS_DIR/${prog}_zp"

    # --- Verify Zolt proof ---
    if "$VERIFIER_BIN" --proof /tmp/zolt_bench_proof.bin --preprocessing /tmp/zolt_bench_preproc.bin >/dev/null 2>&1; then
        echo "VERIFIED" > "$RESULTS_DIR/${prog}_verified"
    else
        echo "FAILED" > "$RESULTS_DIR/${prog}_verified"
    fi

    # --- Jolt ---
    jolt_out=$("$JOLT_BIN" "$elf" 2>&1)
    jt=$(echo "$jolt_out" | extract 'Total:[[:space:]]+([0-9.]+)')
    jp=$(echo "$jolt_out" | extract 'Prove:[[:space:]]+([0-9.]+)')
    cy=$(echo "$jolt_out" | extract '\(([0-9]+) cycles' "?")
    tl=$(echo "$jolt_out" | extract 'padded to ([0-9]+)' "?")
    echo "$jt" > "$RESULTS_DIR/${prog}_jt"
    echo "$jp" > "$RESULTS_DIR/${prog}_jp"
    echo "$cy" > "$RESULTS_DIR/${prog}_cy"
    echo "$tl" > "$RESULTS_DIR/${prog}_tl"

    verified=$(cat "$RESULTS_DIR/${prog}_verified")
    printf "done (%s)\n" "$verified"
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
    [ ! -f "$RESULTS_DIR/${prog}_zt" ] && continue
    zt=$(cat "$RESULTS_DIR/${prog}_zt")
    jt=$(cat "$RESULTS_DIR/${prog}_jt")
    cy=$(cat "$RESULTS_DIR/${prog}_cy")
    tl=$(cat "$RESULTS_DIR/${prog}_tl")

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
