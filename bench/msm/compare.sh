#!/bin/bash
# MSM Benchmark: Zolt (Zig Pippenger) vs Arkworks (Rust Pippenger)
# Head-to-head comparison at Dory-relevant sizes (2..4096 points)
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== Building Zig MSM benchmark ==="
zig build -Doptimize=ReleaseFast 2>&1 | tail -1 || true

echo ""
echo "=== Building Rust MSM benchmark ==="
cargo build --release --manifest-path bench/msm/Cargo.toml 2>&1 | tail -3

echo ""
echo "============================================"
echo "=== Zig (Zolt Pippenger) ==="
echo "============================================"
ZIG_OUT=$(./zig-out/bin/bench-msm 2>&1)
echo "$ZIG_OUT" | grep -v '^\[MSM-BENCH\]'
echo ""

echo "============================================"
echo "=== Rust (Arkworks Pippenger) ==="
echo "============================================"
RUST_OUT=$(./bench/msm/target/release/msm-bench 2>&1)
echo "$RUST_OUT" | grep -v '^\[MSM-BENCH\]'
echo ""

# ── Parse and compare ──
echo "============================================"
echo "=== HEAD-TO-HEAD COMPARISON ==="
echo "============================================"
echo ""
printf "%-6s │ %-22s │ %-22s │ %-22s\n" \
    "N" "G1-Fr (Zig/Rust)" "G1-i128 (Zig/Rust)" "G2-Fr (Zig/Rust)"
echo "───────┼────────────────────────┼────────────────────────┼────────────────────────"

for n in 2 4 8 16 32 64 128 256 512 1024 2048 4096; do
    # Zig sequential values
    z_line=$(echo "$ZIG_OUT" | grep "^\[MSM-BENCH\] n=$n " | head -1)
    z_g1_fr=$(echo "$z_line" | grep -oP 'g1_fr=\K[0-9.]+' | head -1)
    z_g1_i128=$(echo "$z_line" | grep -oP 'g1_i128=\K[0-9.]+' | head -1)
    z_g2_fr=$(echo "$z_line" | grep -oP 'g2_fr=\K[0-9.]+' | head -1)

    # Rust values
    r_line=$(echo "$RUST_OUT" | grep "^\[MSM-BENCH\] n=$n " | head -1)
    r_g1_fr=$(echo "$r_line" | grep -oP 'g1_fr=\K[0-9.]+' | head -1)
    r_g1_i128=$(echo "$r_line" | grep -oP 'g1_i128=\K[0-9.]+' | head -1)
    r_g2_fr=$(echo "$r_line" | grep -oP 'g2_fr=\K[0-9.]+' | head -1)

    # Compute ratios
    g1_fr_ratio=$(awk "BEGIN{if($r_g1_fr+0>0) printf \"%.2fx\", $z_g1_fr/$r_g1_fr; else print \"N/A\"}")
    g1_i128_ratio=$(awk "BEGIN{if($r_g1_i128+0>0) printf \"%.2fx\", $z_g1_i128/$r_g1_i128; else print \"N/A\"}")
    g2_fr_ratio=$(awk "BEGIN{if($r_g2_fr+0>0) printf \"%.2fx\", $z_g2_fr/$r_g2_fr; else print \"N/A\"}")

    printf "%5d  │ %6.3f/%6.3f %7s │ %6.3f/%6.3f %7s │ %6.3f/%6.3f %7s\n" \
        "$n" \
        "$z_g1_fr" "$r_g1_fr" "$g1_fr_ratio" \
        "$z_g1_i128" "$r_g1_i128" "$g1_i128_ratio" \
        "$z_g2_fr" "$r_g2_fr" "$g2_fr_ratio"
done

echo ""
echo "(ratio = Zig/Rust, <1.0 means Zig is faster)"
echo ""

# ── Parallel comparison (Zig only, since arkworks MSM uses Rayon internally) ──
echo "============================================"
echo "=== ZIG PARALLEL vs SEQUENTIAL ==="
echo "============================================"
echo ""
printf "%-6s │ %-24s │ %-24s │ %-24s\n" \
    "N" "G1-Fr (seq/par)" "G1-i128 (seq/par)" "G2-Fr (seq/par)"
echo "───────┼──────────────────────────┼──────────────────────────┼──────────────────────────"

for n in 2 4 8 16 32 64 128 256 512 1024 2048 4096; do
    z_line=$(echo "$ZIG_OUT" | grep "^\[MSM-BENCH\] n=$n " | head -1)
    z_g1_fr=$(echo "$z_line" | grep -oP 'g1_fr=\K[0-9.]+' | head -1)
    z_g1_i128=$(echo "$z_line" | grep -oP 'g1_i128=\K[0-9.]+' | head -1)
    z_g2_fr=$(echo "$z_line" | grep -oP 'g2_fr=\K[0-9.]+' | head -1)
    z_g1_fr_par=$(echo "$z_line" | grep -oP 'g1_fr_par=\K[0-9.]+' | head -1)
    z_g1_i128_par=$(echo "$z_line" | grep -oP 'g1_i128_par=\K[0-9.]+' | head -1)
    z_g2_fr_par=$(echo "$z_line" | grep -oP 'g2_fr_par=\K[0-9.]+' | head -1)

    g1_speedup=$(awk "BEGIN{if($z_g1_fr_par+0>0) printf \"%.2fx\", $z_g1_fr/$z_g1_fr_par; else print \"N/A\"}")
    g1i_speedup=$(awk "BEGIN{if($z_g1_i128_par+0>0) printf \"%.2fx\", $z_g1_i128/$z_g1_i128_par; else print \"N/A\"}")
    g2_speedup=$(awk "BEGIN{if($z_g2_fr_par+0>0) printf \"%.2fx\", $z_g2_fr/$z_g2_fr_par; else print \"N/A\"}")

    printf "%5d  │ %6.3f→%6.3f %7s │ %6.3f→%6.3f %7s │ %6.3f→%6.3f %7s\n" \
        "$n" \
        "$z_g1_fr" "$z_g1_fr_par" "$g1_speedup" \
        "$z_g1_i128" "$z_g1_i128_par" "$g1i_speedup" \
        "$z_g2_fr" "$z_g2_fr_par" "$g2_speedup"
done

echo ""
echo "(speedup = seq/par, >1.0 means parallel is faster)"
