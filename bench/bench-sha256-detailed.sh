#!/usr/bin/env bash
#
# bench-sha256-detailed.sh — Ultra-granular Zolt vs Jolt stage-by-stage benchmark
#
# Runs both provers on a SHA256 ELF and produces a hierarchical
# comparison showing every measurable sub-operation.
#
# Usage:
#   ./bench/bench-sha256-detailed.sh                       # sha256_1024 (default)
#   ./bench/bench-sha256-detailed.sh sha256_2048           # specific program
#   RUNS=3 ./bench/bench-sha256-detailed.sh sha256_1024    # best-of-3
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZOLT_BIN="$ROOT/zig-out/bin/zolt"
JOLT_BIN="$ROOT/jolt-bench/target/release/jolt-bench"
PROG="${1:-sha256_1024}"
ELF="$ROOT/examples/${PROG}.elf"
RUNS="${RUNS:-3}"

# ── Colors ──
RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[0;33m'
CYN='\033[0;36m'
BLD='\033[1m'
DIM='\033[2m'
RST='\033[0m'

# ── Validate ──
[ ! -f "$ELF" ] && echo "ERROR: $ELF not found" && exit 1
[ ! -f "$ZOLT_BIN" ] && echo "ERROR: $ZOLT_BIN not found (run: zig build -Doptimize=ReleaseFast)" && exit 1
[ ! -f "$JOLT_BIN" ] && echo "ERROR: $JOLT_BIN not found (run: cargo build --release -p jolt-bench)" && exit 1

# ── Helpers ──
field() {
    # Extract field=VALUE from a line (handles optional spaces after =)
    # Usage: field "compute" "$line"
    echo "$2" | grep -oP "${1}=\s*\K[0-9.]+" | head -1
}

fieldms() {
    # Extract field=VALUEms from a line (handles spaces like "compute=   57.9ms")
    # Usage: fieldms "compute" "$line"
    echo "$2" | grep -oP "${1}=\s*\K[0-9.]+" | head -1
}

fmtms() {
    # Format number to 1 decimal, right-aligned 8 chars
    [ -z "$1" ] && printf "%8s" "—" && return
    printf "%8.1f" "$1"
}

fmtdelta() {
    local z="$1" j="$2"
    [ -z "$z" ] || [ -z "$j" ] && printf "%8s" "—" && return
    local d=$(awk "BEGIN{printf \"%.1f\", $z - $j}")
    local sign=$(echo "$d" | cut -c1)
    if [ "$sign" = "-" ]; then
        printf "${GRN}%+8.1f${RST}" "$d"
    elif awk "BEGIN{exit !($d > 0.5)}"; then
        printf "${RED}%+8.1f${RST}" "$d"
    else
        printf "%+8.1f" "$d"
    fi
}

fmtratio() {
    local z="$1" j="$2"
    [ -z "$z" ] || [ -z "$j" ] && printf "%7s" "—" && return
    if awk "BEGIN{exit !($j < 0.1)}"; then
        printf "%7s" "—"
        return
    fi
    local r=$(awk "BEGIN{printf \"%.2f\", $z/$j}")
    if awk "BEGIN{exit !($r > 1.5)}"; then
        printf "${RED}%6sx${RST}" "$r"
    elif awk "BEGIN{exit !($r > 1.1)}"; then
        printf "${YEL}%6sx${RST}" "$r"
    elif awk "BEGIN{exit !($r < 0.9)}"; then
        printf "${GRN}%6sx${RST}" "$r"
    else
        printf "%6sx" "$r"
    fi
}

sep() {
    echo -e "  ${DIM}──────────────────────────────────┬──────────┬──────────┬──────────┬─────────${RST}"
}

row() {
    local label="$1" z="$2" j="$3"
    printf "  %-34s│ $(fmtms "$z") │ $(fmtms "$j") │ $(fmtdelta "$z" "$j") │ $(fmtratio "$z" "$j")\n" "$label"
}

header() {
    echo ""
    echo -e "  ${BLD}$1${RST}"
    printf "  %-34s│ %8s │ %8s │ %8s │ %7s\n" "" "Zolt ms" "Jolt ms" "Δ ms" "ratio"
    sep
}

# ══════════════════════════════════════════════════════════════════════════════
# Run provers
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BLD}══════════════════════════════════════════════════════════════════════════════${RST}"
echo -e "${BLD}  SHA256 Fine-Grained Benchmark: ${CYN}${PROG}${RST}"
echo -e "${BLD}  Runs: ${RUNS} (best of N)${RST}"
echo -e "${BLD}══════════════════════════════════════════════════════════════════════════════${RST}"

best_z_total=999999
best_j_total=999999
best_z_out=""
best_j_out=""

for run in $(seq 1 $RUNS); do
    printf "  Run %d/%d..." "$run" "$RUNS"

    # --- Zolt ---
    z_out=$(ZOLT_BENCH=1 "$ZOLT_BIN" prove "$ELF" --jolt-format -o /tmp/zolt_bench_sha.bin \
            --export-preprocessing /tmp/zolt_bench_sha_preproc.bin 2>&1)
    z_total=$(echo "$z_out" | grep -oP 'Total time: \K[0-9.]+' | head -1)
    [ -z "$z_total" ] && z_total=999999

    # --- Jolt ---
    j_out=$(JOLT_BENCH=1 "$JOLT_BIN" "$ELF" 2>&1)
    j_total=$(echo "$j_out" | grep -oP 'prove_total=\K[0-9.]+' | head -1)
    [ -z "$j_total" ] && j_total=999999

    printf " Zolt=%.0fms Jolt=%.0fms\n" "$z_total" "$j_total"

    # Keep best (lowest total prove time) for each
    if awk "BEGIN{exit !($z_total < $best_z_total)}"; then
        best_z_total="$z_total"
        best_z_out="$z_out"
    fi
    if awk "BEGIN{exit !($j_total < $best_j_total)}"; then
        best_j_total="$j_total"
        best_j_out="$j_out"
    fi
done

z_out="$best_z_out"
j_out="$best_j_out"

# ══════════════════════════════════════════════════════════════════════════════
# Parse Zolt output
# ══════════════════════════════════════════════════════════════════════════════

z_bench=$(echo "$z_out" | grep -E '^\[BENCH\]|\[S5-PHASE\]|\[STAGE6-BENCH\]')

# Overhead
z_trace=$(echo "$z_out" | grep -oP 'trace_and_witness=\K[0-9.]+' | head -1)
z_srs=$(echo "$z_out" | grep -oP 'srs_setup=\K[0-9.]+' | head -1)
z_commit=$(echo "$z_out" | grep -oP 'poly_build_and_commit=\K[0-9.]+' | head -1)
z_preprove=$(echo "$z_out" | grep -oP 'pre_prove_setup=\K[0-9.]+' | head -1)
z_wall=$(echo "$z_out" | grep -oP 'Total time: \K[0-9.]+' | head -1)

# Per-stage totals + sub-operations (match "stage=N total=" specifically to avoid instance lines)
for s in 1 2 3 4 5 6 7; do
    line=$(echo "$z_bench" | grep "^\[BENCH\] stage=$s total=" | head -1)
    eval "zs${s}_total=$(field 'total' "$line")"
    eval "zs${s}_init=$(field 'init' "$line")"
    eval "zs${s}_sc=$(field 'sumcheck' "$line")"
    eval "zs${s}_claims=$(field 'claims' "$line")"
done

# Stage 8
z_s8_line=$(echo "$z_bench" | grep "^\[BENCH\] stage=8 " | head -1)
zs8_total=$(field 'total' "$z_s8_line")
zs8_joint=$(field 'joint_poly' "$z_s8_line")
zs8_opening=$(field 'opening' "$z_s8_line")

# Stage 2 instances
z_s2_rwc_line=$(echo "$z_bench" | grep "stage=2 instance=0" | head -1)
z_s2_prod_line=$(echo "$z_bench" | grep "stage=2 instance=1" | head -1)
z_s2_instr_line=$(echo "$z_bench" | grep "stage=2 instance=2" | head -1)
z_s2_raf_line=$(echo "$z_bench" | grep "stage=2 instance=3" | head -1)
z_s2_out_line=$(echo "$z_bench" | grep "stage=2 instance=4" | head -1)

z_s2_rwc_compute=$(fieldms 'compute' "$z_s2_rwc_line")
z_s2_rwc_bind=$(fieldms 'bind' "$z_s2_rwc_line")
z_s2_prod_compute=$(fieldms 'compute' "$z_s2_prod_line")
z_s2_prod_bind=$(fieldms 'bind' "$z_s2_prod_line")
z_s2_instr_compute=$(fieldms 'compute' "$z_s2_instr_line")
z_s2_instr_bind=$(fieldms 'bind' "$z_s2_instr_line")
z_s2_raf_compute=$(fieldms 'compute' "$z_s2_raf_line")
z_s2_raf_bind=$(fieldms 'bind' "$z_s2_raf_line")
z_s2_out_compute=$(fieldms 'compute' "$z_s2_out_line")
z_s2_out_bind=$(fieldms 'bind' "$z_s2_out_line")

# Stage 2 breakdown
z_s2_bk=$(echo "$z_bench" | grep "stage=2 breakdown" | head -1)
z_s2_prover_init=$(fieldms 'prover_init' "$z_s2_bk")
z_s2_round_loop=$(fieldms 'round_loop' "$z_s2_bk")
z_s2_post_loop=$(fieldms 'post_loop' "$z_s2_bk")

# Stage 3 sub
z_s3_detail=$(echo "$z_bench" | grep "stage=3 shift_compute" | head -1)
z_s3_shift_compute=$(fieldms 'shift_compute' "$z_s3_detail")
z_s3_shift_bind=$(fieldms 'shift_bind' "$z_s3_detail")
z_s3_instr_compute=$(fieldms 'instr_compute' "$z_s3_detail")
z_s3_instr_bind=$(fieldms 'instr_bind' "$z_s3_detail")
z_s3_reg_compute=$(fieldms 'reg_compute' "$z_s3_detail")
z_s3_reg_bind=$(fieldms 'reg_bind' "$z_s3_detail")

z_s3_init_detail=$(echo "$z_bench" | grep "stage=3 init:" | head -1)
z_s3_init_shift=$(echo "$z_s3_init_detail" | grep -oP 'shift=\K[0-9.]+' | head -1)
z_s3_init_reg=$(echo "$z_s3_init_detail" | grep -oP 'reg=\K[0-9.]+' | head -1)
z_s3_init_instr=$(echo "$z_s3_init_detail" | grep -oP 'instr=\K[0-9.]+' | head -1)

# Stage 5 phases
z_s5_phase=$(echo "$z_bench" | grep "\[S5-PHASE\]" | head -1)
z_s5_addr_compute=$(fieldms 'addr_compute' "$z_s5_phase")
z_s5_addr_bind=$(fieldms 'addr_bind' "$z_s5_phase")
z_s5_phase_trans=$(fieldms 'phase_trans' "$z_s5_phase")
z_s5_cycle_compute=$(fieldms 'cycle_compute' "$z_s5_phase")
z_s5_cycle_bind=$(fieldms 'cycle_bind' "$z_s5_phase")

# Stage 6 instances
z_s6_bcraf=$(echo "$z_bench" | grep "STAGE6-BENCH.*BcRaf:" | head -1)
z_s6_bool=$(echo "$z_bench" | grep "STAGE6-BENCH.*Bool " | head -1)
z_s6_hamm=$(echo "$z_bench" | grep "STAGE6-BENCH.*Hamm " | head -1)
z_s6_ramra=$(echo "$z_bench" | grep "STAGE6-BENCH.*RamRa:" | head -1)
z_s6_lkra=$(echo "$z_bench" | grep "STAGE6-BENCH.*LkRa " | head -1)
z_s6_inc=$(echo "$z_bench" | grep "STAGE6-BENCH.*Inc  " | head -1)

z_s6_bcraf_compute=$(fieldms 'compute' "$z_s6_bcraf")
z_s6_bcraf_bind=$(fieldms 'bind' "$z_s6_bcraf")
z_s6_bool_compute=$(fieldms 'compute' "$z_s6_bool")
z_s6_bool_bind=$(fieldms 'bind' "$z_s6_bool")
z_s6_hamm_compute=$(fieldms 'compute' "$z_s6_hamm")
z_s6_hamm_bind=$(fieldms 'bind' "$z_s6_hamm")
z_s6_ramra_compute=$(fieldms 'compute' "$z_s6_ramra")
z_s6_ramra_bind=$(fieldms 'bind' "$z_s6_ramra")
z_s6_lkra_compute=$(fieldms 'compute' "$z_s6_lkra")
z_s6_lkra_bind=$(fieldms 'bind' "$z_s6_lkra")
z_s6_inc_compute=$(fieldms 'compute' "$z_s6_inc")
z_s6_inc_bind=$(fieldms 'bind' "$z_s6_inc")

# Stage 6 init breakdown — extract the trailing number (format: "   117.6ms")
z_s6_init_total=$(echo "$z_bench" | grep "STAGE6-BENCH.*Init total:" | head -1 | grep -oP '[0-9.]+(?=ms)' | head -1)
z_s6_init_inc=$(echo "$z_bench" | grep "STAGE6-BENCH.*IncClaim init:" | head -1 | grep -oP '[0-9.]+(?=ms)' | head -1)
z_s6_init_hamming=$(echo "$z_bench" | grep "STAGE6-BENCH.*Hamming init:" | head -1 | grep -oP '[0-9.]+(?=ms)' | head -1)
z_s6_init_ramra=$(echo "$z_bench" | grep "STAGE6-BENCH.*RamRaVirtual init:" | head -1 | grep -oP '[0-9.]+(?=ms)' | head -1)
z_s6_init_lookups=$(echo "$z_bench" | grep "STAGE6-BENCH.*LookupsRa init:" | head -1 | grep -oP '[0-9.]+(?=ms)' | head -1)
z_s6_init_booleanity=$(echo "$z_bench" | grep "STAGE6-BENCH.*Booleanity init:" | head -1 | grep -oP '[0-9.]+(?=ms)' | head -1)
z_s6_init_bcraf=$(echo "$z_bench" | grep "STAGE6-BENCH.*BytecodeRaf init:" | head -1 | grep -oP '[0-9.]+(?=ms)' | head -1)

# ══════════════════════════════════════════════════════════════════════════════
# Parse Jolt output
# ══════════════════════════════════════════════════════════════════════════════

j_bench=$(echo "$j_out" | grep -E '^\[BENCH\]')

# Overhead
j_decode=$(echo "$j_out" | grep -oP 'Decode:\s+\K[0-9.]+' | head -1)
j_trace=$(echo "$j_out" | grep -oP 'Trace:\s+\K[0-9.]+' | head -1)
j_preproc=$(echo "$j_out" | grep -oP 'Preprocessing:\s+\K[0-9.]+' | head -1)
j_prover_gen=$(echo "$j_out" | grep -oP 'Prover gen:\s+\K[0-9.]+' | head -1)
j_commit=$(echo "$j_bench" | grep "stage=commit " | head -1 | grep -oP 'total=\K[0-9.]+')
j_prove_total=$(echo "$j_bench" | grep "prove_total=" | head -1 | grep -oP 'prove_total=\K[0-9.]+')

# Per-stage totals
for s in 1 2 3 4 5 6 7 8; do
    val=$(echo "$j_bench" | grep "stage=$s total=" | head -1 | grep -oP 'total=\K[0-9.]+')
    eval "js${s}_total=$val"
done

# Per-stage init/sumcheck/finalize
for s in 1 2 3 4 5 6 7; do
    line=$(echo "$j_bench" | grep "stage=$s init=" | head -1)
    eval "js${s}_init=$(field 'init' "$line")"
    eval "js${s}_sc=$(field 'sumcheck' "$line")"
    eval "js${s}_finalize=$(field 'finalize' "$line")"
done

# Per-stage sumcheck detail (compute/bind/transcript)
for s in 1 2 3 4 5 6 7; do
    line=$(echo "$j_bench" | grep "stage=$s sumcheck_detail" | head -1)
    eval "js${s}_compute=$(field 'compute' "$line")"
    eval "js${s}_bind=$(field 'bind' "$line")"
    eval "js${s}_transcript=$(field 'transcript' "$line")"
    eval "js${s}_rounds=$(field 'rounds' "$line")"
done

# Stage 8 sub
j_s8_line=$(echo "$j_bench" | grep "stage=8 rlc_build=" | head -1)
js8_rlc=$(field 'rlc_build' "$j_s8_line")
js8_opening=$(field 'opening' "$j_s8_line")

# Stage 2 instances (Jolt)
j_s2_rwc=$(echo "$j_bench" | grep "stage=2.*RamRWC:" | head -1)
j_s2_prod=$(echo "$j_bench" | grep "stage=2.*ProductVirtualRemainder:" | head -1)
j_s2_instr=$(echo "$j_bench" | grep "stage=2.*InstrClaimReduction:" | head -1)
j_s2_raf=$(echo "$j_bench" | grep "stage=2.*RamRafEval:" | head -1)
j_s2_output=$(echo "$j_bench" | grep "stage=2.*RamOutputCheck:" | head -1)

j_s2_rwc_compute=$(fieldms 'compute' "$j_s2_rwc")
j_s2_rwc_bind=$(fieldms 'bind' "$j_s2_rwc")
j_s2_prod_compute=$(fieldms 'compute' "$j_s2_prod")
j_s2_prod_bind=$(fieldms 'bind' "$j_s2_prod")
j_s2_instr_compute=$(fieldms 'compute' "$j_s2_instr")
j_s2_instr_bind=$(fieldms 'bind' "$j_s2_instr")
j_s2_raf_compute=$(fieldms 'compute' "$j_s2_raf")
j_s2_raf_bind=$(fieldms 'bind' "$j_s2_raf")
j_s2_out_compute=$(fieldms 'compute' "$j_s2_output")
j_s2_out_bind=$(fieldms 'bind' "$j_s2_output")

# Stage 3 instances (Jolt)
j_s3_shift=$(echo "$j_bench" | grep "stage=3.*Shift:" | head -1)
j_s3_instr=$(echo "$j_bench" | grep "stage=3.*InstructionInput:" | head -1)
j_s3_reg=$(echo "$j_bench" | grep "stage=3.*RegistersClaimReduction:" | head -1)

j_s3_shift_compute=$(fieldms 'compute' "$j_s3_shift")
j_s3_shift_bind=$(fieldms 'bind' "$j_s3_shift")
j_s3_instr_compute=$(fieldms 'compute' "$j_s3_instr")
j_s3_instr_bind=$(fieldms 'bind' "$j_s3_instr")
j_s3_reg_compute=$(fieldms 'compute' "$j_s3_reg")
j_s3_reg_bind=$(fieldms 'bind' "$j_s3_reg")

# Stage 4 instances (Jolt)
j_s4_rwc=$(echo "$j_bench" | grep "stage=4.*RegistersRWC:" | head -1)
j_s4_val=$(echo "$j_bench" | grep "stage=4.*RamValCheck:" | head -1)

j_s4_rwc_compute=$(fieldms 'compute' "$j_s4_rwc")
j_s4_rwc_bind=$(fieldms 'bind' "$j_s4_rwc")
j_s4_val_compute=$(fieldms 'compute' "$j_s4_val")
j_s4_val_bind=$(fieldms 'bind' "$j_s4_val")

# Stage 5 instances (Jolt)
j_s5_lookups=$(echo "$j_bench" | grep "stage=5.*LookupsReadRaf:" | head -1)
j_s5_ramra=$(echo "$j_bench" | grep "stage=5.*RamRaReduction:" | head -1)
j_s5_regval=$(echo "$j_bench" | grep "stage=5.*RegistersValEval:" | head -1)

j_s5_lookups_compute=$(fieldms 'compute' "$j_s5_lookups")
j_s5_lookups_bind=$(fieldms 'bind' "$j_s5_lookups")
j_s5_ramra_compute=$(fieldms 'compute' "$j_s5_ramra")
j_s5_ramra_bind=$(fieldms 'bind' "$j_s5_ramra")
j_s5_regval_compute=$(fieldms 'compute' "$j_s5_regval")
j_s5_regval_bind=$(fieldms 'bind' "$j_s5_regval")

# Stage 6 instances (Jolt)
j_s6_bcraf=$(echo "$j_bench" | grep "stage=6.*BcRaf:" | head -1)
j_s6_bool=$(echo "$j_bench" | grep "stage=6.*Booleanity:" | head -1)
j_s6_hamm=$(echo "$j_bench" | grep "stage=6.*RamHammingBool:" | head -1)
j_s6_ramra=$(echo "$j_bench" | grep "stage=6.*RamRaVirtual:" | head -1)
j_s6_lkra=$(echo "$j_bench" | grep "stage=6.*LookupsRaVirtual:" | head -1)
j_s6_inc=$(echo "$j_bench" | grep "stage=6.*IncReduction:" | head -1)

j_s6_bcraf_compute=$(fieldms 'compute' "$j_s6_bcraf")
j_s6_bcraf_bind=$(fieldms 'bind' "$j_s6_bcraf")
j_s6_bool_compute=$(fieldms 'compute' "$j_s6_bool")
j_s6_bool_bind=$(fieldms 'bind' "$j_s6_bool")
j_s6_hamm_compute=$(fieldms 'compute' "$j_s6_hamm")
j_s6_hamm_bind=$(fieldms 'bind' "$j_s6_hamm")
j_s6_ramra_compute=$(fieldms 'compute' "$j_s6_ramra")
j_s6_ramra_bind=$(fieldms 'bind' "$j_s6_ramra")
j_s6_lkra_compute=$(fieldms 'compute' "$j_s6_lkra")
j_s6_lkra_bind=$(fieldms 'bind' "$j_s6_lkra")
j_s6_inc_compute=$(fieldms 'compute' "$j_s6_inc")
j_s6_inc_bind=$(fieldms 'bind' "$j_s6_inc")

# ══════════════════════════════════════════════════════════════════════════════
# Display results
# ══════════════════════════════════════════════════════════════════════════════

# ── OVERHEAD ──
header "OVERHEAD (pre-prove)"
j_trace_total=$(awk "BEGIN{printf \"%.1f\", ${j_decode:-0} + ${j_trace:-0}}")
row "Trace + witness gen" "$z_trace" "$j_trace_total"
row "SRS / preprocessing" "$z_srs" "$j_preproc"
row "Poly build + commit (Dory)" "$z_commit" "$j_commit"
j_gen_commit=$(awk "BEGIN{printf \"%.1f\", ${j_prover_gen:-0} + ${j_commit:-0}}")
z_commit_pre=$(awk "BEGIN{printf \"%.1f\", ${z_commit:-0} + ${z_preprove:-0}}")
row "Total pre-prove overhead" "$z_commit_pre" "$j_gen_commit"

# ── STAGE 1: Outer Spartan ──
header "STAGE 1 — Outer Spartan (R1CS Az·Bz=Cz)"
row "Total" "$zs1_total" "$js1_total"
row "  init" "$zs1_init" "$js1_init"
row "  sumcheck" "$zs1_sc" "$js1_sc"
row "    compute" "" "$js1_compute"
row "    bind" "" "$js1_bind"
row "  claims" "$zs1_claims" ""

# ── STAGE 2: Product + RAM ──
header "STAGE 2 — Products + RAM RAF + Output (${js2_rounds:-?} rounds)"
row "Total" "$zs2_total" "$js2_total"
row "  init" "$zs2_init" "$js2_init"
row "  sumcheck" "$zs2_sc" "$js2_sc"
# Compute Zolt aggregate from instances
z_s2_total_compute=$(awk "BEGIN{printf \"%.1f\", ${z_s2_rwc_compute:-0}+${z_s2_prod_compute:-0}+${z_s2_instr_compute:-0}+${z_s2_raf_compute:-0}+${z_s2_out_compute:-0}}")
z_s2_total_bind=$(awk "BEGIN{printf \"%.1f\", ${z_s2_rwc_bind:-0}+${z_s2_prod_bind:-0}+${z_s2_instr_bind:-0}+${z_s2_raf_bind:-0}+${z_s2_out_bind:-0}}")
row "    total compute" "$z_s2_total_compute" "$js2_compute"
row "    total bind" "$z_s2_total_bind" "$js2_bind"
row "  claims" "$zs2_claims" "${js2_finalize:-0}"
echo ""
echo -e "  ${DIM}Per-instance breakdown:${RST}"
printf "  %-34s│ %8s │ %8s │ %8s │ %8s │ %8s │ %8s\n" \
    "  Instance" "Z comp" "J comp" "Δ comp" "Z bind" "J bind" "Δ bind"
sep
for inst in rwc prod instr raf out; do
    case $inst in
        rwc)  label="RamRWC" ;;
        prod) label="ProductVirtual" ;;
        instr) label="InstrLookups" ;;
        raf)  label="RamRafEval" ;;
        out)  label="RamOutputCheck" ;;
    esac
    eval "zc=\$z_s2_${inst}_compute"
    eval "zb=\$z_s2_${inst}_bind"
    eval "jc=\$j_s2_${inst}_compute"
    eval "jb=\$j_s2_${inst}_bind"
    printf "  %-34s│ $(fmtms "$zc") │ $(fmtms "$jc") │ $(fmtdelta "$zc" "$jc") │ $(fmtms "$zb") │ $(fmtms "$jb") │ $(fmtdelta "$zb" "$jb")\n" "    $label"
done

# ── STAGE 3: Shift + Instruction + Registers ──
header "STAGE 3 — Shift + InstrInput + RegClaim (${js3_rounds:-?} rounds)"
row "Total" "$zs3_total" "$js3_total"
# Zolt stage 3 init is tracked inside sumcheck, compute from sub-init
z_s3_init_total=$(awk "BEGIN{printf \"%.1f\", ${z_s3_init_shift:-0} + ${z_s3_init_reg:-0} + ${z_s3_init_instr:-0}}")
row "  init (incl. in sc for Zolt)" "$z_s3_init_total" "$js3_init"
row "    init:shift" "$z_s3_init_shift" ""
row "    init:reg" "$z_s3_init_reg" ""
row "    init:instr" "$z_s3_init_instr" ""
row "  sumcheck" "$zs3_sc" "$js3_sc"
echo ""
echo -e "  ${DIM}Per-instance breakdown:${RST}"
printf "  %-34s│ %8s │ %8s │ %8s │ %8s │ %8s │ %8s\n" \
    "  Instance" "Z comp" "J comp" "Δ comp" "Z bind" "J bind" "Δ bind"
sep
for inst in shift instr reg; do
    case $inst in
        shift) label="Shift" ;;
        instr) label="InstructionInput" ;;
        reg)   label="RegistersClaim" ;;
    esac
    eval "zc=\$z_s3_${inst}_compute"
    eval "zb=\$z_s3_${inst}_bind"
    eval "jc=\$j_s3_${inst}_compute"
    eval "jb=\$j_s3_${inst}_bind"
    printf "  %-34s│ $(fmtms "$zc") │ $(fmtms "$jc") │ $(fmtdelta "$zc" "$jc") │ $(fmtms "$zb") │ $(fmtms "$jb") │ $(fmtdelta "$zb" "$jb")\n" "    $label"
done

# ── STAGE 4: Registers RW + RAM Val ──
header "STAGE 4 — RegistersRWC + RamValCheck (${js4_rounds:-?} rounds)"
row "Total" "$zs4_total" "$js4_total"
row "  init" "$zs4_init" "$js4_init"
row "  sumcheck" "$zs4_sc" "$js4_sc"
row "    total compute" "" "$js4_compute"
row "    total bind" "" "$js4_bind"
row "  claims" "$zs4_claims" "${js4_finalize:-0}"
echo ""
echo -e "  ${DIM}Per-instance breakdown (Jolt only — Zolt lacks sub-instance detail):${RST}"
printf "  %-34s│ %8s │ %8s │ %8s │ %8s\n" \
    "  Instance" "J comp" "J bind" "" ""
sep
printf "  %-34s│ $(fmtms "$j_s4_rwc_compute") │ $(fmtms "$j_s4_rwc_bind") │ %8s │ %8s\n" "    RegistersRWC" "" ""
printf "  %-34s│ $(fmtms "$j_s4_val_compute") │ $(fmtms "$j_s4_val_bind") │ %8s │ %8s\n" "    RamValCheck" "" ""

# ── STAGE 5: Lookups RAF + RAM RA + Registers Val ──
header "STAGE 5 — LookupsRaf + RamRA + RegVal (${js5_rounds:-?} rounds)"
row "Total" "$zs5_total" "$js5_total"
row "  init" "$zs5_init" "$js5_init"
row "  sumcheck" "$zs5_sc" "$js5_sc"
# Compute Zolt aggregate from phases
z_s5_total_compute=$(awk "BEGIN{printf \"%.1f\", ${z_s5_addr_compute:-0}+${z_s5_cycle_compute:-0}}")
z_s5_total_bind=$(awk "BEGIN{printf \"%.1f\", ${z_s5_addr_bind:-0}+${z_s5_cycle_bind:-0}}")
row "    total compute" "$z_s5_total_compute" "$js5_compute"
row "    total bind" "$z_s5_total_bind" "$js5_bind"
echo ""
echo -e "  ${DIM}Zolt phase breakdown:${RST}"
printf "  %-34s│ %8s │\n" "" "Zolt ms"
sep
printf "  %-34s│ $(fmtms "$z_s5_addr_compute") │\n" "    addr_compute"
printf "  %-34s│ $(fmtms "$z_s5_addr_bind") │\n" "    addr_bind"
printf "  %-34s│ $(fmtms "$z_s5_phase_trans") │\n" "    phase_trans"
printf "  %-34s│ $(fmtms "$z_s5_cycle_compute") │\n" "    cycle_compute"
printf "  %-34s│ $(fmtms "$z_s5_cycle_bind") │\n" "    cycle_bind"
echo ""
echo -e "  ${DIM}Jolt per-instance breakdown:${RST}"
printf "  %-34s│ %8s │ %8s\n" "  Instance" "J comp" "J bind"
sep
printf "  %-34s│ $(fmtms "$j_s5_lookups_compute") │ $(fmtms "$j_s5_lookups_bind")\n" "    LookupsReadRaf"
printf "  %-34s│ $(fmtms "$j_s5_ramra_compute") │ $(fmtms "$j_s5_ramra_bind")\n" "    RamRaReduction"
printf "  %-34s│ $(fmtms "$j_s5_regval_compute") │ $(fmtms "$j_s5_regval_bind")\n" "    RegistersValEval"

# ── STAGE 6: Bytecode + Booleanity + Hamming + RA ──
header "STAGE 6 — BcRaf + Bool + Hamm + RamRA + LkRA + Inc (30 rounds)"
row "Total" "$zs6_total" "$js6_total"
row "  init" "$zs6_init" "$js6_init"
row "  sumcheck" "$zs6_sc" "$js6_sc"
row "  claims" "$zs6_claims" "${js6_finalize:-0}"
echo ""
echo -e "  ${DIM}Stage 6 init breakdown (Zolt):${RST}"
printf "  %-34s│ %8s │\n" "" "Zolt ms"
sep
printf "  %-34s│ $(fmtms "$z_s6_init_total") │\n" "    Init TOTAL"
printf "  %-34s│ $(fmtms "$z_s6_init_inc") │\n" "      IncClaim"
printf "  %-34s│ $(fmtms "$z_s6_init_hamming") │\n" "      Hamming"
printf "  %-34s│ $(fmtms "$z_s6_init_ramra") │\n" "      RamRaVirtual"
printf "  %-34s│ $(fmtms "$z_s6_init_lookups") │\n" "      LookupsRa"
printf "  %-34s│ $(fmtms "$z_s6_init_booleanity") │\n" "      Booleanity"
printf "  %-34s│ $(fmtms "$z_s6_init_bcraf") │\n" "      BytecodeRaf"
echo ""
echo -e "  ${DIM}Per-instance sumcheck breakdown:${RST}"
printf "  %-34s│ %8s │ %8s │ %8s │ %8s │ %8s │ %8s\n" \
    "  Instance" "Z comp" "J comp" "Δ comp" "Z bind" "J bind" "Δ bind"
sep
for inst in bcraf bool hamm ramra lkra inc; do
    case $inst in
        bcraf) label="BcRaf" ;;
        bool)  label="Booleanity" ;;
        hamm)  label="Hamming" ;;
        ramra) label="RamRaVirtual" ;;
        lkra)  label="LookupsRaVirtual" ;;
        inc)   label="IncReduction" ;;
    esac
    eval "zc=\$z_s6_${inst}_compute"
    eval "zb=\$z_s6_${inst}_bind"
    eval "jc=\$j_s6_${inst}_compute"
    eval "jb=\$j_s6_${inst}_bind"
    printf "  %-34s│ $(fmtms "$zc") │ $(fmtms "$jc") │ $(fmtdelta "$zc" "$jc") │ $(fmtms "$zb") │ $(fmtms "$jb") │ $(fmtdelta "$zb" "$jb")\n" "    $label"
done

# ── STAGE 7: Hamming Weight ──
header "STAGE 7 — Hamming Weight Reduction"
row "Total" "$zs7_total" "$js7_total"
row "  init" "$zs7_init" "$js7_init"
row "  sumcheck" "$zs7_sc" "$js7_sc"
row "  claims" "$zs7_claims" "${js7_finalize:-0}"

# ── STAGE 8: Dory Opening ──
header "STAGE 8 — Dory Joint Poly + Opening"
row "Total (excl commit)" "$zs8_total" "$js8_total"
row "  joint_poly / rlc_build" "$zs8_joint" "$js8_rlc"
row "  opening proof" "$zs8_opening" "$js8_opening"

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BLD}══════════════════════════════════════════════════════════════════════════════${RST}"
echo -e "${BLD}  SUMMARY${RST}"
echo -e "${BLD}══════════════════════════════════════════════════════════════════════════════${RST}"

header "Prove totals (stages 1-8 + commit)"
z_s17=$(awk "BEGIN{printf \"%.1f\", ${zs1_total:-0}+${zs2_total:-0}+${zs3_total:-0}+${zs4_total:-0}+${zs5_total:-0}+${zs6_total:-0}+${zs7_total:-0}}")
j_s17=$(awk "BEGIN{printf \"%.1f\", ${js1_total:-0}+${js2_total:-0}+${js3_total:-0}+${js4_total:-0}+${js5_total:-0}+${js6_total:-0}+${js7_total:-0}}")
row "Stages 1-7 sumcheck" "$z_s17" "$j_s17"
row "Stage 8 (Dory opening)" "$zs8_total" "$js8_total"
row "Commit (Dory)" "$z_commit" "$j_commit"
z_prove_total=$(awk "BEGIN{printf \"%.1f\", ${z_s17:-0}+${zs8_total:-0}+${z_commit:-0}}")
row "PROVE TOTAL (s1-8 + commit)" "$z_prove_total" "$j_prove_total"
row "Wall clock" "$z_wall" ""

# ── TOP GAPS (absolute delta, descending) ──
echo ""
echo -e "  ${BLD}Top gaps (Zolt − Jolt, descending):${RST}"
echo ""

# Collect all meaningful deltas
declare -A gaps
gaps["Stage 1"]=$(awk "BEGIN{printf \"%.1f\", ${zs1_total:-0} - ${js1_total:-0}}")
gaps["Stage 2"]=$(awk "BEGIN{printf \"%.1f\", ${zs2_total:-0} - ${js2_total:-0}}")
gaps["Stage 3"]=$(awk "BEGIN{printf \"%.1f\", ${zs3_total:-0} - ${js3_total:-0}}")
gaps["Stage 4"]=$(awk "BEGIN{printf \"%.1f\", ${zs4_total:-0} - ${js4_total:-0}}")
gaps["Stage 5"]=$(awk "BEGIN{printf \"%.1f\", ${zs5_total:-0} - ${js5_total:-0}}")
gaps["Stage 6"]=$(awk "BEGIN{printf \"%.1f\", ${zs6_total:-0} - ${js6_total:-0}}")
gaps["Stage 7"]=$(awk "BEGIN{printf \"%.1f\", ${zs7_total:-0} - ${js7_total:-0}}")
gaps["Stage 8 (Dory)"]=$(awk "BEGIN{printf \"%.1f\", ${zs8_total:-0} - ${js8_total:-0}}")
gaps["Commit"]=$(awk "BEGIN{printf \"%.1f\", ${z_commit:-0} - ${j_commit:-0}}")
gaps["S6:Booleanity"]=$(awk "BEGIN{printf \"%.1f\", (${z_s6_bool_compute:-0}+${z_s6_bool_bind:-0}) - (${j_s6_bool_compute:-0}+${j_s6_bool_bind:-0})}")
gaps["S6:LookupsRaVirtual"]=$(awk "BEGIN{printf \"%.1f\", (${z_s6_lkra_compute:-0}+${z_s6_lkra_bind:-0}) - (${j_s6_lkra_compute:-0}+${j_s6_lkra_bind:-0})}")
gaps["S6:RamRaVirtual"]=$(awk "BEGIN{printf \"%.1f\", (${z_s6_ramra_compute:-0}+${z_s6_ramra_bind:-0}) - (${j_s6_ramra_compute:-0}+${j_s6_ramra_bind:-0})}")
gaps["S2:RamRWC bind"]=$(awk "BEGIN{printf \"%.1f\", ${z_s2_rwc_bind:-0} - ${j_s2_rwc_bind:-0}}")
gaps["S2:InstrLookups"]=$(awk "BEGIN{printf \"%.1f\", (${z_s2_instr_compute:-0}+${z_s2_instr_bind:-0}) - (${j_s2_instr_compute:-0}+${j_s2_instr_bind:-0})}")
gaps["S5:phase_trans"]="${z_s5_phase_trans:-0}"  # Zolt-only overhead
gaps["S8:opening"]=$(awk "BEGIN{printf \"%.1f\", ${zs8_opening:-0} - ${js8_opening:-0}}")
gaps["S8:joint_poly/rlc"]=$(awk "BEGIN{printf \"%.1f\", ${zs8_joint:-0} - ${js8_rlc:-0}}")

for key in "${!gaps[@]}"; do
    echo "${gaps[$key]} $key"
done | awk '{d=$1; if(d<0)d=-d; print d, $0}' | sort -rn | head -15 | while read abs delta name; do
    if awk "BEGIN{exit !($delta > 0)}"; then
        printf "    ${RED}%+8.1f ms${RST}  %s\n" "$delta" "$name"
    else
        printf "    ${GRN}%+8.1f ms${RST}  %s\n" "$delta" "$name"
    fi
done

unset gaps

echo ""
echo -e "${BLD}══════════════════════════════════════════════════════════════════════════════${RST}"
echo ""
