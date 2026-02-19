#!/usr/bin/env bash
# Zolt → Jolt Cross-Verification Script
#
# This script generates proofs for all example programs using Zolt
# and verifies them using Jolt's verifier.
#
# Prerequisites:
#   - Zolt built: zig build -Doptimize=ReleaseFast
#   - Jolt repo at ../jolt (or set JOLT_DIR env var)
#
# Usage:
#   ./scripts/verify_all.sh              # Test all programs
#   ./scripts/verify_all.sh fibonacci    # Test specific program
#   ./scripts/verify_all.sh --quick      # Only prove, skip Jolt verification

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOLT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
JOLT_DIR="${JOLT_DIR:-$(cd "$ZOLT_DIR/../jolt" 2>/dev/null && pwd || echo "")}"
ZOLT_BIN="$ZOLT_DIR/zig-out/bin/zolt"
TMP_DIR="/tmp/zolt_verify"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# All example programs
ALL_PROGRAMS=(fibonacci collatz factorial sum signed primes gcd bitwise)

# Parse arguments
QUICK_MODE=false
PROGRAMS=()
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK_MODE=true ;;
        --help|-h)
            echo "Usage: $0 [--quick] [program1 program2 ...]"
            echo ""
            echo "Options:"
            echo "  --quick     Only generate proofs, skip Jolt verification"
            echo "  --help      Show this help"
            echo ""
            echo "Programs: ${ALL_PROGRAMS[*]}"
            exit 0
            ;;
        *) PROGRAMS+=("$arg") ;;
    esac
done

# Default to all programs if none specified
if [ ${#PROGRAMS[@]} -eq 0 ]; then
    PROGRAMS=("${ALL_PROGRAMS[@]}")
fi

# Create temp directory
mkdir -p "$TMP_DIR"

# Check prerequisites
echo "=== Zolt → Jolt Cross-Verification ==="
echo ""

if [ ! -f "$ZOLT_BIN" ]; then
    echo -e "${RED}ERROR: Zolt binary not found at $ZOLT_BIN${NC}"
    echo "Build with: cd $ZOLT_DIR && zig build -Doptimize=ReleaseFast"
    exit 1
fi

if [ "$QUICK_MODE" = false ] && [ -z "$JOLT_DIR" ]; then
    echo -e "${RED}ERROR: Jolt directory not found${NC}"
    echo "Set JOLT_DIR env var or clone Jolt to ../jolt"
    exit 1
fi

# Track results
PASSED=0
FAILED=0
PROVE_TIMES=()

echo "Programs: ${PROGRAMS[*]}"
echo "Mode: $([ "$QUICK_MODE" = true ] && echo "Quick (prove only)" || echo "Full (prove + verify)")"
echo ""

for prog in "${PROGRAMS[@]}"; do
    ELF="$ZOLT_DIR/examples/${prog}.elf"
    PROOF="$TMP_DIR/${prog}_proof.bin"
    PREPROC="$TMP_DIR/${prog}_preprocessing.bin"

    if [ ! -f "$ELF" ]; then
        echo -e "${RED}SKIP $prog: ELF not found at $ELF${NC}"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Generate proof
    echo -n "[$prog] Proving... "
    START_TIME=$(date +%s%N)
    if ! "$ZOLT_BIN" prove "$ELF" --jolt-format -o "$PROOF" --export-preprocessing "$PREPROC" 2>/dev/null; then
        echo -e "${RED}FAILED (proof generation)${NC}"
        FAILED=$((FAILED + 1))
        continue
    fi
    END_TIME=$(date +%s%N)
    ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
    PROOF_SIZE=$(wc -c < "$PROOF")
    echo -e "done (${ELAPSED_MS}ms, ${PROOF_SIZE} bytes)"
    PROVE_TIMES+=("$prog:${ELAPSED_MS}ms")

    if [ "$QUICK_MODE" = true ]; then
        PASSED=$((PASSED + 1))
        continue
    fi

    # Verify with Jolt
    echo -n "[$prog] Verifying with Jolt... "
    cp "$PROOF" /tmp/zolt_proof_dory.bin
    cp "$PREPROC" /tmp/zolt_preprocessing.bin
    cp "${PREPROC}.ram" /tmp/zolt_preprocessing.bin.ram

    VERIFY_OUTPUT=$(cd "$JOLT_DIR" && cargo test --package jolt-core --features zolt-debug \
        zolt_compat_test::tests::test_verify_zolt_proof_with_zolt_preprocessing \
        -- --ignored --nocapture 2>&1)

    if echo "$VERIFY_OUTPUT" | grep -q "SUCCESS: Zolt proof verified by Jolt!"; then
        echo -e "${GREEN}PASSED ✓${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAILED ✗${NC}"
        # Show last few lines for debugging
        echo "$VERIFY_OUTPUT" | tail -10
        FAILED=$((FAILED + 1))
    fi
done

# Summary
echo ""
echo "=== Summary ==="
echo -e "Passed: ${GREEN}${PASSED}${NC}/${#PROGRAMS[@]}"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed: ${RED}${FAILED}${NC}/${#PROGRAMS[@]}"
fi

echo ""
echo "Prove times:"
for t in "${PROVE_TIMES[@]}"; do
    echo "  $t"
done

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All ${PASSED} programs verified successfully!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}${FAILED} program(s) failed!${NC}"
    exit 1
fi
