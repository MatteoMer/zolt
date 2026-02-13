#!/bin/bash
# Regression test script for Jolt-format proof generation and verification
# Tests all 8 example programs

ZOLT_BIN="zig-out/bin/zolt"
VERIFIER_BIN="jolt/target/release/zolt-verifier"
EXAMPLES_DIR="examples"
TMP_DIR="/tmp/zolt_regression"

mkdir -p "$TMP_DIR"

PROGRAMS=(collatz fibonacci factorial gcd bitwise primes signed sum)
PASS_COUNT=0
FAIL_COUNT=0

for prog in "${PROGRAMS[@]}"; do
    echo "=============================="
    echo "Testing: $prog"
    echo "=============================="
    
    PROOF_FILE="$TMP_DIR/${prog}_proof.bin"
    PREPROC_FILE="$TMP_DIR/${prog}_preprocessing.bin"
    
    # Generate Jolt-format proof
    echo "[1/2] Generating proof for $prog..."
    if timeout 3600 $ZOLT_BIN prove --jolt-format -o "$PROOF_FILE" --export-preprocessing "$PREPROC_FILE" "$EXAMPLES_DIR/${prog}.elf" 2>&1; then
        echo "  Proof generated: $(stat -c%s "$PROOF_FILE") bytes"
        echo "  Preprocessing: $(stat -c%s "$PREPROC_FILE") bytes"
        
        # Verify with Jolt verifier
        echo "[2/2] Verifying with Jolt..."
        if $VERIFIER_BIN --proof "$PROOF_FILE" --preprocessing "$PREPROC_FILE" 2>&1; then
            echo "  ✅ $prog PASSED"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo "  ❌ $prog FAILED (verification)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "  ❌ $prog FAILED (proof generation)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "=============================="
echo "RESULTS: $PASS_COUNT passed, $FAIL_COUNT failed out of ${#PROGRAMS[@]}"
echo "=============================="

if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    exit 0
else
    echo "⚠️  Some tests failed."
    exit 1
fi
