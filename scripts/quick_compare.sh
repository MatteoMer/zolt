#!/bin/bash
# Quick side-by-side comparison of Zolt vs Jolt sumcheck

echo "============================================"
echo "       ZOLT vs JOLT SUMCHECK COMPARISON"
echo "============================================"
echo ""

# Extract key values in aligned format
echo "=== INITIAL CLAIM ==="
echo -n "ZOLT: "
grep "STAGE1_INITIAL: claim = " /tmp/zolt.log | head -1 | sed 's/.*claim = //'
echo -n "JOLT: "
grep "STAGE1_INITIAL: claim = " /tmp/jolt.log | head -1 | sed 's/.*claim = //'

echo ""
echo "=== ROUND 0 ==="
echo "--- Current Claim ---"
echo -n "ZOLT: "
grep "STAGE1_ROUND_0: current_claim = " /tmp/zolt.log | head -1 | sed 's/.*current_claim = //'
echo -n "JOLT: "
grep "STAGE1_ROUND_0: current_claim = " /tmp/jolt.log | head -1 | sed 's/.*current_claim = //'

echo ""
echo "--- c0 (little-endian bytes) ---"
echo -n "ZOLT: "
grep "STAGE1_ROUND_0: c0_le = " /tmp/zolt.log | head -1 | sed 's/.*c0_le = //'
echo -n "JOLT: "
grep "STAGE1_ROUND_0: c0_bytes = " /tmp/jolt.log | head -1 | sed 's/.*c0_bytes = //'

echo ""
echo "--- c2 (little-endian bytes) ---"
echo -n "ZOLT: "
grep "STAGE1_ROUND_0: c2_le = " /tmp/zolt.log | head -1 | sed 's/.*c2_le = //'
echo -n "JOLT: "
grep "STAGE1_ROUND_0: c2_bytes = " /tmp/jolt.log | head -1 | sed 's/.*c2_bytes = //'

echo ""
echo "--- Challenge (little-endian bytes) ---"
echo -n "ZOLT: "
grep "STAGE1_ROUND_0: challenge_le = " /tmp/zolt.log | head -1 | sed 's/.*challenge_le = //'
echo -n "JOLT: "
grep "STAGE1_ROUND_0: challenge_bytes = " /tmp/jolt.log | head -1 | sed 's/.*challenge_bytes = //'

echo ""
echo "--- Next Claim ---"
echo -n "ZOLT: "
grep "STAGE1_ROUND_0: next_claim = " /tmp/zolt.log | head -1 | sed 's/.*next_claim = //'
echo -n "JOLT: "
grep "STAGE1_ROUND_0: next_claim = " /tmp/jolt.log | head -1 | sed 's/.*next_claim = //'

echo ""
echo "=== ROUND 1 ==="
echo "--- c0 (little-endian bytes) ---"
echo -n "ZOLT: "
grep "STAGE1_ROUND_1: c0_le = " /tmp/zolt.log | head -1 | sed 's/.*c0_le = //'
echo -n "JOLT: "
grep "STAGE1_ROUND_1: c0_bytes = " /tmp/jolt.log | head -1 | sed 's/.*c0_bytes = //'

echo ""
echo "--- Challenge ---"
echo -n "ZOLT: "
grep "STAGE1_ROUND_1: challenge_le = " /tmp/zolt.log | head -1 | sed 's/.*challenge_le = //'
echo -n "JOLT: "
grep "STAGE1_ROUND_1: challenge_bytes = " /tmp/jolt.log | head -1 | sed 's/.*challenge_bytes = //'

echo ""
echo "=== FINAL OUTPUT ==="
echo -n "ZOLT output_claim: "
grep "output_claim (scaled)" /tmp/zolt.log | tail -1 | sed 's/.*= //'
echo -n "JOLT output_claim: "
grep "SUMCHECK VERIFICATION FAILED" -A1 /tmp/jolt.log | grep "output_claim" | sed 's/.*: *//'
echo -n "JOLT expected:     "
grep "expected_output_claim:" /tmp/jolt.log | tail -1 | sed 's/.*: *//'

echo ""
echo "============================================"
echo "Key things to check:"
echo "1. Do initial claims match?"
echo "2. Do c0/c2/c3 coefficients match at each round?"
echo "3. Do challenges match?"
echo "4. Does final output_claim match expected?"
echo "============================================"
