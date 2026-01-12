#!/bin/bash
# Build Zolt, generate proof, and verify with Jolt

LOGS_DIR="${ZOLT_LOGS_DIR:-$(dirname "$0")/../logs}"
mkdir -p "$LOGS_DIR"

zig build && \
    ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
    --export-preprocessing "$LOGS_DIR/zolt_preprocessing.bin" \
    -o "$LOGS_DIR/zolt_proof_dory.bin" --srs /tmp/jolt_dory_srs.bin && \
    cd ../jolt && cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture 2>&1 | tee "$LOGS_DIR/jolt.log" && cd -
