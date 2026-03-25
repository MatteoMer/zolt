#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Building Zig ThreadPool benchmark ==="
cd "$ROOT"
# Build as a test binary that can import project modules
zig build-exe -OReleaseFast \
    -Mroot="$ROOT/bench/threadpool_vs_rayon/main.zig" \
    --deps root:zolt -Mzolt="$ROOT/src/root.zig" \
    --cache-dir .zig-cache --global-cache-dir "$HOME/.cache/zig" \
    --name tp-bench 2>&1 || {
    echo "Standalone build failed, trying inline approach..."
    # Fallback: compile via main build system
    zig run -OReleaseFast "$ROOT/bench/threadpool_vs_rayon/main.zig" \
        --deps root:zolt -Mzolt="$ROOT/src/root.zig" 2>&1
}

echo ""
echo "=== Building Rust Rayon benchmark ==="
cd "$ROOT/bench/threadpool_vs_rayon"
cargo build --release -q 2>&1

echo ""
echo "=========================================="
echo "  Zig ThreadPool"
echo "=========================================="
"$ROOT/tp-bench" 2>/dev/null || "$ROOT/zig-out/bin/tp-bench"

echo ""
echo "=========================================="
echo "  Rust Rayon"
echo "=========================================="
"$ROOT/bench/threadpool_vs_rayon/target/release/rayon-bench"
