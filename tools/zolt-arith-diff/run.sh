#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

cargo run --release \
  --manifest-path "$ROOT/tools/zolt-arith-diff/arkworks-fixtures/Cargo.toml" \
  -- \
  --out-dir "$ROOT/testdata/zolt-arith-diff"

zig build test-zolt-arith-diff
