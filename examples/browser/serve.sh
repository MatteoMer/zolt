#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Build threaded wasm32 (with atomics + bulk_memory)
echo "Building wasm32-threaded..."
zig build -Dtarget=wasm32-freestanding -Dcpu=generic+atomics+bulk_memory -Doptimize=ReleaseFast
mkdir -p zig-out/wasm32-threaded
cp zig-out/bin/zolt_capi.wasm zig-out/wasm32-threaded/

# Build plain wasm64 and wasm32 (single-threaded fallback)
echo "Building wasm64..."
zig build -Dtarget=wasm64-freestanding -Doptimize=ReleaseFast
mkdir -p zig-out/wasm64
cp zig-out/bin/zolt_capi.wasm zig-out/wasm64/

echo "Building wasm32..."
zig build -Dtarget=wasm32-freestanding -Doptimize=ReleaseFast
mkdir -p zig-out/wasm32
cp zig-out/bin/zolt_capi.wasm zig-out/wasm32/

PORT="${1:-8080}"
echo ""
echo "Serving at http://localhost:$PORT/examples/browser/"
echo "Press Ctrl-C to stop."

# SharedArrayBuffer requires Cross-Origin-Opener-Policy and
# Cross-Origin-Embedder-Policy headers. Use a custom Python handler.
python3 - "$PORT" <<'PYEOF'
import sys, http.server, functools

class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

port = int(sys.argv[1])
server = http.server.HTTPServer(("", port), COOPCOEPHandler)
server.serve_forever()
PYEOF
