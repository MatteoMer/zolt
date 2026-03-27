# Metal GPU Acceleration for Zolt — Initial Report

**Date:** 2026-03-25
**Branch:** `feature/metal-gpu`
**Target:** Apple Silicon (M1/M2/M3/M4), macOS, AWS mac2.metal instances

---

## 0. Setup Guide

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Xcode installed (for Metal compiler — App Store or developer.apple.com)
- Xcode Command Line Tools installed (`xcode-select --install`)
- Zig 0.15.x (via Homebrew: `brew install zig`)

### One-Time Setup

```bash
# 1. Install the Metal Toolchain (required even with Xcode installed)
xcodebuild -downloadComponent MetalToolchain

# 2. Ensure xcode-select points to Command Line Tools (Zig needs this)
sudo xcode-select -s /Library/Developer/CommandLineTools

# 3. Verify both work
zig version                  # should print 0.15.x
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    xcrun --find metal       # should print a path to the metal binary

# 4. Build and test
zig build -Doptimize=ReleaseFast
```

### Why the dual SDK setup?

Zig's linker cannot parse the `.tbd` stubs in the Xcode 26 macOS SDK (link errors for
basic libc symbols like `_abort`, `_malloc`, etc.). The Command Line Tools ship older
SDKs (15.x) that work. The Metal compiler only ships with Xcode. So:

- **Zig builds** use CLT (via `xcode-select`)
- **Metal shader compilation** uses Xcode (via `DEVELOPER_DIR` env var in `build.zig`)

This is automatic — `build.zig` handles the `DEVELOPER_DIR` override. Contributors
just need both Xcode and CLT installed.

### AWS mac2.metal Setup

Same steps, but install Xcode via CLI:

```bash
# On a mac2.metal or mac2-m2pro.metal instance
xcode-select --install
# Download Xcode from developer.apple.com or use xcodes CLI tool
xcodebuild -downloadComponent MetalToolchain
sudo xcode-select -s /Library/Developer/CommandLineTools
```

---

## 1. Why Metal on Apple Silicon

### Unified Memory Changes the Game

Discrete GPUs (NVIDIA/AMD) require PCIe transfers between CPU and GPU memory. For ZK
proving, this is catastrophic: the sumcheck protocol alternates between GPU-parallel
evaluation and CPU-sequential Fiat-Shamir hashing every round. With 19 rounds for a
524K-trace proof, that's 19 × 2 PCIe round-trips per stage — the transfer latency alone
can exceed the compute savings.

Apple Silicon has **unified memory**: CPU and GPU share the same physical DRAM. A Metal
buffer created with `storageModeShared` is directly accessible by both processors with
zero copies. The sumcheck loop becomes:

```
1. GPU: compute round polynomial    (2^n parallel field ops)
2. CPU: Fiat-Shamir hash            (sequential, ~microseconds)
3. GPU: bind polynomial             (2^(n-1) parallel ops)
4. goto 1                           ← no transfer, no sync overhead
```

This makes Apple Silicon the ideal architecture for interactive-style ZK protocols.

### M1 Pro GPU Specs (this machine)

- 14 GPU cores × 128 ALUs = 1,792 execution units
- SIMD group width: 32 threads
- Max threads per threadgroup: 1024
- Shared memory per threadgroup: 32 KB
- Metal 4 (64-bit integers available since MSL 2.2; atomics since Apple GPU Family 8+)
- Memory bandwidth: ~200 GB/s (shared with CPU)

### AWS mac2.metal Instances

- `mac2.metal`: M1, 8 GPU cores, 16 GB unified memory
- `mac2-m2pro.metal`: M2 Pro, up to 19 GPU cores, 32 GB unified memory
- Bare metal — full Metal API access, no virtualization layer

---

## 2. Architecture: Pure-Zig Metal Integration

### The "No Deps, No FFI" Constraint

Zolt uses LLVM carry intrinsics (`llvm.x86.addcarry.u64`) and ARM64 inline assembly
(`asm volatile`) as core language features — not FFI. The same principle applies to Metal
integration: Zig can natively call C ABI functions, and Apple's Objective-C runtime is a
C library (`libobjc.dylib`). We use the platform, not a dependency.

### How It Works

Metal's API is Objective-C, but the Objective-C runtime is just C functions:

```zig
// These are system-level C calls, not FFI wrappers
const objc = @cImport({
    @cInclude("objc/runtime.h");
    @cInclude("objc/message.h");
});

// Metal device creation is a plain C function
extern "c" fn MTLCreateSystemDefaultDevice() ?*anyopaque;

// All method calls go through objc_msgSend
const device = MTLCreateSystemDefaultDevice();
const queue = objc.objc_msgSend(device, sel("newCommandQueue"));
```

This is **zero dependencies**: we link against system frameworks that ship with macOS
(`Metal.framework`, `Foundation.framework`), the same way we link against `libSystem.dylib`
for thread primitives. No Cargo crates, no C wrapper libraries, no vendored code.

### Build Integration

```zig
// In build.zig — link system frameworks
exe.root_module.linkFramework("Metal");
exe.root_module.linkFramework("CoreGraphics");
exe.root_module.linkFramework("Foundation");

// Compile .metal shaders to .metallib at build time
// Use DEVELOPER_DIR override so xcode-select can stay on CLT (Zig needs CLT)
const metal_compile = b.addSystemCommand(&.{
    "xcrun", "metal", "-O3", "-o", "shaders.air", "src/gpu/shaders.metal"
});
metal_compile.setEnvironmentVariable("DEVELOPER_DIR",
    "/Applications/Xcode.app/Contents/Developer");

const metal_link = b.addSystemCommand(&.{
    "xcrun", "metallib", "-o", "shaders.metallib", "shaders.air"
});
metal_link.setEnvironmentVariable("DEVELOPER_DIR",
    "/Applications/Xcode.app/Contents/Developer");
```

MSL shaders are compiled by Apple's `metal` compiler at build time. The resulting
`.metallib` binary is embedded or loaded at runtime. No runtime shader compilation needed
in production (though Metal supports it for development).

**Important: Dual SDK setup.** Zig's linker is incompatible with Xcode 26's macOS SDK
(undefined symbol errors for basic libc). The Metal compiler requires Xcode. Solution:

- `xcode-select` stays pointed at **Command Line Tools** (for Zig)
- Metal tools are invoked via `xcrun` with `DEVELOPER_DIR` override (for shader compilation)
- This is handled automatically in `build.zig` — no manual switching needed

---

## 3. The GPU Compute Layer

### Design Principles

1. **Zolt-specific, not general-purpose.** We don't build a GPU abstraction library.
   We build exactly what the prover needs: field arithmetic dispatch, polynomial
   operations, and MSM. No unused generality.

2. **Thin integration layer.** The Zig-side Metal code is a small module (`src/gpu/`)
   that handles device setup, buffer management, and kernel dispatch. All math lives
   in MSL shaders.

3. **CPU/GPU hybrid.** The GPU handles bulk data-parallel field operations. The CPU
   handles Fiat-Shamir, transcript management, control flow, and small/irregular work.
   The ThreadPool continues to handle CPU parallelism for work that doesn't justify GPU
   dispatch.

4. **Unified buffer pool.** Pre-allocate shared-mode Metal buffers at proof start,
   reuse across stages. No per-round allocation.

### Module Structure

```
src/gpu/
├── device.zig          # Metal device, command queue, pipeline state cache
├── buffer.zig          # Shared-mode buffer pool, zero-copy field element arrays
├── dispatch.zig        # Kernel dispatch helpers (threadgroup sizing, sync)
├── kernels.zig         # Zig-side kernel interfaces (type-safe wrappers)
└── shaders/
    ├── field.metal      # BN254 field arithmetic (mul, add, sub, inv)
    ├── poly.metal       # Polynomial bind, evaluate, fold
    ├── reduce.metal     # Tree reduction (field add, field mul)
    ├── msm.metal        # Multi-scalar multiplication (Pippenger buckets)
    └── sumcheck.metal   # Sumcheck round computation (fused kernel)
```

### Buffer Management

```zig
// Shared-mode buffer — accessible by both CPU and GPU, zero-copy
const GpuBuffer = struct {
    metal_buffer: *anyopaque,   // MTLBuffer
    ptr: [*]F,                  // CPU-visible pointer to same memory
    len: usize,

    /// Read from CPU after GPU writes — no copy, just cache invalidation
    pub fn cpuSlice(self: *GpuBuffer) []F {
        return self.ptr[0..self.len];
    }
};
```

On Apple Silicon, `storageModeShared` buffers are coherent: the CPU can read GPU results
immediately after the command buffer completes. No explicit flush or invalidate needed.

---

## 4. MSL Kernel Design

### 4.1 BN254 Field Arithmetic (32-bit Limb Representation)

**Critical design decision:** Use 8×32-bit limbs on GPU, not 4×64-bit.

Apple Silicon GPUs are optimized for 32-bit operations (graphics heritage). While Metal 4
supports 64-bit integers, the 32-bit ALU throughput is significantly higher. Every major
GPU ZK implementation (Icicle, SPPARK, Barretenberg) uses 32-bit limbs for this reason.

```metal
// BN254 scalar field element: 8 × 32-bit limbs (little-endian)
struct Fp256 {
    uint32_t limbs[8];
};

// Montgomery multiplication (CIOS, 32-bit limbs)
// Input: a, b in Montgomery form
// Output: a * b * R^(-1) mod p  (in Montgomery form)
Fp256 fp_mul(Fp256 a, Fp256 b, constant Fp256& p, constant uint32_t& inv) {
    uint64_t t[9] = {0};  // accumulator (one extra limb)

    for (int i = 0; i < 8; i++) {
        // Multiply-accumulate: t += a * b[i]
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a.limbs[j] * b.limbs[i] + t[j] + carry;
            t[j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        t[8] += carry;

        // Montgomery reduction step
        uint32_t m = (uint32_t)(t[0] * inv);
        carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)m * p.limbs[j] + t[j] + carry;
            t[j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        t[8] += carry;

        // Shift right by one limb
        for (int j = 0; j < 8; j++) t[j] = t[j+1];
        t[8] = 0;
    }

    // Final conditional subtraction
    Fp256 result;
    // ... (subtract p if t >= p)
    return result;
}
```

**Conversion:** Field elements are stored in 4×64-bit format on the CPU side. On GPU
dispatch, we convert to 8×32-bit. This conversion is itself data-parallel and cheap.
Alternatively, we keep a dual representation, or we standardize on 8×32-bit everywhere
(worth benchmarking).

### 4.2 Polynomial Bind Kernel

The innermost operation in sumcheck. Called every round, over 2^(n-round) elements.

```metal
// bindFirst: evals_out[i] = evals[2i] + r * (evals[2i+1] - evals[2i])
kernel void poly_bind(
    device const Fp256* evals   [[buffer(0)]],
    device Fp256*       out     [[buffer(1)]],
    constant Fp256&     r       [[buffer(2)]],
    constant uint&      half_n  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= half_n) return;

    Fp256 lo = evals[tid * 2];
    Fp256 hi = evals[tid * 2 + 1];
    Fp256 diff = fp_sub(hi, lo);
    out[tid] = fp_add(lo, fp_mul(diff, r));
}
```

For SHA256-2048 (524K trace, 19 vars): first round dispatches 262,144 threads. Each
thread does 1 sub + 1 mul + 1 add = 3 field ops. On 1,792 ALUs, this is ~146 waves
of work — plenty to saturate the GPU.

### 4.3 Sumcheck Round Computation (Fused Kernel)

Instead of separate compute + reduce, fuse the entire round polynomial computation into
one kernel with threadgroup-level reduction:

```metal
// Compute sumcheck round polynomial at evaluation points r=0, r=1, r=2
// Then reduce across all threads using threadgroup shared memory
kernel void sumcheck_round(
    device const Fp256* poly_evals  [[buffer(0)]],
    device const Fp256* eq_evals    [[buffer(1)]],
    device Fp256*       round_poly  [[buffer(2)]],  // output: 3 values
    constant uint&      half_n      [[buffer(3)]],
    uint tid     [[thread_position_in_grid]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Each thread handles one pair (2i, 2i+1)
    Fp256 accum[3] = {fp_zero(), fp_zero(), fp_zero()};

    if (tid < half_n) {
        Fp256 p_lo = poly_evals[tid * 2];
        Fp256 p_hi = poly_evals[tid * 2 + 1];
        Fp256 e_lo = eq_evals[tid * 2];
        Fp256 e_hi = eq_evals[tid * 2 + 1];

        // eval at r=0: p_lo * e_lo
        accum[0] = fp_mul(p_lo, e_lo);
        // eval at r=1: p_hi * e_hi
        accum[1] = fp_mul(p_hi, e_hi);
        // eval at r=2: (2*p_hi - p_lo) * (2*e_hi - e_lo)
        accum[2] = fp_mul(
            fp_sub(fp_add(p_hi, p_hi), p_lo),
            fp_sub(fp_add(e_hi, e_hi), e_lo)
        );
    }

    // Threadgroup reduction using shared memory
    threadgroup Fp256 shared[3][256];
    for (int k = 0; k < 3; k++) shared[k][lid] = accum[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduce within threadgroup
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            for (int k = 0; k < 3; k++)
                shared[k][lid] = fp_add(shared[k][lid], shared[k][lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Threadgroup leader writes partial sum
    if (lid == 0) {
        for (int k = 0; k < 3; k++)
            round_poly[tgid * 3 + k] = shared[k][0];
    }
}
```

This fused kernel does everything in one dispatch:
- Evaluate the multilinear polynomial at 3 points (r=0, r=1, r=2)
- Reduce all evaluations to partial sums per threadgroup
- A small CPU-side final reduction sums the threadgroup partials

### 4.4 MSM (Pippenger Buckets)

MSM is the dominant cost in commitment phases (Stage 8: 3.2s for commit). Pippenger's
algorithm has three phases:

1. **Bucket accumulation** — massively parallel (one point per thread)
2. **Bucket reduction** — parallel within each window
3. **Window combination** — sequential (small)

```metal
// Phase 1: Classify each scalar into buckets and accumulate the point
kernel void msm_bucket_accumulate(
    device const AffinePoint* points     [[buffer(0)]],
    device const uint32_t*    scalars    [[buffer(1)]],  // decomposed by window
    device atomic<ProjectivePoint>* buckets [[buffer(2)]],
    constant uint& window_bits           [[buffer(3)]],
    constant uint& window_idx            [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint bucket_idx = extract_window(scalars[tid], window_idx, window_bits);
    if (bucket_idx == 0) return;

    // Atomic point addition into bucket
    // (requires careful implementation — see section 5)
    atomic_point_add(&buckets[bucket_idx - 1], points[tid]);
}
```

The atomic point addition is the tricky part. Alternatives to atomics:
- **Sort-then-reduce**: Sort (scalar, point) pairs by bucket index, then reduce
  contiguous runs. Avoids atomics entirely. Better for GPU.
- **Histogram + scatter**: Two-pass — count bucket sizes, then scatter points into
  bucket arrays, then parallel reduce each bucket.

The sort-then-reduce approach is likely best here. Metal has efficient parallel radix sort
primitives available through SIMD group operations.

### 4.5 UnreducedProductAccum on GPU

Zolt's deferred reduction pattern maps perfectly to GPU:

```metal
// Fused multiply-accumulate without intermediate reduction
// Accumulate products as 9 × 64-bit limbs, reduce once at the end
struct UnreducedAccum {
    uint64_t limbs[9];
};

kernel void fold_accum(
    device const Fp256* a      [[buffer(0)]],
    device const Fp256* b      [[buffer(1)]],
    device Fp256*       out    [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Each thread accumulates a range of products without reduction
    UnreducedAccum accum = {0};
    uint chunk = (n + tg_size - 1) / tg_size;
    uint start = tid * chunk;
    uint end = min(start + chunk, n);

    for (uint i = start; i < end; i++) {
        mul_to_accum(&accum, a[i], b[i]);  // no reduction!
    }

    // Reduce once, then threadgroup-reduce the results
    Fp256 result = reduce_accum(accum);
    // ... threadgroup reduction ...
}
```

---

## 5. Challenges and Design Decisions

### 5.1 32-bit vs 64-bit Limbs

| Aspect | 8×32-bit | 4×64-bit |
|--------|----------|----------|
| ALU utilization | Full (32-bit optimized) | Partial (64-bit emulated on some paths) |
| Registers per element | 8 | 4 (but 64-bit registers may count as 2) |
| Multiply cost | 64 `imadd` ops (full 32-bit) | 16 `mul` ops (but each is 64-bit) |
| CPU compatibility | Requires conversion | Same format as CPU |
| Industry precedent | Icicle, SPPARK, all GPU ZK | None on GPU |

**Recommendation:** 8×32-bit limbs on GPU. The conversion cost is negligible compared
to the throughput gain. Keep 4×64-bit on CPU — the conversion boundary is at the
GPU dispatch interface.

### 5.2 Dispatch Granularity

Not every operation should go to the GPU. The full dispatch overhead (command buffer
creation, encoding, commit, GPU start) is ~10-50μs on Apple Silicon. Rules:

- **< 4K elements**: CPU only (ThreadPool or sequential)
- **4K-64K elements**: Depends on kernel complexity — benchmark to find crossover
- **> 64K elements**: GPU dispatch (massive parallelism wins)

For SHA256-2048 with 524K trace, most operations are well above the threshold.

### 5.3 Synchronization Between CPU and GPU

The sumcheck protocol requires alternating CPU/GPU work:

```
Round 1: GPU compute (262K ops) → CPU hash → GPU bind (262K→131K)
Round 2: GPU compute (131K ops) → CPU hash → GPU bind (131K→65K)
...
Round 7: GPU compute (4K ops) → CPU hash → GPU bind (4K→2K)
Round 8+: CPU only (below 4K threshold — dispatch overhead exceeds compute)
```

On Apple Silicon, the "→" transitions are near-free (unified memory). But we still need
to wait for GPU completion before the CPU reads results. Metal provides:

- **`waitUntilCompleted`**: Block until GPU finishes (simplest)
- **`addCompletedHandler`**: Callback when GPU finishes (for overlap)
- **Shared event signaling**: Fine-grained GPU→CPU sync

For the sumcheck loop, `waitUntilCompleted` is sufficient — the CPU work between GPU
dispatches is tiny (hash + challenge generation), so there's nothing to overlap.

### 5.4 Double Buffering

While one command buffer executes on GPU, we can encode the next one on CPU. This hides
encoding latency:

```
GPU:  [compute round 1] [bind round 1] [compute round 2] [bind round 2]  ...
CPU:  [encode rnd1][hash][encode bind1][hash][encode rnd2]                ...
          ^overlap^         ^overlap^
```

This is a refinement for later phases. Initial implementation uses simple synchronous
dispatch.

---

## 6. Integration with Existing Code

### 6.1 GpuAccelerator Struct

A single entry point that each prover stage can use:

```zig
pub const GpuAccelerator = struct {
    device: *anyopaque,          // MTLDevice
    queue: *anyopaque,           // MTLCommandQueue
    pipelines: PipelineCache,    // Pre-compiled compute pipelines
    buffer_pool: BufferPool,     // Reusable shared-mode buffers

    /// Compute sumcheck round polynomial on GPU
    /// Returns 3 field elements (evaluations at r=0, r=1, r=2)
    pub fn sumcheckRound(self: *GpuAccelerator, poly: []const F, eq: []const F) [3]F

    /// Bind polynomial on GPU (fold 2^n → 2^(n-1))
    pub fn polyBind(self: *GpuAccelerator, evals: []F, r: F) void

    /// Batched field multiply: out[i] = a[i] * b[i]
    pub fn fieldMulBatch(self: *GpuAccelerator, a: []const F, b: []const F, out: []F) void

    /// Multi-scalar multiplication
    pub fn msm(self: *GpuAccelerator, scalars: []const F, points: []const AffinePoint) ProjectivePoint

    /// Check if GPU is available (comptime + runtime)
    pub fn isAvailable() bool
};
```

### 6.2 Transparent Fallback

On non-Apple platforms or when Metal is unavailable, the existing CPU code path runs
unchanged. The switch is comptime where possible:

```zig
const gpu = if (comptime GpuAccelerator.isAvailable())
    try GpuAccelerator.init(allocator)
else
    null;

// In hot path:
if (gpu) |g| {
    g.sumcheckRound(poly, eq);
} else {
    // Existing CPU path — unchanged
    computeRoundPolynomialCPU(poly, eq);
}
```

### 6.3 GPU Across All Stages

Every stage runs on GPU. No exceptions, no "skip — already fast" carve-outs. The goal
is a single execution model: all field arithmetic goes to Metal. This is long-term work
on a pre-production codebase — consistency and simplicity of the execution model matter
more than cherry-picking wins.

| Stage | GPU Opportunity | Expected Speedup |
|-------|----------------|-------------------|
| Stage 1 (Outer) | Sumcheck rounds + commit | 3-5x |
| Stage 2 (Product) | Sumcheck rounds (currently 0 parallelism) | 10-20x |
| Stage 3 (Shift+Regs) | All sub-prover sumcheck loops | 3-5x |
| Stage 4 (Sparse) | Sumcheck rounds (currently 0 parallelism) | 10-20x |
| Stage 5 (Lookups) | Sumcheck rounds | 2-4x |
| Stage 6 (Bytecode) | Sumcheck rounds | 2-4x |
| Stage 7 (Hamming) | Sumcheck rounds (small, but uniform model) | 1-2x |
| Stage 8 (Commit) | MSM + polynomial evaluation | 20-100x |

The win isn't just per-stage speedup. It's that the GPU kernels become the **single
implementation** of field arithmetic in the hot path. As the MSL kernels improve, every
stage benefits. No need to optimize CPU field ops and GPU field ops separately.

---

## 7. Phased Roadmap

### Phase 0: Foundation (the Metal-Zig bridge)

Build the core infrastructure:

- `src/gpu/device.zig` — Metal device init, command queue, pipeline state cache
- `src/gpu/buffer.zig` — Shared-mode buffer pool with zero-copy field element access
- Objective-C runtime bindings (minimal: `objc_msgSend`, selector caching)
- Build system integration (compile `.metal` → `.metallib`, link frameworks)
- Smoke test: dispatch a trivial kernel, verify results on CPU

**Deliverable:** `GpuAccelerator.init()` works, can dispatch a no-op kernel.

### Phase 1: Field Arithmetic on GPU

- MSL implementation of BN254 field ops with 8×32-bit limbs
- Montgomery multiplication (CIOS), add, sub, neg
- Format conversion: 4×64-bit ↔ 8×32-bit
- Correctness tests: compare GPU results against CPU for all field ops
- Batch multiply kernel + benchmark vs CPU ThreadPool

**Deliverable:** `fieldMulBatch()` works correctly and is faster than CPU for N > 16K.

### Phase 2: Polynomial Operations

- `poly_bind` kernel (the sumcheck workhorse)
- `sumcheck_round` fused kernel (compute + reduce in one dispatch)
- Tree reduction kernel (field add)
- UnreducedProductAccum on GPU
- Benchmark: polynomial bind for various sizes (1K → 1M)

**Deliverable:** `sumcheckRound()` and `polyBind()` work, integrated into one prover
stage as proof of concept.

### Phase 3: Prover Integration (All Stages)

- Wire `GpuAccelerator` into every prover stage (1-8)
- Stage 2 (`ProductVirtualRemainderProver`): GPU sumcheck rounds
- Stage 4 (`Stage4GruenProver`, `ValEvaluationProver`): GPU sumcheck rounds
- Stages 1, 3, 5, 6, 7: GPU sumcheck rounds (same kernel, different data)
- Handle the sumcheck loop: GPU compute → CPU hash → GPU bind
- End-to-end proof correctness test (verify against Jolt verifier)
- Benchmark SHA256-2048 with full GPU proving

**Deliverable:** Full proof generation with GPU across all stages. Proofs verify
against upstream Jolt verifier — same bytes, different execution model.

### Phase 4: MSM on GPU

- Pippenger bucket accumulation kernel
- Sort-then-reduce approach (avoid atomics)
- Bucket reduction kernel (tree reduce per bucket)
- Window combination (CPU — small)
- GLV decomposition integration (existing `src/msm/glv.zig`)
- Benchmark: MSM for typical commitment sizes

**Deliverable:** `msm()` on GPU, integrated into Stage 8 commit phase. Stage 8 commit
drops from 3.2s to sub-second.

### Phase 5: Optimization

- Double buffering for sumcheck loop
- Pipeline state caching and warm-up
- Memory pool tuning (pre-allocate for max trace size)
- Adaptive dispatch threshold tuning (find the crossover point per kernel)
- End-to-end benchmarks across all test programs

**Deliverable:** Tuned GPU prover. Full benchmark suite.

---

## 8. Expected Performance Impact

### SHA256-2048 Projections

```
Stage              Current (ms)  Projected (ms)  Speedup
─────────────────  ────────────  ──────────────  ───────
Commit + Stage 1       5185          2000         2.6x
Stage 2                1828           150        12.2x   ← from 0 parallelism to GPU
Stage 3                 824           300         2.7x
Stage 4                1470           100        14.7x   ← from 0 parallelism to GPU
Stage 5                1185           500         2.4x
Stage 6                2456          1000         2.5x
Stage 7                 156            80         2.0x   ← GPU uniform model
Stage 8                2899           500         5.8x   ← MSM on GPU
─────────────────  ────────────  ──────────────  ───────
PROVE TOTAL           16002          4630         3.5x
WALL CLOCK            18840          6964         2.7x
```

These projections are conservative for stages 2/4 (going from sequential CPU to GPU is a
massive jump) and moderate for MSM (well-studied problem with known GPU speedups).

### Comparison Target

Jolt SHA256-2048: 11.6s wall clock.
Projected Zolt with Metal: ~7s wall clock — **1.7x faster than Jolt**.

For larger traces (primes_large, 65K cycles), the GPU advantage compounds: more
parallelism, better ALU saturation. The current 12.1x slowdown should flip to a
significant advantage.

---

## 9. Risks and Considerations

### Platform Lock-in

Metal is Apple-only. This acceleration path works on macOS (local dev) and AWS mac2.metal
(cloud). It does not work on Linux/NVIDIA. If Linux GPU support is needed later, the
architecture (thin integration layer, separate shader files) makes it straightforward to
add a Vulkan compute backend with the same Zig-side API.

### 32-bit Limb Performance on Apple GPU

The actual 32-bit integer multiply throughput on Apple Silicon GPUs is not as well
documented as NVIDIA's. Early benchmarking (Phase 1) will validate the expected throughput.
If 32-bit performance disappoints, we can evaluate 64-bit limbs (Metal 4 supports them)
or a hybrid approach.

### Shader Debugging

MSL compute shader debugging is harder than CPU debugging. Mitigations:
- Extensive CPU-side reference tests (compare GPU output against CPU for every operation)
- Metal GPU capture in Xcode for performance profiling
- Start with simple kernels, build complexity incrementally

### Minimum Dispatch Size

For very small traces (256 cycles), GPU dispatch overhead exceeds the compute savings.
Even within large proofs, the last ~11 sumcheck rounds (below 4K elements) should stay
on CPU. The crossover point depends on kernel complexity and needs benchmarking.

---

## 10. What We Are NOT Doing

- **Not wrapping an existing GPU library.** No Icicle, no SPPARK, no Barretenberg.
  We write our own MSL kernels, designed for exactly what Zolt needs.

- **Not building a general-purpose GPU framework.** No abstraction layers, no backend
  system, no plug-in architecture. One module (`src/gpu/`), one target (Metal), one
  purpose (accelerate the prover).

- **Not optimizing for NVIDIA.** Metal first. If Vulkan/CUDA is needed later, the
  architecture supports it, but we're not designing for it now.

- **Not changing the proof format.** The output is still a Jolt-compatible proof. The
  verifier doesn't know or care that the prover used a GPU.

---

## Appendix: Key References

- Apple Metal Shading Language Specification (Metal 4)
- Apple GPU architecture overview (WWDC sessions on Metal compute)
- Icicle GPU ZK library (32-bit limb patterns): https://github.com/ingonyama-zk/icicle
- Pippenger MSM on GPU (ZPrize competition results)
- Existing Zolt performance analysis: `docs/sha256-2048-perf-analysis.md`
- Existing ThreadPool design: `docs/threadpool-optimization-report.md`
