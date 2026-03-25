# ThreadPool Optimization Report

**Date:** 2026-03-25
**Scope:** Sumcheck inner-loop parallelism + ThreadPool reduce architecture

## What we did

### 1. Sumcheck parallelism (stages 2-4)

Added `parallelReduce` and `parallelForForce` to all sumcheck compute and bind paths
that were previously sequential.

**Files modified:**
- `src/zkvm/ram/val_evaluation.zig` — parallelReduce compute + parallelForForce bind
- `src/zkvm/spartan/sparse_registers.zig` — entry-range parallelReduce + deferred reduction
- `src/zkvm/ram/read_write_checking.zig` — entry-range parallelReduce Phase 1/2 + parallel bind
- `src/zkvm/spartan/product_remainder.zig` — parallelReduce compute + parallel bind
- `src/zkvm/claim_reductions/instruction_lookups.zig` — parallelReduce compute + parallelForForce bind
- `src/zkvm/ram/raf_checking.zig` — parallelReduce compute
- `src/zkvm/ram/output_check.zig` — parallelReduce compute + parallelForForce bind
- `src/zkvm/spartan/stage3_prover.zig` — parallelize Shift/Registers prefix-suffix compute + bind
- `src/zkvm/spartan/stage4_gruen_prover.zig` — add thread_pool, forward to sparse/val
- `src/zkvm/jolt_prover.zig` — wire thread_pool to all sub-provers + [BENCH] output
- `src/zkvm/mod.zig` — [BENCH] overhead phase timing
- `src/main.zig` — [BENCH] total time output

**UnreducedProductAccum** (deferred Montgomery reduction) added to inner loops of:
product_remainder, val_evaluation, raf_checking, sparse_registers. These are the provers
where Jolt uses `mul_to_product_accum` and we didn't.

**Sparse matrix parallelism** uses entry-range splitting (not group-index splitting) to
avoid O(N*threads) scan overhead. Each thread processes complete groups whose first entry
falls within its assigned entry range.

### 2. ThreadPool reduce architecture

Rewrote `reduceImpl` from static pre-chunking to Rayon-equivalent architecture:

**SpinLatch** (`src/utils/thread_pool.zig`):
- 4-byte atomic flag (no futex, no counter)
- Used for internal reduce tree nodes
- `waitWhileWorking`: spin + steal + yield (purely userspace)
- CompletionLatch only at root (for main-thread futex fallback)

**Rayon join pattern**:
- Push right half to deque (stealable)
- Execute left half inline
- After left completes, try to pop right back from own deque
- If right wasn't stolen → run inline with ZERO latch overhead
- If stolen → spin-wait via SpinLatch

**Theft-adaptive splits counter**:
- `splits_remaining` halves each level (like Rayon's `Splitter`)
- `pusher_index` tracks which worker pushed the job
- If executing worker != pusher → stolen → reset splits to `max(num_threads, splits/2)`
- Initial splits = `actual_threads * 2`

**Eliminated per-level overhead**:
- Before: CompletionLatch (128 bytes + futex) per level, phantom right-side Job + latch
- After: SpinLatch (4 bytes) for left child only, right half is direct function call

---

## SHA256-2048 Prover Results

```
Program: SHA256 with 2048-byte input (497K cycles, padded to 524K)
All proofs verified against upstream Jolt verifier.

Stage         Baseline    After       Improvement
─────────     ────────    ─────       ───────────
Stage 2       1828 ms     ~1550 ms    -15%
Stage 4       1470 ms     ~850 ms     -42%
```

---

## ThreadPool vs Rayon Benchmark

```
Workload: parallel reduce Σ a[i]*b[i] over BN254 field element pairs
Config: 100 iters, 5 runs (min-of-runs)
Zig: 9 threads, Rust: 8 threads

N        │ Zig par  Rayon par │ Abs winner         │ Zig ratio  Rayon ratio
─────────┼────────────────────┼────────────────────┼────────────────────────
1K       │ 0.053ms   0.042ms │ Rayon 1.2x faster  │  ~0.9x      ~1.2x
4K       │ 0.093ms   0.169ms │ Zig 1.8x faster    │  ~1.3x      ~1.3x
16K      │ 0.230ms   0.421ms │ Zig 1.8x faster    │  ~2.0x      ~2.3x
64K      │ 0.634ms   0.918ms │ Zig 1.4x faster    │  ~3.0x      ~4.3x
256K     │ 1.599ms   2.764ms │ Zig 1.7x faster    │  ~4.9x      ~5.6x
512K     │ 2.553ms   5.093ms │ Zig 1.9x faster    │  ~6.2x      ~6.0x

Absolute time: Zig wins 5/6
Speedup ratio: varies by run, roughly tied at large sizes
```

Run benchmark: `bash bench/threadpool_vs_rayon/compare.sh`

---

## What Jolt does that we verified against

Checked against actual Jolt code at `~/projects/jolt/` (not assumptions):

| Optimization | Jolt uses it | Zolt uses it | Notes |
|---|---|---|---|
| `mul_to_product_accum` in ProductVirtualRemainder | Yes | **Yes** (added) | |
| `mul_to_product_accum` in RegistersRWC sparse | Yes | Not yet | RWC compute stays sequential |
| `mul_to_product_accum` in RamValCheck | Yes | **Yes** (added) | |
| `mul_to_product_accum` in RafEvaluation | Yes | **Yes** (added) | |
| `mul_to_product_accum` in InstructionLookups | **No** | No | Jolt doesn't use it either |
| `mul_to_product_accum` in OutputSumcheck | **No** (uses par_fold_out_in) | No | Different optimization |
| `par_fold_out_in_unreduced` (Gruen split-eq) | Yes (5 provers) | No | Structural change, not simple drop-in |
| `par_chunk_by` for sparse matrices | Yes | Approximated with entry-range split | |
| Batched sumcheck cross-instance parallelism | **No** (sequential) | No | Neither parallelizes across instances |

---

## Remaining gaps

### 1. Rayon ratio at 64K-256K
Rayon gets ~4-5.5x speedup ratio vs our ~3-5x at these sizes. Our absolute time is faster,
but the ratio gap means we're leaving parallelism on the table. Root cause: Rayon's
`join_context` has lower per-level overhead (SpinLatch probe is a single atomic load, and
the inline-pop-back path avoids latch entirely in the common case). Our implementation
matches this pattern but still has slightly higher overhead from the job identity check.

### 2. `par_fold_out_in_unreduced` equivalent
Jolt's Gruen split-eq has a specialized `par_fold_out_in_unreduced` that combines parallel
folding over E_out blocks WITH deferred Montgomery reduction. This is used in 5 provers
(ProductVirtualRemainder, InstructionInput, OutputSumcheck, Outer, and a couple more).
We don't have an equivalent. Building one would require a parallel fold primitive in the
split-eq polynomial that integrates with our ThreadPool.

### 3. RamRWC sparse compute deferred reduction
The RamReadWriteChecking Phase 1/2 compute uses immediate field ops in the inner merge loop.
Jolt uses `mul_to_product_accum` with nested parallel reduce. We parallelized the outer
group iteration but the inner merge is still sequential with immediate reduction.

### 4. 1K absolute time
We lose to Rayon at N=1K (0.053ms vs 0.042ms). This is pure dispatch overhead for trivially
small workloads. The `effectiveThreads` check uses `MIN_ITEMS_PER_THREAD=256`, so 1024/256=4
threads are used when it should probably stay sequential. Could add a reduce-specific higher
threshold, but this conflicts with the join pattern that benefits small sizes.

---

## Files reference

| File | What |
|---|---|
| `src/utils/thread_pool.zig` | SpinLatch, adaptive reduce, join pattern, theft detection |
| `bench/threadpool_vs_rayon/compare.sh` | Run side-by-side benchmark |
| `bench/threadpool_vs_rayon/main.zig` | Zig ThreadPool benchmark |
| `bench/threadpool_vs_rayon/bench_rayon.rs` | Rust Rayon benchmark |
| `docs/sha256-2048-perf-analysis.md` | Sumcheck stage-by-stage analysis |
| `docs/threadpool-vs-rayon-analysis.md` | Original architecture analysis (pre-implementation) |
