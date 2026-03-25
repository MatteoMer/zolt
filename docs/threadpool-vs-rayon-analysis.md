# ThreadPool vs Rayon: Architecture Analysis

**Date:** 2026-03-25
**Context:** Investigating why Zolt's parallel speedup degrades at large sizes despite
faster sequential and parallel throughput than Rust/Rayon in absolute terms.

## Benchmark Results

```
Workload: parallel reduce Σ a[i]*b[i] over BN254 field elements
Zig: 9 threads (ThreadPool), Rust: 8 threads (Rayon)

N (pairs)  │ Zig seq   Zig par  spdup │ Rust seq  Rust par  spdup │ Par effic.
───────────┼──────────────────────────┼──────────────────────────┼───────────
     1,024 │  0.065ms   0.059ms  1.1x │  0.117ms   0.121ms  1.0x │ Z:39% R:27%
    65,536 │  2.577ms   0.735ms  3.5x │  3.668ms   1.671ms  2.2x │ Z:39% R:27%
   262,144 │  7.761ms   3.243ms  2.4x │ 16.312ms   4.662ms  3.5x │ Z:27% R:44%
   524,288 │ 16.062ms   6.550ms  2.5x │ 29.828ms   8.977ms  3.3x │ Z:27% R:42%
```

**Key observation:** Zig is faster in absolute time at EVERY size. But parallel efficiency
drops from 39% to 27% at large sizes, while Rayon's efficiency INCREASES to 42%.

**This means:** With equal parallel efficiency we'd be ~1.5x faster than Jolt, not 1.6x slower.

---

## ThreadPool Architecture (Zolt)

### What it does well

1. **Chase-Lev work-stealing deques** — lock-free, proven design (2013 PPoPP paper)
2. **64-byte cache-line-aligned Jobs** — no heap allocation per job
3. **Caller participates** — waitWhileWorking() pops/steals while waiting
4. **Adaptive binary splitting in parallelFor** — rangeJobExecute recursively splits
5. **Nested dispatch via TLS** — supports unlimited nesting depth
6. **Generation-based wake** — no thundering herd
7. **Cache-line padded partials** — prevents false sharing in reduce

### The problem: reduceImpl uses static chunking

```
parallelForImpl:  Creates 1 job → recursively splits into binary tree
                  Workers steal subtrees → natural load balancing
                  Depth: ~log2(N/threshold)

reduceImpl:       Creates N_threads jobs upfront → pushes ALL to one deque
                  Workers steal one chunk each → fixed assignment
                  No rebalancing after initial steal
```

In `reduceImpl` (line 675):
```zig
const chunk_size = (len + actual_threads - 1) / actual_threads;
const num_chunks = (len + chunk_size - 1) / chunk_size;

// Push one job per chunk — ALL pushed to caller's single deque
for (0..num_chunks) |chunk_idx| {
    const job = Job.initFrom(ReduceJobPayload, payload, &reduceJobExecute);
    worker.deque.push(job);
}
```

**Problems with this approach at large sizes:**

1. **No work rebalancing.** If one thread's chunk hits L3 cache while another is in L1,
   the slow thread holds up the sequential combine phase. No sub-splitting possible.

2. **Initial steal contention.** All N chunks pushed to one deque. N-1 workers must
   sequentially CAS the `top` field to steal their chunk.

3. **Sequential reduction at end.** After all chunks complete, caller sequentially
   reduces N partial results. For N=9 field elements this is negligible, but the
   pattern prevents tree-shaped parallel reduction.

---

## Rayon Architecture

### How reduce works in Rayon

Rayon uses **recursive binary splitting** for ALL operations including reduce:

```rust
fn helper(len, migrated, splitter, producer, consumer) -> Result {
    if splitter.try_split(len, migrated) {
        let mid = len / 2;
        let (left_prod, right_prod) = producer.split_at(mid);
        let (left_cons, right_cons, reducer) = consumer.split_at(mid);

        let (left, right) = join_context(
            |ctx| helper(mid, ctx.migrated(), splitter, left_prod, left_cons),
            |ctx| helper(len-mid, ctx.migrated(), splitter, right_prod, right_cons),
        );
        reducer.reduce(left, right)  // Binary tree reduction
    } else {
        producer.fold_with(consumer.into_folder()).complete()  // Leaf: fold
    }
}
```

**Key properties:**

1. **Binary tree of tasks.** Reduce creates a recursive tree, not flat chunks.
   Each internal node spawns two children via `join_context()`.

2. **join_context pushes one child, executes the other.** The pushed child can
   be stolen by idle workers. The current thread continues with the other half.
   This naturally distributes work across the tree.

3. **Adaptive splitter with theft detection.** The splitter tracks remaining
   desired splits (initially = num_threads). When a job is STOLEN (migrated=true),
   the splitter RESETS to max(num_threads, splits/2), allowing the thief to
   create more parallelism. This is how Rayon adapts to load imbalance.

4. **Tree-shaped reduction.** Partial results flow up the binary tree via
   `reducer.reduce(left, right)`. This is O(log N) depth, not O(N) sequential.

### Splitting policy

```rust
fn try_split(&mut self, stolen: bool) -> bool {
    if stolen {
        self.splits = max(num_threads, self.splits / 2);  // RESET on theft!
        true
    } else if self.splits > 0 {
        self.splits /= 2;
        true
    } else {
        false
    }
}
```

Initial splits = num_threads. For 8 threads on 524K items:
- Level 0: split [0..524K) → [0..262K) + [262K..524K), push left, continue right
- Level 1: split [262K..524K) → two 131K ranges, push left, continue right
- Level 2: split → two 65K ranges
- Level 3: splits=0, fold 65K items sequentially

**Effective leaf size: ~524K / 2^3 = 65K items per leaf** (8 leaves for 8 threads).

If a thief steals the [0..262K) job, the thief's splitter RESETS and creates
its own sub-tree. This is how Rayon achieves dynamic load balancing — fast
threads steal large subtrees and split them further.

---

## What to implement

### Option A: Adaptive reduce (match Rayon's pattern)

Replace `reduceImpl` with recursive binary splitting that mirrors `rangeJobExecute`:

```
reduceJobExecute:
  if len > threshold:
    push left half as new job (with its own partial)
    tail-recurse on right half
    wait for left, combine via reduce(left_partial, right_partial)
  else:
    compute map(ctx, start, end) → partial result
    store in partial slot, complete latch
```

**Implementation approach:**
- Each job carries a pointer to its parent's partial-result slot (or a per-node slot)
- Binary tree of ReduceJob nodes, each with its own partial
- Leaves compute map(), internal nodes combine via reduce()
- Use the existing CompletionLatch per split for synchronization

**Complexity:** Medium. Main challenge is the per-node partial storage. Could use a
stack-allocated tree of results (bounded by DEQUE_CAP depth), or heap-allocate
a small results array.

**Expected improvement:** At N=524K, efficiency should go from 27% → ~40%+,
making parallel reduce ~1.5x faster. For the prover, this affects every
sumcheck round in every stage.

### Option B: Simpler — over-partition with more chunks

Instead of `num_chunks = actual_threads`, use `num_chunks = actual_threads * 4`:

```zig
const chunk_size = (len + actual_threads * 4 - 1) / (actual_threads * 4);
const num_chunks = (len + chunk_size - 1) / chunk_size;
```

This creates 4x more chunks than threads, so early-finishing threads can steal
remaining chunks. The existing work-stealing in `waitWhileWorking` handles the rest.

**Pros:** Trivial change (one line). Better load balancing.
**Cons:** More deque push/steal overhead. 36 jobs for 9 threads. Sequential combine
over 36 partials (still negligible for field elements).

### Option C: Use parallelForImpl with thread-local accumulators

Avoid reduceImpl entirely. Instead:
- Allocate thread-local accumulator array (one per thread, cache-line padded)
- Use parallelForImpl (which already does adaptive splitting) with a func that
  accumulates into thread_local[worker_index]
- After completion, sequentially combine all accumulators

**Pros:** Reuses the battle-tested adaptive splitting. No new code in ThreadPool.
**Cons:** Requires TLS or atomic worker-index lookup per iteration. Slightly more
complex call sites.

---

## Other findings from the ThreadPool review

### Issues found

1. **Partials array alignment.** `var partials: [MAX_THREADS+1]PaddedPartial = undefined;`
   is stack-allocated but not aligned to cache_line. First partial may false-share
   with adjacent stack variables. Fix: add `align(cache_line)`.

2. **Caller spin asymmetry.** waitWhileWorking spins 32 times, workers spin 64.
   No clear rationale. Should be consistent.

3. **Linear steal search.** O(N_workers) per steal attempt. For 16 workers this is
   fine. Would matter at 64+ workers.

### Not issues (verified)

1. **Deque capacity.** DEQUE_CAP=256 is sufficient. Max observed depth with
   3-level nesting and adaptive splitting: ~54 entries.

2. **Generation overflow.** u32 wraps at 4.3B operations. Not reachable in practice.

3. **False sharing on deque fields.** `bottom` and `top` are in the same struct but
   different cache lines (bottom is written by owner, top by stealers). The Deque
   struct doesn't explicitly pad them, but they're 8 bytes apart in a 16KB buffer.
   Could be improved but unlikely bottleneck.

---

## Recommendation

**Start with Option B** (over-partition). It's a one-line change to `reduceImpl` that
immediately improves load balancing at large sizes. Benchmark the improvement.

**Then implement Option A** (adaptive reduce) for the full Rayon-equivalent behavior.
This is the architecturally correct fix and would bring parallel efficiency from
~27% to ~40%+, making Zolt's parallel reduce definitively faster than Rayon+arkworks
at all sizes.

---

## Files

- ThreadPool: `src/utils/thread_pool.zig`
- Rayon core: `~/.cargo/registry/src/.../rayon-1.11.0/src/`
- Benchmark (Zig): `bench/threadpool_vs_rayon/main.zig`
- Benchmark (Rust): `bench/threadpool_vs_rayon/bench_rayon.rs`
