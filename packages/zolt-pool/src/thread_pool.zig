//! ThreadPool - Chase-Lev Work-Stealing Thread Pool
//!
//! Per-worker Chase-Lev deques (2013 PPoPP paper orderings) with adaptive work splitting.
//! Supports unlimited nested dispatch via TLS-based worker identification.
//! Workers stay productive while waiting: pop own deque → steal from random victim.
//!
//! Same public API as the previous atomic counter implementation — zero call-site changes.

const std = @import("std");
const atomic = std.atomic;
const Futex = std.Thread.Futex;
const Allocator = std.mem.Allocator;

/// Minimum number of elements per thread to justify parallelism.
/// Below this threshold, work runs sequentially on the caller's thread.
const MIN_ITEMS_PER_THREAD: usize = 8;

const MAX_THREADS: usize = 16;
const cache_line = atomic.cache_line;

// Spin tuning constants
const SPIN_ITERS: u32 = 64;
const YIELD_ITERS: u32 = 8;

// Chase-Lev deque capacity (power of 2). Supports ~256 pending jobs per worker.
// Max depth with 3-level nesting and adaptive splitting: ~54 entries.
const DEQUE_CAP: usize = 256;

const WorkerState = enum(u32) {
    spinning = 0,
    yielding = 1,
    parking = 2,
    parked = 3,
    shutdown = 4,
};

// ============================================================================
// Job: 64-byte cache-line-sized work unit
// ============================================================================

const Job = struct {
    execute_fn: *const fn (*Job, *WorkerData) void,
    data: [56]u8 align(8) = undefined,

    fn init(comptime T: type, val: T, execute_fn: *const fn (*Job, *WorkerData) void) Job {
        comptime std.debug.assert(@sizeOf(T) <= 56);
        var job = Job{ .execute_fn = execute_fn };
        @memcpy(job.data[0..@sizeOf(T)], std.mem.asBytes(&val));
        return job;
    }

    fn getPayload(self: *Job, comptime T: type) *T {
        return @ptrCast(@alignCast(&self.data));
    }
};

comptime {
    std.debug.assert(@sizeOf(Job) == 64);
}

// ============================================================================
// CompletionLatch: stack-allocated completion tracking
// ============================================================================

const CompletionLatch = struct {
    remaining: atomic.Value(u32) align(cache_line),
    done: atomic.Value(u32) = atomic.Value(u32).init(0),

    fn init(count: u32) CompletionLatch {
        return .{
            .remaining = atomic.Value(u32).init(count),
        };
    }

    fn addPending(self: *CompletionLatch, n: u32) void {
        _ = self.remaining.fetchAdd(n, .acq_rel);
    }

    fn completeOne(self: *CompletionLatch) void {
        const prev = self.remaining.fetchSub(1, .acq_rel);
        if (prev == 1) {
            self.done.store(1, .release);
            Futex.wake(&self.done, std.math.maxInt(u32));
        }
    }

    fn isDone(self: *const CompletionLatch) bool {
        return self.done.load(.acquire) == 1;
    }

    fn waitWhileWorking(self: *CompletionLatch, worker: *WorkerData, pool: *ThreadPool) void {
        while (!self.isDone()) {
            // Try to pop from own deque (LIFO — cache friendly)
            if (worker.deque.pop()) |job| {
                job.execute_fn(job, worker);
                continue;
            }

            // Try to steal from random victim
            if (worker.tryStealing(pool)) |job| {
                job.execute_fn(job, worker);
                continue;
            }

            // Brief spin before futex wait
            for (0..32) |_| {
                if (self.isDone()) return;
                atomic.spinLoopHint();
            }

            // Futex wait if still not done
            if (!self.isDone()) {
                Futex.timedWait(&self.done, 0, 1_000_000) catch {}; // 1ms timeout
            }
        }
    }
};

// ============================================================================
// SpinLatch: lightweight atomic flag for internal reduce synchronization
// ============================================================================

/// Lightweight spin latch — single atomic flag, no futex.
/// Used for internal reduce tree nodes where the waiting thread is always
/// a worker that can stay productive by stealing work.
const SpinLatch = struct {
    state: atomic.Value(u32) = atomic.Value(u32).init(0),

    fn probe(self: *const SpinLatch) bool {
        return self.state.load(.acquire) == 1;
    }

    fn set(self: *SpinLatch) void {
        self.state.store(1, .release);
    }

    /// Spin + steal while waiting. Purely userspace — no futex syscall.
    fn waitWhileWorking(self: *const SpinLatch, worker: *WorkerData, pool: *ThreadPool) void {
        if (self.probe()) return;
        while (true) {
            // Try own deque (LIFO, cache friendly)
            if (worker.deque.pop()) |job| {
                job.execute_fn(job, worker);
                if (self.probe()) return;
                continue;
            }
            // Try stealing
            if (worker.tryStealing(pool)) |job| {
                job.execute_fn(job, worker);
                if (self.probe()) return;
                continue;
            }
            // Brief spin then yield (no futex)
            for (0..32) |_| {
                if (self.probe()) return;
                atomic.spinLoopHint();
            }
            std.Thread.yield() catch {};
        }
    }
};

// ============================================================================
// Chase-Lev Deque: lock-free SPMC work-stealing deque
// ============================================================================

const Deque = struct {
    buffer: [DEQUE_CAP]*Job align(cache_line) = undefined,
    bottom: atomic.Value(i64) align(cache_line) = atomic.Value(i64).init(0),
    top: atomic.Value(i64) align(cache_line) = atomic.Value(i64).init(0),

    const StealResult = union(enum) {
        success: *Job,
        empty,
        retry,
    };

    /// Owner-only push. O(1), uncontended. Returns false if deque is full.
    fn push(self: *Deque, job: *Job) bool {
        const b = self.bottom.load(.monotonic);
        const t = self.top.load(.acquire);

        if (b - t >= DEQUE_CAP) {
            return false;
        }

        self.buffer[@intCast(@mod(b, DEQUE_CAP))] = job;
        // Release fence folded into bottom store: ensures buffer write is visible before bottom update
        self.bottom.store(b + 1, .release);
        return true;
    }

    /// Owner-only pop. LIFO (cache-friendly). Returns null if empty.
    fn pop(self: *Deque) ?*Job {
        const b = self.bottom.load(.monotonic) - 1;
        // All four pop/steal boundary ops use seq_cst per Chase-Lev (Lê et al., PPoPP 2013)
        self.bottom.store(b, .seq_cst);
        const t = self.top.load(.seq_cst);

        if (t <= b) {
            // Non-empty
            const job = self.buffer[@intCast(@mod(b, DEQUE_CAP))];
            if (t == b) {
                // Last element — race with stealers
                if (self.top.cmpxchgStrong(t, t + 1, .seq_cst, .monotonic)) |_| {
                    // Lost race to stealer
                    self.bottom.store(t + 1, .monotonic);
                    return null;
                }
                self.bottom.store(t + 1, .monotonic);
            }
            return job;
        } else {
            // Empty
            self.bottom.store(t, .monotonic);
            return null;
        }
    }

    /// Any-thread steal. FIFO (fair). CAS-based.
    fn steal(self: *Deque) StealResult {
        // All four pop/steal boundary ops use seq_cst per Chase-Lev (Lê et al., PPoPP 2013)
        const t = self.top.load(.seq_cst);
        const b = self.bottom.load(.seq_cst);

        if (t >= b) return .empty;

        const job = self.buffer[@intCast(@mod(t, DEQUE_CAP))];

        if (self.top.cmpxchgStrong(t, t + 1, .seq_cst, .monotonic)) |_| {
            return .retry;
        }

        return .{ .success = job };
    }
};

// ============================================================================
// WorkerData: per-worker state with deque and TLS
// ============================================================================

const WorkerData = struct {
    deque: Deque align(cache_line) = .{},
    pool_ptr: *ThreadPool = undefined,
    index: usize = 0,
    rng_state: u32 = 0,
    state: atomic.Value(u32) align(cache_line) = atomic.Value(u32).init(@intFromEnum(WorkerState.spinning)),
    /// Nesting depth for the main-thread slot. 0 = free. Prevents SPMC violation
    /// if two non-worker threads try to use the same pool concurrently.
    /// Incremented on each dispatch entry, decremented on exit. Only the first
    /// increment (0→1) claims the slot; only the last decrement (1→0) releases it.
    claim_depth: atomic.Value(u32) = atomic.Value(u32).init(0),

    fn initRng(self: *WorkerData) void {
        // Seed RNG with index + timestamp for randomness
        const ts: u64 = @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())));
        self.rng_state = @truncate((self.index +% 1) *% 2654435761 +% ts);
        if (self.rng_state == 0) self.rng_state = 1;
    }

    fn nextRandom(self: *WorkerData) u32 {
        var x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng_state = x;
        return x;
    }

    fn tryStealing(self: *WorkerData, pool: *ThreadPool) ?*Job {
        const total = pool.thread_count + 1; // workers + main thread slot
        if (total <= 1) return null;

        // Retry full sweep on CAS contention (max 3 sweeps).
        // .retry means work EXISTS but another stealer won the CAS race.
        var retries: u32 = 0;
        while (retries < 3) {
            const start = self.nextRandom() % @as(u32, @intCast(total));
            var got_retry = false;
            for (0..total) |offset| {
                const victim_idx = (@as(usize, start) + offset) % total;
                if (victim_idx == self.index) continue;

                switch (pool.workers[victim_idx].deque.steal()) {
                    .success => |job| return job,
                    .empty => continue,
                    .retry => {
                        got_retry = true;
                        continue; // restart sweep from new random start
                    },
                }
            }
            if (!got_retry) break; // full sweep found nothing
            retries += 1;
        }
        return null;
    }
};

threadlocal var tls_worker: ?*WorkerData = null;

// ============================================================================
// Job Payloads
// ============================================================================

const RangeJobPayload = struct {
    user_fn: *const anyopaque, // type-erased fn pointer
    context_ptr: *const anyopaque,
    start: usize,
    end: usize,
    latch: *CompletionLatch,
    split_threshold: usize,
    parent_spin: ?*SpinLatch,
};

comptime {
    std.debug.assert(@sizeOf(RangeJobPayload) <= 56);
}

const ReduceJobPayload = struct {
    signal_fn: *const anyopaque, // type-erased: signals parent (CompletionLatch or SpinLatch)
    context_ptr: *const anyopaque,
    start: usize,
    end: usize,
    latch: *anyopaque, // *CompletionLatch (root) or *SpinLatch (internal)
    partial_ptr: *anyopaque,
    splits_remaining: u32, // Rayon-style: halves each level, stop at 0
    pusher_index: u32, // worker index that pushed this job (for theft detection)
};

comptime {
    std.debug.assert(@sizeOf(ReduceJobPayload) <= 56);
}

const JoinJobPayload = struct {
    user_fn: *const anyopaque,
    context_ptr: *const anyopaque,
    result_ptr: *anyopaque,
    latch: *SpinLatch,
};

comptime {
    std.debug.assert(@sizeOf(JoinJobPayload) <= 56);
}

// ============================================================================
// ThreadPool
// ============================================================================

pub const ThreadPool = struct {
    workers: [MAX_THREADS + 1]WorkerData,
    threads: [MAX_THREADS]std.Thread,
    thread_count: usize,
    allocator: Allocator,
    generation: atomic.Value(u32) align(cache_line),

    /// Initialize a thread pool with auto-detected CPU count (capped at 16).
    pub fn init(allocator: Allocator) !*ThreadPool {
        const cpu_count = std.Thread.getCpuCount() catch 4;
        const thread_count = @min(cpu_count, MAX_THREADS);
        return initWithCount(allocator, thread_count);
    }

    /// Initialize a thread pool with a specific number of worker threads.
    pub fn initWithCount(allocator: Allocator, thread_count: usize) !*ThreadPool {
        const actual_count = @min(thread_count, MAX_THREADS);

        const self = try allocator.create(ThreadPool);
        errdefer allocator.destroy(self);

        self.* = ThreadPool{
            .workers = [_]WorkerData{.{}} ** (MAX_THREADS + 1),
            .threads = undefined,
            .thread_count = actual_count,
            .allocator = allocator,
            .generation = atomic.Value(u32).init(0),
        };

        // Initialize worker metadata
        for (0..actual_count + 1) |i| {
            self.workers[i].pool_ptr = self;
            self.workers[i].index = i;
            self.workers[i].initRng();
        }

        // Spawn worker threads (indices 0..actual_count-1)
        var spawned: usize = 0;
        errdefer {
            for (self.workers[0..spawned]) |*w| {
                w.state.store(@intFromEnum(WorkerState.shutdown), .release);
            }
            _ = self.generation.fetchAdd(1, .release);
            for (self.workers[0..spawned]) |*w| {
                Futex.wake(&w.state, 1);
            }
            for (self.threads[0..spawned]) |t| {
                t.join();
            }
        }

        for (0..actual_count) |i| {
            self.threads[i] = try std.Thread.spawn(.{}, workerMain, .{ self, i });
            spawned += 1;
        }

        return self;
    }

    pub fn deinit(self: *ThreadPool) void {
        // Signal shutdown
        for (self.workers[0..self.thread_count]) |*w| {
            w.state.store(@intFromEnum(WorkerState.shutdown), .release);
        }
        _ = self.generation.fetchAdd(1, .release);
        for (self.workers[0..self.thread_count]) |*w| {
            Futex.wake(&w.state, 1);
        }
        for (self.threads[0..self.thread_count]) |t| {
            t.join();
        }
        // Clear TLS and claim depth if main thread was using this pool
        if (tls_worker) |w| {
            if (w.pool_ptr == self) {
                w.claim_depth.store(0, .release);
                tls_worker = null;
            }
        }
        self.allocator.destroy(self);
    }

    /// Get the current worker for this thread, initializing the main thread slot if needed.
    /// Only one non-worker thread may use the pool at a time (SPMC deque invariant).
    /// Returns null if the main-thread slot is already claimed by another thread.
    fn getCurrentWorker(self: *ThreadPool) ?*WorkerData {
        if (tls_worker) |w| {
            if (w.pool_ptr == self) {
                // Nested dispatch from same thread — bump depth (never fails)
                if (w.index == self.thread_count) {
                    _ = w.claim_depth.fetchAdd(1, .monotonic);
                }
                return w;
            }
        }
        // Main thread: CAS 0→1 to claim the slot (prevents SPMC violation)
        const w = &self.workers[self.thread_count];
        if (w.claim_depth.cmpxchgStrong(0, 1, .acquire, .monotonic) != null) {
            return null; // Another non-worker thread already owns this slot
        }
        tls_worker = w;
        return w;
    }

    /// Release one nesting level of the main-thread slot. Only truly releases when depth hits 0.
    fn releaseWorkerClaim(worker: *WorkerData) void {
        const prev = worker.claim_depth.fetchSub(1, .release);
        if (prev == 1) {
            // Last nesting level — slot is now free for other threads
            tls_worker = null;
        }
    }

    /// Get pool from TLS (for downstream code that needs pool access inside closures).
    pub fn getPool() ?*ThreadPool {
        if (tls_worker) |w| return w.pool_ptr;
        return null;
    }

    // ========================================================================
    // Public API
    // ========================================================================

    /// Force-parallel for: ignores MIN_ITEMS_PER_THREAD threshold.
    pub fn parallelForForce(
        self: *ThreadPool,
        len: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) void,
    ) void {
        if (len <= 1) {
            for (0..len) |i| func(context, i);
            return;
        }
        self.parallelForImpl(len, context, func, @min(self.thread_count + 1, len));
    }

    /// Fine-grained parallel for: submits each index as its own work item.
    /// Optimal dynamic load balancing for small numbers of heavyweight, heterogeneous tasks.
    pub fn parallelForEach(
        self: *ThreadPool,
        len: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) void,
    ) void {
        if (len <= 1) {
            for (0..len) |i| func(context, i);
            return;
        }
        self.parallelForImplWithThreshold(len, context, func, @min(self.thread_count + 1, len), 1);
    }

    /// Parallel for: apply `func(context, index)` for each index in 0..len.
    pub fn parallelFor(
        self: *ThreadPool,
        len: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) void,
    ) void {
        if (len == 0) return;

        const actual_threads = self.effectiveThreads(len, 1);
        if (actual_threads <= 1) {
            for (0..len) |i| func(context, i);
            return;
        }

        self.parallelForImpl(len, context, func, actual_threads);
    }

    /// Parallel chunks: split `slice` into chunks and call `func(context, chunk, chunk_start_index)`.
    pub fn parallelChunks(
        self: *ThreadPool,
        comptime T: type,
        slice: []T,
        chunk_size_hint: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), []T, usize) void,
    ) void {
        if (slice.len == 0) return;

        const actual_chunk_size = if (chunk_size_hint == 0) blk: {
            const actual_threads = self.effectiveThreads(slice.len, 1);
            break :blk (slice.len + actual_threads - 1) / actual_threads;
        } else chunk_size_hint;

        const n_chunks = (slice.len + actual_chunk_size - 1) / actual_chunk_size;

        if (n_chunks <= 1) {
            func(context, slice, 0);
            return;
        }

        // Wrap chunks as indices into a parallelFor
        const Ctx = @TypeOf(context);
        const ChunkCtx = struct {
            ctx: Ctx,
            slice_ptr: [*]T,
            slice_len: usize,
            chunk_size: usize,
        };
        var chunk_ctx = ChunkCtx{
            .ctx = context,
            .slice_ptr = slice.ptr,
            .slice_len = slice.len,
            .chunk_size = actual_chunk_size,
        };

        self.parallelForForce(n_chunks, &chunk_ctx, struct {
            fn run(cc: *const ChunkCtx, chunk_idx: usize) void {
                const off = chunk_idx * cc.chunk_size;
                const chunk_end = @min(off + cc.chunk_size, cc.slice_len);
                func(cc.ctx, cc.slice_ptr[off..chunk_end], off);
            }
        }.run);
    }

    /// Parallel reduce: split 0..len into chunks, each producing a partial result via
    /// `map(context, start, end) -> R`, then combine with `reduce(a, b) -> R`.
    pub fn parallelReduce(
        self: *ThreadPool,
        comptime R: type,
        len: usize,
        identity: R,
        context: anytype,
        comptime map: fn (@TypeOf(context), usize, usize) R,
        comptime reduce: fn (R, R) R,
    ) R {
        if (len == 0) return identity;

        const actual_threads = self.effectiveThreads(len, 1);
        if (actual_threads <= 1) {
            return map(context, 0, len);
        }

        return self.reduceImpl(R, len, identity, context, map, reduce, actual_threads);
    }

    /// Force-parallel reduce: ignores MIN_ITEMS_PER_THREAD threshold.
    pub fn parallelReduceForce(
        self: *ThreadPool,
        comptime R: type,
        len: usize,
        identity: R,
        context: anytype,
        comptime map: fn (@TypeOf(context), usize, usize) R,
        comptime reduce: fn (R, R) R,
    ) R {
        if (len == 0) return identity;

        const actual_threads = @min(self.thread_count + 1, len);
        if (actual_threads <= 1) {
            return map(context, 0, len);
        }

        return self.reduceImpl(R, len, identity, context, map, reduce, actual_threads);
    }

    /// Join: run two independent functions concurrently and return both results.
    pub fn join(
        self: *ThreadPool,
        comptime RA: type,
        comptime RB: type,
        context_a: anytype,
        comptime func_a: fn (@TypeOf(context_a)) RA,
        context_b: anytype,
        comptime func_b: fn (@TypeOf(context_b)) RB,
    ) struct { RA, RB } {
        if (self.thread_count == 0) {
            return .{ func_a(context_a), func_b(context_b) };
        }

        const CtxB = @TypeOf(context_b);
        var result_b: RB = undefined;

        const JoinCtx = struct {
            ctx_b: CtxB,
            result_ptr: *RB,
        };
        var join_ctx = JoinCtx{
            .ctx_b = context_b,
            .result_ptr = &result_b,
        };

        const worker = self.getCurrentWorker() orelse {
            return .{ func_a(context_a), func_b(context_b) };
        };
        defer if (worker.index == self.thread_count) releaseWorkerClaim(worker);

        var latch = SpinLatch{};

        const join_payload = JoinJobPayload{
            .user_fn = @ptrCast(&struct {
                fn call(jc: *const JoinCtx) void {
                    jc.result_ptr.* = func_b(jc.ctx_b);
                }
            }.call),
            .context_ptr = @ptrCast(&join_ctx),
            .result_ptr = @ptrCast(&result_b),
            .latch = &latch,
        };

        var job = Job.init(JoinJobPayload, join_payload, &joinJobExecute);
        if (!worker.deque.push(&job)) {
            // Deque full: run both sequentially
            return .{ func_a(context_a), func_b(context_b) };
        }

        // Wake all idle workers on first push (deque was empty/near-empty).
        // Previous code only woke 1, causing slow fan-out in recursive joins.
        const b = worker.deque.bottom.load(.monotonic);
        const t = worker.deque.top.load(.monotonic);
        if (b -% t <= 1) {
            self.wakeWorkers(self.thread_count);
        }

        // Caller does task A
        const result_a = func_a(context_a);

        // Try to pop back our job (Rayon pattern: pop before latch check).
        // Common case: job is still in our deque — direct call, no type erasure.
        if (worker.deque.pop()) |job_ptr| {
            if (job_ptr == &job) {
                // Got it back — run inline, no latch/steal overhead
                result_b = func_b(context_b);
                return .{ result_a, result_b };
            }
            // Different job — execute it then wait for ours
            job_ptr.execute_fn(job_ptr, worker);
        }
        // Task B was stolen — wait for completion, staying productive
        if (!latch.probe()) {
            latch.waitWhileWorking(worker, self);
        }

        return .{ result_a, result_b };
    }

    // ========================================================================
    // Internal: parallelFor implementation with adaptive splitting
    // ========================================================================

    fn parallelForImpl(
        self: *ThreadPool,
        len: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) void,
        actual_threads: usize,
    ) void {
        self.parallelForImplWithThreshold(len, context, func, actual_threads, @max(1, len / (actual_threads * 4)));
    }

    fn parallelForImplWithThreshold(
        self: *ThreadPool,
        len: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize) void,
        actual_threads: usize,
        split_threshold: usize,
    ) void {
        const worker = self.getCurrentWorker() orelse {
            for (0..len) |i| func(context, i);
            return;
        };
        defer if (worker.index == self.thread_count) releaseWorkerClaim(worker);

        const Ctx = @TypeOf(context);
        var ctx_copy = context;

        var latch = CompletionLatch.init(1);
        var root_spin = SpinLatch{};

        const range_payload = RangeJobPayload{
            .user_fn = @ptrCast(&struct {
                fn call(ctx: *const Ctx, start: usize, end: usize) void {
                    for (start..end) |i| func(ctx.*, i);
                }
            }.call),
            .context_ptr = @ptrCast(&ctx_copy),
            .start = 0,
            .end = len,
            .latch = &latch,
            .split_threshold = split_threshold,
            .parent_spin = &root_spin,
        };

        var job = Job.init(RangeJobPayload, range_payload, &rangeJobExecute);
        if (!worker.deque.push(&job)) {
            // Deque full: run sequentially
            for (0..len) |i| func(ctx_copy, i);
            return;
        }

        // Wake idle workers
        self.wakeWorkers(@min(actual_threads - 1, self.thread_count));

        // Try to pop back our own job (Rayon join pattern)
        if (worker.deque.pop()) |job_ptr| {
            if (job_ptr == &job) {
                // Got it back — run inline, no steal overhead
                executeRange(
                    range_payload.user_fn,
                    range_payload.context_ptr,
                    0,
                    len,
                    split_threshold,
                    &latch,
                    worker,
                    self,
                );
                return;
            }
            // Different job — execute it then wait
            job_ptr.execute_fn(job_ptr, worker);
        }

        // Wait for root job to fully complete (stack lifetime safety:
        // root_spin is set AFTER rangeJobExecute returns, so &job is still alive)
        root_spin.waitWhileWorking(worker, self);
    }

    // ========================================================================
    // Internal: Reduce implementation (pre-chunked for stable partial indices)
    // ========================================================================

    fn reduceImpl(
        self: *ThreadPool,
        comptime R: type,
        len: usize,
        _: R, // identity (unused — inline root returns complete result)
        context: anytype,
        comptime map: fn (@TypeOf(context), usize, usize) R,
        comptime reduce_fn: fn (R, R) R,
        actual_threads: usize,
    ) R {
        const Ctx = @TypeOf(context);

        // Signal helpers (type-erased for ReduceJobPayload)
        const Signals = struct {
            fn signalCompletion(latch_ptr: *anyopaque) void {
                const latch: *CompletionLatch = @ptrCast(@alignCast(latch_ptr));
                latch.completeOne();
            }
            fn signalSpin(latch_ptr: *anyopaque) void {
                const spin: *SpinLatch = @ptrCast(@alignCast(latch_ptr));
                spin.set();
            }
        };

        // Adaptive binary-split reduce executor with SpinLatch for internal nodes
        const Executor = struct {
            /// Called when a pushed job executes on a worker thread.
            fn execute(job: *Job, worker: *WorkerData) void {
                const p = job.getPayload(ReduceJobPayload);
                const ctx_ptr: *const Ctx = @ptrCast(@alignCast(p.context_ptr));
                const result_ptr: *R = @ptrCast(@alignCast(p.partial_ptr));
                const pool: *ThreadPool = worker.pool_ptr;

                // Rayon-style theft detection: if we're on a different worker
                // than the one that pushed this job, reset splits counter
                const stolen = (worker.index != p.pusher_index);
                var splits = p.splits_remaining;
                if (stolen) {
                    splits = @max(@as(u32, @intCast(pool.thread_count + 1)), splits / 2);
                }

                result_ptr.* = computeRange(ctx_ptr, p.start, p.end, splits, worker, pool);

                // Signal parent: CompletionLatch (root) or SpinLatch (internal)
                const signal: *const fn (*anyopaque) void = @ptrCast(@alignCast(p.signal_fn));
                signal(p.latch);
            }

            /// Specialized wait: tries direct call for jobs matching our Executor,
            /// falls back to indirect call for foreign jobs. This lets LLVM inline
            /// the map function through the direct call path.
            fn spinWait(latch: *SpinLatch, worker: *WorkerData, pool: *ThreadPool) void {
                if (latch.probe()) return;
                while (true) {
                    if (worker.deque.pop()) |job| {
                        executeJobSpecialized(job, worker);
                        if (latch.probe()) return;
                        continue;
                    }
                    if (worker.tryStealing(pool)) |job| {
                        executeJobSpecialized(job, worker);
                        if (latch.probe()) return;
                        continue;
                    }
                    for (0..32) |_| {
                        if (latch.probe()) return;
                        atomic.spinLoopHint();
                    }
                    std.Thread.yield() catch {};
                }
            }

            /// If the job belongs to this Executor, call execute directly (LLVM can inline).
            /// Otherwise fall back to the indirect function pointer.
            inline fn executeJobSpecialized(job: *Job, worker: *WorkerData) void {
                if (job.execute_fn == &execute) {
                    execute(job, worker);
                } else {
                    job.execute_fn(job, worker);
                }
            }

            /// Rayon join pattern: push right half to deque, execute left inline,
            /// try to pop right back. If right wasn't stolen, run it inline
            /// with ZERO latch overhead. If stolen, spin-wait.
            fn computeRange(
                ctx_ptr: *const Ctx,
                start: usize,
                end: usize,
                splits: u32,
                worker: *WorkerData,
                pool: *ThreadPool,
            ) R {
                const range_len = end - start;

                if (splits == 0 or range_len < 2) {
                    return map(ctx_ptr.*, start, end);
                }

                const mid = start + range_len / 2;

                // Push RIGHT half to deque (stealable). Execute LEFT inline.
                var right_result: R = undefined;
                var right_spin = SpinLatch{};

                const right_payload = ReduceJobPayload{
                    .signal_fn = @ptrCast(&Signals.signalSpin),
                    .context_ptr = @ptrCast(ctx_ptr),
                    .start = mid,
                    .end = end,
                    .latch = @ptrCast(&right_spin),
                    .partial_ptr = @ptrCast(&right_result),
                    .splits_remaining = splits / 2,
                    .pusher_index = @intCast(worker.index),
                };

                var right_job = Job.init(ReduceJobPayload, right_payload, &execute);

                const pushed = worker.deque.push(&right_job);
                if (!pushed) {
                    // Deque full: compute right inline (no overhead)
                    right_result = computeRange(ctx_ptr, mid, end, splits / 2, worker, pool);
                }

                // Execute LEFT inline (this is the "task A" in Rayon's join)
                const left_result = computeRange(ctx_ptr, start, mid, splits / 2, worker, pool);

                if (pushed) {
                    // Try to pop right back (it may not have been stolen)
                    if (!right_spin.probe()) {
                        // Not done yet — try to pop from our deque
                        // If we get it back, run inline (zero latch overhead!)
                        if (worker.deque.pop()) |job_ptr| {
                            if (job_ptr == &right_job) {
                                // Got it back! Run inline, no latch overhead
                                right_result = computeRange(ctx_ptr, mid, end, splits / 2, worker, pool);
                            } else {
                                // Got a different job — use specialized dispatch (direct call if ours)
                                executeJobSpecialized(job_ptr, worker);
                                spinWait(&right_spin, worker, pool);
                            }
                        } else {
                            // Deque empty, right was stolen — wait with specialized dispatch
                            spinWait(&right_spin, worker, pool);
                        }
                    }
                    // else: right already completed (stolen and finished fast)
                }

                return reduce_fn(left_result, right_result);
            }
        };

        const worker = self.getCurrentWorker() orelse {
            return map(context, 0, len);
        };
        defer if (worker.index == self.thread_count) releaseWorkerClaim(worker);

        // Wake workers FIRST so they're stealing when we start pushing sub-jobs.
        self.wakeWorkers(@min(actual_threads - 1, self.thread_count));

        // Inline root execution: call computeRange directly instead of push/pop/latch.
        // computeRange pushes right halves to our deque (stealable by woken workers)
        // and recurses into left halves. SpinLatches handle stolen sub-tasks internally.
        // Matches Rayon's Splitter::new() initial splits = num_threads.
        const initial_splits: u32 = @intCast(actual_threads);
        var ctx_copy = context;
        return Executor.computeRange(&ctx_copy, 0, len, initial_splits, worker, self);
    }

    // ========================================================================
    // Job executors
    // ========================================================================

    fn rangeJobExecute(job: *Job, worker: *WorkerData) void {
        const p = job.getPayload(RangeJobPayload);
        const pool: *ThreadPool = worker.pool_ptr;
        executeRange(p.user_fn, p.context_ptr, p.start, p.end, p.split_threshold, p.latch, worker, pool);
        if (p.parent_spin) |spin| spin.set();
    }

    fn executeRange(
        user_fn: *const anyopaque,
        context_ptr: *const anyopaque,
        start: usize,
        end: usize,
        split_threshold: usize,
        latch: *CompletionLatch,
        worker: *WorkerData,
        pool: *ThreadPool,
    ) void {
        const len = end - start;
        if (len <= split_threshold) {
            const TypeErasedFn = *const fn (*const anyopaque, usize, usize) void;
            const call_fn: TypeErasedFn = @ptrCast(@alignCast(user_fn));
            call_fn(context_ptr, start, end);
            latch.completeOne();
            return;
        }
        const mid = start + len / 2;
        var right_spin = SpinLatch{};
        const right_payload = RangeJobPayload{
            .user_fn = user_fn,
            .context_ptr = context_ptr,
            .start = mid,
            .end = end,
            .latch = latch,
            .split_threshold = split_threshold,
            .parent_spin = &right_spin,
        };
        var right_job = Job.init(RangeJobPayload, right_payload, &rangeJobExecute);
        latch.addPending(1);
        const pushed = worker.deque.push(&right_job);
        if (!pushed) {
            executeRange(user_fn, context_ptr, mid, end, split_threshold, latch, worker, pool);
        }
        // Execute LEFT inline
        executeRange(user_fn, context_ptr, start, mid, split_threshold, latch, worker, pool);
        if (pushed) {
            // MUST wait for right before returning (stack lifetime of right_job!)
            if (!right_spin.probe()) {
                if (worker.deque.pop()) |job_ptr| {
                    if (job_ptr == &right_job) {
                        executeRange(user_fn, context_ptr, mid, end, split_threshold, latch, worker, pool);
                        return;
                    }
                    // Different job — execute it then wait
                    job_ptr.execute_fn(job_ptr, worker);
                }
                // Wait for right to complete
                right_spin.waitWhileWorking(worker, pool);
            }
        }
    }

    // reduceJobExecute is now generated at comptime inside reduceImpl
    // (each instantiation captures R, map, reduce via Zig's monomorphization)

    fn joinJobExecute(job: *Job, _: *WorkerData) void {
        const p = job.getPayload(JoinJobPayload);
        const TypeErasedFn = *const fn (*const anyopaque) void;
        const call_fn: TypeErasedFn = @ptrCast(@alignCast(p.user_fn));
        call_fn(p.context_ptr);
        p.latch.set();
    }

    // ========================================================================
    // Worker thread
    // ========================================================================

    fn workerMain(self: *ThreadPool, worker_idx: usize) void {
        const worker = &self.workers[worker_idx];
        tls_worker = worker;
        var last_seen_gen: u32 = 0;

        while (true) {
            if (worker.state.load(.acquire) == @intFromEnum(WorkerState.shutdown)) return;

            // Check generation for wake signal
            const gen = self.generation.load(.acquire);
            if (gen != last_seen_gen) {
                last_seen_gen = gen;
                if (worker.state.load(.acquire) == @intFromEnum(WorkerState.shutdown)) return;
                // Process work: pop own deque, then steal
                self.workerProcessWork(worker);
                continue;
            }

            // Try to find work opportunistically
            if (worker.deque.pop()) |job| {
                job.execute_fn(job, worker);
                continue;
            }
            if (worker.tryStealing(self)) |job| {
                job.execute_fn(job, worker);
                continue;
            }

            // Phase 1: Spin (check deque + steal every 8th iteration)
            var found_work = false;
            for (0..SPIN_ITERS) |spin_i| {
                atomic.spinLoopHint();
                if (worker.state.load(.acquire) == @intFromEnum(WorkerState.shutdown)) return;
                const g = self.generation.load(.acquire);
                if (g != last_seen_gen) {
                    found_work = true;
                    break;
                }
                // Check deque + steal periodically during spin
                if (spin_i % 8 == 7) {
                    if (worker.deque.pop()) |job| {
                        job.execute_fn(job, worker);
                        found_work = true;
                        break;
                    }
                    if (worker.tryStealing(self)) |job| {
                        job.execute_fn(job, worker);
                        found_work = true;
                        break;
                    }
                }
            }
            if (found_work) continue;

            // Phase 2: Yield (with stealing attempts)
            worker.state.store(@intFromEnum(WorkerState.yielding), .release);
            for (0..YIELD_ITERS) |_| {
                std.Thread.yield() catch {};
                if (worker.state.load(.acquire) == @intFromEnum(WorkerState.shutdown)) return;
                const g = self.generation.load(.acquire);
                if (g != last_seen_gen) {
                    found_work = true;
                    break;
                }
                // Try stealing during yield phase
                if (worker.tryStealing(self)) |job| {
                    job.execute_fn(job, worker);
                    found_work = true;
                    break;
                }
            }
            if (found_work) {
                worker.state.store(@intFromEnum(WorkerState.spinning), .release);
                continue;
            }

            // Phase 3: Park
            worker.state.store(@intFromEnum(WorkerState.parking), .release);
            const recheck_gen = self.generation.load(.acquire);
            if (recheck_gen != last_seen_gen) {
                worker.state.store(@intFromEnum(WorkerState.spinning), .release);
                continue;
            }

            worker.state.store(@intFromEnum(WorkerState.parked), .release);
            Futex.wait(&worker.state, @intFromEnum(WorkerState.parked));
            // Woken: state was CAS'd by wakeWorkers or set to shutdown. Don't overwrite.
        }
    }

    /// Process available work until deques are empty (with steal retries).
    /// Uses a hybrid idle strategy matching Rayon's persistence (~32 rounds):
    /// first 8 rounds spin (fast, lets pusher threads push sub-jobs),
    /// remaining 24 rounds yield (gives other threads CPU time).
    fn workerProcessWork(self: *ThreadPool, worker: *WorkerData) void {
        var consecutive_steal_failures: u32 = 0;
        while (true) {
            if (worker.deque.pop()) |job| {
                job.execute_fn(job, worker);
                consecutive_steal_failures = 0;
                continue;
            }
            if (worker.tryStealing(self)) |job| {
                job.execute_fn(job, worker);
                consecutive_steal_failures = 0;
                // Cascading wake: bump generation so other parked workers notice work
                _ = self.generation.fetchAdd(1, .release);
                continue;
            }
            consecutive_steal_failures += 1;
            if (consecutive_steal_failures >= 32) break;
            // Hybrid: spin first (fast ~5ns PAUSE), then yield (lets pushers run)
            if (consecutive_steal_failures <= 8) {
                atomic.spinLoopHint();
            } else {
                std.Thread.yield() catch {};
            }
        }
    }

    /// Wake up to `count` workers using CAS-based selective wake.
    fn wakeWorkers(self: *ThreadPool, count: usize) void {
        if (count == 0) return;

        // Bump generation to signal new work
        _ = self.generation.fetchAdd(1, .release);

        var woken: usize = 0;
        for (self.workers[0..self.thread_count]) |*worker| {
            if (woken >= count) break;

            const state = worker.state.load(.acquire);
            if (state == @intFromEnum(WorkerState.parked)) {
                if (worker.state.cmpxchgStrong(
                    @intFromEnum(WorkerState.parked),
                    @intFromEnum(WorkerState.spinning),
                    .release,
                    .monotonic,
                ) == null) {
                    Futex.wake(&worker.state, 1);
                    woken += 1;
                }
            } else if (state == @intFromEnum(WorkerState.parking)) {
                if (worker.state.cmpxchgStrong(
                    @intFromEnum(WorkerState.parking),
                    @intFromEnum(WorkerState.spinning),
                    .release,
                    .monotonic,
                ) == null) {
                    woken += 1;
                }
            } else {
                // spinning or yielding — they'll see generation change
                woken += 1;
            }
        }
    }

    /// How many threads to actually use given the work size.
    fn effectiveThreads(self: *const ThreadPool, total_items: usize, items_per_unit: usize) usize {
        const work_units = total_items * items_per_unit;
        if (work_units < MIN_ITEMS_PER_THREAD) return 1;
        const max_by_work = work_units / MIN_ITEMS_PER_THREAD;
        return @min(self.thread_count + 1, max_by_work);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ThreadPool: parallelFor basic" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const n = 1024;
    var data: [n]u64 = undefined;
    for (0..n) |i| data[i] = 0;

    const Context = struct {
        data: *[n]u64,
    };
    const ctx = Context{ .data = &data };

    tp.parallelFor(n, ctx, struct {
        fn run(c: Context, i: usize) void {
            c.data[i] = @intCast(i * 2);
        }
    }.run);

    for (0..n) |i| {
        try std.testing.expectEqual(@as(u64, i * 2), data[i]);
    }
}

test "ThreadPool: parallelReduce sum" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const n = 2048;
    var data: [n]u64 = undefined;
    for (0..n) |i| data[i] = @intCast(i);

    const Context = struct {
        data: *const [n]u64,
    };
    const ctx = Context{ .data = &data };

    const sum = tp.parallelReduce(
        u64,
        n,
        0,
        ctx,
        struct {
            fn map(c: Context, start: usize, end: usize) u64 {
                var s: u64 = 0;
                for (start..end) |i| s += c.data[i];
                return s;
            }
        }.map,
        struct {
            fn reduce(a: u64, b: u64) u64 {
                return a + b;
            }
        }.reduce,
    );

    try std.testing.expectEqual(@as(u64, 2047 * 2048 / 2), sum);
}

test "ThreadPool: join" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const result = tp.join(
        u64,
        u64,
        @as(u64, 42),
        struct {
            fn a(x: u64) u64 {
                return x * 2;
            }
        }.a,
        @as(u64, 7),
        struct {
            fn b(x: u64) u64 {
                return x + 10;
            }
        }.b,
    );

    try std.testing.expectEqual(@as(u64, 84), result[0]);
    try std.testing.expectEqual(@as(u64, 17), result[1]);
}

test "ThreadPool: parallelForEach basic" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const n = 37;
    var data: [n]u64 = undefined;
    for (0..n) |i| data[i] = 0;

    const Context = struct { data: *[n]u64 };
    tp.parallelForEach(n, Context{ .data = &data }, struct {
        fn run(c: Context, i: usize) void {
            c.data[i] = @intCast(i + 1);
        }
    }.run);

    for (0..n) |i| {
        try std.testing.expectEqual(@as(u64, i + 1), data[i]);
    }
}

test "ThreadPool: small work runs sequentially" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    var data: [10]u64 = undefined;
    for (0..10) |i| data[i] = 0;

    const Context = struct { data: *[10]u64 };
    tp.parallelFor(10, Context{ .data = &data }, struct {
        fn run(c: Context, i: usize) void {
            c.data[i] = 1;
        }
    }.run);

    for (0..10) |i| {
        try std.testing.expectEqual(@as(u64, 1), data[i]);
    }
}

test "ThreadPool: parallelForForce" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const n = 8;
    var data: [n]u64 = undefined;
    for (0..n) |i| data[i] = 0;

    const Context = struct { data: *[n]u64 };
    tp.parallelForForce(n, Context{ .data = &data }, struct {
        fn run(c: Context, i: usize) void {
            c.data[i] = @intCast(i * 3);
        }
    }.run);

    for (0..n) |i| {
        try std.testing.expectEqual(@as(u64, i * 3), data[i]);
    }
}

test "ThreadPool: parallelReduceForce" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const n = 16;
    const sum = tp.parallelReduceForce(
        u64,
        n,
        0,
        {},
        struct {
            fn map(_: void, start: usize, end: usize) u64 {
                var s: u64 = 0;
                for (start..end) |i| s += @as(u64, @intCast(i));
                return s;
            }
        }.map,
        struct {
            fn reduce(a: u64, b: u64) u64 {
                return a + b;
            }
        }.reduce,
    );

    try std.testing.expectEqual(@as(u64, 15 * 16 / 2), sum);
}

test "ThreadPool: dispatch overhead microbenchmark" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const iters = 1000;
    var timer = try std.time.Timer.start();

    for (0..iters) |_| {
        tp.parallelForForce(16, {}, struct {
            fn run(_: void, _: usize) void {}
        }.run);
    }

    const elapsed_ns = timer.read();
    const per_dispatch_ns = elapsed_ns / iters;
    std.debug.print("\n  Dispatch overhead: {}ns per dispatch ({} dispatches)\n", .{ per_dispatch_ns, iters });
    try std.testing.expect(per_dispatch_ns < 100_000);
}

test "ThreadPool: nested parallelFor" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const outer_n = 4;
    const inner_n = 256;
    var data: [outer_n][inner_n]u64 = undefined;
    for (0..outer_n) |i| {
        for (0..inner_n) |j| data[i][j] = 0;
    }

    const InnerCtx = struct {
        row: *[inner_n]u64,
        pool: *ThreadPool,
    };

    const OuterCtx = struct {
        data: *[outer_n][inner_n]u64,
        pool: *ThreadPool,
    };
    const outer_ctx = OuterCtx{ .data = &data, .pool = tp };

    tp.parallelForForce(outer_n, outer_ctx, struct {
        fn run(c: OuterCtx, i: usize) void {
            const inner_ctx = InnerCtx{ .row = &c.data[i], .pool = c.pool };
            c.pool.parallelFor(inner_n, inner_ctx, struct {
                fn inner_run(ic: InnerCtx, j: usize) void {
                    ic.row[j] = @intCast(j + 1);
                }
            }.inner_run);
        }
    }.run);

    for (0..outer_n) |i| {
        for (0..inner_n) |j| {
            try std.testing.expectEqual(@as(u64, j + 1), data[i][j]);
        }
    }
}

test "ThreadPool: nested join with reduce" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const PoolCtx = struct { pool: *ThreadPool };
    const ctx = PoolCtx{ .pool = tp };

    const result = tp.join(
        u64,
        u64,
        ctx,
        struct {
            fn a(c: PoolCtx) u64 {
                // Inner reduce inside join task A
                return c.pool.parallelReduce(
                    u64,
                    512,
                    0,
                    {},
                    struct {
                        fn map(_: void, start: usize, end: usize) u64 {
                            var s: u64 = 0;
                            for (start..end) |i| s += @as(u64, @intCast(i));
                            return s;
                        }
                    }.map,
                    struct {
                        fn reduce(x: u64, y: u64) u64 {
                            return x + y;
                        }
                    }.reduce,
                );
            }
        }.a,
        ctx,
        struct {
            fn b(c: PoolCtx) u64 {
                // Inner reduce inside join task B
                return c.pool.parallelReduce(
                    u64,
                    256,
                    0,
                    {},
                    struct {
                        fn map(_: void, start: usize, end: usize) u64 {
                            var s: u64 = 0;
                            for (start..end) |i| s += @as(u64, @intCast(i));
                            return s;
                        }
                    }.map,
                    struct {
                        fn reduce(x: u64, y: u64) u64 {
                            return x + y;
                        }
                    }.reduce,
                );
            }
        }.b,
    );

    try std.testing.expectEqual(@as(u64, 511 * 512 / 2), result[0]);
    try std.testing.expectEqual(@as(u64, 255 * 256 / 2), result[1]);
}

test "ThreadPool: 3-level nesting" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    // Level 1: parallelForForce(4)
    // Level 2: join(A, B)
    // Level 3: parallelFor(256) inside each join branch
    const n = 4;
    var sums: [n]u64 = [_]u64{0} ** n;

    const PoolCtx = struct { pool: *ThreadPool, sums: *[n]u64 };
    const ctx = PoolCtx{ .pool = tp, .sums = &sums };

    tp.parallelForForce(n, ctx, struct {
        fn run(c: PoolCtx, i: usize) void {
            const JoinCtx = struct { pool: *ThreadPool };
            const jc = JoinCtx{ .pool = c.pool };
            const result = c.pool.join(
                u64,
                u64,
                jc,
                struct {
                    fn a(j: JoinCtx) u64 {
                        var sum: u64 = 0;
                        const SumCtx = struct { pool: *ThreadPool, sum_ptr: *u64 };
                        var sum_ctx = SumCtx{ .pool = j.pool, .sum_ptr = &sum };
                        j.pool.parallelFor(256, &sum_ctx, struct {
                            fn inner(sc: *const SumCtx, k: usize) void {
                                _ = @atomicRmw(u64, sc.sum_ptr, .Add, @as(u64, @intCast(k)), .monotonic);
                            }
                        }.inner);
                        return sum;
                    }
                }.a,
                jc,
                struct {
                    fn b(j: JoinCtx) u64 {
                        return j.pool.parallelReduce(
                            u64,
                            256,
                            0,
                            {},
                            struct {
                                fn map(_: void, start: usize, end: usize) u64 {
                                    var s: u64 = 0;
                                    for (start..end) |k| s += @as(u64, @intCast(k));
                                    return s;
                                }
                            }.map,
                            struct {
                                fn reduce(x: u64, y: u64) u64 {
                                    return x + y;
                                }
                            }.reduce,
                        );
                    }
                }.b,
            );
            c.sums[i] = result[0] + result[1];
        }
    }.run);

    const expected: u64 = 255 * 256 / 2;
    for (0..n) |i| {
        try std.testing.expectEqual(expected * 2, sums[i]);
    }
}

test "ThreadPool: stress nested dispatches" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    // Each iteration: outer reduce over 32 items, inner parallelFor per chunk.
    // Inner sums global indices (start + i), so total per iteration = sum(0..31) = 496.
    const n = 32;
    const expected_per_iter: u64 = @as(u64, n) * (n - 1) / 2; // sum of 0..31

    for (0..100) |_| {
        const sum = tp.parallelReduceForce(
            u64,
            n,
            0,
            tp,
            struct {
                fn map(pool: *ThreadPool, start: usize, end: usize) u64 {
                    // Nested parallelFor inside reduce — sum global indices
                    var s: u64 = 0;
                    const Ctx = struct { pool: *ThreadPool, sum: *u64, base: usize };
                    var ctx = Ctx{ .pool = pool, .sum = &s, .base = start };
                    pool.parallelFor(end - start, &ctx, struct {
                        fn run(c: *const Ctx, i: usize) void {
                            _ = @atomicRmw(u64, c.sum, .Add, @as(u64, @intCast(c.base + i)), .monotonic);
                        }
                    }.run);
                    return s;
                }
            }.map,
            struct {
                fn reduce(a: u64, b: u64) u64 {
                    return a + b;
                }
            }.reduce,
        );
        try std.testing.expectEqual(expected_per_iter, sum);
    }
}

test "ThreadPool: nested parallelForForce exercises parallelism" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const outer_n = 4;
    const inner_n = 512;
    var data: [outer_n][inner_n]u64 = undefined;
    for (0..outer_n) |i| {
        for (0..inner_n) |j| data[i][j] = 0;
    }

    const OuterCtx = struct {
        data: *[outer_n][inner_n]u64,
        pool: *ThreadPool,
    };

    tp.parallelForForce(outer_n, OuterCtx{ .data = &data, .pool = tp }, struct {
        fn run(c: OuterCtx, i: usize) void {
            const InnerCtx = struct { row: *[inner_n]u64 };
            c.pool.parallelForForce(inner_n, InnerCtx{ .row = &c.data[i] }, struct {
                fn inner_run(ic: InnerCtx, j: usize) void {
                    ic.row[j] = @intCast(j + 1);
                }
            }.inner_run);
        }
    }.run);

    for (0..outer_n) |i| {
        for (0..inner_n) |j| {
            try std.testing.expectEqual(@as(u64, j + 1), data[i][j]);
        }
    }
}

test "ThreadPool: stress addPending race" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    for (0..500) |_| {
        var sum = atomic.Value(u64).init(0);

        const Ctx = struct { sum: *atomic.Value(u64) };
        // parallelForEach uses threshold=1, maximizing splitting (stress for addPending race)
        tp.parallelForEach(1000, Ctx{ .sum = &sum }, struct {
            fn run(c: Ctx, i: usize) void {
                _ = c.sum.fetchAdd(@as(u64, @intCast(i)), .monotonic);
            }
        }.run);

        try std.testing.expectEqual(@as(u64, 999 * 1000 / 2), sum.load(.acquire));
    }
}

test "ThreadPool: parallelChunks basic" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    const n = 1024;
    var data: [n]u32 = undefined;
    for (0..n) |i| data[i] = 0;

    tp.parallelChunks(u32, &data, 128, {}, struct {
        fn run(_: void, chunk: []u32, start_idx: usize) void {
            for (chunk, 0..) |*elem, j| {
                elem.* = @intCast(start_idx + j);
            }
        }
    }.run);

    for (0..n) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i)), data[i]);
    }
}

test "ThreadPool: getPool returns correct pool" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    var seen_pool = atomic.Value(usize).init(0);

    const Ctx = struct { pool: *ThreadPool, seen: *atomic.Value(usize) };
    tp.parallelForForce(8, Ctx{ .pool = tp, .seen = &seen_pool }, struct {
        fn run(c: Ctx, _: usize) void {
            if (ThreadPool.getPool()) |p| {
                if (p == c.pool) {
                    _ = c.seen.fetchAdd(1, .monotonic);
                }
            }
        }
    }.run);

    try std.testing.expectEqual(@as(usize, 8), seen_pool.load(.acquire));
}

test "ThreadPool: join with thread_count=1" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 1);
    defer tp.deinit();

    // With the fix, thread_count=1 means 1 worker + main = 2 threads, so join should parallelize.
    var task_a_ran = atomic.Value(u32).init(0);
    var task_b_ran = atomic.Value(u32).init(0);

    const Ctx = struct { flag: *atomic.Value(u32) };

    const result = tp.join(
        u64,
        u64,
        Ctx{ .flag = &task_a_ran },
        struct {
            fn a(c: Ctx) u64 {
                c.flag.store(1, .release);
                return 42;
            }
        }.a,
        Ctx{ .flag = &task_b_ran },
        struct {
            fn b(c: Ctx) u64 {
                c.flag.store(1, .release);
                return 99;
            }
        }.b,
    );

    try std.testing.expectEqual(@as(u64, 42), result[0]);
    try std.testing.expectEqual(@as(u64, 99), result[1]);
    try std.testing.expectEqual(@as(u32, 1), task_a_ran.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), task_b_ran.load(.acquire));
}

test "ThreadPool: deque overflow fallback" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 2);
    defer tp.deinit();

    // Deep nesting that may overflow deque (DEQUE_CAP=256).
    // parallelForEach uses threshold=1, maximizing splitting to stress deque capacity.
    const n = 1024;
    var data: [n]u64 = undefined;
    for (0..n) |i| data[i] = 0;

    const Ctx = struct { data: *[n]u64 };
    tp.parallelForEach(n, Ctx{ .data = &data }, struct {
        fn run(c: Ctx, i: usize) void {
            c.data[i] = @intCast(i + 1);
        }
    }.run);

    for (0..n) |i| {
        try std.testing.expectEqual(@as(u64, i + 1), data[i]);
    }
}

test "ThreadPool: thread_count=0 sequential fallback" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 0);
    defer tp.deinit();

    // parallelFor should run sequentially
    var sum = atomic.Value(u64).init(0);
    const SumCtx = struct { s: *atomic.Value(u64) };
    tp.parallelFor(100, SumCtx{ .s = &sum }, struct {
        fn run(c: SumCtx, i: usize) void {
            _ = c.s.fetchAdd(@as(u64, @intCast(i)), .monotonic);
        }
    }.run);
    try std.testing.expectEqual(@as(u64, 99 * 100 / 2), sum.load(.acquire));

    // parallelForForce should also run sequentially (len > 1 but thread_count = 0)
    sum.store(0, .monotonic);
    tp.parallelForForce(50, SumCtx{ .s = &sum }, struct {
        fn run(c: SumCtx, i: usize) void {
            _ = c.s.fetchAdd(@as(u64, @intCast(i)), .monotonic);
        }
    }.run);
    try std.testing.expectEqual(@as(u64, 49 * 50 / 2), sum.load(.acquire));

    // parallelReduce should run sequentially
    const result = tp.parallelReduce(
        u64,
        100,
        0,
        {},
        struct {
            fn map(_: void, start: usize, end: usize) u64 {
                var s: u64 = 0;
                for (start..end) |i| s += @as(u64, @intCast(i));
                return s;
            }
        }.map,
        struct {
            fn reduce(a: u64, b: u64) u64 {
                return a + b;
            }
        }.reduce,
    );
    try std.testing.expectEqual(@as(u64, 99 * 100 / 2), result);

    // join should run sequentially
    const jr = tp.join(u64, u64, {}, struct {
        fn a(_: void) u64 {
            return 42;
        }
    }.a, {}, struct {
        fn b(_: void) u64 {
            return 99;
        }
    }.b);
    try std.testing.expectEqual(@as(u64, 42), jr[0]);
    try std.testing.expectEqual(@as(u64, 99), jr[1]);
}

test "ThreadPool: mixed nesting patterns stress" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    // Outer parallelForForce with heterogeneous inner patterns per iteration:
    // - Even indices: join(parallelReduce, parallelFor)
    // - Odd indices: nested parallelForForce
    const n = 8;
    var results: [n]u64 = [_]u64{0} ** n;

    const Ctx = struct { pool: *ThreadPool, results: *[n]u64 };
    const ctx = Ctx{ .pool = tp, .results = &results };

    tp.parallelForForce(n, ctx, struct {
        fn run(c: Ctx, i: usize) void {
            if (i % 2 == 0) {
                // Even: join with inner reduce + inner parallelFor
                const JCtx = struct { pool: *ThreadPool };
                const jc = JCtx{ .pool = c.pool };
                const jr = c.pool.join(u64, u64, jc, struct {
                    fn a(j: JCtx) u64 {
                        return j.pool.parallelReduce(u64, 128, 0, {}, struct {
                            fn map(_: void, start: usize, end: usize) u64 {
                                var s: u64 = 0;
                                for (start..end) |k| s += @as(u64, @intCast(k));
                                return s;
                            }
                        }.map, struct {
                            fn reduce(x: u64, y: u64) u64 {
                                return x + y;
                            }
                        }.reduce);
                    }
                }.a, jc, struct {
                    fn b(j: JCtx) u64 {
                        var s = atomic.Value(u64).init(0);
                        const SCtx = struct { sum: *atomic.Value(u64) };
                        j.pool.parallelForForce(128, SCtx{ .sum = &s }, struct {
                            fn inner(sc: SCtx, k: usize) void {
                                _ = sc.sum.fetchAdd(@as(u64, @intCast(k)), .monotonic);
                            }
                        }.inner);
                        return s.load(.acquire);
                    }
                }.b);
                c.results[i] = jr[0] + jr[1];
            } else {
                // Odd: nested parallelForForce with atomic accumulation
                var s = atomic.Value(u64).init(0);
                const SCtx = struct { pool: *ThreadPool, sum: *atomic.Value(u64) };
                c.pool.parallelForForce(16, SCtx{ .pool = c.pool, .sum = &s }, struct {
                    fn outer(sc: SCtx, j: usize) void {
                        var inner_s = atomic.Value(u64).init(0);
                        const ICtx = struct { sum: *atomic.Value(u64), base: usize };
                        sc.pool.parallelForForce(16, ICtx{ .sum = &inner_s, .base = j * 16 }, struct {
                            fn inner(ic: ICtx, k: usize) void {
                                _ = ic.sum.fetchAdd(@as(u64, @intCast(ic.base + k)), .monotonic);
                            }
                        }.inner);
                        _ = sc.sum.fetchAdd(inner_s.load(.acquire), .monotonic);
                    }
                }.outer);
                c.results[i] = s.load(.acquire);
            }
        }
    }.run);

    const expected_even: u64 = 127 * 128 / 2; // sum(0..127) from reduce + sum(0..127) from parallelFor
    const expected_odd: u64 = 255 * 256 / 2; // sum(0..255) via 16x16 nested grid
    for (0..n) |i| {
        if (i % 2 == 0) {
            try std.testing.expectEqual(expected_even * 2, results[i]);
        } else {
            try std.testing.expectEqual(expected_odd, results[i]);
        }
    }
}

test "ThreadPool: nested reduce inside reduce" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    // Outer reduce: 8 chunks, each chunk's map does an inner reduce over 256 elements.
    const result = tp.parallelReduceForce(
        u64,
        8,
        0,
        tp,
        struct {
            fn map(pool: *ThreadPool, start: usize, end: usize) u64 {
                var total: u64 = 0;
                for (start..end) |chunk| {
                    const base = chunk * 256;
                    total += pool.parallelReduce(
                        u64,
                        256,
                        0,
                        base,
                        struct {
                            fn inner_map(b: usize, s: usize, e: usize) u64 {
                                var sum_val: u64 = 0;
                                for (s..e) |k| sum_val += @as(u64, @intCast(b + k));
                                return sum_val;
                            }
                        }.inner_map,
                        struct {
                            fn reduce(a: u64, b: u64) u64 {
                                return a + b;
                            }
                        }.reduce,
                    );
                }
                return total;
            }
        }.map,
        struct {
            fn reduce(a: u64, b: u64) u64 {
                return a + b;
            }
        }.reduce,
    );

    // Total: sum of (base + k) for base in {0,256,512,...,1792}, k in 0..255
    // = sum of 0..2047 = 2047 * 2048 / 2 = 2096128
    try std.testing.expectEqual(@as(u64, 2047 * 2048 / 2), result);
}

test "ThreadPool: concurrent external callers fallback" {
    var tp = try ThreadPool.initWithCount(std.testing.allocator, 4);
    defer tp.deinit();

    // Two OS threads both try to use the same pool. One should succeed with parallel
    // dispatch, the other should fall back to sequential. Both must produce correct results.
    var sum_a = atomic.Value(u64).init(0);
    var sum_b = atomic.Value(u64).init(0);

    const Work = struct {
        fn run(pool: *ThreadPool, sum: *atomic.Value(u64)) void {
            const WorkCtx = struct { sum: *atomic.Value(u64) };
            pool.parallelForForce(512, WorkCtx{ .sum = sum }, struct {
                fn work(c: WorkCtx, i: usize) void {
                    _ = c.sum.fetchAdd(@as(u64, @intCast(i)), .monotonic);
                }
            }.work);
        }
    };

    const t = try std.Thread.spawn(.{}, Work.run, .{ tp, &sum_b });
    Work.run(tp, &sum_a);
    t.join();

    const expected: u64 = 511 * 512 / 2;
    try std.testing.expectEqual(expected, sum_a.load(.acquire));
    try std.testing.expectEqual(expected, sum_b.load(.acquire));
}
