//! ThreadPool - Atomic counter dispatch thread pool
//!
//! Replaces std.Thread.Pool-based dispatch with zero-allocation atomic counter dispatch.
//! Workers claim chunks via fetchAdd on a shared counter. Generation-based wake with
//! 3-phase spin-park (spin → yield → futex) for low-latency dispatch.
//!
//! Same public API as the previous implementation — zero call-site changes required.

const std = @import("std");
const atomic = std.atomic;
const Futex = std.Thread.Futex;
const Allocator = std.mem.Allocator;

/// Minimum number of elements per thread to justify parallelism.
/// Below this threshold, work runs sequentially on the caller's thread.
const MIN_ITEMS_PER_THREAD: usize = 256;

const MAX_THREADS: usize = 16;
const cache_line = atomic.cache_line;

// Spin tuning constants
const SPIN_ITERS: u32 = 64;
const YIELD_ITERS: u32 = 8;

const WorkerState = enum(u32) {
    spinning = 0,
    yielding = 1,
    parking = 2,
    parked = 3,
    shutdown = 4,
};

const PaddedWorkerState = struct {
    state: atomic.Value(u32) align(cache_line) = atomic.Value(u32).init(@intFromEnum(WorkerState.spinning)),
    last_seen_gen: u32 = 0,
};

/// Type-erased function pointer for dispatch callbacks.
const DispatchFn = *const fn (dispatch: *DispatchState, start: usize, end: usize) void;

const DispatchState = struct {
    // CL0: Written once by caller, read once by each worker
    func_ptr: DispatchFn align(cache_line) = undefined,
    context_ptr: *const anyopaque = undefined,
    total_items: usize = 0,
    chunk_size: usize = 0,
    num_chunks: usize = 0,

    // CL1: Hot contended atomics
    next_chunk: atomic.Value(u32) align(cache_line) = atomic.Value(u32).init(0),
    remaining: atomic.Value(u32) = atomic.Value(u32).init(0),
    done_futex: atomic.Value(u32) = atomic.Value(u32).init(0),

    // CL2: Generation counter (polled by spinning workers)
    generation: atomic.Value(u32) align(cache_line) = atomic.Value(u32).init(0),
};

pub const ThreadPool = struct {
    dispatch: DispatchState,
    dispatch_depth: atomic.Value(u32) align(cache_line),
    workers: [MAX_THREADS]PaddedWorkerState,
    threads: [MAX_THREADS]std.Thread,
    thread_count: usize,
    allocator: Allocator,

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
            .dispatch = .{},
            .dispatch_depth = atomic.Value(u32).init(0),
            .workers = [_]PaddedWorkerState{.{}} ** MAX_THREADS,
            .threads = undefined,
            .thread_count = actual_count,
            .allocator = allocator,
        };

        // Spawn worker threads
        var spawned: usize = 0;
        errdefer {
            // Shutdown any already-spawned threads
            for (self.workers[0..spawned]) |*w| {
                w.state.store(@intFromEnum(WorkerState.shutdown), .release);
            }
            _ = self.dispatch.generation.fetchAdd(1, .release);
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
        // Signal shutdown to all workers
        for (self.workers[0..self.thread_count]) |*w| {
            w.state.store(@intFromEnum(WorkerState.shutdown), .release);
        }
        // Bump generation so spinning/yielding workers see shutdown
        _ = self.dispatch.generation.fetchAdd(1, .release);
        // Wake parked workers
        for (self.workers[0..self.thread_count]) |*w| {
            Futex.wake(&w.state, 1);
        }
        // Join all threads
        for (self.threads[0..self.thread_count]) |t| {
            t.join();
        }
        self.allocator.destroy(self);
    }

    // ========================================================================
    // Public API
    // ========================================================================

    /// Force-parallel for: ignores MIN_ITEMS_PER_THREAD threshold.
    /// Use for heavy operations where even 1 item per thread is worthwhile.
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

        // Nested dispatch detection
        if (self.dispatch_depth.load(.acquire) > 0) {
            for (0..len) |i| func(context, i);
            return;
        }

        const Ctx = @TypeOf(context);
        var ctx_copy = context;

        const Wrapper = struct {
            fn call(d: *DispatchState, start: usize, end: usize) void {
                const ctx: *const Ctx = @ptrCast(@alignCast(d.context_ptr));
                for (start..end) |i| func(ctx.*, i);
            }
        };

        const actual_threads = @min(self.thread_count + 1, len);
        const chunk_size = (len + actual_threads - 1) / actual_threads;
        const num_chunks = (len + chunk_size - 1) / chunk_size;

        self.dispatchAndWait(&Wrapper.call, @ptrCast(&ctx_copy), len, chunk_size, num_chunks, actual_threads);
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

        // Nested dispatch detection
        if (self.dispatch_depth.load(.acquire) > 0) {
            for (0..len) |i| func(context, i);
            return;
        }

        const Ctx = @TypeOf(context);
        var ctx_copy = context;

        const Wrapper = struct {
            fn call(d: *DispatchState, start: usize, end: usize) void {
                const ctx: *const Ctx = @ptrCast(@alignCast(d.context_ptr));
                for (start..end) |i| func(ctx.*, i);
            }
        };

        // chunk_size = 1 for per-item dispatch
        self.dispatchAndWait(&Wrapper.call, @ptrCast(&ctx_copy), len, 1, len, @min(self.thread_count + 1, len));
    }

    /// Parallel for: apply `func(context, index)` for each index in 0..len.
    /// Each invocation is independent and may run on any thread.
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

        // Nested dispatch detection
        if (self.dispatch_depth.load(.acquire) > 0) {
            for (0..len) |i| func(context, i);
            return;
        }

        const Ctx = @TypeOf(context);
        var ctx_copy = context;

        const Wrapper = struct {
            fn call(d: *DispatchState, start: usize, end: usize) void {
                const ctx: *const Ctx = @ptrCast(@alignCast(d.context_ptr));
                for (start..end) |i| func(ctx.*, i);
            }
        };

        const chunk_size = (len + actual_threads - 1) / actual_threads;
        const num_chunks = (len + chunk_size - 1) / chunk_size;

        self.dispatchAndWait(&Wrapper.call, @ptrCast(&ctx_copy), len, chunk_size, num_chunks, actual_threads);
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

        // Nested dispatch detection
        if (self.dispatch_depth.load(.acquire) > 0) {
            var offset: usize = 0;
            while (offset < slice.len) {
                const end = @min(offset + actual_chunk_size, slice.len);
                func(context, slice[offset..end], offset);
                offset = end;
            }
            return;
        }

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

        const Wrapper = struct {
            fn call(d: *DispatchState, start: usize, end: usize) void {
                const cc: *const ChunkCtx = @ptrCast(@alignCast(d.context_ptr));
                // Each "item" is a chunk index
                for (start..end) |chunk_idx| {
                    const off = chunk_idx * cc.chunk_size;
                    const chunk_end = @min(off + cc.chunk_size, cc.slice_len);
                    func(cc.ctx, cc.slice_ptr[off..chunk_end], off);
                }
            }
        };

        self.dispatchAndWait(&Wrapper.call, @ptrCast(&chunk_ctx), n_chunks, 1, n_chunks, @min(self.thread_count + 1, n_chunks));
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
    /// Use this for heavy operations (MSM, Miller loops) where even 1 item per thread is worthwhile.
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
        if (self.thread_count <= 1) {
            return .{ func_a(context_a), func_b(context_b) };
        }

        // Nested dispatch detection
        if (self.dispatch_depth.load(.acquire) > 0) {
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

        const Wrapper = struct {
            fn call(d: *DispatchState, _: usize, _: usize) void {
                const jc: *const JoinCtx = @ptrCast(@alignCast(d.context_ptr));
                jc.result_ptr.* = func_b(jc.ctx_b);
            }
        };

        // Dispatch task B as single chunk to 1 worker (caller does task A, not workLoop)
        self.dispatchAndRun(&Wrapper.call, @ptrCast(&join_ctx), 1, 1, 1, 2);

        // Caller executes task A
        const result_a = func_a(context_a);

        // Wait for task B
        self.waitForCompletion();

        return .{ result_a, result_b };
    }

    // ========================================================================
    // Internal: Reduce implementation
    // ========================================================================

    fn reduceImpl(
        self: *ThreadPool,
        comptime R: type,
        len: usize,
        identity: R,
        context: anytype,
        comptime map: fn (@TypeOf(context), usize, usize) R,
        comptime reduce: fn (R, R) R,
        actual_threads: usize,
    ) R {
        // Nested dispatch detection
        if (self.dispatch_depth.load(.acquire) > 0) {
            return map(context, 0, len);
        }

        const chunk_size = (len + actual_threads - 1) / actual_threads;
        const num_chunks = (len + chunk_size - 1) / chunk_size;

        const Ctx = @TypeOf(context);

        // Padded partial to prevent false sharing between worker threads
        const padded_size = if (@sizeOf(R) <= cache_line) cache_line else ((@sizeOf(R) + cache_line - 1) / cache_line) * cache_line;
        const PaddedPartial = struct {
            value: R,
            _padding: [padded_size - @sizeOf(R)]u8 = undefined,
        };

        var partials: [MAX_THREADS + 1]PaddedPartial = undefined;
        for (0..num_chunks) |i| {
            partials[i].value = identity;
        }

        const ReduceCtx = struct {
            ctx: Ctx,
            partials_ptr: [*]PaddedPartial,
            chunk_size: usize,
            total_items: usize,
        };
        var reduce_ctx = ReduceCtx{
            .ctx = context,
            .partials_ptr = &partials,
            .chunk_size = chunk_size,
            .total_items = len,
        };

        const Wrapper = struct {
            fn call(d: *DispatchState, start: usize, end: usize) void {
                const rc: *const ReduceCtx = @ptrCast(@alignCast(d.context_ptr));
                // Each "item" in dispatch is a chunk index
                for (start..end) |chunk_idx| {
                    const item_start = chunk_idx * rc.chunk_size;
                    const item_end = @min(item_start + rc.chunk_size, rc.total_items);
                    rc.partials_ptr[chunk_idx].value = map(rc.ctx, item_start, item_end);
                }
            }
        };

        // Dispatch chunks (each dispatch item = one reduce chunk)
        self.dispatchAndWait(&Wrapper.call, @ptrCast(&reduce_ctx), num_chunks, 1, num_chunks, @min(self.thread_count + 1, num_chunks));

        // Sequential combine
        var result = partials[0].value;
        for (1..num_chunks) |i| {
            result = reduce(result, partials[i].value);
        }
        return result;
    }

    // ========================================================================
    // Internal: Dispatch core
    // ========================================================================

    /// Setup dispatch, wake workers, caller participates, then wait for completion.
    fn dispatchAndWait(
        self: *ThreadPool,
        func_ptr: *const fn (*DispatchState, usize, usize) void,
        context_ptr: *const anyopaque,
        total_items: usize,
        chunk_size: usize,
        num_chunks: usize,
        actual_threads: usize,
    ) void {
        self.dispatchAndRun(func_ptr, context_ptr, total_items, chunk_size, num_chunks, actual_threads);
        self.workLoop(); // caller participates
        self.waitForCompletion();
    }

    /// Setup dispatch and wake workers, but don't participate or wait.
    fn dispatchAndRun(
        self: *ThreadPool,
        func_ptr: *const fn (*DispatchState, usize, usize) void,
        context_ptr: *const anyopaque,
        total_items: usize,
        chunk_size: usize,
        num_chunks: usize,
        actual_threads: usize,
    ) void {
        _ = self.dispatch_depth.fetchAdd(1, .monotonic);

        // Setup dispatch state
        self.dispatch.func_ptr = func_ptr;
        self.dispatch.context_ptr = context_ptr;
        self.dispatch.total_items = total_items;
        self.dispatch.chunk_size = chunk_size;
        self.dispatch.num_chunks = num_chunks;
        self.dispatch.next_chunk.store(0, .monotonic);
        self.dispatch.remaining.store(@intCast(num_chunks), .monotonic);
        self.dispatch.done_futex.store(0, .monotonic);

        // Publish dispatch state to workers
        _ = self.dispatch.generation.fetchAdd(1, .release);

        // Wake workers (minus 1 for caller, unless this is join where caller doesn't participate via workLoop call here)
        const workers_to_wake = if (actual_threads > 0) actual_threads - 1 else 0;
        self.wakeWorkers(workers_to_wake);
    }

    /// Wait for all chunks to complete.
    fn waitForCompletion(self: *ThreadPool) void {
        while (self.dispatch.done_futex.load(.acquire) != 1) {
            Futex.wait(&self.dispatch.done_futex, 0);
        }
        _ = self.dispatch_depth.fetchSub(1, .release);
    }

    /// Claim and execute chunks from the dispatch.
    fn workLoop(self: *ThreadPool) void {
        while (true) {
            const chunk_idx = self.dispatch.next_chunk.fetchAdd(1, .monotonic);
            if (chunk_idx >= self.dispatch.num_chunks) break;

            const start = chunk_idx * self.dispatch.chunk_size;
            const end = @min(start + self.dispatch.chunk_size, self.dispatch.total_items);
            self.dispatch.func_ptr(&self.dispatch, start, end);

            const prev = self.dispatch.remaining.fetchSub(1, .release);
            if (prev == 1) {
                // Last chunk completed
                self.dispatch.done_futex.store(1, .release);
                Futex.wake(&self.dispatch.done_futex, 1);
                return;
            }
        }
    }

    // ========================================================================
    // Internal: Worker thread
    // ========================================================================

    fn workerMain(self: *ThreadPool, worker_idx: usize) void {
        const worker = &self.workers[worker_idx];
        var last_seen_gen: u32 = 0;

        while (true) {
            // Check for shutdown
            if (worker.state.load(.acquire) == @intFromEnum(WorkerState.shutdown)) return;

            // Check for new work
            const gen = self.dispatch.generation.load(.acquire);
            if (gen != last_seen_gen) {
                last_seen_gen = gen;
                // Re-check shutdown after acquiring generation
                if (worker.state.load(.acquire) == @intFromEnum(WorkerState.shutdown)) return;
                self.workLoop();
                continue;
            }

            // Phase 1: Spin
            var found_work = false;
            for (0..SPIN_ITERS) |_| {
                atomic.spinLoopHint();
                if (worker.state.load(.acquire) == @intFromEnum(WorkerState.shutdown)) return;
                const g = self.dispatch.generation.load(.acquire);
                if (g != last_seen_gen) {
                    found_work = true;
                    break;
                }
            }
            if (found_work) continue;

            // Phase 2: Yield
            worker.state.store(@intFromEnum(WorkerState.yielding), .release);
            for (0..YIELD_ITERS) |_| {
                std.Thread.yield() catch {};
                if (worker.state.load(.acquire) == @intFromEnum(WorkerState.shutdown)) return;
                const g = self.dispatch.generation.load(.acquire);
                if (g != last_seen_gen) {
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

            // Critical: recheck generation after storing parking state
            const recheck_gen = self.dispatch.generation.load(.acquire);
            if (recheck_gen != last_seen_gen) {
                worker.state.store(@intFromEnum(WorkerState.spinning), .release);
                continue;
            }

            worker.state.store(@intFromEnum(WorkerState.parked), .release);
            Futex.wait(&worker.state, @intFromEnum(WorkerState.parked));
            // Woken up: state was CAS'd to spinning by wakeWorkers, or set to
            // shutdown by deinit. Do NOT overwrite — just loop back and recheck.
        }
    }

    /// Wake up to `count` workers using CAS-based selective wake.
    fn wakeWorkers(self: *ThreadPool, count: usize) void {
        if (count == 0) return;

        var woken: usize = 0;
        for (self.workers[0..self.thread_count]) |*worker| {
            if (woken >= count) break;

            const state = worker.state.load(.acquire);
            if (state == @intFromEnum(WorkerState.parked)) {
                // Try to CAS parked → spinning, then futex wake
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
                // Try to CAS parking → spinning (no futex needed)
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
        // +1 for the calling thread which also participates
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

    const n = 37; // simulate ~37 polynomial commits
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
            fn run(_: void, _: usize) void {
                // Empty work — measure dispatch overhead only
            }
        }.run);
    }

    const elapsed_ns = timer.read();
    const per_dispatch_ns = elapsed_ns / iters;
    // Log for informational purposes; no strict assertion
    std.debug.print("\n  Dispatch overhead: {}ns per dispatch ({} dispatches)\n", .{ per_dispatch_ns, iters });

    // Soft assertion: should be well under 100µs per dispatch
    try std.testing.expect(per_dispatch_ns < 100_000);
}
