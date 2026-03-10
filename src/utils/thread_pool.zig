//! ThreadPool - Rayon-like parallel primitives built on std.Thread.Pool
//!
//! Provides parallelFor, parallelChunks, parallelReduce, and join for data-parallel work.
//! Uses a persistent thread pool to avoid spawn/join overhead per call.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Minimum number of elements per thread to justify parallelism.
/// Below this threshold, work runs sequentially on the caller's thread.
const MIN_ITEMS_PER_THREAD: usize = 256;

pub const ThreadPool = struct {
    /// Heap-allocated because std.Thread.Pool cannot be moved after init
    /// (worker threads hold a pointer to it).
    pool: *std.Thread.Pool,
    thread_count: usize,
    allocator: Allocator,

    /// Initialize a thread pool with auto-detected CPU count (capped at 8).
    pub fn init(allocator: Allocator) !ThreadPool {
        const cpu_count = std.Thread.getCpuCount() catch 4;
        const thread_count = @min(cpu_count, 16);
        return initWithCount(allocator, thread_count);
    }

    /// Initialize a thread pool with a specific number of worker threads.
    pub fn initWithCount(allocator: Allocator, thread_count: usize) !ThreadPool {
        const pool = try allocator.create(std.Thread.Pool);
        errdefer allocator.destroy(pool);
        try pool.init(.{
            .allocator = std.heap.page_allocator,
            .n_jobs = thread_count,
        });
        return .{
            .pool = pool,
            .thread_count = thread_count,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadPool) void {
        self.pool.deinit();
        self.allocator.destroy(self.pool);
    }

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

        const Ctx = @TypeOf(context);
        const actual_threads = @min(self.thread_count + 1, len);
        const chunk_size = (len + actual_threads - 1) / actual_threads;
        var wg: std.Thread.WaitGroup = .{};

        const Helper = struct {
            fn chunkWorker(ctx: Ctx, start: usize, end: usize) void {
                for (start..end) |i| func(ctx, i);
            }
        };

        var chunk_start: usize = 0;
        while (chunk_start < len) {
            const chunk_end = @min(chunk_start + chunk_size, len);
            self.pool.spawnWg(&wg, Helper.chunkWorker, .{ context, chunk_start, chunk_end });
            chunk_start = chunk_end;
        }

        self.pool.waitAndWork(&wg);
    }

    /// Fine-grained parallel for: submits each index as its own work item.
    /// Unlike parallelForForce which chunks items, this gives optimal dynamic
    /// load balancing for small numbers of heavyweight, heterogeneous tasks.
    /// Use parallelForForce for large N with uniform items.
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

        const Ctx = @TypeOf(context);
        var wg: std.Thread.WaitGroup = .{};

        const Helper = struct {
            fn itemWorker(ctx: Ctx, i: usize) void {
                func(ctx, i);
            }
        };

        for (0..len) |i| {
            self.pool.spawnWg(&wg, Helper.itemWorker, .{ context, i });
        }

        self.pool.waitAndWork(&wg);
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

        const Ctx = @TypeOf(context);
        const chunk_size = (len + actual_threads - 1) / actual_threads;
        var wg: std.Thread.WaitGroup = .{};

        const Helper = struct {
            fn chunkWorker(ctx: Ctx, start: usize, end: usize) void {
                for (start..end) |i| func(ctx, i);
            }
        };

        var chunk_start: usize = 0;
        while (chunk_start < len) {
            const chunk_end = @min(chunk_start + chunk_size, len);
            self.pool.spawnWg(&wg, Helper.chunkWorker, .{ context, chunk_start, chunk_end });
            chunk_start = chunk_end;
        }

        self.pool.waitAndWork(&wg);
    }

    /// Parallel chunks: split `slice` into chunks and call `func(context, chunk, chunk_start_index)`.
    /// Each chunk is processed independently on a worker thread.
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

        const Ctx = @TypeOf(context);
        var wg: std.Thread.WaitGroup = .{};

        const Helper = struct {
            fn chunkWorker(ctx: Ctx, chunk: []T, off: usize) void {
                func(ctx, chunk, off);
            }
        };

        var offset: usize = 0;
        while (offset < slice.len) {
            const end = @min(offset + actual_chunk_size, slice.len);
            self.pool.spawnWg(&wg, Helper.chunkWorker, .{ context, slice[offset..end], offset });
            offset = end;
        }

        self.pool.waitAndWork(&wg);
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

        const chunk_size = (len + actual_threads - 1) / actual_threads;
        const n_chunks = (len + chunk_size - 1) / chunk_size;

        const partials = self.allocator.alloc(R, n_chunks) catch {
            return map(context, 0, len);
        };
        defer self.allocator.free(partials);
        @memset(partials, identity);

        const Ctx = @TypeOf(context);
        var wg: std.Thread.WaitGroup = .{};

        const Helper = struct {
            fn chunkWorker(ctx: Ctx, out: *R, start: usize, end: usize) void {
                out.* = map(ctx, start, end);
            }
        };

        for (0..n_chunks) |chunk_idx| {
            const start = chunk_idx * chunk_size;
            const end = @min(start + chunk_size, len);
            self.pool.spawnWg(&wg, Helper.chunkWorker, .{ context, &partials[chunk_idx], start, end });
        }

        self.pool.waitAndWork(&wg);

        var result = partials[0];
        for (partials[1..]) |p| {
            result = reduce(result, p);
        }
        return result;
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

        var result_b: RB = undefined;
        var wg: std.Thread.WaitGroup = .{};

        const CtxB = @TypeOf(context_b);
        const Helper = struct {
            fn worker(ctx: CtxB, out: *RB) void {
                out.* = func_b(ctx);
            }
        };

        self.pool.spawnWg(&wg, Helper.worker, .{ context_b, &result_b });
        const result_a = func_a(context_a);
        self.pool.waitAndWork(&wg);

        return .{ result_a, result_b };
    }

    /// How many threads to actually use given the work size.
    fn effectiveThreads(self: *const ThreadPool, total_items: usize, items_per_unit: usize) usize {
        const work_units = total_items * items_per_unit;
        if (work_units < MIN_ITEMS_PER_THREAD) return 1;
        const max_by_work = work_units / MIN_ITEMS_PER_THREAD;
        // +1 for the calling thread which also participates via waitAndWork
        return @min(self.thread_count + 1, max_by_work);
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

        // Use all available threads, 1 item per thread minimum
        const actual_threads = @min(self.thread_count + 1, len);
        if (actual_threads <= 1) {
            return map(context, 0, len);
        }

        const chunk_size = (len + actual_threads - 1) / actual_threads;
        const n_chunks = (len + chunk_size - 1) / chunk_size;

        const partials = self.allocator.alloc(R, n_chunks) catch {
            return map(context, 0, len);
        };
        defer self.allocator.free(partials);
        @memset(partials, identity);

        const Ctx = @TypeOf(context);
        var wg: std.Thread.WaitGroup = .{};

        const Helper = struct {
            fn chunkWorker(ctx: Ctx, out: *R, start: usize, end: usize) void {
                out.* = map(ctx, start, end);
            }
        };

        var offset: usize = 0;
        var chunk_idx: usize = 0;
        while (offset < len) {
            const end = @min(offset + chunk_size, len);
            self.pool.spawnWg(&wg, Helper.chunkWorker, .{ context, &partials[chunk_idx], offset, end });
            offset = end;
            chunk_idx += 1;
        }

        self.pool.waitAndWork(&wg);

        var result = partials[0];
        for (1..chunk_idx) |i| {
            result = reduce(result, partials[i]);
        }
        return result;
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
