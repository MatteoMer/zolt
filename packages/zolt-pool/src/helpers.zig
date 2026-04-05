//! Parallel helper utilities that reduce boilerplate around optional ThreadPool usage.
//!
//! These helpers handle the common pattern of "if pool exists, use parallel execution,
//! otherwise fall back to sequential" which appears 60+ times across the codebase.

const ThreadPool = @import("thread_pool.zig").ThreadPool;

/// Optional parallelReduce — falls back to sequential map if pool is null.
///
/// Replaces the pattern:
///   if (self.pool) |tp| tp.parallelReduce(R, len, identity, ctx, mapFn, reduceFn) else mapFn(ctx, 0, len);
pub fn parallelReduceOptional(
    comptime R: type,
    pool: ?*ThreadPool,
    len: usize,
    identity: R,
    ctx: anytype,
    mapFn: anytype,
    reduceFn: anytype,
) R {
    return if (pool) |p|
        p.parallelReduce(R, len, identity, ctx, mapFn, reduceFn)
    else
        mapFn(ctx, 0, len);
}

/// Optional parallelReduceForce — falls back to sequential map if pool is null.
///
/// Like parallelReduceOptional but uses force semantics (ignores MIN_ITEMS_PER_THREAD).
/// Replaces the pattern:
///   if (self.pool) |tp| tp.parallelReduceForce(R, len, identity, ctx, mapFn, reduceFn) else mapFn(ctx, 0, len);
pub fn parallelReduceForceOptional(
    comptime R: type,
    pool: ?*ThreadPool,
    len: usize,
    identity: R,
    ctx: anytype,
    mapFn: anytype,
    reduceFn: anytype,
) R {
    return if (pool) |p|
        p.parallelReduceForce(R, len, identity, ctx, mapFn, reduceFn)
    else
        mapFn(ctx, 0, len);
}

/// Optional parallelFor with force semantics — falls back to sequential loop if pool is null.
///
/// Replaces the pattern:
///   if (self.pool) |p| { p.parallelForForce(len, ctx, f); } else { for (0..len) |i| f(ctx, i); }
pub fn parallelForOptional(
    pool: ?*ThreadPool,
    len: usize,
    ctx: anytype,
    f: anytype,
) void {
    if (pool) |p| {
        p.parallelForForce(len, ctx, f);
    } else {
        for (0..len) |i| {
            f(ctx, i);
        }
    }
}
