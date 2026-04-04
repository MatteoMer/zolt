//! Parallel sample sort using ThreadPool.
//!
//! For large arrays (>= 65536 elements), uses parallel sample sort:
//! 1. Sample splitters from input
//! 2. Classify elements into P buckets via binary search
//! 3. Local pdq sort each bucket in parallel
//!
//! For small arrays, falls back to sequential std.sort.pdq.

const std = @import("std");
const ThreadPool = @import("thread_pool.zig").ThreadPool;

/// Parallel sort using sample sort + pdq for large arrays.
/// Falls back to sequential pdq for small arrays.
pub fn parallelSort(
    comptime T: type,
    items: []T,
    pool: *ThreadPool,
    context: anytype,
    comptime lessThanFn: fn (@TypeOf(context), T, T) bool,
) void {
    const Context = @TypeOf(context);
    const MIN_PARALLEL_SIZE = 65536;

    if (items.len < MIN_PARALLEL_SIZE) {
        std.sort.pdq(T, items, context, lessThanFn);
        return;
    }

    // Determine number of buckets = number of threads + 1
    const P = pool.thread_count + 1;
    if (P <= 1) {
        std.sort.pdq(T, items, context, lessThanFn);
        return;
    }

    // Step 1: Sample splitters
    // Take every 64th element as a sample
    const sample_stride = 64;
    const n_samples = items.len / sample_stride;
    if (n_samples < P) {
        std.sort.pdq(T, items, context, lessThanFn);
        return;
    }

    const samples = pool.allocator.alloc(T, n_samples) catch {
        std.sort.pdq(T, items, context, lessThanFn);
        return;
    };
    defer pool.allocator.free(samples);

    for (0..n_samples) |i| {
        samples[i] = items[i * sample_stride];
    }
    std.sort.pdq(T, samples, context, lessThanFn);

    // Pick P-1 evenly spaced splitters
    const n_splitters = P - 1;
    const splitters = pool.allocator.alloc(T, n_splitters) catch {
        std.sort.pdq(T, items, context, lessThanFn);
        return;
    };
    defer pool.allocator.free(splitters);

    for (0..n_splitters) |i| {
        const idx = (i + 1) * n_samples / P;
        splitters[i] = samples[@min(idx, n_samples - 1)];
    }

    // Step 2: Count elements per bucket per thread chunk
    const chunk_size = (items.len + P - 1) / P;
    const n_chunks = (items.len + chunk_size - 1) / chunk_size;

    // count_matrix[chunk][bucket]
    const count_matrix = pool.allocator.alloc([]usize, n_chunks) catch {
        std.sort.pdq(T, items, context, lessThanFn);
        return;
    };
    defer {
        for (count_matrix[0..n_chunks]) |row| pool.allocator.free(row);
        pool.allocator.free(count_matrix);
    }
    for (0..n_chunks) |c| {
        count_matrix[c] = pool.allocator.alloc(usize, P) catch {
            std.sort.pdq(T, items, context, lessThanFn);
            return;
        };
        @memset(count_matrix[c], 0);
    }

    // Classify in parallel
    const ClassifyCtx = struct {
        items_ptr: []const T,
        splitters_ptr: []const T,
        count_mat: [][]usize,
        chunk_sz: usize,
        ctx: Context,

        fn classify(self: @This(), chunk_idx: usize) void {
            const start = chunk_idx * self.chunk_sz;
            const end = @min(start + self.chunk_sz, self.items_ptr.len);
            const counts = self.count_mat[chunk_idx];

            for (start..end) |i| {
                const bucket = binarySearchBucket(T, Context, self.splitters_ptr, self.items_ptr[i], self.ctx, lessThanFn);
                counts[bucket] += 1;
            }
        }
    };
    const classify_ctx = ClassifyCtx{
        .items_ptr = items,
        .splitters_ptr = splitters,
        .count_mat = count_matrix,
        .chunk_sz = chunk_size,
        .ctx = context,
    };
    pool.parallelForForce(n_chunks, classify_ctx, ClassifyCtx.classify);

    // Step 3: Prefix sum to compute global offsets
    // bucket_offsets[b] = start of bucket b in output
    const bucket_offsets = pool.allocator.alloc(usize, P + 1) catch {
        std.sort.pdq(T, items, context, lessThanFn);
        return;
    };
    defer pool.allocator.free(bucket_offsets);

    bucket_offsets[0] = 0;
    for (0..P) |b| {
        var sum: usize = 0;
        for (0..n_chunks) |c| {
            sum += count_matrix[c][b];
        }
        bucket_offsets[b + 1] = bucket_offsets[b] + sum;
    }

    // Compute per-chunk write positions: write_pos[chunk][bucket] = start offset for this chunk's bucket
    const write_pos = pool.allocator.alloc([]usize, n_chunks) catch {
        std.sort.pdq(T, items, context, lessThanFn);
        return;
    };
    defer {
        for (write_pos[0..n_chunks]) |row| pool.allocator.free(row);
        pool.allocator.free(write_pos);
    }
    for (0..n_chunks) |c| {
        write_pos[c] = pool.allocator.alloc(usize, P) catch {
            std.sort.pdq(T, items, context, lessThanFn);
            return;
        };
    }

    for (0..P) |b| {
        var pos = bucket_offsets[b];
        for (0..n_chunks) |c| {
            write_pos[c][b] = pos;
            pos += count_matrix[c][b];
        }
    }

    // Step 4: Scatter elements to output buffer in parallel
    const output = pool.allocator.alloc(T, items.len) catch {
        std.sort.pdq(T, items, context, lessThanFn);
        return;
    };
    defer pool.allocator.free(output);

    // Scatter is sequential per-chunk to avoid atomic writes (each chunk has unique write positions)
    for (0..n_chunks) |c| {
        const start = c * chunk_size;
        const end = @min(start + chunk_size, items.len);
        const wp = write_pos[c];
        for (start..end) |i| {
            const bucket = binarySearchBucket(T, Context, splitters, items[i], context, lessThanFn);
            output[wp[bucket]] = items[i];
            wp[bucket] += 1;
        }
    }

    // Step 5: Sort each bucket in parallel with pdq
    const SortCtx = struct {
        out: []T,
        b_offsets: []const usize,
        ctx: Context,

        fn sortBucket(self: @This(), bucket: usize) void {
            const bstart = self.b_offsets[bucket];
            const bend = self.b_offsets[bucket + 1];
            if (bend > bstart) {
                std.sort.pdq(T, self.out[bstart..bend], self.ctx, lessThanFn);
            }
        }
    };
    pool.parallelForForce(P, SortCtx{
        .out = output,
        .b_offsets = bucket_offsets,
        .ctx = context,
    }, SortCtx.sortBucket);

    // Step 6: Copy sorted output back
    @memcpy(items, output);
}

/// Binary search for the bucket index of an element given sorted splitters.
/// Returns the index of the first splitter that is > element (or n_splitters if all <=).
fn binarySearchBucket(
    comptime T: type,
    comptime Context: type,
    splitters: []const T,
    element: T,
    context: Context,
    comptime lessThanFn: fn (Context, T, T) bool,
) usize {
    var lo: usize = 0;
    var hi: usize = splitters.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (lessThanFn(context, element, splitters[mid])) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}
