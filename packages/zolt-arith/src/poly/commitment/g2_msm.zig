//! G2 Multi-Scalar Multiplication (MSM) for BN254.
//!
//! Pippenger bucket method with wNAF signed digits, supporting both
//! single-threaded and parallel (per-window) execution.

const std = @import("std");
const pairing = @import("../../field/pairing.zig");
const field = @import("../../field/mod.zig");
const msm = @import("../../msm/mod.zig");
const glv = msm.glv;
const ThreadPool = @import("zolt_pool").ThreadPool;

const Fp = field.BN254BaseField;
pub const G2Point = pairing.G2Point;
const G2Projective = pairing.G2Projective;

/// Public wrapper for G2 MSM benchmarking.
pub fn msmG2Bench(comptime F: type, g2_vec: []const G2Point, scalars: []const F, tp: ?*ThreadPool) G2Point {
    return msmG2(F, g2_vec, scalars, tp);
}

/// MSM for G2 points using Pippenger's bucket method with wNAF.
/// For small inputs (< 8), falls back to naive GLV scalar mul.
pub fn msmG2(comptime F: type, g2_vec: []const G2Point, scalars: []const F, tp: ?*ThreadPool) G2Point {
    const n = @min(g2_vec.len, scalars.len);
    if (n == 0) return G2Point.identity();

    // Small inputs: naive GLV
    if (n < 8) {
        var result = G2Projective.identity();
        for (0..n) |i| {
            const scaled = glv.glvScalarMulG2(g2_vec[i], scalars[i]);
            result = result.add(scaled);
        }
        return result.toAffine();
    }

    if (tp != null and n >= 256) {
        return pippengerMsmG2Parallel(F, g2_vec[0..n], scalars[0..n], tp.?);
    }

    return pippengerMsmG2(F, g2_vec[0..n], scalars[0..n]);
}

/// Pippenger MSM for G2 with wNAF signed digits + Jacobian buckets
fn pippengerMsmG2(comptime F: type, bases: []const G2Point, scalars: []const F) G2Point {
    const SCALAR_BITS: usize = 256;
    const MAX_DIGITS: usize = 87;

    const c = g2OptimalWindowSize(bases.len);
    const num_scalar_windows = (SCALAR_BITS + c - 1) / c;
    const num_windows = num_scalar_windows + 1; // +1 for wNAF carry
    const num_buckets = (@as(usize, 1) << @as(u6, @intCast(c))) / 2; // wNAF: 2^(c-1) buckets

    // Compute wNAF digits for all scalars
    const stack_threshold = 128;
    var stack_digits: [stack_threshold][MAX_DIGITS]i32 = undefined;
    var heap_digits: ?[][MAX_DIGITS]i32 = null;
    defer if (heap_digits) |buf| std.heap.page_allocator.free(buf);

    const all_digits: [][MAX_DIGITS]i32 = if (scalars.len <= stack_threshold)
        stack_digits[0..scalars.len]
    else blk: {
        heap_digits = std.heap.page_allocator.alloc([MAX_DIGITS]i32, scalars.len) catch {
            // Fallback to naive
            var result = G2Projective.identity();
            for (0..scalars.len) |i| {
                result = result.add(glv.glvScalarMulG2(bases[i], scalars[i]));
            }
            return result.toAffine();
        };
        break :blk heap_digits.?;
    };

    for (scalars, 0..) |s, i| {
        all_digits[i] = makeG2Digits(s.fromMontgomery().limbs, c, num_scalar_windows);
    }

    // Allocate buckets
    var heap_buckets: ?[]G2Projective = null;
    defer if (heap_buckets) |buf| std.heap.page_allocator.free(buf);

    var stack_buckets: [128]G2Projective = undefined;
    const buckets: []G2Projective = if (num_buckets <= 128)
        stack_buckets[0..num_buckets]
    else blk: {
        heap_buckets = std.heap.page_allocator.alloc(G2Projective, num_buckets) catch {
            var result = G2Projective.identity();
            for (0..scalars.len) |i| {
                result = result.add(glv.glvScalarMulG2(bases[i], scalars[i]));
            }
            return result.toAffine();
        };
        break :blk heap_buckets.?;
    };

    var final_result = G2Projective.identity();

    var window_idx: usize = num_windows;
    while (window_idx > 0) {
        window_idx -= 1;

        if (!final_result.isIdentity()) {
            var k: usize = 0;
            while (k < c) : (k += 1) {
                final_result = final_result.double();
            }
        }

        // Reset buckets
        for (0..num_buckets) |j| {
            buckets[j] = G2Projective.identity();
        }

        // Accumulate using wNAF signed digits
        for (bases, 0..) |base, idx| {
            if (base.infinity) continue;
            const digit = all_digits[idx][window_idx];
            if (digit > 0) {
                const bidx: usize = @intCast(digit - 1);
                buckets[bidx] = buckets[bidx].addAffine(base);
            } else if (digit < 0) {
                const bidx: usize = @intCast(-digit - 1);
                buckets[bidx] = buckets[bidx].addAffine(base.neg());
            }
        }

        // Running sum reduction
        var running_sum = G2Projective.identity();
        var window_sum = G2Projective.identity();

        var bucket_idx: usize = num_buckets;
        while (bucket_idx > 0) {
            bucket_idx -= 1;
            running_sum = running_sum.add(buckets[bucket_idx]);
            window_sum = window_sum.add(running_sum);
        }

        final_result = final_result.add(window_sum);
    }

    return final_result.toAffine();
}

/// Pippenger MSM for G2 with parallel window processing
fn pippengerMsmG2Parallel(comptime F: type, bases: []const G2Point, scalars: []const F, tp: *ThreadPool) G2Point {
    const SCALAR_BITS: usize = 256;
    const MAX_DIGITS: usize = 87;

    const c = g2OptimalWindowSize(bases.len);
    const num_scalar_windows = (SCALAR_BITS + c - 1) / c;
    const num_windows = num_scalar_windows + 1;
    const num_buckets = (@as(usize, 1) << @as(u6, @intCast(c))) / 2;

    // Compute wNAF digits
    const heap_digits = std.heap.page_allocator.alloc([MAX_DIGITS]i32, scalars.len) catch
        return pippengerMsmG2(F, bases, scalars);
    defer std.heap.page_allocator.free(heap_digits);

    for (scalars, 0..) |s, i| {
        heap_digits[i] = makeG2Digits(s.fromMontgomery().limbs, c, num_scalar_windows);
    }

    // Allocate per-window buckets and window sums
    const all_buckets = std.heap.page_allocator.alloc(G2Projective, num_windows * num_buckets) catch
        return pippengerMsmG2(F, bases, scalars);
    defer std.heap.page_allocator.free(all_buckets);

    const window_sums = std.heap.page_allocator.alloc(G2Projective, num_windows) catch
        return pippengerMsmG2(F, bases, scalars);
    defer std.heap.page_allocator.free(window_sums);

    // Phase 1: Process all windows in parallel
    const ParCtx = struct {
        all_digits: [][MAX_DIGITS]i32,
        all_buckets: []G2Projective,
        window_sums: []G2Projective,
        bases: []const G2Point,
        num_buckets: usize,
    };
    const ctx = ParCtx{
        .all_digits = heap_digits,
        .all_buckets = all_buckets,
        .window_sums = window_sums,
        .bases = bases,
        .num_buckets = num_buckets,
    };

    tp.parallelForForce(num_windows, ctx, struct {
        fn f(cx: ParCtx, win_idx: usize) void {
            const bucket_offset = win_idx * cx.num_buckets;
            const buckets = cx.all_buckets[bucket_offset .. bucket_offset + cx.num_buckets];

            for (0..cx.num_buckets) |j| {
                buckets[j] = G2Projective.identity();
            }

            for (cx.bases, 0..) |base, idx| {
                if (base.infinity) continue;
                const digit = cx.all_digits[idx][win_idx];
                if (digit > 0) {
                    const bidx: usize = @intCast(digit - 1);
                    buckets[bidx] = buckets[bidx].addAffine(base);
                } else if (digit < 0) {
                    const bidx: usize = @intCast(-digit - 1);
                    buckets[bidx] = buckets[bidx].addAffine(base.neg());
                }
            }

            var running_sum = G2Projective.identity();
            var window_sum = G2Projective.identity();
            var bucket_idx: usize = cx.num_buckets;
            while (bucket_idx > 0) {
                bucket_idx -= 1;
                running_sum = running_sum.add(buckets[bucket_idx]);
                window_sum = window_sum.add(running_sum);
            }

            cx.window_sums[win_idx] = window_sum;
        }
    }.f);

    // Phase 2: Combine window sums sequentially (high to low)
    var final_result = window_sums[num_windows - 1];
    var window_idx: usize = num_windows - 1;
    while (window_idx > 0) {
        window_idx -= 1;
        var k: usize = 0;
        while (k < c) : (k += 1) {
            final_result = final_result.double();
        }
        final_result = final_result.add(window_sums[window_idx]);
    }

    return final_result.toAffine();
}

/// wNAF digit decomposition for G2 Pippenger
pub fn makeG2Digits(limbs: [4]u64, c: usize, num_digits: usize) [87]i32 {
    const MAX_G2_DIGITS = 87;
    var digits: [MAX_G2_DIGITS]i32 = undefined;
    const radix: u64 = @as(u64, 1) << @as(u6, @intCast(c));
    const window_mask: u64 = radix - 1;
    const half_radix: u64 = radix >> 1;
    var carry: u64 = 0;

    for (0..num_digits) |i| {
        const bit_offset = i * c;
        const u64_idx = bit_offset / 64;
        const bit_idx = @as(u6, @intCast(bit_offset % 64));

        var bit_buf: u64 = if (u64_idx < 4) limbs[u64_idx] >> bit_idx else 0;
        const bit_idx_usize: usize = @as(usize, bit_idx);
        if (bit_idx_usize + c > 64 and u64_idx + 1 < 4) {
            bit_buf |= limbs[u64_idx + 1] << @as(u6, @intCast(64 - bit_idx_usize));
        }

        const coef = carry + (bit_buf & window_mask);
        carry = if (coef >= half_radix) 1 else 0;
        const digit: i32 = @as(i32, @intCast(coef)) - @as(i32, @intCast(carry)) * @as(i32, @intCast(radix));
        digits[i] = digit;
    }
    // Final carry as extra digit
    if (num_digits < MAX_G2_DIGITS) {
        digits[num_digits] = @intCast(carry);
    }
    return digits;
}

/// Optimal window size for G2 Pippenger
pub fn g2OptimalWindowSize(n: usize) usize {
    if (n < 32) return 3;
    if (n < 256) return 5;
    if (n < 2048) return 6;
    if (n < 16384) return 7;
    if (n < 131072) return 8;
    return 9;
}
