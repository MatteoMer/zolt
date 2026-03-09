//! Batch Affine Point Additions for One-Hot Polynomials
//!
//! For one-hot polynomials where each cycle selects exactly one basis point,
//! we can compute G1 sums using batch affine additions with shared batch inversion
//! instead of full MSM. This is significantly faster when the number of nonzero
//! entries per row is small.
//!
//! Reference: jolt-optimizations/src/batch_addition.rs

const std = @import("std");
const Allocator = std.mem.Allocator;
const msm = @import("mod.zig");
const field = @import("../field/mod.zig");

const Fp = field.BN254BaseField;
const G1Point = msm.AffinePoint(Fp);

/// Sum selected basis points using batch affine additions with repeated halving.
/// Returns identity if `indices` is empty.
pub fn batchG1Additions(bases: []const G1Point, indices: []const u16) G1Point {
    if (indices.len == 0) return G1Point.identity();
    if (indices.len == 1) return bases[indices[0]];

    // For small sets, just do naive addition
    var acc = bases[indices[0]];
    for (indices[1..]) |idx| {
        acc = acc.add(bases[idx]);
    }
    return acc;
}

/// Sum K sets of basis points with shared batch inversion per halving round.
///
/// Algorithm (matching Jolt `batch_g1_additions_multi`):
/// 1. Gather working sets from indices
/// 2. Loop while any set has >1 point:
///    - Collect all denominators (p2.x - p1.x) across all sets
///    - Single batchInversion on all denominators
///    - Apply affine chord formula
///    - Carry odd-length set tails forward
/// 3. Return working_sets[k][0] for each k
pub fn batchG1AdditionsMulti(
    bases: []const G1Point,
    index_sets: []const []const u16,
    allocator: Allocator,
) ![]G1Point {
    const k = index_sets.len;
    const results = try allocator.alloc(G1Point, k);

    // Count total points needed for working storage
    var total_points: usize = 0;
    for (index_sets) |indices| {
        total_points += indices.len;
    }

    if (total_points == 0) {
        @memset(results, G1Point.identity());
        return results;
    }

    // Allocate flat working buffer for all sets
    const working_buf = try allocator.alloc(G1Point, total_points);
    defer allocator.free(working_buf);

    // Track each set's offset and current length in the working buffer
    const set_offsets = try allocator.alloc(usize, k);
    defer allocator.free(set_offsets);
    const set_lengths = try allocator.alloc(usize, k);
    defer allocator.free(set_lengths);

    // Populate working buffer
    var offset: usize = 0;
    for (index_sets, 0..) |indices, i| {
        set_offsets[i] = offset;
        set_lengths[i] = indices.len;
        for (indices, 0..) |idx, j| {
            working_buf[offset + j] = bases[idx];
        }
        offset += indices.len;
    }

    // Scratch buffers for batch inversion (max possible denominators = total_points/2)
    const max_denoms = total_points;
    const denoms = try allocator.alloc(Fp, max_denoms);
    defer allocator.free(denoms);
    const scratch = try allocator.alloc(Fp, max_denoms);
    defer allocator.free(scratch);

    // Repeated halving loop
    while (true) {
        // Check if any set still has >1 point
        var any_multi = false;
        for (set_lengths) |len| {
            if (len > 1) {
                any_multi = true;
                break;
            }
        }
        if (!any_multi) break;

        // Collect denominators from all pairs across all sets
        var num_denoms: usize = 0;

        // Build a map from denom index -> (set_index, pair_index) for applying results
        // We need to track which pairs correspond to which denominators
        // Simple approach: iterate sets twice (once for denoms, once to apply)

        // First pass: collect denominators
        for (0..k) |si| {
            const len = set_lengths[si];
            if (len <= 1) continue;
            const base_off = set_offsets[si];
            const num_pairs = len / 2;
            for (0..num_pairs) |pi| {
                const p1 = working_buf[base_off + 2 * pi];
                const p2 = working_buf[base_off + 2 * pi + 1];
                if (p1.infinity or p2.infinity) {
                    denoms[num_denoms] = Fp.one(); // placeholder, won't be used
                } else if (p1.x.eql(p2.x)) {
                    // Same x: either doubling or cancellation
                    if (p1.y.eql(p2.y)) {
                        // Doubling: denominator is 2*y
                        denoms[num_denoms] = p1.y.add(p1.y);
                    } else {
                        // Points cancel (p1 = -p2)
                        denoms[num_denoms] = Fp.one(); // placeholder
                    }
                } else {
                    denoms[num_denoms] = p2.x.sub(p1.x);
                }
                num_denoms += 1;
            }
        }

        // Batch invert all denominators
        if (num_denoms > 0) {
            Fp.batchInversion(denoms[0..num_denoms], scratch[0..num_denoms]);
        }

        // Second pass: apply chord formula using inverted denominators
        var denom_idx: usize = 0;
        for (0..k) |si| {
            const len = set_lengths[si];
            if (len <= 1) continue;
            const base_off = set_offsets[si];
            const num_pairs = len / 2;
            const has_odd = (len & 1) == 1;

            for (0..num_pairs) |pi| {
                const p1 = working_buf[base_off + 2 * pi];
                const p2 = working_buf[base_off + 2 * pi + 1];
                const inv = denoms[denom_idx];
                denom_idx += 1;

                if (p1.infinity) {
                    working_buf[base_off + pi] = p2;
                } else if (p2.infinity) {
                    working_buf[base_off + pi] = p1;
                } else if (p1.x.eql(p2.x)) {
                    if (p1.y.eql(p2.y)) {
                        // Double: lambda = 3x^2 / 2y
                        if (p1.y.isZero()) {
                            working_buf[base_off + pi] = G1Point.identity();
                        } else {
                            const x_sq = p1.x.square();
                            const lambda = x_sq.add(x_sq).add(x_sq).mul(inv);
                            const x3 = lambda.square().sub(p1.x).sub(p2.x);
                            const y3 = lambda.mul(p1.x.sub(x3)).sub(p1.y);
                            working_buf[base_off + pi] = G1Point.fromCoords(x3, y3);
                        }
                    } else {
                        // Cancel
                        working_buf[base_off + pi] = G1Point.identity();
                    }
                } else {
                    // Standard chord: lambda = (y2-y1) * inv
                    const lambda = p2.y.sub(p1.y).mul(inv);
                    const x3 = lambda.square().sub(p1.x).sub(p2.x);
                    const y3 = lambda.mul(p1.x.sub(x3)).sub(p1.y);
                    working_buf[base_off + pi] = G1Point.fromCoords(x3, y3);
                }
            }

            // Carry odd tail forward
            if (has_odd) {
                working_buf[base_off + num_pairs] = working_buf[base_off + len - 1];
            }

            // Update length: ceil(len/2)
            set_lengths[si] = num_pairs + @as(usize, if (has_odd) 1 else 0);
        }
    }

    // Extract results
    for (0..k) |si| {
        if (set_lengths[si] == 0) {
            results[si] = G1Point.identity();
        } else {
            results[si] = working_buf[set_offsets[si]];
        }
    }

    return results;
}
