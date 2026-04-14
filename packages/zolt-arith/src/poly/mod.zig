//! Polynomial operations for Jolt
//!
//! This module provides polynomial types and operations used in the Jolt proof system,
//! including multilinear polynomials, equality polynomials, and commitment schemes.

const std = @import("std");

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;
const field = @import("../field/mod.zig");

pub const commitment = @import("commitment/mod.zig");
pub const split_eq = @import("split_eq.zig");
pub const multiquadratic = @import("multiquadratic.zig");
pub const lt_poly = @import("lt_poly.zig");
pub const interpolation = @import("interpolation.zig");
pub const product_tree = @import("product_tree.zig");

// Re-export commonly used types
pub const GruenSplitEqPolynomial = split_eq.GruenSplitEqPolynomial;
pub const MultiquadraticPolynomial = multiquadratic.MultiquadraticPolynomial;
pub const GridValue = multiquadratic.GridValue;

/// Dense multilinear polynomial representation
///
/// A multilinear polynomial over n variables is stored as 2^n evaluations
/// at the boolean hypercube points.
pub fn DensePolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Evaluations at boolean hypercube points
        evaluations: []F,
        /// Number of variables
        num_vars: usize,
        /// Allocator used for this polynomial
        allocator: Allocator,
        /// Pre-allocated scratch buffer for parallel bind (double-buffer technique)
        scratch: ?[]F = null,

        /// Create a new polynomial from evaluations
        pub fn init(allocator: Allocator, evaluations: []const F) !Self {
            const num_vars = std.math.log2_int(usize, evaluations.len);
            std.debug.assert(evaluations.len == (@as(usize, 1) << num_vars));

            const evals = try allocator.alloc(F, evaluations.len);
            @memcpy(evals, evaluations);

            return .{
                .evaluations = evals,
                .num_vars = num_vars,
                .allocator = allocator,
            };
        }

        /// Create a polynomial that takes ownership of an existing evaluations slice.
        /// The caller must not free the slice — it will be freed by deinit().
        pub fn initOwned(allocator: Allocator, evaluations: []F) Self {
            const num_vars = if (evaluations.len <= 1) 0 else std.math.log2_int(usize, evaluations.len);
            return .{
                .evaluations = evaluations,
                .num_vars = num_vars,
                .allocator = allocator,
            };
        }

        /// Create a zero polynomial with n variables
        pub fn zero(allocator: Allocator, num_vars: usize) !Self {
            const size = @as(usize, 1) << num_vars;
            const evals = try allocator.alloc(F, size);
            @memset(evals, F.zero());

            return .{
                .evaluations = evals,
                .num_vars = num_vars,
                .allocator = allocator,
            };
        }

        /// Free the polynomial
        pub fn deinit(self: *Self) void {
            if (self.scratch) |s| self.allocator.free(s);
            self.allocator.free(self.evaluations);
        }

        /// Get the number of evaluations (2^num_vars)
        pub fn len(self: *const Self) usize {
            return self.evaluations.len;
        }

        /// Evaluate the polynomial at a point
        pub fn evaluate(self: *const Self, point: []const F) F {
            std.debug.assert(point.len == self.num_vars);

            // Use the multilinear extension formula
            var result = F.zero();
            for (self.evaluations, 0..) |eval, i| {
                var term = eval;
                for (0..self.num_vars) |j| {
                    const shift_amount: std.math.Log2Int(usize) = @intCast(j);
                    const bit = (i >> shift_amount) & 1;
                    if (bit == 1) {
                        term = term.mul(point[j]);
                    } else {
                        term = term.mul(F.one().sub(point[j]));
                    }
                }
                result = result.add(term);
            }
            return result;
        }

        /// Add two polynomials
        pub fn add(self: *const Self, other: *const Self) !Self {
            std.debug.assert(self.num_vars == other.num_vars);
            const evals = try self.allocator.alloc(F, self.evaluations.len);

            for (0..self.evaluations.len) |i| {
                evals[i] = self.evaluations[i].add(other.evaluations[i]);
            }

            return .{
                .evaluations = evals,
                .num_vars = self.num_vars,
                .allocator = self.allocator,
            };
        }

        /// Multiply by a scalar
        pub fn scale(self: *const Self, scalar: F) !Self {
            const evals = try self.allocator.alloc(F, self.evaluations.len);

            for (0..self.evaluations.len) |i| {
                evals[i] = self.evaluations[i].mul(scalar);
            }

            return .{
                .evaluations = evals,
                .num_vars = self.num_vars,
                .allocator = self.allocator,
            };
        }

        /// Bind the first (highest index) variable to a value
        /// This reduces the polynomial from n to n-1 variables
        /// Uses HighToLow binding order (top variable bound first)
        pub fn bindFirst(self: *const Self, value: F) !Self {
            std.debug.assert(self.num_vars > 0);

            const new_size = self.boundLen() / 2;
            const evals = try self.allocator.alloc(F, new_size);

            const one_minus_value = F.one().sub(value);

            for (0..new_size) |i| {
                // f(x_1, ..., x_n) evaluated at x_n = value (top variable)
                // = (1 - value) * f(x_1, ..., 0) + value * f(x_1, ..., 1)
                const low = self.evaluations[i].mul(one_minus_value);
                const high = self.evaluations[i + new_size].mul(value);
                evals[i] = low.add(high);
            }

            return .{
                .evaluations = evals,
                .num_vars = self.num_vars - 1,
                .allocator = self.allocator,
            };
        }

        /// Bind the last (lowest index) variable to a value - in-place
        /// This reduces the polynomial from n to n-1 variables
        /// Uses LowToHigh binding order (bottom variable bound first)
        ///
        /// Matches Jolt's bound_poly_var_bot:
        ///   Z[i] = Z[2*i] + r * (Z[2*i+1] - Z[2*i])
        ///
        /// This is the correct binding order for the outer streaming sumcheck
        /// linear phase where we bind cycle variables from low to high.
        pub fn bindLow(self: *Self, value: F) void {
            std.debug.assert(self.num_vars > 0);

            // Use boundLen() not evaluations.len — after prior binds, evaluations.len
            // stays at the original allocated size while the valid data shrinks.
            const new_size = self.boundLen() / 2;

            for (0..new_size) |i| {
                // f(x_0, x_1, ..., x_{n-1}) evaluated at x_0 = value (bottom variable)
                // new[i] = old[2*i] + r * (old[2*i+1] - old[2*i])
                // = (1 - r) * old[2*i] + r * old[2*i+1]
                const low = self.evaluations[2 * i];
                const high = self.evaluations[2 * i + 1];
                self.evaluations[i] = low.add(value.mul(high.sub(low)));
            }

            self.num_vars -= 1;
        }

        /// Bind the last (lowest index) variable to a value - parallel double-buffer
        /// Uses scratch buffer to avoid data races: reads from evaluations, writes to scratch,
        /// then swaps pointers. Requires scratch buffer to be pre-allocated.
        pub fn bindLowParallel(self: *Self, value: F, tp: *@import("zolt_pool").ThreadPool) void {
            const new_size = self.boundLen() / 2;
            const scratch = self.scratch orelse {
                self.bindLow(value); // fallback to sequential
                return;
            };

            const Ctx = struct { src: []const F, dst: []F, r: F };
            tp.parallelForForce(
                new_size,
                Ctx{ .src = self.evaluations, .dst = scratch, .r = value },
                struct {
                    fn f(ctx: Ctx, i: usize) void {
                        const low = ctx.src[2 * i];
                        const high = ctx.src[2 * i + 1];
                        ctx.dst[i] = low.add(ctx.r.mul(high.sub(low)));
                    }
                }.f,
            );

            // Swap evaluations and scratch (pointer swap, O(1))
            const tmp = self.evaluations;
            self.evaluations = scratch;
            self.scratch = tmp;
            self.num_vars -= 1;
        }

        /// Get the length (number of evaluations after binding)
        pub fn boundLen(self: *const Self) usize {
            return @as(usize, 1) << @intCast(self.num_vars);
        }
    };
}

/// Equality polynomial eq(x, r)
///
/// eq(x, r) = prod_{i=1}^{n} (x_i * r_i + (1 - x_i) * (1 - r_i))
///
/// This polynomial evaluates to 1 when x = r and 0 when x and r differ
/// in any coordinate on the boolean hypercube.
pub fn EqPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The point r that defines the equality polynomial
        r: []F,
        allocator: Allocator,

        /// Create an equality polynomial for point r
        pub fn init(allocator: Allocator, r: []const F) !Self {
            const r_copy = try allocator.alloc(F, r.len);
            @memcpy(r_copy, r);

            return .{
                .r = r_copy,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.r);
        }

        /// Evaluate eq(x, r) at a point x
        pub fn evaluate(self: *const Self, x: []const F) F {
            std.debug.assert(x.len == self.r.len);

            var result = F.one();
            for (0..self.r.len) |i| {
                // x_i * r_i + (1 - x_i) * (1 - r_i)
                const xi_ri = x[i].mul(self.r[i]);
                const one_minus_xi = F.one().sub(x[i]);
                const one_minus_ri = F.one().sub(self.r[i]);
                const term = xi_ri.add(one_minus_xi.mul(one_minus_ri));
                result = result.mul(term);
            }
            return result;
        }

        /// Compute all evaluations on the boolean hypercube
        ///
        /// Uses BIG-ENDIAN indexing to match Jolt's EqPolynomial::evals():
        /// - result[i] = eq(r, x) where x is the binary representation of i
        /// - bit 0 (MSB of i) corresponds to r[0]
        /// - bit n-1 (LSB of i) corresponds to r[n-1]
        ///
        /// This matches Jolt's convention where for r = [r0, r1, ..., r_{n-1}]:
        /// - index 0 → eq(r, [0, 0, ..., 0])
        /// - index 1 → eq(r, [0, 0, ..., 1])  (LSB = 1, so last var x_{n-1} = 1)
        /// - etc.
        pub fn evals(self: *const Self, allocator: Allocator) ![]F {
            return evalsSliceWithScaling(F, allocator, self.r, null);
        }

        /// Compute eq evaluations over a slice with optional scaling factor
        /// This is the core implementation used by both instance and static methods.
        ///
        /// Matches Jolt's EqPolynomial::evals_parallel algorithm:
        /// - Iterates through r in reverse order (big-endian)
        /// - For each step, doubles the active region size
        /// - result[0..size] is the active region
        /// - For each i in active region: result[i+size] = result[i] * r[j], result[i] -= result[i+size]
        pub fn evalsSliceWithScaling(comptime FieldType: type, allocator: Allocator, r: []const FieldType, scaling_factor: ?FieldType) ![]FieldType {
            const n = r.len;
            const final_size = @as(usize, 1) << @as(std.math.Log2Int(usize), @intCast(n));
            const result = try allocator.alloc(FieldType, final_size);
            buildEqTableInPlace(r, result, scaling_factor);
            return result;
        }

        /// Compute eq evaluations with optional parallel inner loops.
        /// Same as evalsSliceWithScaling but parallelizes large levels via ThreadPool.
        pub fn evalsSliceWithScalingParallel(comptime FieldType: type, allocator: Allocator, r: []const FieldType, scaling_factor: ?FieldType, pool: ?*@import("zolt_pool").ThreadPool) ![]FieldType {
            const n = r.len;
            if (n == 0) {
                const result = try allocator.alloc(FieldType, 1);
                result[0] = scaling_factor orelse FieldType.one();
                return result;
            }

            const final_size = @as(usize, 1) << @as(std.math.Log2Int(usize), @intCast(n));
            const result = try allocator.alloc(FieldType, final_size);

            // No memset needed: every entry is written before read by the expansion loop below.
            result[0] = scaling_factor orelse FieldType.one();

            // BE convention: iterate r in reverse
            var size: usize = 1;
            var j: usize = n;
            while (j > 0) {
                j -= 1;
                const r_j = r[j];

                if (pool != null and size >= 256) {
                    const TP = @import("zolt_pool").ThreadPool;
                    const Ctx = struct {
                        res: []FieldType,
                        rj: FieldType,
                        sz: usize,
                    };
                    const ctx = Ctx{ .res = result, .rj = r_j, .sz = size };
                    const p: *TP = pool.?;
                    p.parallelForForce(size, ctx, struct {
                        fn f(c: Ctx, i: usize) void {
                            const x = c.res[i];
                            const y = x.mul(c.rj);
                            c.res[i + c.sz] = y;
                            c.res[i] = x.sub(y);
                        }
                    }.f);
                } else {
                    for (0..size) |i| {
                        const x = result[i];
                        const y = x.mul(r_j);
                        result[i + size] = y;
                        result[i] = x.sub(y);
                    }
                }

                size *= 2;
            }

            return result;
        }

        /// Build eq table in-place into a pre-allocated output buffer.
        /// out must have length 2^r.len. No temporary allocation.
        /// Supports an optional scaling factor: out[0] starts as scale instead of 1.
        pub fn buildEqTableInPlace(r: []const F, out: []F, scaling_factor: ?F) void {
            const n = r.len;
            std.debug.assert(out.len == @as(usize, 1) << @intCast(n));
            // No memset needed: the loop writes every entry before reading it.
            // At each level, we read out[0..level_size-1] (already written) and
            // write out[0..2*level_size-1]. After n levels all 2^n entries are set.
            out[0] = scaling_factor orelse F.one();
            var level_size: usize = 1;
            var j: usize = n;
            while (j > 0) {
                j -= 1;
                const r_j = r[j];
                for (0..level_size) |i| {
                    const x = out[i];
                    const y = x.mul(r_j);
                    out[i + level_size] = y;
                    out[i] = x.sub(y);
                }
                level_size *= 2;
            }
        }

        /// Build eq+1 table in-place: out[j] = eq(r, j-1) with out[0] = 0.
        /// Uses the identity eq+1(r, j) = eq(r, j-1).
        /// out must have length 2^r.len. No temporary allocation.
        pub fn buildEqPlusOneTableInPlace(r: []const F, out: []F) void {
            const n = r.len;
            const size = out.len;
            std.debug.assert(size == @as(usize, 1) << @intCast(n));
            // eq_plus_one[j] = eq(r, j-1) for j > 0, eq_plus_one[0] = 0.
            // This implements: Σ_j eq_plus_one[j] * f(j) = Σ_j eq(r, j-1) * f(j)
            //                = Σ_{j'} eq(r, j') * f(j'+1)
            // Which is the "shift forward" used in the shift sumcheck.
            // Build eq table in-place, then shift RIGHT by 1.
            buildEqTableInPlace(r, out, null);
            var k: usize = size - 1;
            while (k > 0) : (k -= 1) {
                out[k] = out[k - 1];
            }
            out[0] = F.zero();
        }

        /// Build both eq and eq+1 tables in-place. No temporary allocation.
        /// eq_out[j] = eq(r, j), eq_plus_one_out[j] = eq(r, j-1) with [0] = 0.
        pub fn buildEqAndEqPlusOneInPlace(r: []const F, eq_out: []F, eq_plus_one_out: []F) void {
            const size = eq_out.len;
            std.debug.assert(size == @as(usize, 1) << @intCast(r.len));
            std.debug.assert(eq_plus_one_out.len == size);
            buildEqTableInPlace(r, eq_out, null);
            eq_plus_one_out[0] = F.zero();
            @memcpy(eq_plus_one_out[1..], eq_out[0 .. size - 1]);
        }

        /// Bind the first variable and reduce polynomial size in-place
        /// After binding n variables, evals() will return 2^(original_len - n) values
        pub fn bind(self: *Self, value: F) void {
            if (self.r.len == 0) return;

            const new_len = self.r.len - 1;
            // Shift r values down (bind first variable)
            for (0..new_len) |i| {
                // Interpolate: new_r[i] = (1 - value) * r[i] + value * r[i+1]
                // Actually for eq polynomial binding, we just drop the first r
                // But we need to update the "current" evaluation
                self.r[i] = self.r[i + 1];
            }
            // We can't actually resize, so we track this differently
            // For now, leave r unchanged but the caller should track bound count
            _ = value;
        }

        /// Evaluate eq(x, r) using the MLE formula (static version)
        pub fn mle(r: []const F, x: []const F) F {
            std.debug.assert(r.len == x.len);
            var result = F.one();
            for (0..r.len) |i| {
                const ri_xi = r[i].mul(x[i]);
                const one_minus_ri = F.one().sub(r[i]);
                const one_minus_xi = F.one().sub(x[i]);
                result = result.mul(ri_xi.add(one_minus_ri.mul(one_minus_xi)));
            }
            return result;
        }
    };
}

/// EqPlusOne polynomial eq+1(x, y)
///
/// This MLE evaluates to 1 when y = x + 1 (binary increment).
/// For x in the range [0, 2^l - 2], eq+1(x, y) = 1 iff y = x + 1.
/// When x is all 1s, the result is 0 (there's no successor in the range).
///
/// Used in Jolt's ShiftSumcheck to relate values at consecutive indices.
pub fn EqPlusOnePolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The point x that defines eq+1(x, ·)
        x: []F,
        allocator: Allocator,

        /// Create an EqPlusOne polynomial for point x
        /// x is assumed to be in BIG_ENDIAN order (MSB first)
        pub fn init(allocator: Allocator, x: []const F) !Self {
            const x_copy = try allocator.alloc(F, x.len);
            @memcpy(x_copy, x);

            return .{
                .x = x_copy,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.x);
        }

        /// Evaluate eq+1(x, y) at point y
        /// Both x and y are in BIG_ENDIAN order (x[0] is MSB)
        ///
        /// eq+1(x, y) = 1 iff y = x + 1 in binary.
        /// If x + 1 = y, then:
        ///   - Let k be the index of the rightmost 0 bit in x (counting from LSB)
        ///   - In y, all bits below k are 0 (from 1s in x that carry over)
        ///   - Bit k in y is 1 (from the carry)
        ///   - Bits above k are the same in x and y
        pub fn evaluate(self: *const Self, y: []const F) F {
            const l = self.x.len;
            std.debug.assert(y.len == l);

            var result = F.zero();

            // Sum over all possible positions k for the "flip" bit
            // k is the suffix length of 1s in x (0-indexed from LSB)
            for (0..l) |k| {
                // lower_bits_product: bits 0..k-1 (from LSB) are 1 in x, 0 in y
                // In BIG_ENDIAN: these are indices l-1, l-2, ..., l-k
                var lower_bits_product = F.one();
                for (0..k) |i| {
                    const idx = l - 1 - i;
                    // x[idx] should be 1, y[idx] should be 0
                    lower_bits_product = lower_bits_product.mul(self.x[idx].mul(F.one().sub(y[idx])));
                }

                // kth_bit_product: bit k is 0 in x, 1 in y
                // In BIG_ENDIAN: index l-1-k
                const kth_idx = l - 1 - k;
                const kth_bit_product = F.one().sub(self.x[kth_idx]).mul(y[kth_idx]);

                // higher_bits_product: bits k+1..l-1 are the same in x and y
                // In BIG_ENDIAN: indices 0..l-2-k
                var higher_bits_product = F.one();
                for ((k + 1)..l) |i| {
                    const idx = l - 1 - i;
                    // x[idx] == y[idx] means: x*y + (1-x)*(1-y)
                    const xi_yi = self.x[idx].mul(y[idx]);
                    const one_minus_xi = F.one().sub(self.x[idx]);
                    const one_minus_yi = F.one().sub(y[idx]);
                    higher_bits_product = higher_bits_product.mul(xi_yi.add(one_minus_xi.mul(one_minus_yi)));
                }

                result = result.add(lower_bits_product.mul(kth_bit_product).mul(higher_bits_product));
            }

            return result;
        }

        /// Compute eq+1(x, y) directly (static version)
        pub fn mle(x: []const F, y: []const F) F {
            const l = x.len;
            std.debug.assert(y.len == l);

            var result = F.zero();

            for (0..l) |k| {
                var lower_bits_product = F.one();
                for (0..k) |i| {
                    const idx = l - 1 - i;
                    lower_bits_product = lower_bits_product.mul(x[idx].mul(F.one().sub(y[idx])));
                }

                const kth_idx = l - 1 - k;
                const kth_bit_product = F.one().sub(x[kth_idx]).mul(y[kth_idx]);

                var higher_bits_product = F.one();
                for ((k + 1)..l) |i| {
                    const idx = l - 1 - i;
                    const xi_yi = x[idx].mul(y[idx]);
                    const one_minus_xi = F.one().sub(x[idx]);
                    const one_minus_yi = F.one().sub(y[idx]);
                    higher_bits_product = higher_bits_product.mul(xi_yi.add(one_minus_xi.mul(one_minus_yi)));
                }

                result = result.add(lower_bits_product.mul(kth_bit_product).mul(higher_bits_product));
            }

            return result;
        }

        /// Bind the first variable (MSB) to a value, reducing the polynomial
        /// This is used in sumcheck rounds
        pub fn bind(self: *Self, value: F) void {
            _ = self;
            _ = value;
            // For eq+1, binding is complex. For now, just reduce size tracking.
            // A proper implementation would update internal state.
            // This is a placeholder - the actual computation happens in evaluate().
        }
    };
}

/// Prefix-suffix decomposition of eq+1 polynomial for sumcheck optimization.
///
/// Decomposes eq+1((r_hi, r_lo), (y_hi, y_lo)) as:
///   prefix_0(r_lo, y_lo) * suffix_0(r_hi, y_hi) +
///   prefix_1(r_lo, y_lo) * suffix_1(r_hi, y_hi)
///
/// This allows efficient sumcheck computation by operating on smaller polynomials.
/// The decomposition is correct because:
/// - prefix_0 = eq+1(r_lo, j) for all j
/// - suffix_0 = eq(r_hi, j) for all j
/// - prefix_1 = is_max(r_lo) * delta(j=0) (sparse: only non-zero at index 0)
/// - suffix_1 = eq+1(r_hi, j) for all j
///
/// where is_max(r_lo) = eq((1,1,...,1), r_lo)
pub fn EqPlusOnePrefixSuffixPoly(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Evals of eq+1(r_lo, j) for all j in {0, 1, ..., 2^(n/2)-1}
        prefix_0: []F,
        /// Evals of eq(r_hi, j) for all j in {0, 1, ..., 2^(n/2)-1}
        suffix_0: []F,
        /// Evals of is_max(r_lo) * delta(j=0) - only index 0 is non-zero
        prefix_1: []F,
        /// Evals of eq+1(r_hi, j) for all j
        suffix_1: []F,

        allocator: Allocator,

        /// Create prefix-suffix decomposition for point r (in BIG_ENDIAN order)
        /// r is split at the midpoint: r = (r_hi || r_lo)
        pub fn init(allocator: Allocator, r: []const F) !Self {
            std.debug.assert(r.len >= 2);
            const mid = r.len / 2;
            const r_hi = r[0..mid];
            const r_lo = r[mid..];

            const n_lo = r_lo.len;
            const n_hi = r_hi.len;
            const size_lo: usize = @as(usize, 1) << @intCast(n_lo);
            const size_hi: usize = @as(usize, 1) << @intCast(n_hi);

            // Allocate buffers
            const prefix_0 = try allocator.alloc(F, size_lo);
            const suffix_0 = try allocator.alloc(F, size_hi);
            const prefix_1 = try allocator.alloc(F, size_lo);
            const suffix_1 = try allocator.alloc(F, size_hi);

            // Compute is_max(r_lo) = eq((1,1,...,1), r_lo)
            var is_max_eval = F.one();
            for (0..n_lo) |i| {
                // eq((1), r_lo[i]) = r_lo[i] * 1 + (1-r_lo[i]) * 0 = r_lo[i]
                is_max_eval = is_max_eval.mul(r_lo[i]);
            }

            // Initialize prefix_1: only non-zero at index 0
            @memset(prefix_1, F.zero());
            prefix_1[0] = is_max_eval;

            // Compute eq+1(r_lo, j) and eq+1(r_hi, j) for all j
            // Also compute eq(r_hi, j) = suffix_0
            EqPolynomial(F).buildEqPlusOneTableInPlace(r_lo, prefix_0);
            EqPolynomial(F).buildEqAndEqPlusOneInPlace(r_hi, suffix_0, suffix_1);

            return Self{
                .prefix_0 = prefix_0,
                .suffix_0 = suffix_0,
                .prefix_1 = prefix_1,
                .suffix_1 = suffix_1,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.prefix_0);
            self.allocator.free(self.suffix_0);
            self.allocator.free(self.prefix_1);
            self.allocator.free(self.suffix_1);
        }

        /// Get the size of the prefix arrays
        pub fn prefixSize(self: *const Self) usize {
            return self.prefix_0.len;
        }

        /// Get the size of the suffix arrays
        pub fn suffixSize(self: *const Self) usize {
            return self.suffix_0.len;
        }
    };
}

/// Univariate polynomial (used in sumcheck)
pub fn UniPoly(comptime F: type) type {
    // Delegate interpolation and product-tree methods to separate modules.
    const Interp = interpolation.Interpolation(F);
    const PTree = product_tree.ProductTree(F);

    return struct {
        const Self = @This();

        /// Precomputed field constants to avoid redundant inverse() calls per round.
        /// Each inverse costs ~256 Montgomery muls via Fermat's little theorem.
        pub const INV2: F = Interp.INV2;
        pub const INV6: F = Interp.INV6;

        /// Coefficients in monomial basis (constant term first)
        coeffs: []F,
        allocator: Allocator,

        /// Create from coefficients
        pub fn init(allocator_arg: Allocator, coeffs_arg: []const F) !Self {
            const c = try allocator_arg.alloc(F, coeffs_arg.len);
            @memcpy(c, coeffs_arg);

            return .{
                .coeffs = c,
                .allocator = allocator_arg,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.coeffs);
        }

        /// Evaluate at a point using Horner's method
        pub fn evaluate(self: *const Self, x: F) F {
            if (self.coeffs.len == 0) return F.zero();

            var result = self.coeffs[self.coeffs.len - 1];
            var i = self.coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(self.coeffs[i]);
            }
            return result;
        }

        /// Get the degree
        pub fn degree(self: *const Self) usize {
            if (self.coeffs.len == 0) return 0;
            return self.coeffs.len - 1;
        }

        // =============================================================
        // Interpolation delegations (see poly/interpolation.zig)
        // =============================================================

        pub const interpolateDegree3 = Interp.interpolateDegree3;
        pub const evalsToCompressed = Interp.evalsToCompressed;
        pub const toomCookToCoeffs = Interp.toomCookToCoeffs;
        pub const toomCookToCompressed = Interp.toomCookToCompressed;
        pub const evaluateToomCookAt = Interp.evaluateToomCookAt;
        pub const fromEvalsAndHint = Interp.fromEvalsAndHint;
        pub const fromEvalsToom = Interp.fromEvalsToom;
        pub const toomCookToCompressedGeneral = Interp.toomCookToCompressedGeneral;
        pub const evaluateToomCookGeneralAt = Interp.evaluateToomCookGeneralAt;
        pub const fromEvalsVandermonde = Interp.fromEvalsVandermonde;
        pub const vandermondeToCompressed = Interp.vandermondeToCompressed;
        pub const evaluateVandermondeAt = Interp.evaluateVandermondeAt;
        pub const evalFromHint = Interp.evalFromHint;
        pub const evalFromHintGeneral = Interp.evalFromHintGeneral;
        pub const evalFromEvalsDeg2 = Interp.evalFromEvalsDeg2;
        pub const evalFromEvalsDeg3 = Interp.evalFromEvalsDeg3;
        pub const evalFromEvalsGeneral = Interp.evalFromEvalsGeneral;

        // =============================================================
        // Product-tree delegations (see poly/product_tree.zig)
        // =============================================================

        pub const evalLinearProd9 = PTree.evalLinearProd9;
        pub const evalProd9Accumulate = PTree.evalProd9Accumulate;
        pub const evalLinearProd10 = PTree.evalLinearProd10;
        pub const finishMlesProductSumFromEvals = PTree.finishMlesProductSumFromEvals;
        pub const toCompressed = PTree.toCompressed;
    };
}

test "EqPolynomial partition of unity" {
    const F = field.BN254Scalar;
    const testing = std.testing;

    // Test 1: with r = [1/2, 1/3] (simple values)
    {
        const half = F.one().mul(UniPoly(F).INV2);
        const third = F.one().mul(F.fromU64(3).inverse().?);
        const r = [_]F{ half, third };

        var eq_poly = try EqPolynomial(F).init(testing.allocator, &r);
        defer eq_poly.deinit();

        const evals = try eq_poly.evals(testing.allocator);
        defer testing.allocator.free(evals);

        // Sum should equal 1
        var sum = F.zero();
        for (evals) |ev| {
            sum = sum.add(ev);
        }

        // Check partition of unity
        try testing.expect(sum.eql(F.one()));
    }

    // Test 2: with larger r values (like in the inner_sum_prod test)
    {
        const r = [_]F{ F.fromU64(5555), F.fromU64(6666) };

        var eq_poly = try EqPolynomial(F).init(testing.allocator, &r);
        defer eq_poly.deinit();

        const evals = try eq_poly.evals(testing.allocator);
        defer testing.allocator.free(evals);

        // Sum should equal 1
        var sum = F.zero();
        for (evals) |ev| {
            sum = sum.add(ev);
        }

        // FORCED Print for debugging
        dbg("\n\n=== PARTITION TEST ===\n", .{});
        dbg("r[0] (5555 as Montgomery):\n", .{});
        for (r[0].limbs) |limb| dbg("  {x:016}\n", .{limb});
        dbg("r[1] (6666 as Montgomery):\n", .{});
        for (r[1].limbs) |limb| dbg("  {x:016}\n", .{limb});
        dbg("Eq evals:\n", .{});
        for (evals, 0..) |ev, i| {
            dbg("  [{d}]: ", .{i});
            for (ev.limbs) |limb| dbg("{x:016} ", .{limb});
            dbg("\n", .{});
        }
        dbg("Sum:\n  ", .{});
        for (sum.limbs) |limb| dbg("{x:016} ", .{limb});
        dbg("\nOne:\n  ", .{});
        for (F.one().limbs) |limb| dbg("{x:016} ", .{limb});
        dbg("\nSum == One? {}\n", .{sum.eql(F.one())});
        dbg("=== END PARTITION TEST ===\n\n", .{});

        // Check partition of unity
        try testing.expect(sum.eql(F.one()));
    }
}

test "dense polynomial basic" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create a simple polynomial with 2 variables (4 evaluations)
    const evals = [_]F{
        F.fromU64(1), // f(0,0)
        F.fromU64(2), // f(1,0)
        F.fromU64(3), // f(0,1)
        F.fromU64(4), // f(1,1)
    };

    var poly = try DensePolynomial(F).init(allocator, &evals);
    defer poly.deinit();

    try std.testing.expectEqual(@as(usize, 2), poly.num_vars);
    try std.testing.expectEqual(@as(usize, 4), poly.len());
}

test "dense polynomial bindLow" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create a polynomial with 2 variables (4 evaluations)
    // Layout with x_0 as LSB: evals[i] = f(b_0, b_1) where i = b_0 + 2*b_1
    // So:
    //   evals[0] = f(0,0) = 1
    //   evals[1] = f(1,0) = 2
    //   evals[2] = f(0,1) = 3
    //   evals[3] = f(1,1) = 4
    const evals = [_]F{
        F.fromU64(1), // f(0,0)
        F.fromU64(2), // f(1,0)
        F.fromU64(3), // f(0,1)
        F.fromU64(4), // f(1,1)
    };

    var poly = try DensePolynomial(F).init(allocator, &evals);
    defer poly.deinit();

    // Bind x_0 (the low variable) to r = 3
    const r = F.fromU64(3);
    poly.bindLow(r);

    // After binding x_0 = r:
    //   new[0] = f(r, 0) = (1-r)*f(0,0) + r*f(1,0) = (1-3)*1 + 3*2 = -2 + 6 = 4
    //   new[1] = f(r, 1) = (1-r)*f(0,1) + r*f(1,1) = (1-3)*3 + 3*4 = -6 + 12 = 6
    // But using Jolt's formula: new[i] = old[2i] + r*(old[2i+1] - old[2i])
    //   new[0] = 1 + 3*(2 - 1) = 1 + 3 = 4 ✓
    //   new[1] = 3 + 3*(4 - 3) = 3 + 3 = 6 ✓

    try std.testing.expectEqual(@as(usize, 1), poly.num_vars);
    try std.testing.expectEqual(@as(usize, 2), poly.boundLen());
    try std.testing.expect(poly.evaluations[0].eql(F.fromU64(4)));
    try std.testing.expect(poly.evaluations[1].eql(F.fromU64(6)));
}

test "dense polynomial bindLow matches Jolt's bound_poly_var_bot" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // 3 variables: 8 evaluations
    const evals = [_]F{
        F.fromU64(10), // f(0,0,0)
        F.fromU64(20), // f(1,0,0)
        F.fromU64(30), // f(0,1,0)
        F.fromU64(40), // f(1,1,0)
        F.fromU64(50), // f(0,0,1)
        F.fromU64(60), // f(1,0,1)
        F.fromU64(70), // f(0,1,1)
        F.fromU64(80), // f(1,1,1)
    };

    var poly = try DensePolynomial(F).init(allocator, &evals);
    defer poly.deinit();

    // Bind x_0 to r = 5
    const r = F.fromU64(5);
    poly.bindLow(r);

    // Expected: new[i] = old[2i] + r*(old[2i+1] - old[2i])
    // new[0] = 10 + 5*(20-10) = 10 + 50 = 60
    // new[1] = 30 + 5*(40-30) = 30 + 50 = 80
    // new[2] = 50 + 5*(60-50) = 50 + 50 = 100
    // new[3] = 70 + 5*(80-70) = 70 + 50 = 120

    try std.testing.expectEqual(@as(usize, 2), poly.num_vars);
    try std.testing.expect(poly.evaluations[0].eql(F.fromU64(60)));
    try std.testing.expect(poly.evaluations[1].eql(F.fromU64(80)));
    try std.testing.expect(poly.evaluations[2].eql(F.fromU64(100)));
    try std.testing.expect(poly.evaluations[3].eql(F.fromU64(120)));
}

test "EqPlusOnePolynomial basic" {
    const F = field.BN254Scalar;

    // Test with 2-bit values
    // eq+1(x, y) = 1 iff y = x + 1
    // x = [0, 0] (binary 0) => y should be [0, 1] (binary 1)
    // x = [0, 1] (binary 1) => y should be [1, 0] (binary 2)
    // x = [1, 0] (binary 2) => y should be [1, 1] (binary 3)
    // x = [1, 1] (binary 3) => no valid y in range (returns 0)

    const zero = F.zero();
    const one = F.one();

    // Test: x = [0, 0] (binary 0), y = [0, 1] (binary 1) should give 1
    {
        const x = [_]F{ zero, zero };
        const y = [_]F{ zero, one };
        const result = EqPlusOnePolynomial(F).mle(&x, &y);
        try std.testing.expect(result.eql(one));
    }

    // Test: x = [0, 1] (binary 1), y = [1, 0] (binary 2) should give 1
    {
        const x = [_]F{ zero, one };
        const y = [_]F{ one, zero };
        const result = EqPlusOnePolynomial(F).mle(&x, &y);
        try std.testing.expect(result.eql(one));
    }

    // Test: x = [1, 0] (binary 2), y = [1, 1] (binary 3) should give 1
    {
        const x = [_]F{ one, zero };
        const y = [_]F{ one, one };
        const result = EqPlusOnePolynomial(F).mle(&x, &y);
        try std.testing.expect(result.eql(one));
    }

    // Test: x = [1, 1] (binary 3), y = anything should give 0 (no successor)
    {
        const x = [_]F{ one, one };
        const y = [_]F{ zero, zero };
        const result = EqPlusOnePolynomial(F).mle(&x, &y);
        try std.testing.expect(result.eql(zero));
    }

    // Test: x = [0, 0], y = [1, 0] should give 0 (not successor)
    {
        const x = [_]F{ zero, zero };
        const y = [_]F{ one, zero };
        const result = EqPlusOnePolynomial(F).mle(&x, &y);
        try std.testing.expect(result.eql(zero));
    }
}

test "EqPolynomial buildEqTableInPlace matches per-point mle" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // n = 0: single entry should be 1
    {
        var out: [1]F = undefined;
        EqPolynomial(F).buildEqTableInPlace(&.{}, &out, null);
        try std.testing.expect(out[0].eql(F.one()));
    }

    // n = 0 with scaling factor
    {
        const scale = F.fromU64(42);
        var out: [1]F = undefined;
        EqPolynomial(F).buildEqTableInPlace(&.{}, &out, scale);
        try std.testing.expect(out[0].eql(scale));
    }

    for (1..13) |n| {
        const size = @as(usize, 1) << @intCast(n);

        const r = try allocator.alloc(F, n);
        defer allocator.free(r);
        for (0..n) |i| r[i] = F.fromU64(@as(u64, i * 7 + 3));

        // Build via in-place builder (no scaling)
        const out = try allocator.alloc(F, size);
        defer allocator.free(out);
        EqPolynomial(F).buildEqTableInPlace(r, out, null);

        // Verify against per-point mle() reference for all 2^n entries
        const j_bits = try allocator.alloc(F, n);
        defer allocator.free(j_bits);
        for (0..size) |j| {
            for (0..n) |k| {
                const bit_pos: std.math.Log2Int(usize) = @intCast(n - 1 - k);
                j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
            }
            const expected = EqPolynomial(F).mle(r, j_bits);
            try std.testing.expect(out[j].eql(expected));
        }

        // Also test with a scaling factor: out[j] == scale * eq(r, j)
        const scale = F.fromU64(17);
        const out_scaled = try allocator.alloc(F, size);
        defer allocator.free(out_scaled);
        EqPolynomial(F).buildEqTableInPlace(r, out_scaled, scale);

        for (0..size) |j| {
            for (0..n) |k| {
                const bit_pos: std.math.Log2Int(usize) = @intCast(n - 1 - k);
                j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
            }
            const expected_scaled = scale.mul(EqPolynomial(F).mle(r, j_bits));
            try std.testing.expect(out_scaled[j].eql(expected_scaled));
        }
    }
}

test "EqPolynomial buildEqPlusOneTableInPlace matches per-point mle" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // n = 0: single entry should be 0 (eq+1 at index 0 is always 0)
    {
        var out: [1]F = undefined;
        EqPolynomial(F).buildEqPlusOneTableInPlace(&.{}, &out);
        try std.testing.expect(out[0].eql(F.zero()));
    }

    for (1..13) |n| {
        const size = @as(usize, 1) << @intCast(n);

        const r = try allocator.alloc(F, n);
        defer allocator.free(r);
        for (0..n) |i| r[i] = F.fromU64(@as(u64, i * 7 + 3));

        const out = try allocator.alloc(F, size);
        defer allocator.free(out);
        EqPolynomial(F).buildEqPlusOneTableInPlace(r, out);

        // Verify eq+1[0] = 0
        try std.testing.expect(out[0].eql(F.zero()));

        // Verify against per-point MLE for all 2^n entries
        const j_bits = try allocator.alloc(F, n);
        defer allocator.free(j_bits);
        for (0..size) |j| {
            for (0..n) |k| {
                const bit_pos: std.math.Log2Int(usize) = @intCast(n - 1 - k);
                j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
            }
            const expected = EqPlusOnePolynomial(F).mle(r, j_bits);
            try std.testing.expect(out[j].eql(expected));
        }
    }
}

test "EqPolynomial buildEqAndEqPlusOneInPlace matches per-point mle" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // n = 0: eq_out = [1], eq_plus_one_out = [0]
    {
        var eq_out: [1]F = undefined;
        var eqp1_out: [1]F = undefined;
        EqPolynomial(F).buildEqAndEqPlusOneInPlace(&.{}, &eq_out, &eqp1_out);
        try std.testing.expect(eq_out[0].eql(F.one()));
        try std.testing.expect(eqp1_out[0].eql(F.zero()));
    }

    for (1..13) |n| {
        const size = @as(usize, 1) << @intCast(n);

        const r = try allocator.alloc(F, n);
        defer allocator.free(r);
        for (0..n) |i| r[i] = F.fromU64(@as(u64, i * 7 + 3));

        const eq_out = try allocator.alloc(F, size);
        defer allocator.free(eq_out);
        const eq_plus_one_out = try allocator.alloc(F, size);
        defer allocator.free(eq_plus_one_out);
        EqPolynomial(F).buildEqAndEqPlusOneInPlace(r, eq_out, eq_plus_one_out);

        const j_bits = try allocator.alloc(F, n);
        defer allocator.free(j_bits);
        for (0..size) |j| {
            for (0..n) |k| {
                const bit_pos: std.math.Log2Int(usize) = @intCast(n - 1 - k);
                j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
            }
            // eq table matches EqPolynomial.mle
            const expected_eq = EqPolynomial(F).mle(r, j_bits);
            try std.testing.expect(eq_out[j].eql(expected_eq));
            // eq+1 table matches EqPlusOnePolynomial.mle
            const expected_eqp1 = EqPlusOnePolynomial(F).mle(r, j_bits);
            try std.testing.expect(eq_plus_one_out[j].eql(expected_eqp1));
        }
    }
}

test "EqPlusOnePrefixSuffixPoly batch construction matches per-point mle" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Test for all n = 2..12, including odd n for asymmetric splits
    for (2..13) |n| {
        // Generate deterministic r values (BE order, as required by init)
        const r = try allocator.alloc(F, n);
        defer allocator.free(r);
        for (0..n) |i| {
            r[i] = F.fromU64(@as(u64, i * 7 + 3));
        }

        // Build via the actual EqPlusOnePrefixSuffixPoly.init code path
        var poly = try EqPlusOnePrefixSuffixPoly(F).init(allocator, r);
        defer poly.deinit();

        const mid = n / 2;
        const r_hi = r[0..mid];
        const r_lo = r[mid..];
        const size_lo: usize = @as(usize, 1) << @intCast(r_lo.len);
        const size_hi: usize = @as(usize, 1) << @intCast(r_hi.len);

        // Verify prefix_0 = eq+1(r_lo, j) for all j via per-point mle
        const j_bits = try allocator.alloc(F, @max(r_lo.len, r_hi.len));
        defer allocator.free(j_bits);
        for (0..size_lo) |j| {
            for (0..r_lo.len) |k| {
                const bit_pos: std.math.Log2Int(usize) = @intCast(r_lo.len - 1 - k);
                j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
            }
            const expected = EqPlusOnePolynomial(F).mle(r_lo, j_bits[0..r_lo.len]);
            try std.testing.expect(poly.prefix_0[j].eql(expected));
        }

        // Verify prefix_1 = is_max(r_lo) * delta(j=0)
        // is_max(r_lo) = eq((1,1,...,1), r_lo) = product of r_lo[i]
        {
            var is_max_expected = F.one();
            for (0..r_lo.len) |i| is_max_expected = is_max_expected.mul(r_lo[i]);
            try std.testing.expect(poly.prefix_1[0].eql(is_max_expected));
            for (1..size_lo) |j| {
                try std.testing.expect(poly.prefix_1[j].eql(F.zero()));
            }
        }

        // Verify suffix_0 = eq(r_hi, j) for all j
        for (0..size_hi) |j| {
            for (0..r_hi.len) |k| {
                const bit_pos: std.math.Log2Int(usize) = @intCast(r_hi.len - 1 - k);
                j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
            }
            const expected = EqPolynomial(F).mle(r_hi, j_bits[0..r_hi.len]);
            try std.testing.expect(poly.suffix_0[j].eql(expected));
        }

        // Verify suffix_1 = eq+1(r_hi, j) for all j
        for (0..size_hi) |j| {
            for (0..r_hi.len) |k| {
                const bit_pos: std.math.Log2Int(usize) = @intCast(r_hi.len - 1 - k);
                j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
            }
            const expected = EqPlusOnePolynomial(F).mle(r_hi, j_bits[0..r_hi.len]);
            try std.testing.expect(poly.suffix_1[j].eql(expected));
        }

        // Verify the decomposition identity for ALL full-size indices:
        // eq+1(r, j) == prefix_0[j_lo] * suffix_0[j_hi] + prefix_1[j_lo] * suffix_1[j_hi]
        const full_size: usize = @as(usize, 1) << @intCast(n);
        const full_j_bits = try allocator.alloc(F, n);
        defer allocator.free(full_j_bits);
        for (0..full_size) |j| {
            const j_hi = j >> @intCast(r_lo.len);
            const j_lo = j & (size_lo - 1);
            const actual = poly.prefix_0[j_lo].mul(poly.suffix_0[j_hi])
                .add(poly.prefix_1[j_lo].mul(poly.suffix_1[j_hi]));
            for (0..n) |k| {
                const bit_pos: std.math.Log2Int(usize) = @intCast(n - 1 - k);
                full_j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
            }
            const expected_full = EqPlusOnePolynomial(F).mle(r, full_j_bits);
            try std.testing.expect(actual.eql(expected_full));
        }
    }
}

test "EqPolynomial batch builders with large field elements" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Use large field elements that exercise modular reduction.
    // These are near the BN254 scalar field modulus (~2^254).
    const large_r = [_]F{
        F.fromU128(0xfedcba9876543210_fedcba9876543210),
        F.fromU128(0x123456789abcdef0_123456789abcdef0),
        F.fromU128(0xa0a0a0a0a0a0a0a0_b1b1b1b1b1b1b1b1),
        F.fromU128(0xffffffffffffffff_fffffffffffffffe),
        F.fromU128(0x8000000000000000_0000000000000001),
    };

    const n = large_r.len;
    const size = @as(usize, 1) << @intCast(n);

    const j_bits = try allocator.alloc(F, n);
    defer allocator.free(j_bits);

    // buildEqTableInPlace
    const eq_out = try allocator.alloc(F, size);
    defer allocator.free(eq_out);
    EqPolynomial(F).buildEqTableInPlace(&large_r, eq_out, null);
    for (0..size) |j| {
        for (0..n) |k| {
            const bit_pos: std.math.Log2Int(usize) = @intCast(n - 1 - k);
            j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
        }
        try std.testing.expect(eq_out[j].eql(EqPolynomial(F).mle(&large_r, j_bits)));
    }

    // buildEqTableInPlace with scaling
    const scale = F.fromU128(0xdeadbeefcafebabe_1234567890abcdef);
    const eq_scaled = try allocator.alloc(F, size);
    defer allocator.free(eq_scaled);
    EqPolynomial(F).buildEqTableInPlace(&large_r, eq_scaled, scale);
    for (0..size) |j| {
        try std.testing.expect(eq_scaled[j].eql(scale.mul(eq_out[j])));
    }

    // buildEqPlusOneTableInPlace
    const eqp1_out = try allocator.alloc(F, size);
    defer allocator.free(eqp1_out);
    EqPolynomial(F).buildEqPlusOneTableInPlace(&large_r, eqp1_out);
    for (0..size) |j| {
        for (0..n) |k| {
            const bit_pos: std.math.Log2Int(usize) = @intCast(n - 1 - k);
            j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
        }
        try std.testing.expect(eqp1_out[j].eql(EqPlusOnePolynomial(F).mle(&large_r, j_bits)));
    }

    // buildEqAndEqPlusOneInPlace
    const eq2 = try allocator.alloc(F, size);
    defer allocator.free(eq2);
    const eqp1_2 = try allocator.alloc(F, size);
    defer allocator.free(eqp1_2);
    EqPolynomial(F).buildEqAndEqPlusOneInPlace(&large_r, eq2, eqp1_2);
    for (0..size) |j| {
        try std.testing.expect(eq2[j].eql(eq_out[j]));
        try std.testing.expect(eqp1_2[j].eql(eqp1_out[j]));
    }
}

test {
    std.testing.refAllDecls(split_eq);
    std.testing.refAllDecls(multiquadratic);
    std.testing.refAllDecls(interpolation);
    std.testing.refAllDecls(product_tree);
}
