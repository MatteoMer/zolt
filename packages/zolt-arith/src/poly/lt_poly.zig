//! LtPolynomial: Lazy less-than polynomial using sqrt(T) decomposition.
//!
//! Instead of materializing a full T-sized LT table, stores 3 arrays of
//! size sqrt(T) and reconstructs values on-the-fly via:
//!   LT(i, r) = LT_hi(i_hi, r_hi) + EQ_hi(i_hi, r_hi) * LT_lo(i_lo, r_lo)
//!
//! Matches Jolt's LtPolynomial (jolt-core/src/poly/lt_poly.rs).

const std = @import("std");
const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;

pub fn LtPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        lt_lo: []F, // lt_evals over low bits of r_cycle, size 2^n_lo_vars
        lt_hi: []F, // lt_evals over high bits of r_cycle, size 2^n_hi_vars
        eq_hi: []F, // eq_evals over high bits of r_cycle, size 2^n_hi_vars
        n_lo_vars: usize, // number of low variables remaining (decrements on bind)
        allocator: Allocator,

        /// Initialize from r_cycle (BIG_ENDIAN: MSB first).
        /// Splits r_cycle into r_hi (first n/2) and r_lo (last n-n/2).
        /// Builds lt_lo, lt_hi, eq_hi arrays each of size sqrt(T).
        pub fn init(allocator: Allocator, r_cycle: []const F, tp: ?*ThreadPool) !Self {
            const n = r_cycle.len;
            const n_hi = n / 2;
            const n_lo = n - n_hi;
            const r_hi = r_cycle[0..n_hi]; // MSB side
            const r_lo = r_cycle[n_hi..n]; // LSB side

            const lt_lo = try computeLtEvals(allocator, r_lo, tp);
            const lt_hi = try computeLtEvals(allocator, r_hi, tp);

            const EqPoly = @import("mod.zig").EqPolynomial(F);
            const eq_hi = try EqPoly.evalsSliceWithScalingParallel(F, allocator, r_hi, null, tp);

            return Self{
                .lt_lo = lt_lo,
                .lt_hi = lt_hi,
                .eq_hi = eq_hi,
                .n_lo_vars = n_lo,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.lt_lo);
            self.allocator.free(self.lt_hi);
            self.allocator.free(self.eq_hi);
        }

        /// Reconstruct LT value at index i on-the-fly.
        /// LT(i) = LT_hi(i_hi) + EQ_hi(i_hi) * LT_lo(i_lo)
        pub inline fn getBoundCoeff(self: *const Self, i: usize) F {
            const i_hi = i >> @intCast(self.n_lo_vars);
            const i_lo = i & ((@as(usize, 1) << @intCast(self.n_lo_vars)) - 1);
            return self.lt_hi[i_hi].add(self.eq_hi[i_hi].mul(self.lt_lo[i_lo]));
        }

        /// Bind one variable (LowToHigh order).
        /// First n_lo_vars rounds: bind lt_lo.
        /// Then: bind both lt_hi and eq_hi.
        pub fn bind(self: *Self, r: F) void {
            if (self.n_lo_vars > 0) {
                const len = @as(usize, 1) << @intCast(self.n_lo_vars);
                const half = len / 2;
                for (0..half) |i| {
                    self.lt_lo[i] = self.lt_lo[2 * i].add(r.mul(self.lt_lo[2 * i + 1].sub(self.lt_lo[2 * i])));
                }
                self.n_lo_vars -= 1;
            } else {
                const len = self.lt_hi.len;
                const half = len / 2;
                if (half == 0) return;
                for (0..half) |i| {
                    self.lt_hi[i] = self.lt_hi[2 * i].add(r.mul(self.lt_hi[2 * i + 1].sub(self.lt_hi[2 * i])));
                    self.eq_hi[i] = self.eq_hi[2 * i].add(r.mul(self.eq_hi[2 * i + 1].sub(self.eq_hi[2 * i])));
                }
            }
        }

        /// Final scalar after all variables are bound.
        pub fn finalClaim(self: *const Self) F {
            return self.lt_hi[0].add(self.eq_hi[0].mul(self.lt_lo[0]));
        }

        /// Compute LT evaluations using the butterfly algorithm.
        /// Same as computeAllLtEvals but standalone for reuse.
        /// r is in BIG_ENDIAN order (MSB first).
        fn computeLtEvals(allocator: Allocator, r: []const F, tp: ?*ThreadPool) ![]F {
            const n = r.len;
            if (n == 0) {
                const evals = try allocator.alloc(F, 1);
                evals[0] = F.zero();
                return evals;
            }
            const size = @as(usize, 1) << @intCast(n);
            const evals = try allocator.alloc(F, size);
            @memset(evals, F.zero());

            // Process r from LSB (r[n-1]) to MSB (r[0]).
            // LT butterfly: right[j] = left[j] * r_i; left[j] += r_i - right[j]
            for (0..n) |i| {
                const ri = r[n - 1 - i];
                const half = @as(usize, 1) << @intCast(i);
                const left = evals[0..half];
                const right = evals[half .. 2 * half];

                if (tp != null and half >= 256) {
                    const Ctx = struct { l: []F, rr: []F, r_val: F };
                    const ctx = Ctx{ .l = left, .rr = right, .r_val = ri };
                    tp.?.parallelForForce(half, ctx, struct {
                        fn f(c: Ctx, j: usize) void {
                            const y = c.l[j].mul(c.r_val);
                            c.rr[j] = y;
                            c.l[j] = c.l[j].add(c.r_val.sub(y));
                        }
                    }.f);
                } else {
                    for (0..half) |j| {
                        const y = left[j].mul(ri);
                        right[j] = y;
                        left[j] = left[j].add(ri.sub(y));
                    }
                }
            }

            return evals;
        }
    };
}
