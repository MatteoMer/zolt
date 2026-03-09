//! OneHotCoeff lookup table for sparse register binding.
//!
//! During Phase 1 (cycle-major) binding, ra/wa coefficients are "one-hot":
//! they take values from a small set {0, gamma, gamma^2, gamma+gamma^2} for ra
//! and {0, 1} for wa. Instead of storing full field elements, we store u16 indices
//! into a lookup table.
//!
//! On each bind(r), the table squares in size (N -> N^2):
//!   new_table[a_idx * old_len + b_idx] = old[b_idx] + r * (old[a_idx] - old[b_idx])
//!
//! When the table reaches MAX_LOOKUP_TABLE_SIZE (2^16), indices are resolved to field elements.

const std = @import("std");

/// Maximum lookup table size before saturation.
const MAX_LOOKUP_TABLE_SIZE: usize = 1 << 16; // 65536

/// Index into a OneHotCoeffLookupTable, stored as u16 to save memory.
pub const LookupTableIndex = struct {
    idx: u16,

    pub fn zero() LookupTableIndex {
        return .{ .idx = 0 };
    }

    pub fn fromU16(val: u16) LookupTableIndex {
        return .{ .idx = val };
    }
};

/// Lookup table for one-hot coefficient values that grows during sumcheck.
///
/// Starts with a small set of initial values (e.g., [0, gamma, gamma^2, gamma + gamma^2])
/// and expands by binding with random challenges. Saturates at MAX_LOOKUP_TABLE_SIZE.
pub fn OneHotCoeffLookupTable(comptime F: type) type {
    return struct {
        const Self = @This();

        lookup_table: []F,
        allocator: std.mem.Allocator,

        /// Initialize with arbitrary initial coefficients (must be power-of-two length).
        pub fn init(allocator: std.mem.Allocator, init_coeffs: []const F) !Self {
            std.debug.assert(std.math.isPowerOfTwo(init_coeffs.len));
            const table = try allocator.alloc(F, init_coeffs.len);
            @memcpy(table, init_coeffs);
            return Self{ .lookup_table = table, .allocator = allocator };
        }

        /// Initialize ra lookup table with 4 entries: [0, gamma, gamma^2, gamma+gamma^2].
        pub fn initRa(allocator: std.mem.Allocator, gamma: F) !Self {
            const table = try allocator.alloc(F, 4);
            const gamma_sq = gamma.mul(gamma);
            table[0] = F.zero();
            table[1] = gamma;
            table[2] = gamma_sq;
            table[3] = gamma.add(gamma_sq);
            return Self{ .lookup_table = table, .allocator = allocator };
        }

        /// Initialize wa lookup table with 2 entries: [0, 1].
        pub fn initWa(allocator: std.mem.Allocator) !Self {
            const table = try allocator.alloc(F, 2);
            table[0] = F.zero();
            table[1] = F.one();
            return Self{ .lookup_table = table, .allocator = allocator };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.lookup_table);
        }

        /// Returns true if the table has reached maximum size and cannot grow further.
        pub fn isSaturated(self: *const Self) bool {
            return self.lookup_table.len >= MAX_LOOKUP_TABLE_SIZE;
        }

        /// Returns the log2 of the current table size.
        pub fn indexBitwidth(self: *const Self) u5 {
            return @intCast(std.math.log2_int(usize, self.lookup_table.len));
        }

        /// Dereference a lookup index to a field element.
        pub fn deref(self: *const Self, idx: LookupTableIndex) F {
            return self.lookup_table[@as(usize, idx.idx)];
        }

        /// Bind the lookup table with challenge r, squaring its size (N -> N^2).
        ///
        /// new[a * old_len + b] = old[b] + r * (old[a] - old[b])
        ///
        /// This matches upstream Jolt: flat_map(|a| map(|b| b + r*(a-b)))
        pub fn bind(self: *Self, r: F) !void {
            std.debug.assert(self.lookup_table.len < MAX_LOOKUP_TABLE_SIZE);

            const old_len = self.lookup_table.len;
            const new_len = old_len * old_len;
            const new_table = try self.allocator.alloc(F, new_len);

            for (0..old_len) |a| {
                const a_val = self.lookup_table[a];
                for (0..old_len) |b| {
                    const b_val = self.lookup_table[b];
                    // b + r * (a - b)
                    new_table[a * old_len + b] = b_val.add(r.mul(a_val.sub(b_val)));
                }
            }

            self.allocator.free(self.lookup_table);
            self.lookup_table = new_table;
        }

        /// Bind two LookupTableIndex values, producing a new combined index.
        ///
        /// The lookup table must have already been bound (its size is now N^2).
        /// The bitwidth used is log2(sqrt(new_table_len)) = log2(old_table_len).
        ///
        /// new_idx = odd << bitwidth | even
        ///
        /// When one side is missing (None), its index defaults to 0 (the zero entry).
        pub fn bindIndices(
            self: *const Self,
            even: ?LookupTableIndex,
            odd: ?LookupTableIndex,
        ) LookupTableIndex {
            // After table bind, bitwidth = log2(table.len).
            // Index composition uses half that (the pre-bind bitwidth).
            const bitwidth = self.indexBitwidth();
            std.debug.assert(bitwidth <= 16);
            // The pre-bind bitwidth is half the current bitwidth.
            const half_bitwidth: u4 = @intCast(bitwidth / 2);

            const even_idx: u16 = if (even) |e| e.idx else 0;
            const odd_idx: u16 = if (odd) |o| o.idx else 0;

            return LookupTableIndex.fromU16((odd_idx << half_bitwidth) | even_idx);
        }

        /// Bind two LookupTableIndex values, using the current (pre-bind) bitwidth.
        ///
        /// Call this BEFORE the table is bound for this round if you want to compose
        /// indices that will be looked up in the post-bind table.
        pub fn bindIndicesPreBind(
            self: *const Self,
            even: ?LookupTableIndex,
            odd: ?LookupTableIndex,
        ) LookupTableIndex {
            const bitwidth: u4 = @intCast(self.indexBitwidth());

            const even_idx: u16 = if (even) |e| e.idx else 0;
            const odd_idx: u16 = if (odd) |o| o.idx else 0;

            return LookupTableIndex.fromU16((odd_idx << bitwidth) | even_idx);
        }

        /// Compute sumcheck evaluations [f(0), f(1) - f(0)] from two lookup indices.
        pub fn evals(
            self: *const Self,
            even: ?LookupTableIndex,
            odd: ?LookupTableIndex,
        ) [2]F {
            const even_val = if (even) |e| self.lookup_table[@as(usize, e.idx)] else F.zero();
            const odd_val = if (odd) |o| self.lookup_table[@as(usize, o.idx)] else F.zero();

            if (even != null and odd != null) {
                return .{ even_val, odd_val.sub(even_val) };
            } else if (even != null) {
                return .{ even_val, even_val.neg() };
            } else if (odd != null) {
                return .{ F.zero(), odd_val };
            } else {
                unreachable; // Both entries are None
            }
        }

        /// Compute sumcheck evaluations [f(0), f(1) - f(0)] from direct field elements.
        pub fn evalsField(
            even: ?F,
            odd: ?F,
        ) [2]F {
            const even_val = even orelse F.zero();
            const odd_val = odd orelse F.zero();

            if (even != null and odd != null) {
                return .{ even_val, odd_val.sub(even_val) };
            } else if (even != null) {
                return .{ even_val, even_val.neg() };
            } else if (odd != null) {
                return .{ F.zero(), odd_val };
            } else {
                unreachable; // Both entries are None
            }
        }

        /// Bind two field elements with challenge r.
        pub fn bindField(
            even: ?F,
            odd: ?F,
            r: F,
        ) F {
            const even_val = even orelse F.zero();
            const odd_val = odd orelse F.zero();

            if (even != null and odd != null) {
                // even + r * (odd - even)
                return even_val.add(r.mul(odd_val.sub(even_val)));
            } else if (even != null) {
                // even * (1 - r)
                return even_val.sub(r.mul(even_val));
            } else if (odd != null) {
                // r * odd
                return r.mul(odd_val);
            } else {
                unreachable; // Both entries are None
            }
        }
    };
}
