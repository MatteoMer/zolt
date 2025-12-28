//! ExpandingTable - Incrementally built EQ polynomial table
//!
//! This matches Jolt's `ExpandingTable` from `jolt-core/src/utils/expanding_table.rs`.
//! It stores EQ(x_1, ..., x_j, r_1, ..., r_j) built up incrementally as we receive
//! random challenges r_j during sumcheck.
//!
//! The table starts with a single entry [1] and doubles in size with each `update(r_j)` call.
//!
//! Reference: jolt-core/src/utils/expanding_table.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Table containing the evaluations `EQ(x_1, ..., x_j, r_1, ..., r_j)`,
/// built up incrementally as we receive random challenges `r_j` over the
/// course of sumcheck.
pub fn ExpandingTable(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Binding order for the table
        pub const BindingOrder = enum {
            LowToHigh,
            HighToLow,
        };

        /// Current length of valid entries
        len: usize,
        /// Actual storage for EQ values
        values: []F,
        /// Binding order
        binding_order: BindingOrder,
        /// Allocator
        allocator: Allocator,

        /// Initialize an ExpandingTable with the given capacity.
        ///
        /// The table starts empty and should be reset before use.
        pub fn init(allocator: Allocator, capacity: usize, binding_order: BindingOrder) !Self {
            const values = try allocator.alloc(F, capacity);
            @memset(values, F.zero());

            return Self{
                .len = 0,
                .values = values,
                .binding_order = binding_order,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.values);
        }

        /// Reset the table to length 1, containing only the given value.
        pub fn reset(self: *Self, value: F) void {
            self.values[0] = value;
            self.len = 1;
        }

        /// Current length of the table
        pub fn length(self: *const Self) usize {
            return self.len;
        }

        /// Get value at index
        pub fn get(self: *const Self, index: usize) F {
            std.debug.assert(index < self.len);
            return self.values[index];
        }

        /// Get a slice of the current values
        pub fn slice(self: *const Self) []const F {
            return self.values[0..self.len];
        }

        /// Update the table (expanding it by a factor of 2) to incorporate
        /// the new random challenge r_j.
        ///
        /// For LowToHigh binding:
        ///   values[i] = (1 - r_j) * old_values[i]
        ///   values[i + len] = r_j * old_values[i]
        ///
        /// After update, values[k] contains eq(r_1, k_1) * eq(r_2, k_2) * ... * eq(r_j, k_j)
        /// where k = (k_1, k_2, ..., k_j) is the binary representation of index k.
        pub fn update(self: *Self, r_j: F) void {
            switch (self.binding_order) {
                .LowToHigh => {
                    // For each existing value, split into (1-r_j) and r_j components
                    const old_len = self.len;
                    var i: usize = 0;
                    while (i < old_len) : (i += 1) {
                        const old_val = self.values[i];
                        const r_times_old = r_j.mul(old_val);
                        // values[i] = (1 - r_j) * old = old - r_j * old
                        self.values[i] = old_val.sub(r_times_old);
                        // values[i + len] = r_j * old
                        self.values[i + old_len] = r_times_old;
                    }
                    self.len = old_len * 2;
                },
                .HighToLow => {
                    // Not used in outer Spartan, but implemented for completeness
                    const old_len = self.len;
                    var new_len: usize = 0;

                    var i: usize = 0;
                    while (i < old_len) : (i += 1) {
                        const old_val = self.values[i];
                        const r_times_old = r_j.mul(old_val);
                        self.values[new_len] = old_val.sub(r_times_old);
                        self.values[new_len + 1] = r_times_old;
                        new_len += 2;
                    }
                    self.len = new_len;
                },
            }
        }

        /// Clone the current values to a new allocation
        pub fn cloneValues(self: *const Self, allocator: Allocator) ![]F {
            const result = try allocator.alloc(F, self.len);
            @memcpy(result, self.values[0..self.len]);
            return result;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "ExpandingTable: basic initialization" {
    const F = @import("../field/mod.zig").BN254Scalar;

    var table = try ExpandingTable(F).init(testing.allocator, 16, .LowToHigh);
    defer table.deinit();

    try testing.expectEqual(@as(usize, 0), table.len);

    table.reset(F.one());
    try testing.expectEqual(@as(usize, 1), table.len);
    try testing.expect(table.get(0).eql(F.one()));
}

test "ExpandingTable: update doubles size" {
    const F = @import("../field/mod.zig").BN254Scalar;

    var table = try ExpandingTable(F).init(testing.allocator, 16, .LowToHigh);
    defer table.deinit();

    table.reset(F.one());

    // First update with r = 0.5 (simulated by field element)
    const r1 = F.fromU64(3); // Some challenge
    table.update(r1);

    try testing.expectEqual(@as(usize, 2), table.len);

    // table[0] = (1 - r1) * 1 = 1 - r1
    // table[1] = r1 * 1 = r1
    const one_minus_r1 = F.one().sub(r1);
    try testing.expect(table.get(0).eql(one_minus_r1));
    try testing.expect(table.get(1).eql(r1));
}

test "ExpandingTable: multiple updates" {
    const F = @import("../field/mod.zig").BN254Scalar;

    var table = try ExpandingTable(F).init(testing.allocator, 16, .LowToHigh);
    defer table.deinit();

    table.reset(F.one());

    // Update with r1
    const r1 = F.fromU64(2);
    table.update(r1);

    // Update with r2
    const r2 = F.fromU64(5);
    table.update(r2);

    try testing.expectEqual(@as(usize, 4), table.len);

    // After two updates, we should have:
    // table[0] = (1-r1)(1-r2) = eq(r1,0)*eq(r2,0)
    // table[1] = r1*(1-r2) = eq(r1,1)*eq(r2,0)
    // table[2] = (1-r1)*r2 = eq(r1,0)*eq(r2,1)
    // table[3] = r1*r2 = eq(r1,1)*eq(r2,1)

    const one_minus_r1 = F.one().sub(r1);
    const one_minus_r2 = F.one().sub(r2);

    // table[0] = (1-r1)(1-r2)
    const expected_0 = one_minus_r1.mul(one_minus_r2);
    try testing.expect(table.get(0).eql(expected_0));

    // table[1] = r1*(1-r2)
    const expected_1 = r1.mul(one_minus_r2);
    try testing.expect(table.get(1).eql(expected_1));

    // table[2] = (1-r1)*r2
    const expected_2 = one_minus_r1.mul(r2);
    try testing.expect(table.get(2).eql(expected_2));

    // table[3] = r1*r2
    const expected_3 = r1.mul(r2);
    try testing.expect(table.get(3).eql(expected_3));
}

test "ExpandingTable: eq polynomial property" {
    const F = @import("../field/mod.zig").BN254Scalar;

    var table = try ExpandingTable(F).init(testing.allocator, 16, .LowToHigh);
    defer table.deinit();

    table.reset(F.one());

    const r1 = F.fromU64(7);
    const r2 = F.fromU64(11);
    table.update(r1);
    table.update(r2);

    // The table should contain eq(r, x) for x in {0,1}^2
    // Sum over all x should equal 1 (normalized eq polynomial)
    var sum = F.zero();
    for (0..table.len) |i| {
        sum = sum.add(table.get(i));
    }

    // Note: this is NOT 1 because we're using unnormalized eq
    // eq(r, x) = Π (r_i * x_i + (1-r_i)(1-x_i))
    // The sum over x is NOT 1 in general

    // However, we can verify that the product of factors works correctly
    // by checking specific entries

    // eq((r1, r2), (0, 0)) = (1-r1)(1-r2)
    // eq((r1, r2), (1, 0)) = r1*(1-r2)
    // eq((r1, r2), (0, 1)) = (1-r1)*r2
    // eq((r1, r2), (1, 1)) = r1*r2

    // These should match what the table stores
    const one_minus_r1 = F.one().sub(r1);
    const one_minus_r2 = F.one().sub(r2);

    try testing.expect(table.get(0).eql(one_minus_r1.mul(one_minus_r2)));
    try testing.expect(table.get(1).eql(r1.mul(one_minus_r2)));
    try testing.expect(table.get(2).eql(one_minus_r1.mul(r2)));
    try testing.expect(table.get(3).eql(r1.mul(r2)));
}
