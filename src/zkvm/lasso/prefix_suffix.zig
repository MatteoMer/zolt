//! Prefix-Suffix Decomposition for Lookup Tables
//!
//! This module implements the prefix-suffix decomposition technique used in Lasso
//! to efficiently evaluate structured lookup tables. The key insight is that many
//! lookup tables have a special structure where the MLE can be decomposed as:
//!
//!   Val(k) = Σ_i P_i(k_prefix) · Q_i(k_suffix)
//!
//! where:
//! - k = (k_prefix || k_suffix) is the lookup index split into prefix and suffix
//! - P_i are "prefix polynomials" that depend only on the high-order bits
//! - Q_i are "suffix polynomials" that depend only on the low-order bits
//!
//! The decomposition enables efficient MLE evaluation during sumcheck:
//! - Prefix polynomials are cached and reused across multiple phases
//! - Suffix polynomials are recomputed each phase (but only over smaller domains)
//!
//! This module provides:
//! - PrefixPolynomial: Cached multilinear polynomial over prefix variables
//! - SuffixPolynomial: Dynamically computed polynomial over suffix variables
//! - PrefixSuffixDecomposition: Combines prefixes and suffixes for table evaluation
//!
//! Reference: Jolt paper Section 7.3 and jolt-core/src/poly/prefix_suffix.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Suffix types that can be computed for lookup tables
/// These correspond to the different ways operand bits combine
pub const SuffixType = enum {
    /// The constant 1 (for identity)
    One,
    /// AND of corresponding bits: x_i * y_i
    And,
    /// NOT-AND: (1 - x_i) * y_i
    NotAnd,
    /// XOR: x_i + y_i - 2*x_i*y_i
    Xor,
    /// OR: x_i + y_i - x_i*y_i
    Or,
    /// Just the right operand: y_i
    RightOperand,
    /// Just the left operand: x_i
    LeftOperand,
    /// Less than comparison result
    LessThan,
    /// Greater than comparison result
    GreaterThan,
    /// Equality comparison result
    Eq,
    /// Not equal
    NotEq,
    /// Sign extension
    SignExtension,

    /// Evaluate the suffix polynomial at a single point
    pub fn evaluate(self: SuffixType, comptime F: type, x: F, y: F) F {
        return switch (self) {
            .One => F.one(),
            .And => x.mul(y),
            .NotAnd => F.one().sub(x).mul(y),
            .Xor => {
                // x XOR y = x + y - 2xy
                const xy = x.mul(y);
                const two = F.fromU64(2);
                return x.add(y).sub(two.mul(xy));
            },
            .Or => {
                // x OR y = x + y - xy
                return x.add(y).sub(x.mul(y));
            },
            .RightOperand => y,
            .LeftOperand => x,
            .LessThan => {
                // For comparing bits: x < y in a single bit means x=0 and y=1
                return F.one().sub(x).mul(y);
            },
            .GreaterThan => {
                // x > y means x=1 and y=0
                return x.mul(F.one().sub(y));
            },
            .Eq => {
                // x = y means (x AND y) OR (NOT x AND NOT y)
                // = xy + (1-x)(1-y) = xy + 1 - x - y + xy = 1 - x - y + 2xy
                const one = F.one();
                const two = F.fromU64(2);
                return one.sub(x).sub(y).add(two.mul(x.mul(y)));
            },
            .NotEq => {
                // NOT (x = y) = x XOR y
                const xy = x.mul(y);
                const two = F.fromU64(2);
                return x.add(y).sub(two.mul(xy));
            },
            .SignExtension => {
                // Sign extension depends on the most significant bit
                // For now, just return x (placeholder for more complex logic)
                return x;
            },
        };
    }
};

/// Prefix types for lookup table decomposition
pub const PrefixType = enum {
    /// Lower word of the result
    LowerWord,
    /// Upper word of the result
    UpperWord,
    /// Equality check prefix
    Eq,
    /// AND operation prefix
    And,
    /// OR operation prefix
    Or,
    /// XOR operation prefix
    Xor,
    /// Less than comparison prefix
    LessThan,
    /// Left operand is zero
    LeftOperandIsZero,
    /// Right operand is zero
    RightOperandIsZero,
    /// Sign extension prefix
    SignExtension,
    /// Shift left prefix
    LeftShift,
    /// Shift right prefix
    RightShift,
};

/// A cached prefix polynomial evaluation
pub fn PrefixPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Cached evaluations over the prefix hypercube
        evaluations: []F,
        /// Number of prefix variables
        num_vars: usize,
        /// Prefix type identifier
        prefix_type: PrefixType,
        allocator: Allocator,

        /// Create a new prefix polynomial
        pub fn init(allocator: Allocator, num_vars: usize, prefix_type: PrefixType) !Self {
            const size = @as(usize, 1) << @intCast(num_vars);
            const evaluations = try allocator.alloc(F, size);
            @memset(evaluations, F.zero());

            return Self{
                .evaluations = evaluations,
                .num_vars = num_vars,
                .prefix_type = prefix_type,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.evaluations);
        }

        /// Get the evaluation at a specific prefix index
        pub fn get(self: *const Self, index: usize) F {
            return self.evaluations[index];
        }

        /// Set the evaluation at a specific prefix index
        pub fn set(self: *Self, index: usize, value: F) void {
            self.evaluations[index] = value;
        }

        /// Bind the first variable to a challenge value
        /// Returns a new prefix polynomial with one fewer variable
        pub fn bind(self: *const Self, challenge: F) !Self {
            std.debug.assert(self.num_vars > 0);

            const new_size = self.evaluations.len / 2;
            const new_evals = try self.allocator.alloc(F, new_size);

            const one_minus_c = F.one().sub(challenge);

            for (0..new_size) |i| {
                const low = self.evaluations[i];
                const high = self.evaluations[i + new_size];
                new_evals[i] = low.mul(one_minus_c).add(high.mul(challenge));
            }

            return Self{
                .evaluations = new_evals,
                .num_vars = self.num_vars - 1,
                .prefix_type = self.prefix_type,
                .allocator = self.allocator,
            };
        }

        /// Evaluate the prefix polynomial at a point
        pub fn evaluate(self: *const Self, point: []const F) F {
            std.debug.assert(point.len == self.num_vars);

            var result = F.zero();
            for (self.evaluations, 0..) |eval, i| {
                var term = eval;
                for (0..self.num_vars) |j| {
                    const bit = (i >> @intCast(j)) & 1;
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

        /// Clone the prefix polynomial
        pub fn clone(self: *const Self) !Self {
            const new_evals = try self.allocator.alloc(F, self.evaluations.len);
            @memcpy(new_evals, self.evaluations);

            return Self{
                .evaluations = new_evals,
                .num_vars = self.num_vars,
                .prefix_type = self.prefix_type,
                .allocator = self.allocator,
            };
        }
    };
}

/// A suffix polynomial computed dynamically
pub fn SuffixPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Coefficients of the suffix polynomial
        /// For linear suffixes, this is [constant, coefficient]
        coeffs: []F,
        /// Suffix type
        suffix_type: SuffixType,
        allocator: Allocator,

        /// Create a new suffix polynomial
        pub fn init(allocator: Allocator, suffix_type: SuffixType) !Self {
            // Most suffix polynomials are linear in each variable
            const coeffs = try allocator.alloc(F, 2);
            coeffs[0] = F.zero();
            coeffs[1] = F.zero();

            return Self{
                .coeffs = coeffs,
                .suffix_type = suffix_type,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.coeffs);
        }

        /// Evaluate the suffix at given operand evaluations
        pub fn evaluate(self: *const Self, x_eval: F, y_eval: F) F {
            return self.suffix_type.evaluate(F, x_eval, y_eval);
        }
    };
}

/// Prefix-Suffix Decomposition for a lookup table
///
/// Decomposes a table's MLE as: Val(k) = Σ_i P_i(k_prefix) · Q_i(k_suffix)
pub fn PrefixSuffixDecomposition(comptime F: type, comptime ORDER: usize) type {
    return struct {
        const Self = @This();

        /// Prefix polynomials (cached)
        prefixes: [ORDER]?PrefixPolynomial(F),
        /// Current suffix evaluations for each term
        suffix_evals: [ORDER]F,

        /// Number of prefix variables
        prefix_vars: usize,
        /// Number of suffix variables
        suffix_vars: usize,
        /// Total chunk length for each phase
        chunk_len: usize,

        /// Current phase (which chunk we're processing)
        phase: usize,
        /// Current round within the phase
        round: usize,

        allocator: Allocator,

        /// Create a new prefix-suffix decomposition
        pub fn init(allocator: Allocator, prefix_vars: usize, suffix_vars: usize) !Self {
            return Self{
                .prefixes = [_]?PrefixPolynomial(F){null} ** ORDER,
                .suffix_evals = [_]F{F.zero()} ** ORDER,
                .prefix_vars = prefix_vars,
                .suffix_vars = suffix_vars,
                .chunk_len = @as(usize, 1) << @intCast(suffix_vars),
                .phase = 0,
                .round = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (&self.prefixes) |*p| {
                if (p.*) |*prefix| {
                    prefix.deinit();
                    p.* = null;
                }
            }
        }

        /// Set a prefix polynomial at a given index
        pub fn setPrefix(self: *Self, index: usize, prefix: PrefixPolynomial(F)) void {
            std.debug.assert(index < ORDER);
            if (self.prefixes[index]) |*old| {
                old.deinit();
            }
            self.prefixes[index] = prefix;
        }

        /// Set a suffix evaluation at a given index
        pub fn setSuffixEval(self: *Self, index: usize, value: F) void {
            std.debug.assert(index < ORDER);
            self.suffix_evals[index] = value;
        }

        /// Compute the combined evaluation: Σ_i P_i(prefix) · Q_i
        pub fn evaluate(self: *const Self, prefix_point: []const F) F {
            var result = F.zero();

            for (0..ORDER) |i| {
                if (self.prefixes[i]) |*prefix| {
                    const p_eval = prefix.evaluate(prefix_point);
                    result = result.add(p_eval.mul(self.suffix_evals[i]));
                }
            }

            return result;
        }

        /// Bind a variable in the current round
        pub fn bind(self: *Self, challenge: F) !void {
            // Bind the challenge in each active prefix
            for (&self.prefixes) |*p| {
                if (p.*) |*prefix| {
                    if (prefix.num_vars > 0) {
                        const new_prefix = try prefix.bind(challenge);
                        prefix.deinit();
                        p.* = new_prefix;
                    }
                }
            }
            self.round += 1;
        }

        /// Initialize suffix evaluations for a new phase
        pub fn initSuffixes(
            self: *Self,
            u_evals: []const F,
            lookup_indices: []const u128,
            suffix_type: SuffixType,
        ) void {
            // Accumulate suffix evaluations based on EQ weights
            @memset(&self.suffix_evals, F.zero());

            for (u_evals, lookup_indices) |u, idx| {
                // Get the suffix portion of the lookup index
                const suffix = idx % self.chunk_len;
                const prefix = idx / self.chunk_len;

                // Evaluate the suffix polynomial for this lookup
                // For simple cases, we accumulate directly into suffix_evals
                _ = prefix;
                _ = suffix;
                _ = suffix_type;

                // For now, just accumulate u directly
                // Real implementation would compute suffix_type.evaluate(...)
                self.suffix_evals[0] = self.suffix_evals[0].add(u);
            }
        }

        /// Advance to the next phase
        pub fn nextPhase(self: *Self) void {
            self.phase += 1;
            self.round = 0;
        }

        /// Get the current prefix size (2^remaining_prefix_vars)
        pub fn currentPrefixSize(self: *const Self) usize {
            if (self.prefixes[0]) |prefix| {
                return prefix.evaluations.len;
            }
            return 0;
        }
    };
}

/// A registry for caching and sharing prefix polynomials across decompositions
pub fn PrefixRegistry(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Cached prefix polynomials indexed by type
        cache: std.AutoHashMap(PrefixType, PrefixPolynomial(F)),
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .cache = std.AutoHashMap(PrefixType, PrefixPolynomial(F)).init(allocator),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            var iter = self.cache.iterator();
            while (iter.next()) |entry| {
                var poly = entry.value_ptr.*;
                poly.deinit();
            }
            self.cache.deinit();
        }

        /// Get or create a prefix polynomial of the given type
        pub fn getOrCreate(self: *Self, prefix_type: PrefixType, num_vars: usize) !*PrefixPolynomial(F) {
            const result = try self.cache.getOrPut(prefix_type);
            if (!result.found_existing) {
                result.value_ptr.* = try PrefixPolynomial(F).init(self.allocator, num_vars, prefix_type);
            }
            return result.value_ptr;
        }

        /// Check if a prefix type is cached
        pub fn contains(self: *const Self, prefix_type: PrefixType) bool {
            return self.cache.contains(prefix_type);
        }
    };
}

test "suffix type evaluation" {
    const F = @import("../../field/mod.zig").BN254Scalar;

    const x = F.fromU64(3);
    const y = F.fromU64(5);

    // Test One
    const one_result = SuffixType.One.evaluate(F, x, y);
    try std.testing.expect(one_result.eql(F.one()));

    // Test And: x * y = 15
    const and_result = SuffixType.And.evaluate(F, x, y);
    try std.testing.expect(and_result.eql(x.mul(y)));

    // Test RightOperand: y
    const right_result = SuffixType.RightOperand.evaluate(F, x, y);
    try std.testing.expect(right_result.eql(y));

    // Test LeftOperand: x
    const left_result = SuffixType.LeftOperand.evaluate(F, x, y);
    try std.testing.expect(left_result.eql(x));
}

test "prefix polynomial basic" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var prefix = try PrefixPolynomial(F).init(allocator, 2, .And);
    defer prefix.deinit();

    // Set some values
    prefix.set(0, F.fromU64(1));
    prefix.set(1, F.fromU64(2));
    prefix.set(2, F.fromU64(3));
    prefix.set(3, F.fromU64(4));

    try std.testing.expect(prefix.get(0).eql(F.fromU64(1)));
    try std.testing.expect(prefix.get(3).eql(F.fromU64(4)));
}

test "prefix polynomial bind" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var prefix = try PrefixPolynomial(F).init(allocator, 2, .And);

    // Set values: [1, 2, 3, 4]
    prefix.set(0, F.fromU64(1));
    prefix.set(1, F.fromU64(2));
    prefix.set(2, F.fromU64(3));
    prefix.set(3, F.fromU64(4));

    // Bind first variable to challenge = 2
    const challenge = F.fromU64(2);
    var bound = try prefix.bind(challenge);
    prefix.deinit();
    defer bound.deinit();

    // After binding: new[i] = old[i]*(1-c) + old[i+2]*c
    // new[0] = 1*(1-2) + 3*2 = -1 + 6 = 5
    // new[1] = 2*(1-2) + 4*2 = -2 + 8 = 6
    try std.testing.expectEqual(@as(usize, 1), bound.num_vars);
    try std.testing.expect(bound.get(0).eql(F.fromU64(5)));
    try std.testing.expect(bound.get(1).eql(F.fromU64(6)));
}

test "prefix suffix decomposition basic" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var decomp = try PrefixSuffixDecomposition(F, 2).init(allocator, 2, 2);
    defer decomp.deinit();

    try std.testing.expectEqual(@as(usize, 2), decomp.prefix_vars);
    try std.testing.expectEqual(@as(usize, 2), decomp.suffix_vars);
    try std.testing.expectEqual(@as(usize, 4), decomp.chunk_len);
}

test "prefix registry" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var registry = PrefixRegistry(F).init(allocator);
    defer registry.deinit();

    // Get or create a prefix
    const prefix = try registry.getOrCreate(.And, 3);
    try std.testing.expectEqual(@as(usize, 3), prefix.num_vars);

    // Should return the same cached instance
    const prefix2 = try registry.getOrCreate(.And, 3);
    try std.testing.expectEqual(prefix, prefix2);

    // Different type should create new
    _ = try registry.getOrCreate(.Or, 3);
    try std.testing.expect(registry.contains(.And));
    try std.testing.expect(registry.contains(.Or));
}
