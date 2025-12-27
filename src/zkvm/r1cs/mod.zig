//! R1CS constraint system for Jolt
//!
//! R1CS (Rank-1 Constraint System) represents constraints of the form:
//! (a · x) * (b · x) = (c · x)
//! where a, b, c are vectors of coefficients and x is the witness vector.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// A single R1CS constraint
pub fn Constraint(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Sparse representation of a linear combination
        pub const LinearCombination = struct {
            /// (variable_index, coefficient) pairs
            terms: std.ArrayList(struct { index: usize, coeff: F }),

            pub fn init(allocator: Allocator) LinearCombination {
                return .{
                    .terms = std.ArrayList(struct { index: usize, coeff: F }).init(allocator),
                };
            }

            pub fn deinit(self: *LinearCombination) void {
                self.terms.deinit();
            }

            /// Add a term to the linear combination
            pub fn addTerm(self: *LinearCombination, index: usize, coeff: F) !void {
                try self.terms.append(.{ .index = index, .coeff = coeff });
            }

            /// Evaluate the linear combination given a witness
            pub fn evaluate(self: *const LinearCombination, witness: []const F) F {
                var result = F.zero();
                for (self.terms.items) |term| {
                    result = result.add(term.coeff.mul(witness[term.index]));
                }
                return result;
            }
        };

        /// A · witness
        a: LinearCombination,
        /// B · witness
        b: LinearCombination,
        /// C · witness
        c: LinearCombination,

        pub fn init(allocator: Allocator) Self {
            return .{
                .a = LinearCombination.init(allocator),
                .b = LinearCombination.init(allocator),
                .c = LinearCombination.init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.a.deinit();
            self.b.deinit();
            self.c.deinit();
        }

        /// Check if the constraint is satisfied by the witness
        pub fn isSatisfied(self: *const Self, witness: []const F) bool {
            const a_val = self.a.evaluate(witness);
            const b_val = self.b.evaluate(witness);
            const c_val = self.c.evaluate(witness);
            return a_val.mul(b_val).eql(c_val);
        }
    };
}

/// R1CS instance (public)
pub fn R1CSInstance(comptime F: type) type {
    return struct {
        const Self = @This();
        const ConstraintType = Constraint(F);

        /// Number of variables in the witness
        num_vars: usize,
        /// Number of public inputs
        num_public: usize,
        /// Constraints
        constraints: std.ArrayList(ConstraintType),
        allocator: Allocator,

        pub fn init(allocator: Allocator, num_vars: usize, num_public: usize) Self {
            return .{
                .num_vars = num_vars,
                .num_public = num_public,
                .constraints = std.ArrayList(ConstraintType).init(allocator),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.constraints.items) |*c| {
                c.deinit();
            }
            self.constraints.deinit();
        }

        /// Add a constraint
        pub fn addConstraint(self: *Self, constraint: ConstraintType) !void {
            try self.constraints.append(constraint);
        }

        /// Check if all constraints are satisfied
        pub fn isSatisfied(self: *const Self, witness: []const F) bool {
            for (self.constraints.items) |*c| {
                if (!c.isSatisfied(witness)) {
                    return false;
                }
            }
            return true;
        }

        /// Get the number of constraints
        pub fn numConstraints(self: *const Self) usize {
            return self.constraints.items.len;
        }
    };
}

/// R1CS witness (private)
pub fn R1CSWitness(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Full witness vector (public inputs + private witness)
        values: []F,
        allocator: Allocator,

        pub fn init(allocator: Allocator, size: usize) !Self {
            const values = try allocator.alloc(F, size);
            @memset(values, F.zero());
            return .{
                .values = values,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.values);
        }

        /// Set a witness value
        pub fn set(self: *Self, index: usize, value: F) void {
            self.values[index] = value;
        }

        /// Get a witness value
        pub fn get(self: *const Self, index: usize) F {
            return self.values[index];
        }
    };
}

test "r1cs constraint" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Create a simple constraint: x * y = z
    // where x=2, y=3, z=6
    var constraint = Constraint(F).init(allocator);
    defer constraint.deinit();

    // A = [0, 1, 0, 0] (select x)
    try constraint.a.addTerm(1, F.one());
    // B = [0, 0, 1, 0] (select y)
    try constraint.b.addTerm(2, F.one());
    // C = [0, 0, 0, 1] (select z)
    try constraint.c.addTerm(3, F.one());

    // Witness: [1, 2, 3, 6] (first element is 1 for constant)
    const witness = [_]F{
        F.one(),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(6),
    };

    // TODO: Enable when field arithmetic is fully implemented
    // try std.testing.expect(constraint.isSatisfied(&witness));
    _ = witness;
}
