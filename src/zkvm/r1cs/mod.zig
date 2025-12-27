//! R1CS constraint system for Jolt
//!
//! R1CS (Rank-1 Constraint System) represents constraints of the form:
//! (a · x) * (b · x) = (c · x)
//! where a, b, c are vectors of coefficients and x is the witness vector.
//!
//! This module includes:
//! - Basic R1CS data structures (Constraint, R1CSInstance, R1CSWitness)
//! - Uniform constraint definitions for Jolt (19 constraints per cycle)
//! - Witness generation from execution traces

const std = @import("std");
const Allocator = std.mem.Allocator;

// Export constraint generation
pub const constraints = @import("constraints.zig");
pub const R1CSInputIndex = constraints.R1CSInputIndex;
pub const UniformTerm = constraints.Term;
pub const LC = constraints.LC;
pub const UniformConstraint = constraints.UniformConstraint;
pub const UNIFORM_CONSTRAINTS = constraints.UNIFORM_CONSTRAINTS;
pub const R1CSCycleInputs = constraints.R1CSCycleInputs;
pub const R1CSWitnessGenerator = constraints.R1CSWitnessGenerator;

/// Term in a linear combination (field-specific version)
pub fn Term(comptime F: type) type {
    return struct {
        index: usize,
        coeff: F,
    };
}

/// A single R1CS constraint
pub fn Constraint(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Sparse representation of a linear combination
        pub const LinearCombination = struct {
            /// (variable_index, coefficient) pairs
            terms: []Term(F),
            len: usize,
            capacity: usize,
            allocator: Allocator,

            pub fn init(allocator: Allocator) LinearCombination {
                return .{
                    .terms = &[_]Term(F){},
                    .len = 0,
                    .capacity = 0,
                    .allocator = allocator,
                };
            }

            pub fn deinit(self: *LinearCombination) void {
                if (self.capacity > 0) {
                    self.allocator.free(self.terms[0..self.capacity]);
                }
            }

            /// Add a term to the linear combination
            pub fn addTerm(self: *LinearCombination, index: usize, coeff: F) !void {
                if (self.len >= self.capacity) {
                    const new_cap = if (self.capacity == 0) 4 else self.capacity * 2;
                    const new_terms = try self.allocator.alloc(Term(F), new_cap);
                    if (self.capacity > 0) {
                        @memcpy(new_terms[0..self.len], self.terms[0..self.len]);
                        self.allocator.free(self.terms[0..self.capacity]);
                    }
                    self.terms = new_terms;
                    self.capacity = new_cap;
                }
                self.terms[self.len] = .{ .index = index, .coeff = coeff };
                self.len += 1;
            }

            /// Evaluate the linear combination given a witness
            pub fn evaluate(self: *const LinearCombination, witness: []const F) F {
                var result = F.zero();
                for (self.terms[0..self.len]) |term| {
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
        constraints: []ConstraintType,
        constraints_len: usize,
        constraints_capacity: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, num_vars: usize, num_public: usize) Self {
            return .{
                .num_vars = num_vars,
                .num_public = num_public,
                .constraints = &[_]ConstraintType{},
                .constraints_len = 0,
                .constraints_capacity = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.constraints[0..self.constraints_len]) |*c| {
                c.deinit();
            }
            if (self.constraints_capacity > 0) {
                self.allocator.free(self.constraints[0..self.constraints_capacity]);
            }
        }

        /// Add a constraint
        pub fn addConstraint(self: *Self, constraint: ConstraintType) !void {
            if (self.constraints_len >= self.constraints_capacity) {
                const new_cap = if (self.constraints_capacity == 0) 8 else self.constraints_capacity * 2;
                const new_constraints = try self.allocator.alloc(ConstraintType, new_cap);
                if (self.constraints_capacity > 0) {
                    @memcpy(new_constraints[0..self.constraints_len], self.constraints[0..self.constraints_len]);
                    self.allocator.free(self.constraints[0..self.constraints_capacity]);
                }
                self.constraints = new_constraints;
                self.constraints_capacity = new_cap;
            }
            self.constraints[self.constraints_len] = constraint;
            self.constraints_len += 1;
        }

        /// Check if all constraints are satisfied
        pub fn isSatisfied(self: *const Self, witness: []const F) bool {
            for (self.constraints[0..self.constraints_len]) |*c| {
                if (!c.isSatisfied(witness)) {
                    return false;
                }
            }
            return true;
        }

        /// Get the number of constraints
        pub fn numConstraints(self: *const Self) usize {
            return self.constraints_len;
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

/// Dense matrix representation for R1CS (used in Spartan)
pub fn SparseMatrix(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Row indices
        row_indices: []usize,
        /// Column indices
        col_indices: []usize,
        /// Values
        values: []F,
        /// Number of non-zero entries
        nnz: usize,
        /// Number of rows
        num_rows: usize,
        /// Number of columns
        num_cols: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, num_rows: usize, num_cols: usize) Self {
            return .{
                .row_indices = &[_]usize{},
                .col_indices = &[_]usize{},
                .values = &[_]F{},
                .nnz = 0,
                .num_rows = num_rows,
                .num_cols = num_cols,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.nnz > 0) {
                self.allocator.free(self.row_indices);
                self.allocator.free(self.col_indices);
                self.allocator.free(self.values);
            }
        }

        /// Create from R1CS constraints
        pub fn fromR1CS(
            allocator: Allocator,
            instance: *const R1CSInstance(F),
            comptime which: enum { A, B, C },
        ) !Self {
            // Count non-zero entries
            var nnz: usize = 0;
            for (instance.constraints[0..instance.constraints_len]) |*c| {
                const lc = switch (which) {
                    .A => &c.a,
                    .B => &c.b,
                    .C => &c.c,
                };
                nnz += lc.len;
            }

            if (nnz == 0) {
                return Self.init(allocator, instance.constraints_len, instance.num_vars);
            }

            const row_indices = try allocator.alloc(usize, nnz);
            const col_indices = try allocator.alloc(usize, nnz);
            const values = try allocator.alloc(F, nnz);

            var idx: usize = 0;
            for (instance.constraints[0..instance.constraints_len], 0..) |*c, row| {
                const lc = switch (which) {
                    .A => &c.a,
                    .B => &c.b,
                    .C => &c.c,
                };
                for (lc.terms[0..lc.len]) |term| {
                    row_indices[idx] = row;
                    col_indices[idx] = term.index;
                    values[idx] = term.coeff;
                    idx += 1;
                }
            }

            return .{
                .row_indices = row_indices,
                .col_indices = col_indices,
                .values = values,
                .nnz = nnz,
                .num_rows = instance.constraints_len,
                .num_cols = instance.num_vars,
                .allocator = allocator,
            };
        }

        /// Multiply matrix by vector: result = M * v
        pub fn mulVec(self: *const Self, v: []const F, allocator: Allocator) ![]F {
            const result = try allocator.alloc(F, self.num_rows);
            @memset(result, F.zero());

            for (0..self.nnz) |i| {
                const row = self.row_indices[i];
                const col = self.col_indices[i];
                result[row] = result[row].add(self.values[i].mul(v[col]));
            }

            return result;
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

    try std.testing.expect(constraint.isSatisfied(&witness));
}

test "r1cs instance" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var instance = R1CSInstance(F).init(allocator, 4, 1);
    defer instance.deinit();

    // Add constraint: x * y = z
    var c1 = Constraint(F).init(allocator);
    try c1.a.addTerm(1, F.one());
    try c1.b.addTerm(2, F.one());
    try c1.c.addTerm(3, F.one());
    try instance.addConstraint(c1);

    const witness = [_]F{
        F.one(),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(6),
    };

    try std.testing.expect(instance.isSatisfied(&witness));
    try std.testing.expectEqual(@as(usize, 1), instance.numConstraints());
}
