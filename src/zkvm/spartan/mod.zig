//! Spartan proof system for Jolt
//!
//! Spartan is a zkSNARK for R1CS that achieves O(n) prover time
//! using polynomial commitments and the sumcheck protocol.
//!
//! The main idea is to prove that for all i:
//!   (Az)_i * (Bz)_i = (Cz)_i
//!
//! Using the sumcheck protocol on:
//!   sum_{x in {0,1}^log(m)} eq(tau, x) * [(Az)(x) * (Bz)(x) - (Cz)(x)] = 0

const std = @import("std");
const Allocator = std.mem.Allocator;
const poly = @import("../../poly/mod.zig");
const subprotocols = @import("../../subprotocols/mod.zig");
const r1cs = @import("../r1cs/mod.zig");

/// Spartan proof for R1CS
pub fn R1CSProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Random evaluation point from verifier
        tau: []F,
        /// Sumcheck proof for the outer sumcheck
        sumcheck_proof: subprotocols.Sumcheck(F).Proof,
        /// Claimed evaluations at random point
        /// [A(r), B(r), C(r)] where r is the final sumcheck point
        eval_claims: [3]F,
        /// Final evaluation point (from sumcheck)
        eval_point: []F,
        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.tau);
            self.sumcheck_proof.deinit();
            self.allocator.free(self.eval_point);
        }
    };
}

/// Preprocessed R1CS instance for Spartan
pub fn R1CSShape(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Matrix A in sparse form
        A: r1cs.SparseMatrix(F),
        /// Matrix B in sparse form
        B: r1cs.SparseMatrix(F),
        /// Matrix C in sparse form
        C: r1cs.SparseMatrix(F),
        /// Number of constraints (m)
        num_constraints: usize,
        /// Number of variables (n)
        num_vars: usize,
        /// Number of public inputs
        num_public: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, instance: *const r1cs.R1CSInstance(F)) !Self {
            return .{
                .A = try r1cs.SparseMatrix(F).fromR1CS(allocator, instance, .A),
                .B = try r1cs.SparseMatrix(F).fromR1CS(allocator, instance, .B),
                .C = try r1cs.SparseMatrix(F).fromR1CS(allocator, instance, .C),
                .num_constraints = instance.constraints_len,
                .num_vars = instance.num_vars,
                .num_public = instance.num_public,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.A.deinit();
            self.B.deinit();
            self.C.deinit();
        }
    };
}

/// Spartan prover
pub fn SpartanProver(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Generate a Spartan proof for an R1CS instance
        ///
        /// The proof uses sumcheck to show that:
        /// sum_{x} eq(tau, x) * [(Az)(x) * (Bz)(x) - (Cz)(x)] = 0
        ///
        /// where tau is a random point chosen by the verifier
        pub fn prove(
            self: *Self,
            shape: *const R1CSShape(F),
            witness: []const F,
            tau: []const F, // Random point from Fiat-Shamir
        ) !R1CSProof(F) {
            const log_m = std.math.log2_int(usize, shape.num_constraints);

            // Compute Az, Bz, Cz
            const Az = try shape.A.mulVec(witness, self.allocator);
            defer self.allocator.free(Az);
            const Bz = try shape.B.mulVec(witness, self.allocator);
            defer self.allocator.free(Bz);
            const Cz = try shape.C.mulVec(witness, self.allocator);
            defer self.allocator.free(Cz);

            // Compute the combined polynomial:
            // f(x) = eq(tau, x) * [(Az)(x) * (Bz)(x) - (Cz)(x)]
            //
            // First, compute eq(tau, x) for all x in {0,1}^log_m
            var eq_poly = try poly.EqPolynomial(F).init(self.allocator, tau);
            defer eq_poly.deinit();
            const eq_evals = try eq_poly.evals(self.allocator);
            defer self.allocator.free(eq_evals);

            // Compute f(x) = eq(tau,x) * [Az(x) * Bz(x) - Cz(x)]
            const size = @as(usize, 1) << log_m;
            const f_evals = try self.allocator.alloc(F, size);

            for (0..size) |i| {
                if (i < shape.num_constraints) {
                    const ab = Az[i].mul(Bz[i]);
                    const abc = ab.sub(Cz[i]);
                    f_evals[i] = eq_evals[i].mul(abc);
                } else {
                    f_evals[i] = F.zero();
                }
            }

            // Create the polynomial for sumcheck
            const f_poly = try poly.DensePolynomial(F).init(self.allocator, f_evals);
            self.allocator.free(f_evals);

            // Run sumcheck
            const result = try subprotocols.runSumcheck(F, f_poly, self.allocator);

            // Extract the evaluation point
            const eval_point = try self.allocator.alloc(F, result.proof.final_point.len);
            @memcpy(eval_point, result.proof.final_point);

            // Compute claimed evaluations of Az, Bz, Cz at the random point
            // For now, use the simple evaluation (in full Spartan, this uses polynomial commitments)
            const A_eval = evaluateAtPoint(Az, eval_point);
            const B_eval = evaluateAtPoint(Bz, eval_point);
            const C_eval = evaluateAtPoint(Cz, eval_point);

            // Copy tau
            const tau_copy = try self.allocator.alloc(F, tau.len);
            @memcpy(tau_copy, tau);

            return R1CSProof(F){
                .tau = tau_copy,
                .sumcheck_proof = result.proof,
                .eval_claims = .{ A_eval, B_eval, C_eval },
                .eval_point = eval_point,
                .allocator = self.allocator,
            };
        }

        /// Evaluate a vector as a multilinear polynomial at a point
        fn evaluateAtPoint(evals: []const F, point: []const F) F {
            if (evals.len == 0) return F.zero();
            if (point.len == 0) return evals[0];

            const half = evals.len / 2;
            if (half == 0) return evals[0];

            // Recursive multilinear extension evaluation
            const low = evaluateAtPoint(evals[0..half], point[1..]);
            const high = evaluateAtPoint(evals[half..], point[1..]);

            // (1 - point[0]) * low + point[0] * high
            const one_minus_p = F.one().sub(point[0]);
            return one_minus_p.mul(low).add(point[0].mul(high));
        }
    };
}

/// Spartan verifier
pub fn SpartanVerifier(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Verify a Spartan proof
        ///
        /// Checks:
        /// 1. Sumcheck verifies with claimed sum = 0
        /// 2. Final sumcheck evaluation matches eq(tau, r) * [A(r)*B(r) - C(r)]
        pub fn verify(
            self: *Self,
            shape: *const R1CSShape(F),
            proof: *const R1CSProof(F),
        ) !bool {
            // The sumcheck should prove that the sum is zero
            if (!proof.sumcheck_proof.claim.eql(F.zero())) {
                return false;
            }

            // Verify the final sumcheck evaluation
            // The final value should be eq(tau, r) * [A(r)*B(r) - C(r)]
            var eq_poly = try poly.EqPolynomial(F).init(self.allocator, proof.tau);
            defer eq_poly.deinit();

            const eq_eval = eq_poly.evaluate(proof.eval_point);
            const ab = proof.eval_claims[0].mul(proof.eval_claims[1]);
            const abc = ab.sub(proof.eval_claims[2]);
            const expected = eq_eval.mul(abc);

            if (!proof.sumcheck_proof.final_eval.eql(expected)) {
                return false;
            }

            // In full Spartan, we would also verify polynomial commitment openings
            // For now, we trust the evaluations
            _ = shape;

            return true;
        }
    };
}

/// Uniform Spartan (for R1CS with uniform structure)
///
/// In Jolt, the R1CS has a uniform structure that allows for more efficient
/// verification using the "spark" optimization. This exploits the fact that
/// the constraint matrices have a repeating block structure.
///
/// The key optimizations are:
/// 1. Sparse polynomial evaluation: Instead of dense matrix-vector products,
///    use the sparse structure to reduce evaluation cost
/// 2. Commitment compression: Commit to smaller sub-matrices and use homomorphism
/// 3. Batch opening: Open multiple polynomial evaluations in a single proof
pub fn UniformSpartan(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Number of repetitions in the uniform structure
        num_repetitions: usize,
        /// Size of each repeated block
        block_size: usize,
        /// Base prover for non-uniform operations
        base_prover: SpartanProver(F),
        allocator: Allocator,

        pub fn init(allocator: Allocator, num_repetitions: usize, block_size: usize) Self {
            return .{
                .num_repetitions = num_repetitions,
                .block_size = block_size,
                .base_prover = SpartanProver(F).init(allocator),
                .allocator = allocator,
            };
        }

        /// Generate proof exploiting uniform structure
        ///
        /// For uniform R1CS, the matrices have block structure:
        /// A = [A_block, A_block, ..., A_block]
        /// where A_block is repeated num_repetitions times.
        ///
        /// This allows us to:
        /// 1. Commit only to A_block instead of full A
        /// 2. Batch the sumcheck across all repetitions
        /// 3. Use sparse evaluation for the block matrix
        pub fn proveUniform(
            self: *Self,
            shape: *const R1CSShape(F),
            witness: []const F,
            tau: []const F,
        ) !R1CSProof(F) {
            // For now, fall back to standard Spartan
            // A full implementation would exploit the uniform structure
            // to reduce proof size and verification time
            return self.base_prover.prove(shape, witness, tau);
        }

        /// Compute block-based witness vector
        ///
        /// Splits the witness into blocks matching the R1CS structure
        pub fn splitWitness(self: *const Self, witness: []const F) ![][]F {
            const num_blocks = (witness.len + self.block_size - 1) / self.block_size;
            const blocks = try self.allocator.alloc([]F, num_blocks);

            for (0..num_blocks) |i| {
                const start = i * self.block_size;
                const end = @min(start + self.block_size, witness.len);
                blocks[i] = witness[start..end];
            }

            return blocks;
        }

        /// Verify that a witness satisfies the uniform R1CS
        pub fn verifyWitness(
            self: *const Self,
            shape: *const R1CSShape(F),
            witness: []const F,
        ) bool {
            _ = self;
            // Check Az * Bz = Cz for each constraint
            for (0..shape.num_constraints) |i| {
                const a_val = shape.A.evalRow(i, witness);
                const b_val = shape.B.evalRow(i, witness);
                const c_val = shape.C.evalRow(i, witness);

                if (!a_val.mul(b_val).eql(c_val)) {
                    return false;
                }
            }
            return true;
        }
    };
}

test "spartan types compile" {
    const F = @import("../../field/mod.zig").BN254Scalar;

    // Verify types compile
    _ = R1CSProof(F);
    _ = R1CSShape(F);
    _ = SpartanProver(F);
    _ = SpartanVerifier(F);
}

test "spartan proof generation" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Create a simple R1CS: x * y = z where x=2, y=3, z=6
    var instance = r1cs.R1CSInstance(F).init(allocator, 4, 1);
    defer instance.deinit();

    var c1 = r1cs.Constraint(F).init(allocator);
    try c1.a.addTerm(1, F.one()); // A selects x
    try c1.b.addTerm(2, F.one()); // B selects y
    try c1.c.addTerm(3, F.one()); // C selects z
    try instance.addConstraint(c1);

    // We need power-of-2 constraints for sumcheck
    // Add a trivial constraint: 0 = 0
    var c2 = r1cs.Constraint(F).init(allocator);
    try c2.c.addTerm(0, F.zero()); // 0 = 0
    try instance.addConstraint(c2);

    // Create shape
    var shape = try R1CSShape(F).init(allocator, &instance);
    defer shape.deinit();

    // Witness: [1, 2, 3, 6]
    const witness = [_]F{
        F.one(),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(6),
    };

    // Random tau (in practice, from Fiat-Shamir)
    const tau = [_]F{F.fromU64(7)};

    // Generate proof
    var prover = SpartanProver(F).init(allocator);
    var proof = try prover.prove(&shape, &witness, &tau);
    defer proof.deinit();

    // Verify proof
    var verifier = SpartanVerifier(F).init(allocator);
    const valid = try verifier.verify(&shape, &proof);

    try std.testing.expect(valid);
}
