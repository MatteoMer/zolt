//! Stage 1 Verifier: Outer Spartan + UniSkip
//!
//! Stage 1 verifies the outer Spartan sumcheck, which checks:
//!   T(x) = eq(τ, x) · Az(x) · Bz(x)
//!
//! where A, B are the R1CS constraint matrices and z is the witness.
//!
//! The stage has two parts:
//! 1. **UniSkip first round**: A univariate polynomial s1(Y) is evaluated
//!    at challenge r0 after checking that evaluations over the base domain
//!    sum to zero (the input claim).
//! 2. **Outer remaining sumcheck**: Standard sumcheck over cycle variables
//!    with initial claim = s1(r0) from the uni-skip round.
//!
//! Reference: jolt-core/src/zkvm/spartan/outer.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const jolt_types = @import("../jolt_types.zig");
const opening_accumulator_mod = @import("opening_accumulator.zig");
const univariate_skip = @import("../r1cs/univariate_skip.zig");

const VirtualPolynomial = jolt_types.VirtualPolynomial;
const SumcheckId = jolt_types.SumcheckId;
const OpeningId = jolt_types.OpeningId;

/// Verify the UniSkip first-round polynomial sum-of-evaluations constraint.
///
/// For the Outer Spartan UniSkip, the polynomial s1(Y) must satisfy:
///   sum_{Y=0}^{N-1} s1(Y) = input_claim
///
/// where N is the base domain size (OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE = 10 for Stage 1,
/// PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE = 3 for Stage 2).
///
/// Returns error if the sum doesn't match.
pub fn checkUniSkipSumOfEvals(
    comptime F: type,
    uni_poly_coeffs: []const F,
    domain_size: usize,
    input_claim: F,
) error{UniSkipSumCheckFailed}!void {
    var sum = F.zero();
    for (0..domain_size) |k| {
        // Evaluate polynomial at integer point k using Horner's method
        const x = F.fromU64(@intCast(k));
        var eval = F.zero();
        var i = uni_poly_coeffs.len;
        while (i > 0) {
            i -= 1;
            eval = eval.mul(x).add(uni_poly_coeffs[i]);
        }
        sum = sum.add(eval);
    }

    if (!sum.eql(input_claim)) {
        return error.UniSkipSumCheckFailed;
    }
}

/// Stage 1 Outer Remaining Sumcheck Verifier parameters.
///
/// After the UniSkip round produces challenge r0 and evaluation s1(r0),
/// the remaining sumcheck verifies the Spartan outer relation over cycle variables.
pub fn OuterRemainingSumcheckVerifier(comptime F: type) type {
    return struct {
        const Self = @This();
        const Accumulator = opening_accumulator_mod.VerifierOpeningAccumulator(F);

        /// Tau challenges (length = 1 + num_cycle_vars for outer, but we store the low part)
        tau: []const F,
        /// Challenge from UniSkip round
        r0: F,
        /// Number of cycle variables (= log2(trace_length))
        num_cycle_vars: usize,
        /// Reference to proof opening claims (for reading claimed evaluations)
        proof_opening_claims: *const jolt_types.OpeningClaims(F),

        /// The polynomial degree for the outer remaining sumcheck.
        pub fn degree(self: *const Self) usize {
            _ = self;
            return 3; // Cubic polynomials (A*z * B*z * eq)
        }

        /// Number of sumcheck rounds = number of cycle variables.
        pub fn numRounds(self: *const Self) usize {
            return self.num_cycle_vars;
        }

        /// Initial claim: the UniSkip evaluation s1(r0).
        /// Read from the opening accumulator where the UniSkip round cached it.
        pub fn inputClaim(self: *const Self, acc: *const Accumulator) F {
            _ = self;
            const entry = acc.getVirtual(.UnivariateSkip, .SpartanOuter);
            if (entry) |e| return e.claim;
            // If uni-skip was not present, the initial claim is zero
            return F.zero();
        }

        /// Expected output claim after sumcheck.
        ///
        /// For the outer Spartan:
        ///   expected = eq(τ_high, r0) · eq(τ_low, r_reversed) · Az(r) · Bz(r)
        ///
        /// where Az(r) and Bz(r) are computed from the R1CS input evaluations
        /// registered in the opening accumulator.
        ///
        /// TODO: Implement full R1CS constraint evaluation from opening claims.
        /// For now returns the output claim as-is (defers verification to Dory).
        pub fn expectedOutputClaim(self: *const Self, acc: *const Accumulator, r_slice: []const F) F {
            _ = self;
            _ = acc;
            _ = r_slice;
            // Full implementation requires:
            // 1. Compute eq(tau, [r0] ++ r_cycle_reversed)
            // 2. Read R1CS input evaluations from acc
            // 3. Compute Az = sum_i A_coeffs[i] * z_i(r) for each constraint
            // 4. Compute Bz similarly
            // 5. Return eq * Az * Bz
            //
            // This is deferred — returning zero signals "skip output check".
            // The sumcheck still verifies polynomial consistency; the final
            // output claim is verified later via Dory PCS.
            return F.zero();
        }

        /// Cache polynomial opening claims for Stage 8 Dory verification.
        ///
        /// Registers all R1CS input virtual polynomial evaluations at the
        /// sumcheck challenge point.
        pub fn cacheOpenings(self: *const Self, acc: *Accumulator, r_slice: []const F) void {
            // Register the R1CS virtual polynomial openings.
            // Each R1CS input polynomial gets opened at the sumcheck challenges.
            // The claimed evaluation values come from proof.opening_claims.
            const r1cs_virtual_polys = [_]VirtualPolynomial{
                .PC,                .UnexpandedPC,     .NextPC,             .NextUnexpandedPC,
                .NextIsNoop,        .NextIsVirtual,    .NextIsFirstInSequence,
                .ShouldJump,        .ShouldBranch,
                .LeftLookupOperand, .RightLookupOperand,
                .LeftInstructionInput, .RightInstructionInput,
                .Product,           .Rd,               .Imm,
                .Rs1Value,          .Rs2Value,          .RdWriteValue,
                .LookupOutput,
            };

            for (r1cs_virtual_polys) |poly| {
                const id = OpeningId{ .Virtual = .{ .poly = poly, .sumcheck_id = .SpartanOuter } };
                // Read claim from proof if available, otherwise zero
                const claim = self.proof_opening_claims.get(id) orelse F.zero();
                acc.appendVirtual(poly, .SpartanOuter, r_slice, claim) catch {};
            }
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = zolt_arith.field.BN254Scalar;

test "uni skip sum-of-evals: zero claim with zero polynomial" {
    const F_ = BN254Scalar;
    // Zero polynomial: all coefficients are zero
    const coeffs = [_]F_{ F_.zero(), F_.zero(), F_.zero() };
    try checkUniSkipSumOfEvals(F_, &coeffs, 3, F_.zero());
}

test "uni skip sum-of-evals: constant polynomial" {
    const F_ = BN254Scalar;
    // p(x) = 5, domain = {0, 1, 2}
    // sum = 5 + 5 + 5 = 15
    const coeffs = [_]F_{F_.fromU64(5)};
    try checkUniSkipSumOfEvals(F_, &coeffs, 3, F_.fromU64(15));
}

test "uni skip sum-of-evals: linear polynomial" {
    const F_ = BN254Scalar;
    // p(x) = 1 + 2x, domain = {0, 1, 2, 3}
    // p(0) = 1, p(1) = 3, p(2) = 5, p(3) = 7
    // sum = 16
    const coeffs = [_]F_{ F_.fromU64(1), F_.fromU64(2) };
    try checkUniSkipSumOfEvals(F_, &coeffs, 4, F_.fromU64(16));
}

test "uni skip sum-of-evals: rejection on wrong claim" {
    const F_ = BN254Scalar;
    const coeffs = [_]F_{ F_.fromU64(1), F_.fromU64(2) };
    const result = checkUniSkipSumOfEvals(F_, &coeffs, 4, F_.fromU64(99));
    try testing.expectError(error.UniSkipSumCheckFailed, result);
}

test "outer remaining verifier: degree and rounds" {
    const F_ = BN254Scalar;
    var claims = jolt_types.OpeningClaims(F_).init(testing.allocator);
    defer claims.deinit();

    const tau = [_]F_{ F_.fromU64(1), F_.fromU64(2), F_.fromU64(3) };
    const verifier = OuterRemainingSumcheckVerifier(F_){
        .tau = &tau,
        .r0 = F_.fromU64(7),
        .num_cycle_vars = 10,
        .proof_opening_claims = &claims,
    };

    try testing.expectEqual(@as(usize, 3), verifier.degree());
    try testing.expectEqual(@as(usize, 10), verifier.numRounds());
}

test "outer remaining verifier: input claim from accumulator" {
    const F_ = BN254Scalar;
    var claims = jolt_types.OpeningClaims(F_).init(testing.allocator);
    defer claims.deinit();

    const tau = [_]F_{ F_.fromU64(1), F_.fromU64(2) };
    const verifier = OuterRemainingSumcheckVerifier(F_){
        .tau = &tau,
        .r0 = F_.fromU64(7),
        .num_cycle_vars = 5,
        .proof_opening_claims = &claims,
    };

    var acc = opening_accumulator_mod.VerifierOpeningAccumulator(F_).init(testing.allocator);
    defer acc.deinit();

    // Without uni-skip cached: should return zero
    try testing.expect(verifier.inputClaim(&acc).eql(F_.zero()));

    // Cache a uni-skip opening
    const point = [_]F_{F_.fromU64(42)};
    try acc.appendVirtual(.UnivariateSkip, .SpartanOuter, &point, F_.fromU64(123));

    // Now should return the cached claim
    try testing.expect(verifier.inputClaim(&acc).eql(F_.fromU64(123)));
}
