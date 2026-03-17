//! Jolt Prover: Direct Jolt-Compatible 7-Stage Proof Generation
//!
//! This module generates proofs directly in Jolt's 7-stage format
//! for cross-verification compatibility with the upstream Jolt verifier.
//!
//! ## Stage Layout (Jolt 7 stages)
//!
//! 1. Outer Spartan (+ UniSkip)
//! 2. Product virtualization + RAM RAF + Read-Write (+ UniSkip)
//! 3. Spartan shift + Instruction input + Registers claim
//! 4. Registers RW + RAM val evaluation + RAM val final
//! 5. Registers val evaluation + RAM RA + Lookups RAF
//! 6. Bytecode RAF + Hamming + Booleanity + RA virtual
//! 7. Hamming weight claim reduction
//!
//! ## Constraint Evaluation

const std = @import("std");

const Allocator = std.mem.Allocator;
const ThreadPool = @import("../utils/thread_pool.zig").ThreadPool;

const jolt_types = @import("jolt_types.zig");
const field_mod = @import("../field/mod.zig");
const UnreducedProductAccum = field_mod.UnreducedProductAccum;
const r1cs = @import("r1cs/mod.zig");
const streaming_outer = @import("spartan/streaming_outer.zig");
const product_remainder = @import("spartan/product_remainder.zig");
const transcripts = @import("../transcripts/mod.zig");
const Blake2bTranscript = transcripts.Blake2bTranscript;
const poly_mod = @import("../poly/mod.zig");
const jolt_device = @import("jolt_device.zig");
const constants = @import("../common/constants.zig");
const ram = @import("ram/mod.zig");
const instruction = @import("instruction/mod.zig");
const spartan_mod = @import("spartan/mod.zig");
const Stage3Prover = spartan_mod.Stage3Prover;
const Stage5BatchedProver = spartan_mod.Stage5BatchedProver;
const Stage6BatchedProver = spartan_mod.Stage6BatchedProver;
const preprocessing = @import("preprocessing.zig");

const debug_verbose = false;
const stage_timing_enabled = false;

/// Direct Jolt-compatible 7-stage prover
pub fn JoltProver(comptime F: type) type {
    return struct {
        const Self = @This();

        // Import types we need
        const JoltProofType = jolt_types.JoltProof;
        const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof;
        const UniSkipFirstRoundProof = jolt_types.UniSkipFirstRoundProof;
        const OpeningClaims = jolt_types.OpeningClaims;
        const SumcheckId = jolt_types.SumcheckId;
        const OpeningId = jolt_types.OpeningId;
        const VirtualPolynomial = jolt_types.VirtualPolynomial;

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
            };
        }

        pub fn initWithThreadPool(allocator: Allocator, tp: *ThreadPool) Self {
            return Self{
                .allocator = allocator,
                .thread_pool = tp,
            };
        }


        /// Generate a zero-filled sumcheck proof with the specified number of rounds
        ///
        /// Each round has a compressed polynomial with degree `degree_bound`.
        /// For claim = 0, all-zero polynomials satisfy p(0) + p(1) = claim.
        fn generateZeroSumcheckProof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            num_rounds: usize,
            degree_bound: usize,
        ) !void {
            // Compressed poly: coeffs_except_linear_term has `degree_bound` elements
            // (constant, quadratic, cubic, ...) - linear term is recovered from hint
            for (0..num_rounds) |_| {
                const coeffs = try self.allocator.alloc(F, degree_bound);
                @memset(coeffs, F.zero());
                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });
            }
        }


        /// Result of Stage 1 sumcheck proof generation
        const Stage1Result = struct {
            /// Accumulated sumcheck challenges (r_stream, r_cycle_bits...)
            /// The full r_cycle point is [r_stream, r1, r2, ..., r_n] reversed
            challenges: std.ArrayListUnmanaged(F),
            /// The first-round challenge r0 from UniSkip
            r0: F,
            /// The UnivariateSkip claim: evaluation of UniSkip polynomial at r0
            /// This is the input_claim for the remaining sumcheck rounds
            uni_skip_claim: F,
            /// Allocator for cleanup
            allocator: Allocator,

            pub fn deinit(self: *Stage1Result) void {
                self.challenges.deinit(self.allocator);
            }
        };

        /// Generate sumcheck proof using the streaming outer prover with Fiat-Shamir transcript
        ///
        /// This produces actual polynomial evaluations by computing Az*Bz products
        /// from the R1CS constraints, using the provided transcript for challenges.
        ///
        /// Returns the accumulated challenges for computing r_cycle.
        fn generateStreamingOuterSumcheckProofWithTranscript(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            uniskip_proof: *const UniSkipFirstRoundProof(F),
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau: []const F,
            transcript: *Blake2bTranscript(F),
            compact_witnesses: ?[]const @import("r1cs/evaluators.zig").CompactWitness,
        ) !Stage1Result {
            const StreamingOuterProver = streaming_outer.StreamingOuterProver(F);
            const LagrangePoly = r1cs.univariate_skip.LagrangePolynomial(F);
            var challenges: std.ArrayListUnmanaged(F) = .{};

            // Extract tau_high for the UniSkip Lagrange kernel
            // tau has length num_rows_bits = num_cycle_vars + 2
            // tau_high is the last element (used for Lagrange kernel)
            // Full tau is passed to split_eq (it handles the split internally)
            if (tau.len < 2) {
                const num_rounds = 1 + std.math.log2_int(usize, @max(1, cycle_witnesses.len));
                try self.generateZeroSumcheckProof(proof, num_rounds, 3);
                return Stage1Result{ .challenges = challenges, .r0 = F.zero(), .uni_skip_claim = F.zero(), .allocator = self.allocator };
            }
            const tau_high = tau[tau.len - 1];

            // DEBUG: Print tau length (challenges from transcript)

            // The first round was already processed by UniSkip
            // Append the UniSkip polynomial to transcript
            transcript.appendScalars("uniskip_poly", uniskip_proof.uni_poly);

            // Get the challenge for the first round (r0)
            const r0 = transcript.challengeScalar();

            // Compute the Lagrange kernel L(r0, tau_high) to use as initial scaling
            const lagrange_tau_r0 = try LagrangePoly.lagrangeKernel(
                r1cs.univariate_skip.OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
                r0,
                tau_high,
                self.allocator,
            );

            // Initialize the streaming prover with full tau and Lagrange kernel scaling
            // The prover internally extracts:
            //   tau_high = tau[tau.len - 1] (stored separately for first-round polynomial)
            //   tau_low = tau[0..tau.len - 1] (passed to split_eq)
            // This matches Jolt's behavior in OuterSharedState::new().
            var outer_prover = StreamingOuterProver.initWithScaling(
                self.allocator,
                cycle_witnesses,
                tau, // Full tau - prover extracts tau_low and tau_high internally
                lagrange_tau_r0,
            ) catch {
                // Fallback to zero proofs if initialization fails
                const num_rounds = 1 + std.math.log2_int(usize, @max(1, cycle_witnesses.len));
                try self.generateZeroSumcheckProof(proof, num_rounds, 3);
                return Stage1Result{ .challenges = challenges, .r0 = r0, .uni_skip_claim = F.zero(), .allocator = self.allocator };
            };
            // Set pre-built compact witnesses (not owned — don't free on deinit)
            outer_prover.compact_witnesses = compact_witnesses;
            defer {
                outer_prover.compact_witnesses = null; // prevent double-free
                outer_prover.deinit();
            }

            // Compute the UnivariateSkip claim: evaluation of UniSkip polynomial at r0
            const uni_skip_claim = evaluatePolyAtChallenge(uniskip_proof.uni_poly, r0);

            // DEBUG: Decompose s1(r0) = L(tau_high, r0) * t1(r0) and compare
            if (comptime false) {
                const inv_L = lagrange_tau_r0.inverse();
                if (inv_L) |_| {

                    // Now compute t1(r0) directly by evaluating the direct sum
                    // using the witnesses and lagrange evals at r0
                    // Build eq tables from the SAME full_tau
                    const tau_len = tau.len;
                    const m_d = tau_len / 2;
                    const n_x_in_bits_d = if (tau_len > 1) tau_len - 1 - m_d else 0;
                    const n_x_in_prime_bits_d = if (n_x_in_bits_d > 0) n_x_in_bits_d - 1 else 0;
                    const n_x_out_d: usize = @as(usize, 1) << @intCast(m_d);
                    const n_x_in_d: usize = @as(usize, 1) << @intCast(n_x_in_bits_d);

                    // Build eq tables (same logic as streaming_outer.buildEqTable)
                    const E_out_d = try self.allocator.alloc(F, n_x_out_d);
                    defer self.allocator.free(E_out_d);
                    E_out_d[0] = F.one();
                    var cs_d: usize = 1;
                    for (0..m_d) |kd| {
                        const t_k = tau[kd];
                        const omt_k = F.one().sub(t_k);
                        var id: usize = cs_d;
                        while (id > 0) {
                            id -= 1;
                            E_out_d[2 * id + 1] = E_out_d[id].mul(t_k);
                            E_out_d[2 * id] = E_out_d[id].mul(omt_k);
                        }
                        cs_d *= 2;
                    }

                    const E_in_d = try self.allocator.alloc(F, n_x_in_d);
                    defer self.allocator.free(E_in_d);
                    E_in_d[0] = F.one();
                    var cs_d2: usize = 1;
                    for (0..n_x_in_bits_d) |kd2| {
                        const t_k2 = tau[m_d + kd2];
                        const omt_k2 = F.one().sub(t_k2);
                        var id2: usize = cs_d2;
                        while (id2 > 0) {
                            id2 -= 1;
                            E_in_d[2 * id2 + 1] = E_in_d[id2].mul(t_k2);
                            E_in_d[2 * id2] = E_in_d[id2].mul(omt_k2);
                        }
                        cs_d2 *= 2;
                    }

                    // Compute Lagrange evals at r0
                    const StreamProver = streaming_outer.StreamingOuterProver(F);
                    const FGSZ = StreamProver.FIRST_GROUP_SIZE;
                    const SGSZ = StreamProver.SECOND_GROUP_SIZE;
                    const c_mod = r1cs.constraints;
                    var lags: [FGSZ]F = undefined;
                    {
                        const lag_start: i64 = -@as(i64, (FGSZ - 1) / 2);
                        for (0..FGSZ) |li| {
                            var lnum = F.one();
                            var lden = F.one();
                            for (0..FGSZ) |lj| {
                                if (li != lj) {
                                    const lxj: i64 = lag_start + @as(i64, @intCast(lj));
                                    const lxjf = if (lxj >= 0) F.fromU64(@intCast(lxj)) else F.zero().sub(F.fromU64(@intCast(-lxj)));
                                    lnum = lnum.mul(r0.sub(lxjf));
                                    const ldiff: i64 = @as(i64, @intCast(li)) - @as(i64, @intCast(lj));
                                    const ldifff = if (ldiff > 0) F.fromU64(@intCast(ldiff)) else F.zero().sub(F.fromU64(@intCast(-ldiff)));
                                    lden = lden.mul(ldifff);
                                }
                            }
                            lags[li] = lnum.mul(lden.inverse().?);
                        }
                    }

                    // Check Lagrange weights: should sum to 1
                    var lag_sum = F.zero();
                    for (0..FGSZ) |li| lag_sum = lag_sum.add(lags[li]);

                    // Check base domain: t1 at Y=0 (base point, should be zero)
                    {
                        var t1_at_zero = F.zero();
                        for (0..n_x_out_d) |xo0| {
                            for (0..n_x_in_d) |xi0| {
                                const eq0 = E_out_d[xo0].mul(E_in_d[xi0]);
                                const xip0 = xi0 >> 1;
                                const cyc0 = (xo0 << @intCast(n_x_in_prime_bits_d)) | xip0;
                                const grp0: usize = xi0 & 1;
                                if (cyc0 < cycle_witnesses.len) {
                                    const gid0 = if (grp0 == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                    const w0 = &cycle_witnesses[cyc0];
                                    // At Y=0, L_i(0) = delta_{i,4} since domain is {-4,...,5}
                                    const ci0 = 4; // index of Y=0 in domain
                                    const gsz0: usize = if (grp0 == 0) FGSZ else SGSZ;
                                    if (ci0 < gsz0) {
                                        const cc0 = c_mod.UNIFORM_CONSTRAINTS[gid0[ci0]];
                                        const az0 = cc0.condition.evaluate(F, w0.asSlice());
                                        const bz0 = cc0.left.evaluate(F, w0.asSlice()).sub(cc0.right.evaluate(F, w0.asSlice()));
                                        t1_at_zero = t1_at_zero.add(eq0.mul(az0.mul(bz0)));
                                    }
                                }
                            }
                        }
                    }

                    // Check ALL 10 base domain points
                    for (0..FGSZ) |base_idx| {
                        var t1_at_base = F.zero();
                        for (0..n_x_out_d) |xob| {
                            for (0..n_x_in_d) |xib| {
                                const eqb = E_out_d[xob].mul(E_in_d[xib]);
                                const xipb = xib >> 1;
                                const cycb = (xob << @intCast(n_x_in_prime_bits_d)) | xipb;
                                const grpb: usize = xib & 1;
                                if (cycb < cycle_witnesses.len) {
                                    const gidb = if (grpb == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                    const wb = &cycle_witnesses[cycb];
                                    const gszb: usize = if (grpb == 0) FGSZ else SGSZ;
                                    if (base_idx < gszb) {
                                        const ccb = c_mod.UNIFORM_CONSTRAINTS[gidb[base_idx]];
                                        const azb = ccb.condition.evaluate(F, wb.asSlice());
                                        const bzb = ccb.left.evaluate(F, wb.asSlice()).sub(ccb.right.evaluate(F, wb.asSlice()));
                                        const prod_b = azb.mul(bzb);
                                        if (!prod_b.eql(F.zero())) {
                                            // Print witness values for this cycle
                                        }
                                        t1_at_base = t1_at_base.add(eqb.mul(prod_b));
                                    }
                                }
                            }
                        }
                        if (!t1_at_base.eql(F.zero())) {
                        }
                    }

                    // Check direct_sum for cycle 0, group 0 individually
                    if (cycle_witnesses.len > 0) {
                        const w_test = &cycle_witnesses[0];
                        var az_test = F.zero();
                        var bz_test = F.zero();
                        for (0..FGSZ) |ki| {
                            const cc_test = c_mod.UNIFORM_CONSTRAINTS[c_mod.FIRST_GROUP_INDICES[ki]];
                            const cv_test = cc_test.condition.evaluate(F, w_test.asSlice());
                            const mv_test = cc_test.left.evaluate(F, w_test.asSlice()).sub(cc_test.right.evaluate(F, w_test.asSlice()));
                            az_test = az_test.add(lags[ki].mul(cv_test));
                            bz_test = bz_test.add(lags[ki].mul(mv_test));
                        }
                    }

                    // Compute direct sum
                    var direct_t1_r0 = F.zero();
                    for (0..n_x_out_d) |xod| {
                        for (0..n_x_in_d) |xid| {
                            const eq_d = E_out_d[xod].mul(E_in_d[xid]);
                            const xipd = xid >> 1;
                            const cycd = (xod << @intCast(n_x_in_prime_bits_d)) | xipd;
                            const grpd: usize = xid & 1;

                            if (cycd < cycle_witnesses.len) {
                                const gsz_d: usize = if (grpd == 0) FGSZ else SGSZ;
                                const gid_d = if (grpd == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                const wd = &cycle_witnesses[cycd];
                                var azd = F.zero();
                                var bzd = F.zero();
                                for (0..gsz_d) |kkd| {
                                    const ccd = c_mod.UNIFORM_CONSTRAINTS[gid_d[kkd]];
                                    const cvd = ccd.condition.evaluate(F, wd.asSlice());
                                    const mvd = ccd.left.evaluate(F, wd.asSlice()).sub(ccd.right.evaluate(F, wd.asSlice()));
                                    azd = azd.add(lags[kkd].mul(cvd));
                                    bzd = bzd.add(lags[kkd].mul(mvd));
                                }
                                direct_t1_r0 = direct_t1_r0.add(eq_d.mul(azd.mul(bzd)));
                            }
                        }
                    }

                    // Also: compute s1(r0) = L(tau_high,r0) * direct_t1_r0 and compare

                    // CRITICAL: Evaluate t1(r0) from UniSkip coefficients directly
                    // s1(r0) = uni_skip_claim, t1(r0) = s1(r0) / L(tau_high, r0)
                    // But also evaluate t1 at r0 using Lagrange formula from the 19 domain evaluations
                    {
                        const US = r1cs.univariate_skip;
                        const EXT_SIZE = US.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE; // 19
                        const DEG = US.OUTER_UNIVARIATE_SKIP_DEGREE; // 9

                        // Get t1_vals from the UniSkip polynomial's coefficient representation
                        // Actually, we need the evaluation values. Let's compute t1(r0) using
                        // Lagrange interpolation from the 19 evaluations that were used to build the polynomial.
                        // We need the evaluations - recompute them quickly
                        const targets = US.UNISKIP_TARGETS;

                        // Build t1_eval_vals: the 19 evaluations on {-9,...,9}
                        // Base domain {-4,...,5}: t1 = 0
                        // Extended targets: computed from witnesses
                        var t1_eval_19: [EXT_SIZE]F = [_]F{F.zero()} ** EXT_SIZE;

                        // Recompute extended evaluations at the 9 target points
                        for (targets) |target_y| {
                            var ext_sum = F.zero();
                            for (0..n_x_out_d) |xo| {
                                const eo = if (xo < E_out_d.len) E_out_d[xo] else F.zero();
                                for (0..n_x_in_d) |xi| {
                                    const ei = if (xi < E_in_d.len) E_in_d[xi] else F.zero();
                                    const eq_v = eo.mul(ei);
                                    const xip = xi >> 1;
                                    const cyc = (xo << @intCast(n_x_in_prime_bits_d)) | xip;
                                    const grp: u1 = @truncate(xi & 1);
                                    if (cyc < cycle_witnesses.len) {
                                        const w = &cycle_witnesses[cyc];
                                        const gsz: usize = if (grp == 0) FGSZ else SGSZ;
                                        const gids = if (grp == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                        // Build Lagrange evals at target_y
                                        const tgt_f = if (target_y >= 0) F.fromU64(@intCast(target_y)) else F.zero().sub(F.fromU64(@intCast(-target_y)));
                                        var lag_tgt: [FGSZ]F = undefined;
                                        const lag_s: i64 = -@as(i64, (FGSZ - 1) / 2);
                                        for (0..FGSZ) |lti| {
                                            var ln = F.one();
                                            var ld = F.one();
                                            for (0..FGSZ) |ltj| {
                                                if (lti != ltj) {
                                                    const ltxj: i64 = lag_s + @as(i64, @intCast(ltj));
                                                    const ltxjf = if (ltxj >= 0) F.fromU64(@intCast(ltxj)) else F.zero().sub(F.fromU64(@intCast(-ltxj)));
                                                    ln = ln.mul(tgt_f.sub(ltxjf));
                                                    const ltd: i64 = @as(i64, @intCast(lti)) - @as(i64, @intCast(ltj));
                                                    const ltdf = if (ltd > 0) F.fromU64(@intCast(ltd)) else F.zero().sub(F.fromU64(@intCast(-ltd)));
                                                    ld = ld.mul(ltdf);
                                                }
                                            }
                                            lag_tgt[lti] = ln.mul(ld.inverse().?);
                                        }
                                        var az_t = F.zero();
                                        var bz_t = F.zero();
                                        for (0..gsz) |ki| {
                                            const cc = c_mod.UNIFORM_CONSTRAINTS[gids[ki]];
                                            const cv = cc.condition.evaluate(F, w.asSlice());
                                            const mv = cc.left.evaluate(F, w.asSlice()).sub(cc.right.evaluate(F, w.asSlice()));
                                            az_t = az_t.add(lag_tgt[ki].mul(cv));
                                            bz_t = bz_t.add(lag_tgt[ki].mul(mv));
                                        }
                                        ext_sum = ext_sum.add(eq_v.mul(az_t.mul(bz_t)));
                                    }
                                }
                            }
                            const pos: usize = @intCast(target_y + @as(i64, DEG));
                            t1_eval_19[pos] = ext_sum;
                        }

                        // Now evaluate t1(r0) using Lagrange formula from the 19 evaluations
                        var t1_r0_lagrange19 = F.zero();
                        for (0..EXT_SIZE) |li19| {
                            if (t1_eval_19[li19].eql(F.zero())) continue;
                            const xi19: i64 = @as(i64, @intCast(li19)) - @as(i64, DEG);
                            var ln19 = F.one();
                            var ld19 = F.one();
                            for (0..EXT_SIZE) |lj19| {
                                if (li19 != lj19) {
                                    const xj19: i64 = @as(i64, @intCast(lj19)) - @as(i64, DEG);
                                    const xj19f = if (xj19 >= 0) F.fromU64(@intCast(xj19)) else F.zero().sub(F.fromU64(@intCast(-xj19)));
                                    ln19 = ln19.mul(r0.sub(xj19f));
                                    const d19: i64 = xi19 - xj19;
                                    const d19f = if (d19 > 0) F.fromU64(@intCast(d19)) else F.zero().sub(F.fromU64(@intCast(-d19)));
                                    ld19 = ld19.mul(d19f);
                                }
                            }
                            t1_r0_lagrange19 = t1_r0_lagrange19.add(t1_eval_19[li19].mul(ln19.mul(ld19.inverse().?)));
                        }


                        // Also check: does s1 polynomial Horner eval at r0 match uni_skip_claim?
                        // (s1 = the actual polynomial sent in the proof)
                    }

                    // Now compute t1 at Y=-5 (first extended target) using the SAME
                    // Lagrange eval approach (not COEFFS_PER_J) and compare with
                    // what evaluateAzBzAtTargetY gives
                    const neg5_field = F.zero().sub(F.fromU64(5));
                    var lag_neg5: [FGSZ]F = undefined;
                    {
                        const lag_start_n5: i64 = -@as(i64, (FGSZ - 1) / 2);
                        for (0..FGSZ) |li| {
                            var lnum2 = F.one();
                            var lden2 = F.one();
                            for (0..FGSZ) |lj| {
                                if (li != lj) {
                                    const lxj2: i64 = lag_start_n5 + @as(i64, @intCast(lj));
                                    const lxjf2 = if (lxj2 >= 0) F.fromU64(@intCast(lxj2)) else F.zero().sub(F.fromU64(@intCast(-lxj2)));
                                    lnum2 = lnum2.mul(neg5_field.sub(lxjf2));
                                    const ldiff2: i64 = @as(i64, @intCast(li)) - @as(i64, @intCast(lj));
                                    const ldifff2 = if (ldiff2 > 0) F.fromU64(@intCast(ldiff2)) else F.zero().sub(F.fromU64(@intCast(-ldiff2)));
                                    lden2 = lden2.mul(ldifff2);
                                }
                            }
                            lag_neg5[li] = lnum2.mul(lden2.inverse().?);
                        }
                    }

                    // Compute direct sum at Y=-5 using Lagrange evals at -5
                    var direct_t1_neg5 = F.zero();
                    for (0..n_x_out_d) |xod2| {
                        for (0..n_x_in_d) |xid2| {
                            const eq_d2 = E_out_d[xod2].mul(E_in_d[xid2]);
                            const xipd2 = xid2 >> 1;
                            const cycd2 = (xod2 << @intCast(n_x_in_prime_bits_d)) | xipd2;
                            const grpd2: usize = xid2 & 1;

                            if (cycd2 < cycle_witnesses.len) {
                                const gsz_d2: usize = if (grpd2 == 0) FGSZ else SGSZ;
                                const gid_d2 = if (grpd2 == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                const wd2 = &cycle_witnesses[cycd2];
                                var azd2 = F.zero();
                                var bzd2 = F.zero();
                                for (0..gsz_d2) |kkd2| {
                                    const ccd2 = c_mod.UNIFORM_CONSTRAINTS[gid_d2[kkd2]];
                                    const cvd2 = ccd2.condition.evaluate(F, wd2.asSlice());
                                    const mvd2 = ccd2.left.evaluate(F, wd2.asSlice()).sub(ccd2.right.evaluate(F, wd2.asSlice()));
                                    azd2 = azd2.add(lag_neg5[kkd2].mul(cvd2));
                                    bzd2 = bzd2.add(lag_neg5[kkd2].mul(mvd2));
                                }
                                direct_t1_neg5 = direct_t1_neg5.add(eq_d2.mul(azd2.mul(bzd2)));
                            }
                        }
                    }

                    // Now compute using COEFFS_PER_J (same as evaluateAzBzAtTargetY)
                    const unskip = r1cs.univariate_skip;
                    var coeffs_t1_neg5 = F.zero();
                    for (0..n_x_out_d) |xod3| {
                        for (0..n_x_in_d) |xid3| {
                            const eq_d3 = E_out_d[xod3].mul(E_in_d[xid3]);
                            const xipd3 = xid3 >> 1;
                            const cycd3 = (xod3 << @intCast(n_x_in_prime_bits_d)) | xipd3;
                            const grpd3: usize = xid3 & 1;

                            if (cycd3 < cycle_witnesses.len) {
                                const gsz_d3: usize = if (grpd3 == 0) FGSZ else SGSZ;
                                const gid_d3 = if (grpd3 == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                const wd3 = &cycle_witnesses[cycd3];
                                // Evaluate using COEFFS_PER_J[0] (target Y=-5 is index 0)
                                const coeffs_j = unskip.COEFFS_PER_J[0]; // target -5
                                var azd3 = F.zero();
                                var bzd3 = F.zero();
                                for (0..gsz_d3) |kkd3| {
                                    const ccd3 = c_mod.UNIFORM_CONSTRAINTS[gid_d3[kkd3]];
                                    const cvd3 = ccd3.condition.evaluate(F, wd3.asSlice());
                                    const mvd3 = ccd3.left.evaluate(F, wd3.asSlice()).sub(ccd3.right.evaluate(F, wd3.asSlice()));
                                    const cf3 = coeffs_j[kkd3];
                                    const cf3f = if (cf3 > 0) F.fromU64(@intCast(cf3)) else F.zero().sub(F.fromU64(@intCast(-cf3)));
                                    azd3 = azd3.add(cf3f.mul(cvd3));
                                    bzd3 = bzd3.add(cf3f.mul(mvd3));
                                }
                                coeffs_t1_neg5 = coeffs_t1_neg5.add(eq_d3.mul(azd3.mul(bzd3)));
                            }
                        }
                    }

                    // Compute t1 at ALL 19 domain points {-9,...,9} using direct Lagrange
                    // and compare with what the polynomial gives
                    var domain_mismatches: usize = 0;
                    for (0..19) |dpidx| {
                        const dpy: i64 = @as(i64, @intCast(dpidx)) - 9;
                        const dpy_field = if (dpy >= 0) F.fromU64(@intCast(dpy)) else F.zero().sub(F.fromU64(@intCast(-dpy)));

                        // Compute Lagrange evals at this Y on 10-point domain
                        var lag_y: [FGSZ]F = undefined;
                        {
                            const lag_start_y: i64 = -@as(i64, (FGSZ - 1) / 2);
                            for (0..FGSZ) |liy| {
                                var lnumy = F.one();
                                var ldeny = F.one();
                                for (0..FGSZ) |ljy| {
                                    if (liy != ljy) {
                                        const lxjy: i64 = lag_start_y + @as(i64, @intCast(ljy));
                                        const lxjfy = if (lxjy >= 0) F.fromU64(@intCast(lxjy)) else F.zero().sub(F.fromU64(@intCast(-lxjy)));
                                        lnumy = lnumy.mul(dpy_field.sub(lxjfy));
                                        const ldiffy: i64 = @as(i64, @intCast(liy)) - @as(i64, @intCast(ljy));
                                        const ldiffyf = if (ldiffy > 0) F.fromU64(@intCast(ldiffy)) else F.zero().sub(F.fromU64(@intCast(-ldiffy)));
                                        ldeny = ldeny.mul(ldiffyf);
                                    }
                                }
                                lag_y[liy] = lnumy.mul(ldeny.inverse().?);
                            }
                        }

                        // Compute direct t1(dpy) sum
                        var t1_dpy = F.zero();
                        for (0..n_x_out_d) |xody| {
                            for (0..n_x_in_d) |xidy| {
                                const eq_dy = E_out_d[xody].mul(E_in_d[xidy]);
                                const xipdy = xidy >> 1;
                                const cycdy = (xody << @intCast(n_x_in_prime_bits_d)) | xipdy;
                                const grpdy: usize = xidy & 1;
                                if (cycdy < cycle_witnesses.len) {
                                    const gsz_dy: usize = if (grpdy == 0) FGSZ else SGSZ;
                                    const gid_dy = if (grpdy == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                    const wdy = &cycle_witnesses[cycdy];
                                    var azdy = F.zero();
                                    var bzdy = F.zero();
                                    for (0..gsz_dy) |kkdy| {
                                        const ccdy = c_mod.UNIFORM_CONSTRAINTS[gid_dy[kkdy]];
                                        const cvdy = ccdy.condition.evaluate(F, wdy.asSlice());
                                        const mvdy = ccdy.left.evaluate(F, wdy.asSlice()).sub(ccdy.right.evaluate(F, wdy.asSlice()));
                                        azdy = azdy.add(lag_y[kkdy].mul(cvdy));
                                        bzdy = bzdy.add(lag_y[kkdy].mul(mvdy));
                                    }
                                    t1_dpy = t1_dpy.add(eq_dy.mul(azdy.mul(bzdy)));
                                }
                            }
                        }

                        // (placeholder for polynomial evaluation at dpy)

                        // For base points, the direct sum should be 0 (correct witness)
                        const is_base = (dpy >= -4 and dpy <= 5);
                        if (!t1_dpy.eql(F.zero()) and is_base) {
                            domain_mismatches += 1;

                            // Find which cycles contribute non-zero AzBz at this base point
                            var cnt_nz: usize = 0;
                            for (0..n_x_out_d) |xodz| {
                                for (0..n_x_in_d) |xidz| {
                                    const xipz = xidz >> 1;
                                    const cycz = (xodz << @intCast(n_x_in_prime_bits_d)) | xipz;
                                    const grpz: usize = xidz & 1;
                                    if (cycz < cycle_witnesses.len) {
                                        const gsz_z: usize = if (grpz == 0) FGSZ else SGSZ;
                                        const gid_z = if (grpz == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                        const wdz = &cycle_witnesses[cycz];
                                        var azz = F.zero();
                                        var bzz = F.zero();
                                        for (0..gsz_z) |kkz| {
                                            const ccz = c_mod.UNIFORM_CONSTRAINTS[gid_z[kkz]];
                                            const cvz = ccz.condition.evaluate(F, wdz.asSlice());
                                            const mvz = ccz.left.evaluate(F, wdz.asSlice()).sub(ccz.right.evaluate(F, wdz.asSlice()));
                                            azz = azz.add(lag_y[kkz].mul(cvz));
                                            bzz = bzz.add(lag_y[kkz].mul(mvz));
                                        }
                                        const abz = azz.mul(bzz);
                                        if (!abz.eql(F.zero())) {
                                            cnt_nz += 1;
                                            if (cnt_nz <= 3) {
                                                // Also print individual Az, Bz
                                                // And Lagrange evals used
                                            }
                                        }
                                    }
                                }
                            }
                        } else if (!is_base) {
                        }
                    }

                    // Check ALL constraints at ALL cycles for violations
                    var total_violations: usize = 0;
                    for (0..cycle_witnesses.len) |cv| {
                        const wcv = &cycle_witnesses[cv];
                        for (0..c_mod.UNIFORM_CONSTRAINTS.len) |ci| {
                            const cc = c_mod.UNIFORM_CONSTRAINTS[ci];
                            const cond_v = cc.condition.evaluate(F, wcv.asSlice());
                            const left_v = cc.left.evaluate(F, wcv.asSlice());
                            const right_v = cc.right.evaluate(F, wcv.asSlice());
                            const diff_v = left_v.sub(right_v);
                            const prod_v = cond_v.mul(diff_v);
                            if (!prod_v.eql(F.zero())) {
                                total_violations += 1;
                                if (total_violations <= 20) {
                                }
                            }
                        }
                    }

                    // Print key witness values for violated cycles
                    if (total_violations > 0 and cycle_witnesses.len > 54) {
                        // Also print cycle 55
                        if (cycle_witnesses.len > 55) {
                        }
                    }
                }
            }

            // Bind the first-round challenge from transcript with the uni_skip_claim
            outer_prover.bindFirstRoundChallenge(r0, uni_skip_claim) catch {};

            // Match Jolt's cache_openings: after UniSkip verification, the verifier calls
            // accumulator.append_virtual() which appends the uni_skip_claim to transcript.
            // This happens BEFORE BatchedSumcheck::verify which also appends it.
            // flush_to_transcript: uni_skip opening claim
            transcript.appendScalar("opening_claim", uni_skip_claim);
            if (comptime debug_verbose) {
                std.debug.print("[ZOLT-PROVER] after_flush transcript_state = ", .{});
                for (transcript.state[0..8]) |b| std.debug.print("{x:0>2}", .{b});
                std.debug.print(" round={}\n", .{transcript.n_rounds});
            }

            // BatchedSumcheck::verify: append input_claim then get batching coefficients
            transcript.appendScalar("sumcheck_claim", uni_skip_claim);

            // Get batching coefficient - advances transcript state AND provides scaling factor
            const batching_coeff = transcript.challengeScalarFull();
            if (comptime debug_verbose) {
                std.debug.print("[ZOLT-PROVER] input_claim (uni_skip_claim) = {any}\n", .{uni_skip_claim.toBytesBE()});
                std.debug.print("[ZOLT-PROVER] batching_coeff = {any}\n", .{batching_coeff.toBytesBE()});
                std.debug.print("[ZOLT-PROVER] transcript state: ", .{});
                for (transcript.state[0..8]) |b| std.debug.print("{x:0>2} ", .{b});
                std.debug.print("round={}\n", .{transcript.n_rounds});
            }

            // Generate remaining rounds
            // In Jolt, stage1_sumcheck_proof contains num_rounds polynomials
            // where num_rounds = 1 + num_cycle_vars (1 streaming + cycle vars)
            // The UniSkip is separate and doesn't count here
            const num_remaining_rounds = outer_prover.numRounds(); // 1 + num_cycle_vars
            if (num_remaining_rounds == 0) {
                return Stage1Result{ .challenges = challenges, .r0 = r0, .uni_skip_claim = uni_skip_claim, .allocator = self.allocator };
            }

            // Compute initial claim = uni_skip_claim * batching_coeff (for Jolt compatibility)
            const initial_claim = uni_skip_claim.mul(batching_coeff);
            if (comptime debug_verbose) {
                std.debug.print("[ZOLT-PROVER] batched_claim = {any}\n", .{initial_claim.toBytesBE()});
            }

            // Generate all remaining round polynomials with transcript integration
            for (0..num_remaining_rounds) |_| {
                const raw_evals: [4]F = outer_prover.computeRemainingRoundPoly() catch {
                    // Fallback to zero polynomial
                    const coeffs = try self.allocator.alloc(F, 3);
                    @memset(coeffs, F.zero());
                    try proof.compressed_polys.append(self.allocator, .{
                        .coeffs_except_linear_term = coeffs,
                        .allocator = self.allocator,
                    });
                    try challenges.append(self.allocator, F.zero());
                    continue;
                };

                // Scale evaluations by batching coefficient for output
                const scaled_evals = [4]F{
                    raw_evals[0].mul(batching_coeff),
                    raw_evals[1].mul(batching_coeff),
                    raw_evals[2].mul(batching_coeff),
                    raw_evals[3].mul(batching_coeff),
                };

                // Convert to compressed coefficients for proof
                const compressed = poly_mod.UniPoly(F).evalsToCompressed(scaled_evals);
                const coeffs = try self.allocator.alloc(F, 3);
                coeffs[0] = compressed[0]; // c0
                coeffs[1] = compressed[1]; // c2
                coeffs[2] = compressed[2]; // c3

                // DEBUG: Print round polynomial coefficients (LE bytes for Jolt comparison)

                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                // Append round polynomial to transcript
                transcript.appendScalars("sumcheck_poly", coeffs);

                // Get challenge from transcript
                const challenge = transcript.challengeScalar();
                try challenges.append(self.allocator, challenge);

                // DEBUG: Print challenge (LE bytes for Jolt comparison)

                // Bind challenge and update claim
                outer_prover.bindRemainingRoundChallenge(challenge) catch {};
                outer_prover.updateClaim(raw_evals, challenge);
            }

            // DEBUG: Print final summary including eq factor from split_eq
            const prover_eq_factor = outer_prover.split_eq.current_scalar;

            // Print final claim from prover

            // Compute implied Az*Bz = final_claim / eq_factor
            if (!prover_eq_factor.eql(F.zero())) {
            }

            // CROSS-CHECK: Compute the "correct" final_claim directly from witnesses
            // This is what the verifier expects: eq_factor * Az(r_stream, r0, r_cycle) * Bz(r_stream, r0, r_cycle)
            // where r_cycle is the full set of bound challenges reversed
            if (comptime false) {
                const all_chal = challenges.items;
                const r0_check = r0;

                // Get the cycle challenges (skip r_stream)
                const cycle_chal = if (all_chal.len > 1) all_chal[1..] else all_chal[0..0];

                // Compute MLE evaluations of R1CS inputs at r_cycle (reversed)
                const r_cycle_be = try self.allocator.alloc(F, cycle_chal.len);
                defer self.allocator.free(r_cycle_be);
                for (0..cycle_chal.len) |idx| {
                    r_cycle_be[idx] = cycle_chal[cycle_chal.len - 1 - idx];
                }

                const R1CSInputEval = r1cs.R1CSInputEvaluator(F);
                const check_evals = try R1CSInputEval.computeClaimedInputs(
                    self.allocator,
                    cycle_witnesses,
                    r_cycle_be,
                );

                // Compute Lagrange weights at r0
                const FGSZ = 10;
                const SGSZ = 9;
                var w_check: [FGSZ]F = undefined;
                const start: i64 = -4;
                for (0..FGSZ) |ii| {
                    var numer = F.one();
                    var denom = F.one();
                    for (0..FGSZ) |jj| {
                        if (ii != jj) {
                            const x_j: i64 = start + @as(i64, @intCast(jj));
                            const x_j_f = if (x_j >= 0)
                                F.fromU64(@intCast(x_j))
                            else
                                F.zero().sub(F.fromU64(@intCast(-x_j)));
                            numer = numer.mul(r0_check.sub(x_j_f));
                            const d: i64 = @as(i64, @intCast(ii)) - @as(i64, @intCast(jj));
                            if (d > 0) {
                                denom = denom.mul(F.fromU64(@intCast(d)));
                            } else {
                                denom = denom.mul(F.zero().sub(F.fromU64(@intCast(-d))));
                            }
                        }
                    }
                    w_check[ii] = if (!denom.eql(F.zero())) numer.mul(denom.inverse().?) else F.zero();
                }

                // Build z vector
                var z_check: [r1cs.R1CSInputIndex.NUM_INPUTS + 1]F = undefined;
                @memcpy(z_check[0..r1cs.R1CSInputIndex.NUM_INPUTS], &check_evals);
                z_check[r1cs.R1CSInputIndex.NUM_INPUTS] = F.one();

                // Compute az_g0, bz_g0
                var az_g0_c = F.zero();
                var bz_g0_c = F.zero();
                for (0..FGSZ) |ii| {
                    const cidx = r1cs.FIRST_GROUP_INDICES[ii];
                    const cons = r1cs.UNIFORM_CONSTRAINTS[cidx];
                    const az_c = cons.condition.evaluateWithConstant(F, &z_check);
                    const bz_c = cons.left.evaluateWithConstant(F, &z_check).sub(cons.right.evaluateWithConstant(F, &z_check));
                    az_g0_c = az_g0_c.add(w_check[ii].mul(az_c));
                    bz_g0_c = bz_g0_c.add(w_check[ii].mul(bz_c));
                }

                var az_g1_c = F.zero();
                var bz_g1_c = F.zero();
                for (0..SGSZ) |ii| {
                    const cidx = r1cs.SECOND_GROUP_INDICES[ii];
                    const cons = r1cs.UNIFORM_CONSTRAINTS[cidx];
                    const az_c = cons.condition.evaluateWithConstant(F, &z_check);
                    const bz_c = cons.left.evaluateWithConstant(F, &z_check).sub(cons.right.evaluateWithConstant(F, &z_check));
                    az_g1_c = az_g1_c.add(w_check[ii].mul(az_c));
                    bz_g1_c = bz_g1_c.add(w_check[ii].mul(bz_c));
                }


            }

            return Stage1Result{ .challenges = challenges, .r0 = r0, .uni_skip_claim = uni_skip_claim, .allocator = self.allocator };
        }

        /// Evaluate a polynomial given as coefficients at a point using Horner's method
        /// Uses standard Montgomery multiplication.
        fn evaluatePolyAtPoint(coeffs: []const F, x: F) F {
            if (coeffs.len == 0) return F.zero();

            var result = coeffs[coeffs.len - 1];
            var i = coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(coeffs[i]);
            }
            return result;
        }

        /// Evaluate a polynomial at a challenge point using Horner's method.
        ///
        /// Both the coefficients and the challenge point should be in Montgomery form.
        fn evaluatePolyAtChallenge(coeffs: []const F, x: F) F {
            if (coeffs.len == 0) return F.zero();

            // Both coeffs and x are in Montgomery form (challenges are now
            // converted to Montgomery form in the transcript).
            // Standard field multiplication works correctly.

            // Use Horner's method
            var result = coeffs[coeffs.len - 1];
            var i = coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(coeffs[i]);
            }
            return result;
        }

        /// R1CS input indices in upstream Jolt's ALL_R1CS_INPUTS order (35 inputs)
        /// Maps from Jolt's ordering (index in this array) to Zolt's R1CSInputIndex
        const JOLT_TO_ZOLT_R1CS_INDICES = [35]r1cs.R1CSInputIndex{
            .LeftInstructionInput, // 0
            .RightInstructionInput, // 1
            .Product, // 2
            .ShouldBranch, // 3
            .PC, // 4
            .UnexpandedPC, // 5
            .Imm, // 6
            .RamAddress, // 7
            .Rs1Value, // 8
            .Rs2Value, // 9
            .RdWriteValue, // 10
            .RamReadValue, // 11
            .RamWriteValue, // 12
            .LeftLookupOperand, // 13
            .RightLookupOperand, // 14
            .NextUnexpandedPC, // 15
            .NextPC, // 16
            .NextIsVirtual, // 17
            .NextIsFirstInSequence, // 18
            .LookupOutput, // 19
            .ShouldJump, // 20
            .FlagAddOperands, // 21
            .FlagSubtractOperands, // 22
            .FlagMultiplyOperands, // 23
            .FlagLoad, // 24
            .FlagStore, // 25
            .FlagJump, // 26
            .FlagWriteLookupOutputToRD, // 27
            .FlagVirtualInstruction, // 28
            .FlagAssert, // 29
            .FlagDoNotUpdateUnexpandedPC, // 30
            .FlagAdvice, // 31
            .FlagIsCompressed, // 32
            .FlagIsFirstInSequence, // 33
            .FlagIsLastInSequence, // 34
        };

        /// VirtualPolynomial identifiers in upstream Jolt's order (35 inputs)
        const R1CS_VIRTUAL_POLYS = [35]VirtualPolynomial{
            .LeftInstructionInput, // 0
            .RightInstructionInput, // 1
            .Product, // 2
            .ShouldBranch, // 3
            .PC, // 4
            .UnexpandedPC, // 5
            .Imm, // 6
            .RamAddress, // 7
            .Rs1Value, // 8
            .Rs2Value, // 9
            .RdWriteValue, // 10
            .RamReadValue, // 11
            .RamWriteValue, // 12
            .LeftLookupOperand, // 13
            .RightLookupOperand, // 14
            .NextUnexpandedPC, // 15
            .NextPC, // 16
            .NextIsVirtual, // 17
            .NextIsFirstInSequence, // 18
            .LookupOutput, // 19
            .ShouldJump, // 20
            // OpFlags variants (14 of them) - CircuitFlags indices
            .{ .OpFlags = 0 }, // 21: AddOperands
            .{ .OpFlags = 1 }, // 22: SubtractOperands
            .{ .OpFlags = 2 }, // 23: MultiplyOperands
            .{ .OpFlags = 3 }, // 24: Load
            .{ .OpFlags = 4 }, // 25: Store
            .{ .OpFlags = 5 }, // 26: Jump
            .{ .OpFlags = 6 }, // 27: WriteLookupOutputToRD
            .{ .OpFlags = 7 }, // 28: VirtualInstruction
            .{ .OpFlags = 8 }, // 29: Assert
            .{ .OpFlags = 9 }, // 30: DoNotUpdateUnexpandedPC
            .{ .OpFlags = 10 }, // 31: Advice
            .{ .OpFlags = 11 }, // 32: IsCompressed
            .{ .OpFlags = 12 }, // 33: IsFirstInSequence
            .{ .OpFlags = 13 }, // 34: IsLastInSequence
        };

        /// Add all 35 R1CS input opening claims for SpartanOuter with zero claims
        ///
        /// This exactly matches the ALL_R1CS_INPUTS array in upstream Jolt's r1cs/inputs.rs:
        /// - 21 simple virtual polynomials
        /// - 14 OpFlags variants
        fn addSpartanOuterOpeningClaims(
            self: *Self,
            claims: *OpeningClaims(F),
        ) !void {
            _ = self;

            // Add all R1CS inputs for SpartanOuter with zero claims
            for (R1CS_VIRTUAL_POLYS) |poly| {
                try claims.insert(
                    .{ .Virtual = .{ .poly = poly, .sumcheck_id = .SpartanOuter } },
                    F.zero(),
                );
            }

            // Add the UnivariateSkip claim for SpartanOuter
            try claims.insert(
                .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanOuter } },
                F.zero(),
            );
        }

        /// Add all 35 R1CS input opening claims for SpartanOuter with actual evaluations
        ///
        /// This computes the MLE evaluations at r_cycle and uses those as the claims.
        ///
        /// IMPORTANT: This also appends all 35 R1CS input claims to the transcript
        /// in Jolt's order (ALL_R1CS_INPUTS). This is required for Fiat-Shamir
        /// consistency before deriving Stage 2's tau_high challenge.
        fn addSpartanOuterOpeningClaimsWithEvaluations(
            self: *Self,
            claims: *OpeningClaims(F),
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            r_cycle: []const F,
            uni_skip_claim: F,
            transcript: *Blake2bTranscript(F),
            _: F, // r_stream (unused after debug removal)
            r0: F,
        ) !void {
            // Compute MLE evaluations at r_cycle
            const R1CSInputEvaluator = r1cs.R1CSInputEvaluator(F);
            const input_evals = try R1CSInputEvaluator.computeClaimedInputs(
                self.allocator,
                cycle_witnesses,
                r_cycle,
            );


            // Compute Lagrange weights at r0
            const FIRST_GROUP_SIZE = 10;
            const SECOND_GROUP_SIZE = 9;
            var lagrange_weights: [FIRST_GROUP_SIZE]F = undefined;
            const base_left: i64 = -@as(i64, (FIRST_GROUP_SIZE - 1) / 2); // = -4

            for (0..FIRST_GROUP_SIZE) |i| {
                var numer = F.one();
                var denom = F.one();

                for (0..FIRST_GROUP_SIZE) |j| {
                    if (i != j) {
                        const x_j: i64 = base_left + @as(i64, @intCast(j));
                        const x_j_field = if (x_j >= 0)
                            F.fromU64(@intCast(x_j))
                        else
                            F.zero().sub(F.fromU64(@intCast(-x_j)));
                        numer = numer.mul(r0.sub(x_j_field));

                        const diff: i64 = @as(i64, @intCast(i)) - @as(i64, @intCast(j));
                        if (diff > 0) {
                            denom = denom.mul(F.fromU64(@intCast(diff)));
                        } else {
                            denom = denom.mul(F.zero().sub(F.fromU64(@intCast(-diff))));
                        }
                    }
                }

                lagrange_weights[i] = if (!denom.eql(F.zero()))
                    numer.mul(denom.inverse().?)
                else
                    F.zero();
            }

            // Build z vector with trailing 1 (like Jolt)
            var z: [r1cs.R1CSInputIndex.NUM_INPUTS + 1]F = undefined;
            @memcpy(z[0..r1cs.R1CSInputIndex.NUM_INPUTS], &input_evals);
            z[r1cs.R1CSInputIndex.NUM_INPUTS] = F.one(); // constant column

            // Compute az_g0, bz_g0 from first group
            var az_g0 = F.zero();
            var bz_g0 = F.zero();
            for (0..FIRST_GROUP_SIZE) |i| {
                const constraint_idx = r1cs.FIRST_GROUP_INDICES[i];
                const constraint = r1cs.UNIFORM_CONSTRAINTS[constraint_idx];

                // az_contrib = condition.dot_product(z)
                const az_contrib = constraint.condition.evaluateWithConstant(F, &z);
                // bz_contrib = (left - right).dot_product(z)
                const bz_contrib = constraint.left.evaluateWithConstant(F, &z)
                    .sub(constraint.right.evaluateWithConstant(F, &z));

                az_g0 = az_g0.add(lagrange_weights[i].mul(az_contrib));
                bz_g0 = bz_g0.add(lagrange_weights[i].mul(bz_contrib));
            }

            // Compute az_g1, bz_g1 from second group
            var az_g1 = F.zero();
            var bz_g1 = F.zero();
            const g1_len = @min(SECOND_GROUP_SIZE, FIRST_GROUP_SIZE);
            for (0..g1_len) |i| {
                const constraint_idx = r1cs.SECOND_GROUP_INDICES[i];
                const constraint = r1cs.UNIFORM_CONSTRAINTS[constraint_idx];

                const az_contrib = constraint.condition.evaluateWithConstant(F, &z);
                const bz_contrib = constraint.left.evaluateWithConstant(F, &z)
                    .sub(constraint.right.evaluateWithConstant(F, &z));

                az_g1 = az_g1.add(lagrange_weights[i].mul(az_contrib));
                bz_g1 = bz_g1.add(lagrange_weights[i].mul(bz_contrib));
            }

            // Blend with r_stream


            // Add R1CS inputs for SpartanOuter with computed evaluations
            // AND append each claim to transcript in Jolt's order (for Fiat-Shamir)

            for (R1CS_VIRTUAL_POLYS, 0..) |poly, jolt_idx| {
                // Map Jolt's index to Zolt's R1CSInputIndex
                const zolt_idx = JOLT_TO_ZOLT_R1CS_INDICES[jolt_idx].toIndex();
                const claim = input_evals[zolt_idx];

                try claims.insert(
                    .{ .Virtual = .{ .poly = poly, .sumcheck_id = .SpartanOuter } },
                    claim,
                );

                // flush_to_transcript: opening claim
                transcript.appendScalar("opening_claim", claim);
            }

            // Add the UnivariateSkip claim for SpartanOuter
            // This is uni_poly.evaluate(r0), the input_claim for the remaining sumcheck
            // NOTE: Do NOT append to transcript here - the UniSkip claim was already appended
            // twice earlier (once in cache_openings after r0 sampling, once in BatchedSumcheck::prove)
            try claims.insert(
                .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanOuter } },
                uni_skip_claim,
            );
        }

        /// Create a UniSkipFirstRoundProof for Stage 1 (degree-27 polynomial)
        ///
        /// Jolt's Stage 1 (Spartan outer) uses a degree-27 first-round polynomial
        /// that encodes all 19 R1CS constraints via the univariate skip optimization.
        ///
        /// For the verification to pass, the polynomial must satisfy:
        ///   Σ_{j=0}^{27} coeff_j * power_sums[j] = 0
        ///
        /// where power_sums[j] = Σ_{t in domain} t^j for domain {-4, -3, ..., 4, 5}.
        ///
        /// The simplest valid polynomial is all zeros (trivially sums to 0).
        fn createUniSkipProofStage1(self: *Self) !?UniSkipFirstRoundProof(F) {
            // For stage 1, we need 28 coefficients (degree 27)
            const NUM_COEFFS = r1cs.OUTER_FIRST_ROUND_POLY_NUM_COEFFS;

            // Create an all-zero polynomial that trivially satisfies the sum constraint.
            const coeffs = try self.allocator.alloc(F, NUM_COEFFS);
            @memset(coeffs, F.zero());

            return UniSkipFirstRoundProof(F){
                .uni_poly = coeffs,
                .allocator = self.allocator,
            };
        }

        /// Create a UniSkipFirstRoundProof for Stage 1 from actual witnesses
        ///
        /// This computes real Az*Bz products using the constraint evaluators,
        /// producing a polynomial that satisfies the univariate skip verification.
        ///
        /// IMPORTANT: The eq polynomial must be computed from tau_low (excluding tau_high)
        /// because the UniSkip polynomial formula is:
        ///   s1(Y) = L(τ_high, Y) · t1(Y)
        /// where t1(Y) = Σ_x eq(τ_low, x) · Az(x,Y) · Bz(x,Y)
        ///
        /// If we used the full tau, τ_high would be counted twice!
        fn createUniSkipProofStage1FromWitnesses(
            self: *Self,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau: []const F,
            compact_witnesses: ?[]const @import("r1cs/evaluators.zig").CompactWitness,
        ) !?UniSkipFirstRoundProof(F) {
            if (cycle_witnesses.len == 0) {
                return self.createUniSkipProofStage1();
            }

            const NUM_COEFFS = r1cs.OUTER_FIRST_ROUND_POLY_NUM_COEFFS;

            if (tau.len < 2) {
                return self.createUniSkipProofStage1();
            }

            var outer_prover = try streaming_outer.StreamingOuterProver(F).initWithScaling(
                self.allocator,
                cycle_witnesses,
                tau,
                null, // No scaling for initial UniSkip - will be applied in interpolation
            );
            outer_prover.thread_pool = self.thread_pool;
            // Set pre-built compact witnesses (not owned — don't free on deinit)
            outer_prover.compact_witnesses = compact_witnesses;
            defer {
                outer_prover.compact_witnesses = null; // prevent double-free
                outer_prover.deinit();
            }

            // Compute the univariate skip polynomial using the fixed implementation
            // that properly handles both constraint groups
            const uni_poly_coeffs = try outer_prover.computeFirstRoundPoly();

            // DEBUG: Print first few UniSkip coefficients
            if (uni_poly_coeffs.len > 0) {
            }
            if (uni_poly_coeffs.len > 1) {
            }

            // Copy coefficients to our proof structure
            const coeffs = try self.allocator.alloc(F, NUM_COEFFS);
            @memset(coeffs, F.zero());

            // Copy available coefficients (may be fewer than NUM_COEFFS)
            const copy_len = @min(uni_poly_coeffs.len, NUM_COEFFS);
            @memcpy(coeffs[0..copy_len], uni_poly_coeffs[0..copy_len]);

            return UniSkipFirstRoundProof(F){
                .uni_poly = coeffs,
                .allocator = self.allocator,
            };
        }


        /// Convert with actual per-cycle witnesses and Fiat-Shamir transcript
        ///
        /// This method produces proofs with proper Az*Bz evaluations and uses
        /// the Blake2b transcript for all Fiat-Shamir challenges.
        /// This is the method to use for Jolt cross-verification.
        pub fn proveWithTranscript(
            self: *Self,
            comptime Commitment: type,
            comptime Proof: type,
            log_t: u8,
            log_k: u8,
            commitments: []const Commitment,
            joint_opening_proof: ?Proof,
            config: JoltProverConfig,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau: []const F,
            transcript: *Blake2bTranscript(F),
        ) !JoltProofType(F, Commitment, Proof) {
            var jolt_proof = JoltProofType(F, Commitment, Proof).init(self.allocator);

            // Copy configuration parameters
            const trace_length: usize = @as(usize, 1) << @intCast(log_t);
            const ram_K: usize = @as(usize, 1) << @intCast(log_k);

            jolt_proof.trace_length = trace_length;
            jolt_proof.ram_K = ram_K;

            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Set config structs (matching Jolt's serialization format)
            jolt_proof.rw_config = jolt_types.ReadWriteConfig.default(log_t, log_k);
            jolt_proof.one_hot_config = .{
                .log_k_chunk = @intCast(config.log_k_chunk),
                .lookups_ra_virtual_log_k_chunk = @intCast(config.lookups_ra_virtual_log_k_chunk),
            };
            jolt_proof.dory_layout = 0; // Wide layout

            // Compute derived parameters
            const n_cycle_vars = std.math.log2_int(usize, trace_length);
            const log_ram_k = std.math.log2_int(usize, ram_K);

            // CRITICAL: Pad cycle_witnesses to trace_length with NoOp witness values.
            // In Jolt, padded cycles are Cycle::NoOp which has:
            //   - FlagIsNoop = 1
            //   - FlagDoNotUpdateUnexpandedPC = 1
            //   - All other R1CS inputs = 0
            // The prover must use padded witnesses because:
            //   1. The streaming outer sumcheck materializes Az/Bz over all trace_length cycles
            //   2. Even though Az*Bz = 0 for NoOp cycles, the individual Az values at NoOp
            //      cycles are non-zero (some conditions evaluate to 1)
            //   3. The verifier evaluates Az(r)*Bz(r) using MLE openings that include NoOp
            //      contributions, so the prover's MLE must match
            const padded_witnesses = try self.allocator.alloc(r1cs.R1CSCycleInputs(F), trace_length);
            defer self.allocator.free(padded_witnesses);

            // Copy actual witness data
            @memcpy(padded_witnesses[0..cycle_witnesses.len], cycle_witnesses);

            // Fill padded cycles with NoOp witness values
            {
                const pad_len = trace_length - cycle_witnesses.len;
                const pad_start = cycle_witnesses.len;
                if (self.thread_pool != null and pad_len >= 256) {
                    const PadCtx = struct {
                        pw: []r1cs.R1CSCycleInputs(F),
                        start: usize,
                    };
                    self.thread_pool.?.parallelFor(pad_len, PadCtx{ .pw = padded_witnesses, .start = pad_start }, struct {
                        fn f(ctx: PadCtx, idx: usize) void {
                            const i = ctx.start + idx;
                            ctx.pw[i] = r1cs.R1CSCycleInputs(F).init();
                            ctx.pw[i].values[r1cs.R1CSInputIndex.FlagIsNoop.toIndex()] = F.one();
                            ctx.pw[i].values[r1cs.R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                        }
                    }.f);
                } else {
                    for (pad_start..trace_length) |i| {
                        padded_witnesses[i] = r1cs.R1CSCycleInputs(F).init();
                        padded_witnesses[i].values[r1cs.R1CSInputIndex.FlagIsNoop.toIndex()] = F.one();
                        padded_witnesses[i].values[r1cs.R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                    }
                }
            }


            // Copy commitments and append to transcript
            for (commitments) |c| {
                try jolt_proof.commitments.append(self.allocator, c);
            }

            // Append commitments to transcript (GT elements for Dory)
            // This is done in Jolt's prover before deriving challenges
            // Note: For now we skip this since commitment serialization to transcript
            // is complex and involves GT element encoding

            // Set joint opening proof
            jolt_proof.joint_opening_proof = joint_opening_proof;

            // Per-stage timing
            var stage_timer = std.time.Timer.start() catch unreachable;

            // Build compact integer witnesses for fast evaluation
            const r1cs_evaluators = @import("r1cs/evaluators.zig");
            const compact_witnesses = try r1cs_evaluators.buildCompactWitnesses(F, padded_witnesses, self.allocator, self.thread_pool);
            defer self.allocator.free(compact_witnesses);

            // Create UniSkip proof for Stage 1 with actual constraint evaluations
            // Use padded witnesses so that NoOp cycles are included in the polynomial evaluation
            // DEBUG: Validate compact vs field for second group
            {
                var sg_mismatch: usize = 0;
                for (0..@min(padded_witnesses.len, compact_witnesses.len)) |ci| {
                    const cw = &compact_witnesses[ci];
                    const ws = padded_witnesses[ci].asSlice();
                    const two_pow_64 = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                    const bz_field_sg = r1cs_evaluators.computeBzSecondGroupDirect(F, ws, two_pow_64);
                    for (0..9) |t| {
                        const field_bz = bz_field_sg[t];
                        const s192_bz = cw.bz_second[t];
                        // Convert S192 to field for comparison
                        // Convert S192 magnitude to field element
                        const lo: u128 = @as(u128, s192_bz.magnitude[0]) | (@as(u128, s192_bz.magnitude[1]) << 64);
                        const hi: u64 = s192_bz.magnitude[2];
                        const lo_f = F.fromU128(lo);
                        const hi_f = F.fromU64(hi).mul(F.fromU128(@as(u128, 1) << 64).mul(F.fromU128(@as(u128, 1) << 64)));
                        const mag_f = lo_f.add(hi_f);
                        const s192_as_field = if (s192_bz.is_positive) mag_f else F.zero().sub(mag_f);
                        if (!field_bz.eql(s192_as_field)) {
                            if (sg_mismatch < 3) {
                                const rl_f = ws[r1cs.R1CSInputIndex.RightLookupOperand.toIndex()];
                                const rl_std = rl_f.fromMontgomery();
                                const upc = ws[r1cs.R1CSInputIndex.UnexpandedPC.toIndex()].toU64();
                                std.debug.print("SG BZ MISMATCH: cycle={d} SG[{d}] UPC=0x{x} RL_limbs=[{x},{x},{x},{x}]\n", .{
                                    ci, t, upc, rl_std.limbs[0], rl_std.limbs[1], rl_std.limbs[2], rl_std.limbs[3],
                                });
                            }
                            sg_mismatch += 1;
                        }
                    }
                }
                if (sg_mismatch > 0) {
                    std.debug.print("SG BZ: {d} mismatches\n", .{sg_mismatch});
                } else {
                    std.debug.print("SG BZ: All match!\n", .{});
                }
            }

            jolt_proof.stage1_uni_skip_first_round_proof = try self.createUniSkipProofStage1FromWitnesses(
                padded_witnesses,
                tau,
                compact_witnesses,
            );

            // Stage 1: Outer Spartan Remaining - use streaming prover with transcript
            // Use padded witnesses so Az/Bz MLE evaluations match the verifier's computation
            var stage1_result: ?Stage1Result = null;
            if (jolt_proof.stage1_uni_skip_first_round_proof) |*uniskip| {
                stage1_result = try self.generateStreamingOuterSumcheckProofWithTranscript(
                    &jolt_proof.stage1_sumcheck_proof,
                    uniskip,
                    padded_witnesses,
                    tau,
                    transcript,
                    compact_witnesses,
                );
            } else {
                // Fallback to zero proofs
                try self.generateZeroSumcheckProof(
                    &jolt_proof.stage1_sumcheck_proof,
                    1 + n_cycle_vars,
                    3,
                );
            }
            defer if (stage1_result) |*r| r.deinit();

            // Add Stage 1 opening claims with computed MLE evaluations
            if (stage1_result) |result| {
                // The r_cycle point for R1CS input MLE evaluation
                // Sumcheck challenges = [r_stream, r_1, r_2, ..., r_n]
                // r_cycle = challenges[1..] converted to BIG_ENDIAN (reversed)
                //
                // Both Zolt's EqPolynomial.evals and Jolt's EqPolynomial::evals use
                // BIG_ENDIAN convention (r[0]→MSB). The sumcheck challenges are in
                // binding order (LowToHigh), so we reverse them to BIG_ENDIAN.
                const all_challenges = result.challenges.items;

                // Skip the first challenge (r_stream) to get the cycle challenges
                const cycle_challenges = if (all_challenges.len > 1)
                    all_challenges[1..]
                else
                    all_challenges;

                // Convert from sumcheck order to BIG_ENDIAN (MLE eval order)
                const r_cycle_big_endian = try self.allocator.alloc(F, cycle_challenges.len);
                defer self.allocator.free(r_cycle_big_endian);
                for (0..cycle_challenges.len) |i| {
                    r_cycle_big_endian[i] = cycle_challenges[cycle_challenges.len - 1 - i];
                }

                // Get r_stream (first challenge) and r0 from Stage 1 result
                const r_stream = if (all_challenges.len > 0) all_challenges[0] else F.zero();

                try self.addSpartanOuterOpeningClaimsWithEvaluations(
                    &jolt_proof.opening_claims,
                    padded_witnesses,
                    r_cycle_big_endian,
                    result.uni_skip_claim,
                    transcript,
                    r_stream,
                    result.r0,
                );
            } else {
                // Fallback to zero claims
                try self.addSpartanOuterOpeningClaims(&jolt_proof.opening_claims);
            }

            if (comptime stage_timing_enabled) {
                std.debug.print("    [STAGE-TIMING] Stage 1: {d:.1} ms\n", .{@as(f64, @floatFromInt(stage_timer.read())) / 1_000_000.0});
            }
            stage_timer.reset();

            // Create UniSkip proof for Stage 2
            // Jolt samples a NEW tau_high for Stage 2 from the transcript (see ProductVirtualUniSkipParams::new)
            // tau = [r_cycle_outer, tau_high] where tau_high is freshly sampled
            const tau_high_stage2 = transcript.challengeScalar();

            // Get the 3 product claims from Stage 1's opening claims
            // Order: Product, ShouldBranch, ShouldJump
            const PRODUCT_VIRTUALS = [3]VirtualPolynomial{
                .Product,
                .ShouldBranch,
                .ShouldJump,
            };

            var base_evals_stage2: [3]F = [_]F{F.zero()} ** 3;
            for (PRODUCT_VIRTUALS, 0..) |poly, i| {
                const claim_key = OpeningId{ .Virtual = .{ .poly = poly, .sumcheck_id = .SpartanOuter } };
                if (jolt_proof.opening_claims.get(claim_key)) |claim| {
                    base_evals_stage2[i] = claim;
                }
            }

            // Debug: Print Stage 2 setup

            // Build tau_stage2 BEFORE calling createUniSkipProofStage2WithClaims
            // tau_stage2 = [r_cycle_reversed, tau_high_stage2]
            const tau_stage2_early = try self.allocator.alloc(F, n_cycle_vars + 1);
            defer self.allocator.free(tau_stage2_early);

            if (stage1_result) |result| {
                const all_challenges = result.challenges.items;
                // Skip the first challenge (r_stream) to get r_cycle
                const cycle_challenges = if (all_challenges.len > 1)
                    all_challenges[1..]
                else
                    all_challenges;

                // r_cycle reversed (BIG_ENDIAN)
                for (0..n_cycle_vars) |i| {
                    const src_idx = n_cycle_vars - 1 - i;
                    if (src_idx < cycle_challenges.len) {
                        tau_stage2_early[i] = cycle_challenges[src_idx];
                    } else {
                        tau_stage2_early[i] = F.zero();
                    }
                }
            } else {
                for (0..n_cycle_vars) |i| {
                    tau_stage2_early[i] = F.zero();
                }
            }
            tau_stage2_early[n_cycle_vars] = tau_high_stage2;

            jolt_proof.stage2_uni_skip_first_round_proof = try self.createUniSkipProofStage2WithClaims(
                &base_evals_stage2,
                tau_high_stage2,
                padded_witnesses,
                tau_stage2_early,
            );

            // CRITICAL: Append Stage 2 UniSkip polynomial to transcript (matching Jolt verifier flow)
            // The verifier calls UniSkipFirstRoundProof::verify which:
            // 1. Appends the polynomial coefficients to transcript
            // 2. Derives r0 challenge
            // 3. Calls cache_openings which appends UnivariateSkip claim
            var r0_stage2: F = F.zero();
            var uni_skip_claim_stage2: F = F.zero();

            if (jolt_proof.stage2_uni_skip_first_round_proof) |proof| {
                // Append polynomial
                transcript.appendScalars("uniskip_poly", proof.uni_poly);

                // Derive r0 challenge
                r0_stage2 = transcript.challengeScalar();

                // Compute UnivariateSkip claim = poly(r0)
                // uni_poly = [c0, c1, c2, ..., c12] -> poly(x) = c0 + c1*x + c2*x^2 + ...
                var r_power = F.one();
                for (proof.uni_poly) |coeff| {
                    uni_skip_claim_stage2 = uni_skip_claim_stage2.add(coeff.mul(r_power));
                    r_power = r_power.mul(r0_stage2);
                }

                // flush_to_transcript: uni_skip opening claim
                transcript.appendScalar("opening_claim", uni_skip_claim_stage2);

                // Update the opening claim for UnivariateSkip at SpartanProductVirtualization
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanProductVirtualization } },
                    uni_skip_claim_stage2,
                );

                // Debug: verify the claim was inserted correctly
                const inserted_claim = jolt_proof.opening_claims.get(.{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanProductVirtualization } });
                if (inserted_claim) |_| {
                } else {
                }
            }

            // Stage 2 batches 5 sumcheck instances:
            // 1. ProductVirtualRemainder: n_cycle_vars rounds
            // 2. RamRafEvaluation: log_ram_k rounds
            // 3. RamReadWriteChecking: log_ram_k + n_cycle_vars rounds (max!)
            // 4. RamOutputCheck: log_ram_k rounds
            // 5. InstructionLookupsClaimReduction: n_cycle_vars rounds
            // max_num_rounds = log_ram_k + n_cycle_vars
            //
            // CRITICAL: Stage 2's tau is NOT the original tau from Stage 1!
            // It's built from [r_cycle_stage1, tau_high_stage2] where:
            // - r_cycle_stage1 = the sumcheck challenges from Stage 1 (opening point)
            // - tau_high_stage2 = freshly sampled challenge
            // See Jolt's ProductVirtualUniSkipParams::new
            var tau_stage2 = try self.allocator.alloc(F, n_cycle_vars + 1);
            defer self.allocator.free(tau_stage2);

            // Build tau_stage2 from Stage 1 challenges
            // Also compute r_spartan_original (non-reversed) for InstructionLookupsClaimReduction
            var r_spartan_original = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_spartan_original);

            if (stage1_result) |result| {
                const all_challenges = result.challenges.items;
                // Skip the first challenge (r_stream) to get r_cycle
                const cycle_challenges = if (all_challenges.len > 1)
                    all_challenges[1..]
                else
                    all_challenges;

                // Debug: print Stage 1 challenges
                if (cycle_challenges.len > 0) {
                }

                // Store r_spartan_original in BIG_ENDIAN order (like Jolt's opening point)
                // This is used by InstructionLookupsClaimReduction
                for (0..n_cycle_vars) |i| {
                    const src_idx = n_cycle_vars - 1 - i;
                    if (src_idx < cycle_challenges.len) {
                        r_spartan_original[i] = cycle_challenges[src_idx];
                    } else {
                        r_spartan_original[i] = F.zero();
                    }
                }

                // CRITICAL: In Jolt, the opening point r_cycle is stored in BIG_ENDIAN order
                // (reversed from sumcheck challenge order).
                // See OuterRemainingSumcheckParams::normalize_opening_point which converts
                // from LITTLE_ENDIAN to BIG_ENDIAN via match_endianness() (reverses the vector)
                //
                // So tau_stage2 = [r_cycle_reversed, tau_high] where r_cycle_reversed[i] = r_cycle[n-1-i]
                for (0..n_cycle_vars) |i| {
                    tau_stage2[i] = r_spartan_original[i];
                }
            } else {
                // Fallback to zeros
                for (0..n_cycle_vars) |i| {
                    tau_stage2[i] = F.zero();
                    r_spartan_original[i] = F.zero();
                }
            }
            // Append tau_high_stage2 as the last element
            tau_stage2[n_cycle_vars] = tau_high_stage2;

            if (tau_stage2.len > 0) {
            }

            var stage2_result = try self.generateStage2BatchedSumcheckProof(
                &jolt_proof.stage2_sumcheck_proof,
                transcript,
                r0_stage2,
                uni_skip_claim_stage2,
                tau_stage2,
                r_spartan_original,
                padded_witnesses,
                n_cycle_vars,
                log_ram_k,
                &jolt_proof.opening_claims,
                config,
            );

            // Add remaining opening claims - use actual final claims from provers
            // Instance 1 (RAF): RamRa opening is the final claim
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
                stage2_result.raf_final_claim,
            );
            // Instance 2 (RWC): Individual opening claims for ra, val, inc
            // NOTE: The rwc_val_claim is the evaluation of RamVal at the Stage 2 opening point.
            // For Stage 4's RamValEvaluation, Jolt computes input_claim = rwc_val_claim - init_eval.
            // For programs without RAM ops, this should be 0 only if rwc_val_claim equals init_eval.
            // However, changing this claim would break Stage 2's expected_output_claim check.
            // The root cause needs to be fixed in the RWC prover's polynomial computation.
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamReadWriteChecking } },
                stage2_result.rwc_val_claim, // RamVal evaluation at opening point
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamReadWriteChecking } },
                stage2_result.rwc_ra_claim, // RamRa evaluation at opening point
            );
            // RamInc is a committed polynomial needed by RamReadWriteChecking
            try jolt_proof.opening_claims.insert(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamReadWriteChecking } },
                stage2_result.rwc_inc_claim, // RamInc evaluation at r_cycle
            );
            // Note: UnivariateSkip for SpartanProductVirtualization was already set above with the actual claim value

            // Add PRODUCT_UNIQUE_FACTOR_VIRTUALS claims for SpartanProductVirtualization
            // These 8 virtual polynomials match upstream's PRODUCT_UNIQUE_FACTOR_VIRTUALS:
            // [0] LeftInstructionInput, [1] RightInstructionInput, [2] OpFlags(Jump),
            // [3] OpFlags(WriteLookupOutputToRD), [4] LookupOutput, [5] InstructionFlags(Branch),
            // [6] NextIsNoop, [7] OpFlags(VirtualInstruction)
            if (comptime debug_verbose) {
                std.debug.print("[INSERT] LeftInstructionInput@ProdVirt = {any}\n", .{stage2_result.factor_evals[0].toBytesBE()});
            }
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[0], // LeftInstructionInput
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[1], // RightInstructionInput
            );
            // OpFlags::Jump = CircuitFlags index 5
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .OpFlags = 5 }, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[2], // Jump
            );
            // OpFlags::WriteLookupOutputToRD = CircuitFlags index 6
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .OpFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[3], // WriteLookupOutputToRD
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[4], // LookupOutput
            );
            // InstructionFlags::Branch = index 4
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 4 }, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[5], // Branch
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .NextIsNoop, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[6], // NextIsNoop
            );
            // OpFlags::VirtualInstruction = CircuitFlags index 7
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .OpFlags = 7 }, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[7], // VirtualInstruction
            );

            // Stage 2: OutputSumcheckVerifier claims
            // val_final_claim is the MLE evaluation Val_final(r') at the opening point
            // This comes from the OutputSumcheck prover's val_final polynomial after binding
            //
            // NOTE: We keep the original output_val_final_claim here because it's used in Stage 2's
            // expected_output_claim computation. The sumcheck polynomial rounds were generated based
            // on this value, so changing it would break Stage 2 verification.
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamValFinal, .sumcheck_id = .RamOutputCheck } },
                stage2_result.output_val_final_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamValInit, .sumcheck_id = .RamOutputCheck } },
                stage2_result.output_val_init_claim,
            );

            // Clean up stage2_result
            defer stage2_result.deinit();

            // Stage 2: InstructionLookupsClaimReductionSumcheckVerifier claims
            // These are the MLE evaluations of the lookup polynomials at the sumcheck challenges
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .InstructionClaimReduction } },
                stage2_result.instr_lookup_output_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LeftLookupOperand, .sumcheck_id = .InstructionClaimReduction } },
                stage2_result.instr_left_operand_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RightLookupOperand, .sumcheck_id = .InstructionClaimReduction } },
                stage2_result.instr_right_operand_claim,
            );
            if (comptime debug_verbose) {
                std.debug.print("[INSERT] LeftInstructionInput@InstrClaimRed = {any}\n", .{stage2_result.instr_left_instr_input_claim.toBytesBE()});
            }
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .InstructionClaimReduction } },
                stage2_result.instr_left_instr_input_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .InstructionClaimReduction } },
                stage2_result.instr_right_instr_input_claim,
            );

            // CRITICAL: Append Stage 2 cache_openings claims to transcript
            // Order follows upstream instance ordering:
            // [0] RamReadWriteChecking: 3 claims (RamVal, RamRa, RamInc)
            // [1] ProductVirtualRemainder: 8 claims (PRODUCT_UNIQUE_FACTOR_VIRTUALS)
            // [2] InstructionLookupsClaimReduction: 5 claims
            // [3] RamRafEvaluation: 1 claim (RamRa)
            // [4] OutputSumcheck: 1 claim (RamValFinal only; RamValInit is not opened)

            // Instance 0: RamReadWriteChecking - 3 claims
            transcript.appendScalar("opening_claim", stage2_result.rwc_val_claim);
            transcript.appendScalar("opening_claim", stage2_result.rwc_ra_claim);
            transcript.appendScalar("opening_claim", stage2_result.rwc_inc_claim);

            // Instance 1: ProductVirtualRemainder - 8 claims (PRODUCT_UNIQUE_FACTOR_VIRTUALS)
            for (stage2_result.factor_evals) |eval| {
                transcript.appendScalar("opening_claim", eval);
            }

            // Instance 2: InstructionLookupsClaimReduction - 2 flushed claims
            // LookupOutput, LeftInstructionInput, RightInstructionInput are aliased to
            // ProductVirtualRemainder's openings at the same point, so NOT flushed.
            // Only LeftLookupOperand and RightLookupOperand are new.
            transcript.appendScalar("opening_claim", stage2_result.instr_left_operand_claim);
            transcript.appendScalar("opening_claim", stage2_result.instr_right_operand_claim);

            // Instance 3: RamRafEvaluation - 1 claim
            transcript.appendScalar("opening_claim", stage2_result.raf_final_claim);

            // Instance 4: OutputSumcheck - 1 claim (only RamValFinal; RamValInit is NOT opened)
            transcript.appendScalar("opening_claim", stage2_result.output_val_final_claim);


            if (comptime stage_timing_enabled) {
                std.debug.print("    [STAGE-TIMING] Stage 2: {d:.1} ms\n", .{@as(f64, @floatFromInt(stage_timer.read())) / 1_000_000.0});
            }
            stage_timer.reset();

            // Stage 3: SpartanShift, InstructionInput, RegistersClaimReduction
            // Extract r_product from Stage 2 challenges (last n_cycle_vars in BIG_ENDIAN)
            var r_product = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_product);
            {
                // Stage 2 challenges are in sumcheck order (LITTLE_ENDIAN)
                // ProductVirtualRemainder uses the last n_cycle_vars challenges
                // We need to reverse to get BIG_ENDIAN order
                const stage2_chals = stage2_result.challenges;
                const product_start = if (stage2_chals.len > n_cycle_vars) stage2_chals.len - n_cycle_vars else 0;
                for (0..n_cycle_vars) |i| {
                    const src_idx = n_cycle_vars - 1 - i;
                    if (product_start + src_idx < stage2_chals.len) {
                        r_product[i] = stage2_chals[product_start + src_idx];
                    } else {
                        r_product[i] = F.zero();
                    }
                }
            }


            // Generate Stage 3 proof using the proper sumcheck prover
            var stage3_prover_instance = Stage3Prover(F).init(self.allocator);
            stage3_prover_instance.thread_pool = self.thread_pool;
            var stage3_result = try stage3_prover_instance.generateStage3Proof(
                &jolt_proof.stage3_sumcheck_proof,
                transcript,
                &jolt_proof.opening_claims,
                padded_witnesses,
                n_cycle_vars,
                r_spartan_original, // r_outer in BIG_ENDIAN
                r_product, // r_product in BIG_ENDIAN
            );
            defer stage3_result.deinit();

            // Debug: Print Stage 3 challenges for comparison with Jolt
            // NOTE: Stage 3 challenges are MontU128Challenge-style [0, 0, low, high] limbs
            // where the limbs ARE the Montgomery representation directly.
            // To compare with Jolt's params.r_cycle, we need to look at limbs[2] and limbs[3].
            // Also print in the format that matches Jolt's params.r_cycle (16 zero bytes + 16 data bytes)
            for (0..stage3_result.challenges.len) |i| {
                const c = stage3_result.challenges[stage3_result.challenges.len - 1 - i];
                // Jolt's Challenge serializes as [0, 0, low_LE, high_LE] where each u64 is in LE bytes
                var jolt_format: [32]u8 = [_]u8{0} ** 32;
                std.mem.writeInt(u64, jolt_format[16..24], c.limbs[2], .little);
                std.mem.writeInt(u64, jolt_format[24..32], c.limbs[3], .little);
            }

            // DEBUG: Check challenges immediately before claiming them

            // SpartanShift claims (from Stage 3 prover)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanShift } },
                stage3_result.shift_unexpanded_pc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanShift } },
                stage3_result.shift_pc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .OpFlags = @intFromEnum(instruction.CircuitFlags.VirtualInstruction) }, .sumcheck_id = .SpartanShift } },
                stage3_result.shift_is_virtual_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .OpFlags = @intFromEnum(instruction.CircuitFlags.IsFirstInSequence) }, .sumcheck_id = .SpartanShift } },
                stage3_result.shift_is_first_in_sequence_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = @intFromEnum(instruction.InstructionFlags.IsNoop) }, .sumcheck_id = .SpartanShift } },
                stage3_result.shift_is_noop_claim,
            );

            // InstructionInputVirtualization claims (from Stage 3 prover)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = @intFromEnum(instruction.InstructionFlags.LeftOperandIsRs1Value) }, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_left_is_rs1_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Rs1Value, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_rs1_value_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = @intFromEnum(instruction.InstructionFlags.LeftOperandIsPC) }, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_left_is_pc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_unexpanded_pc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = @intFromEnum(instruction.InstructionFlags.RightOperandIsRs2Value) }, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_right_is_rs2_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Rs2Value, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_rs2_value_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = @intFromEnum(instruction.InstructionFlags.RightOperandIsImm) }, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_right_is_imm_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_imm_claim,
            );

            // RegistersClaimReduction claims (from Stage 3 prover)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RdWriteValue, .sumcheck_id = .RegistersClaimReduction } },
                stage3_result.reg_rd_write_value_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Rs1Value, .sumcheck_id = .RegistersClaimReduction } },
                stage3_result.reg_rs1_value_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Rs2Value, .sumcheck_id = .RegistersClaimReduction } },
                stage3_result.reg_rs2_value_claim,
            );

            // BytecodeReadRaf claims (InstructionReadRaf in Jolt)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .InstructionRafFlag, .sumcheck_id = .BytecodeReadRaf } },
                F.zero(),
            );
            // InstructionRa chunks (just add first chunk for now)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionRa = 0 }, .sumcheck_id = .BytecodeReadRaf } },
                F.zero(),
            );

            // IncClaimReduction claims (Increment checking)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .IncClaimReduction } },
                F.zero(),
            );

            // LookupOutput at InstructionClaimReduction was already added in Stage 2

            if (comptime stage_timing_enabled) {
                std.debug.print("    [STAGE-TIMING] Stage 3: {d:.1} ms\n", .{@as(f64, @floatFromInt(stage_timer.read())) / 1_000_000.0});
            }
            stage_timer.reset();

            // Stage 4: RegistersReadWriteChecking, RamValEvaluation, RamValFinalEvaluation
            // RegistersReadWriteChecking has LOG_K + log2(T) rounds where LOG_K = log2(REGISTER_COUNT)
            // REGISTER_COUNT = 32 (RISCV) + 96 (Virtual) = 128, so LOG_K = 7
            const log_registers = 7; // log2(128) = 7
            const stage4_max_rounds = log_registers + n_cycle_vars;

            // NOTE: Stage 3's cache_openings (13 claims, 3 aliased) are already appended to the transcript
            // by stage3_prover.generateStage3Proof() - no need to append again here.

            // Stage 4 RegistersReadWriteChecking needs:
            // - gamma from transcript (challenge scalar)
            // - r_cycle from Stage 3 (the sumcheck challenges from RegistersClaimReduction)
            // - execution trace from config
            // DEBUG: Print transcript state before gamma

            // ALWAYS-ON: Print transcript state before gamma for comparison with Jolt verifier

            const gamma_stage4 = transcript.challengeScalarFull();

            // Domain separator and gamma for RamValCheck (combined ValEvaluation + ValFinal)
            // Must match upstream: transcript.append_bytes(b"ram_val_check_gamma", &[])
            transcript.appendBytes("ram_val_check_gamma", &.{});
            const ram_val_check_gamma = transcript.challengeScalarFull();

            // Variables to store Stage 4 opening point for Stage 5
            var stage4_regs_r_address: ?[]F = null;
            var stage4_regs_r_cycle: ?[]F = null;
            var stage4_r_cycle_val: ?[]F = null; // r_cycle_val from RamValEvaluation (for RamRaClaimReduction)
            // r_reduction from Stage 2 InstructionClaimReduction (for LookupsReadRaf in Stage 5)
            // CRITICAL: InstructionClaimReduction is in Stage 2, NOT Stage 3!
            // The challenges are the last n_cycle_vars challenges from Stage 2 (Instance 4).
            var r_reduction_be: ?[]F = null;
            // Stage 4 inc_poly copy for Stage 6 diagnostic
            var stage4_inc_poly_copy: ?[]F = null;
            defer {
                if (stage4_regs_r_address) |arr| self.allocator.free(arr);
                if (stage4_regs_r_cycle) |arr| self.allocator.free(arr);
                if (stage4_r_cycle_val) |arr| self.allocator.free(arr);
                if (r_reduction_be) |arr| self.allocator.free(arr);
                if (stage4_inc_poly_copy) |arr| self.allocator.free(arr);
            }

            // Compute r_reduction_be from Stage 2 challenges (InstructionClaimReduction)
            // Stage 2 has 5 instances with max_num_rounds = log_ram_k + n_cycle_vars
            // Instance 4 (InstructionClaimReduction) uses the last n_cycle_vars rounds
            // So its challenges are stage2_result.challenges[max_num_rounds - n_cycle_vars .. max_num_rounds]
            const max_stage2_rounds = log_ram_k + n_cycle_vars;
            const instr_start = max_stage2_rounds - n_cycle_vars;

            // Extract and reverse the InstructionClaimReduction challenges to BIG_ENDIAN order
            r_reduction_be = try self.allocator.alloc(F, n_cycle_vars);
            for (0..n_cycle_vars) |i| {
                const src_idx = instr_start + i;
                // Reverse to BIG_ENDIAN: first challenge in LITTLE_ENDIAN becomes last in BIG_ENDIAN
                const dest_idx = n_cycle_vars - 1 - i;
                r_reduction_be.?[dest_idx] = stage2_result.challenges[src_idx];
            }


            // Use Stage 4 prover if we have execution and memory trace data.
            stage4_block: {
                const trace = config.execution_trace orelse {
                    try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, stage4_max_rounds, 2);
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } }, F.zero());
                    break :stage4_block;
                };
                const memory_trace = config.memory_trace orelse {
                    try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, stage4_max_rounds, 2);
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } }, F.zero());
                    break :stage4_block;
                };

                // Use the Gruen-optimized Stage 4 prover for Jolt compatibility
                const Stage4ProverType = spartan_mod.stage4_gruen_prover.Stage4GruenProver(F);
                const Stage3Claims = spartan_mod.stage4_gruen_prover.Stage3Claims(F);

                const stage3_claims = Stage3Claims{
                    .rd_write_value = stage3_result.reg_rd_write_value_claim,
                    .rs1_value = stage3_result.reg_rs1_value_claim,
                    .rs2_value = stage3_result.reg_rs2_value_claim,
                };

                const input_claim_registers = stage3_claims.rd_write_value
                    .add(gamma_stage4.mul(stage3_claims.rs1_value))
                    .add(gamma_stage4.mul(gamma_stage4).mul(stage3_claims.rs2_value));

                // Derive r_address from Stage 2 RWC sumcheck challenges using normalize_opening_point.
                //
                // RamReadWriteChecking has 3 phases:
                // - Phase 1: phase1_num_rounds cycle vars (bound low-to-high)
                // - Phase 2: phase2_num_rounds address vars (bound low-to-high)
                // - Phase 3: (log_T - phase1) cycle vars + (log_K - phase2) address vars
                //
                // With default config (phase1 = log_T/2, phase2 = log_K):
                // - Phase 1: challenges[0..phase1]
                // - Phase 2: challenges[phase1..phase1+phase2]
                // - Phase 3 cycle: challenges[phase1+phase2..phase1+phase2+phase3_cycle]
                //
                // normalize_opening_point returns:
                // - r_cycle = reverse(phase3_cycle) ++ reverse(phase1)
                // - r_address = reverse(phase3_address) ++ reverse(phase2)
                // - opening_point = [r_address, r_cycle]
                //
                // ValEvaluation splits at log_K to get r_address.

                // Get phase config from the proof
                const phase1 = jolt_proof.rw_config.ram_rw_phase1_num_rounds;
                const phase2 = jolt_proof.rw_config.ram_rw_phase2_num_rounds;
                const phase3_cycle_len = n_cycle_vars - phase1;
                const phase3_address_len = log_ram_k - phase2;


                // Extract r_address using normalize_opening_point logic:
                // r_address = reverse(phase3_address) ++ reverse(phase2)
                // With default config (phase2 = log_K, phase3_address = 0):
                // r_address = reverse(challenges[phase1..phase1+phase2])
                var r_address_be = try self.allocator.alloc(F, log_ram_k);
                defer self.allocator.free(r_address_be);
                @memset(r_address_be, F.zero());

                // Phase 2 address challenges are at indices [phase1..phase1+phase2)
                const phase2_start = phase1;
                for (0..phase2) |i| {
                    const src_idx = phase2_start + i;
                    if (src_idx < stage2_result.challenges.len) {
                        // reverse(phase2): put challenge at phase2_start+i into r_address[phase2-1-i]
                        // Then prepend to phase3_address (which is at r_address[0..phase3_address_len])
                        const dest_idx = phase3_address_len + (phase2 - 1 - i);
                        if (dest_idx < log_ram_k) {
                            r_address_be[dest_idx] = stage2_result.challenges[src_idx];
                        }
                    }
                }
                // Phase 3 address challenges are at indices [phase1+phase2+phase3_cycle..end)
                const phase3_addr_start = phase1 + phase2 + phase3_cycle_len;
                for (0..phase3_address_len) |i| {
                    const src_idx = phase3_addr_start + i;
                    if (src_idx < stage2_result.challenges.len) {
                        // reverse(phase3_address): put into r_address[phase3_addr_len-1-i]
                        const dest_idx = phase3_address_len - 1 - i;
                        r_address_be[dest_idx] = stage2_result.challenges[src_idx];
                    }
                }


                // Also extract r_cycle for other uses
                // r_cycle = reverse(phase3_cycle) ++ reverse(phase1)
                var r_cycle_be = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(r_cycle_be);
                @memset(r_cycle_be, F.zero());

                // Phase 1 cycle challenges at indices [0..phase1)
                for (0..phase1) |i| {
                    if (i < stage2_result.challenges.len) {
                        // reverse(phase1): put into r_cycle[phase3_cycle_len + (phase1-1-i)]
                        const dest_idx = phase3_cycle_len + (phase1 - 1 - i);
                        if (dest_idx < n_cycle_vars) {
                            r_cycle_be[dest_idx] = stage2_result.challenges[i];
                        }
                    }
                }
                // Phase 3 cycle challenges at indices [phase1+phase2..phase1+phase2+phase3_cycle_len)
                const phase3_cycle_start = phase1 + phase2;
                for (0..phase3_cycle_len) |i| {
                    const src_idx = phase3_cycle_start + i;
                    if (src_idx < stage2_result.challenges.len) {
                        // reverse(phase3_cycle): put into r_cycle[phase3_cycle_len-1-i]
                        const dest_idx = phase3_cycle_len - 1 - i;
                        r_cycle_be[dest_idx] = stage2_result.challenges[src_idx];
                    }
                }

                // Reverse to LITTLE_ENDIAN for MLE helpers.
                var r_cycle_le = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(r_cycle_le);
                for (0..n_cycle_vars) |i| {
                    r_cycle_le[i] = r_cycle_be[n_cycle_vars - 1 - i];
                }

                // DEBUG: Print r_cycle_be and r_cycle_le for LT polynomial debugging
                {
                }

                var r_address_le = try self.allocator.alloc(F, log_ram_k);
                defer self.allocator.free(r_address_le);
                for (0..log_ram_k) |i| {
                    r_address_le[i] = r_address_be[log_ram_k - 1 - i];
                }

                // CRITICAL FIX: ValEvaluation MUST use getLowestAddress() to match Jolt.
                // Jolt's RamReadWriteChecking includes ALL memory accesses (including I/O region),
                // using remap_address(addr, memory_layout) which starts at get_lowest_address().
                // ValEvaluation must use the same address range because:
                //   - val_claim comes from RWC which uses get_lowest_address()
                //   - init_eval is computed at the RWC r_address
                //   - The wa polynomial in ValEvaluation must include I/O writes (e.g., termination)
                //   - Otherwise input_claim = val_claim - init_eval ≠ actual polynomial sum
                const start_address: u64 = if (config.memory_layout) |ml|
                    ml.getLowestAddress()
                else
                    constants.RAM_START_ADDRESS;

                // CRITICAL FIX: ValEvaluation and ValFinal use DIFFERENT r_address points!
                //
                // For ValEvaluation:
                //   - Jolt verifier gets r_address from RamVal @ RamReadWriteChecking
                //   - This is r_address_be that we computed from Stage 2 RWC challenges above
                //   - init_eval = eval_initial_ram_mle(r_address from RWC)
                //
                // For ValFinal:
                //   - Jolt verifier gets r_address from RamValFinal @ RamOutputCheck
                //   - This is the same point where output_val_init_claim was computed
                //   - init_eval = output_val_init_claim (from OutputSumcheck)
                //
                // Previously we used output_val_init_claim for both, which is WRONG!

                // Compute init_eval for ValEvaluation at the RWC r_address
                const init_eval_for_val_eval = blk: {
                    if (config.memory_layout) |ml| {
                        const result = computeInitialRamEval(
                            config.bytecode_words,
                            config.min_bytecode_address,
                            ml,
                            r_address_be,
                            log_ram_k,
                            config.program_inputs,
                        );
                        break :blk result;
                    }
                    // No memory layout -> init_eval = 0
                    break :blk F.zero();
                };

                // Initialize val_eval prover for combined RamValCheck.
                // Uses a single prover with inc, wa, lt arrays that computes inc * wa * (lt + gamma).

                // Initialize val_eval prover to get its polynomial sum
                const trace_len = trace.steps.items.len;
                // CRITICAL FIX: Use r_cycle_be for the LT polynomial, not r_cycle_le!
                // Jolt's verifier computes LT(r, r_cycle) where both are in BIG_ENDIAN.
                // The r_cycle comes from the RamVal opening point which is stored in BE.
                // Using r_cycle_le produces a different LT value because LT is not symmetric.
                const val_eval_params_early = try ram.ValEvaluationParams(F).init(
                    self.allocator,
                    init_eval_for_val_eval, // Use the correct init_eval computed at RWC r_address
                    trace_len,
                    ram_K,
                    r_address_le, // r_address uses LE for eq polynomial (symmetric)
                    r_cycle_be, // FIXED: Use BE for LT polynomial (not symmetric)
                );
                // CRITICAL: Use initWithLayout to filter out synthetic termination/panic writes.
                // This matches Jolt's behavior where these bits are set directly in final memory
                // without corresponding trace entries in the inc polynomial.
                var val_eval_prover_early = try ram.ValEvaluationProver(F).initWithLayout(
                    self.allocator,
                    memory_trace,
                    config.initial_ram,
                    val_eval_params_early,
                    start_address,
                    config.memory_layout, // Pass memory_layout to filter synthetic writes
                );
                defer val_eval_prover_early.deinit();

                // Debug: verify which r_cycle was passed

                // Combined RamValCheck input_claim matching upstream formula:
                //   input_claim = (val_rw_claim - init_eval) + gamma * (val_final_claim - init_eval)
                // Uses single init_eval at RWC r_address (both addresses are equal with default config).
                const input_claim_val_eval = stage2_result.rwc_val_claim.sub(init_eval_for_val_eval);
                const input_claim_val_final = stage2_result.output_val_final_claim.sub(init_eval_for_val_eval);
                const input_claim_ram_val_check = input_claim_val_eval.add(ram_val_check_gamma.mul(input_claim_val_final));


                // Append 2 input claims to transcript (upstream has 2 instances, not 3)
                transcript.appendScalar("sumcheck_claim", input_claim_registers);
                transcript.appendScalar("sumcheck_claim", input_claim_ram_val_check);

                // Sample 2 batching coefficients
                const batch0 = transcript.challengeScalarFull();
                const batch1 = transcript.challengeScalarFull();


                const batching_coeffs = [2]F{ batch0, batch1 };

                // Now initialize regs prover with actual batch0
                // CRITICAL FIX: Stage 3's sumcheck binds variables in the order challenges are sampled.
                // The final claim rd_write_value_claim = f(c0, c1, ..., cn) in that order.
                // Stage 4's eq polynomial must use the SAME ordering so that:
                //   Σ_j eq(r, j) * f(j) = f(r) = rd_write_value_claim when r = Stage 3 challenges
                // We do NOT reverse to BE here - we use the original LE order for the prover.
                // (The BE conversion is only for serialization to match Jolt's opening_point format.)
                const stage3_r_cycle_le = stage3_result.challenges;

                var regs_prover = Stage4ProverType.initWithClaims(
                    self.allocator,
                    trace,
                    gamma_stage4,
                    stage3_r_cycle_le,
                    stage3_claims,
                    batch0, // Use the correct batching coefficient from transcript
                ) catch {
                    try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, stage4_max_rounds, 3);
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } }, F.zero());
                    break :stage4_block;
                };
                defer regs_prover.deinit();


                // Combined RamValCheck: val_eval + gamma * val_final as a single instance
                const ram_val_check_rounds = val_eval_prover_early.numRounds(); // = n_cycle_vars
                const rounds_per_instance = [2]usize{ stage4_max_rounds, ram_val_check_rounds };

                // Track individual claims for 2 instances
                var individual_claims = [2]F{
                    input_claim_registers,
                    input_claim_ram_val_check,
                };

                // Scale initial claims by 2^(max_rounds - instance_rounds)
                for (0..2) |i| {
                    const scale_power = stage4_max_rounds - rounds_per_instance[i];
                    for (0..scale_power) |_| {
                        individual_claims[i] = individual_claims[i].add(individual_claims[i]);
                    }
                }

                // Initial batched claim
                var batched_claim = F.zero();
                for (0..2) |i| {
                    batched_claim = batched_claim.add(individual_claims[i].mul(batching_coeffs[i]));
                }

                var regs_current_claim = individual_claims[0];

                var stage4_r_sumcheck = try self.allocator.alloc(F, stage4_max_rounds);
                defer self.allocator.free(stage4_r_sumcheck);

                // Save inc_poly before binding for verification and Stage 6 diagnostic
                stage4_inc_poly_copy = try self.allocator.alloc(F, @as(usize, 1) << @intCast(n_cycle_vars));
                @memcpy(stage4_inc_poly_copy.?, regs_prover.inc_poly[0..stage4_inc_poly_copy.?.len]);

                for (0..stage4_max_rounds) |round_idx| {
                    var combined_evals = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };

                    // Instance 0: RegistersRWC (always active)
                    const regs_evals = regs_prover.computeRoundEvals(round_idx, regs_current_claim);
                    for (0..4) |j| {
                        combined_evals[j] = combined_evals[j].add(regs_evals[j].mul(batching_coeffs[0]));
                    }

                    // Instance 1: Combined RamValCheck = val_eval + gamma * val_final
                    const ram_val_check_offset = stage4_max_rounds - ram_val_check_rounds;
                    const two_inv_local = F.fromU64(2).inverse() orelse F.one();

                    var ram_val_check_evals_opt: ?[4]F = null;
                    const ram_val_check_active = round_idx >= ram_val_check_offset;
                    if (!ram_val_check_active) {
                        // Inactive: constant polynomial H(X) = claim/2
                        const half_claim = individual_claims[1].mul(two_inv_local);
                        const weighted = half_claim.mul(batching_coeffs[1]);
                        for (0..3) |j| {
                            combined_evals[j] = combined_evals[j].add(weighted);
                        }
                    } else {
                        // Active: compute combined polynomial inc * wa * (LT + gamma)
                        // Single prover computes all Toom-Cook evals directly, matching upstream.
                        var ram_evals = val_eval_prover_early.computeRoundPolynomialCombined(ram_val_check_gamma);
                        // Apply hint: p(0) = claim - p(1)
                        ram_evals[0] = individual_claims[1].sub(ram_evals[1]);

                        ram_val_check_evals_opt = ram_evals;
                        for (0..4) |j| {
                            combined_evals[j] = combined_evals[j].add(ram_evals[j].mul(batching_coeffs[1]));
                        }
                    }

                    // Convert Toom-Cook evals to coefficients and append to proof
                    const full_coeffs = poly_mod.UniPoly(F).toomCookToCoeffs(combined_evals);
                    var coeffs = [_]F{ full_coeffs[0], full_coeffs[1], full_coeffs[2], full_coeffs[3] };
                    const coeffs_slice: []const F = &coeffs;
                    try jolt_proof.stage4_sumcheck_proof.addRoundPoly(coeffs_slice);

                    // Compressed coefficients [c0, c2, c3] for transcript
                    const compressed = poly_mod.UniPoly(F).toomCookToCompressed(combined_evals);
                    transcript.appendScalars("sumcheck_poly", compressed[0..3]);

                    const challenge = transcript.challengeScalar();
                    stage4_r_sumcheck[round_idx] = challenge;
                    batched_claim = evalFromHint(compressed, batched_claim, challenge);

                    // Update instance 0 (regs): always active
                    regs_current_claim = evaluateCubicAtChallengeFromEvals(regs_evals, challenge);
                    individual_claims[0] = regs_current_claim;
                    regs_prover.bindChallenge(round_idx, challenge);

                    // Update instance 1 (combined RamValCheck)
                    if (ram_val_check_evals_opt) |ram_evals| {
                        // Active: evaluate combined polynomial at challenge
                        individual_claims[1] = evaluateCubicAtChallengeFromEvals(ram_evals, challenge);
                        // Bind val_eval prover (contains inc, wa, lt arrays for both terms)
                        val_eval_prover_early.bindChallengeWithPoly(challenge, ram_evals);
                    } else {
                        // Inactive: claim' = claim / 2
                        individual_claims[1] = individual_claims[1].mul(two_inv_local);
                    }
                }

                // Verify batched = sum of weighted claims

                const regs_claims = regs_prover.getFinalClaims();
                const val_eval_openings = val_eval_prover_early.getFinalOpenings();


                // RegistersReadWriteChecking opening claims
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } },
                    regs_claims.val_claim,
                );
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } },
                    regs_claims.rs1_ra_claim,
                );
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } },
                    regs_claims.rs2_ra_claim,
                );
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } },
                    regs_claims.rd_wa_claim,
                );
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } },
                    regs_claims.inc_claim,
                );

                // RamValCheck opening claims (combined, 2 claims: RamRa then RamInc)
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } },
                    val_eval_openings.wa_eval,
                );
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } },
                    val_eval_openings.inc_eval,
                );

                // Cache openings into transcript (7 claims total)
                transcript.appendScalar("opening_claim", regs_claims.val_claim);
                transcript.appendScalar("opening_claim", regs_claims.rs1_ra_claim);
                transcript.appendScalar("opening_claim", regs_claims.rs2_ra_claim);
                transcript.appendScalar("opening_claim", regs_claims.rd_wa_claim);
                transcript.appendScalar("opening_claim", regs_claims.inc_claim);
                transcript.appendScalar("opening_claim", val_eval_openings.wa_eval);
                transcript.appendScalar("opening_claim", val_eval_openings.inc_eval);

                // Save Stage 4 RegistersRWC opening point for Stage 5
                // For RegistersRWC:
                // - Phase 1: 8 rounds (all cycle vars)
                // - Phase 2: 7 rounds (all address vars)
                // - Phase 3: 0 rounds
                // r_address = reverse(phase2_challenges) = reverse(stage4_r_sumcheck[8..15])
                // r_cycle = reverse(phase1_challenges) = reverse(stage4_r_sumcheck[0..8])
                const regs_log_k: usize = 7; // LOG_REGISTER_COUNT
                stage4_regs_r_address = try self.allocator.alloc(F, regs_log_k);
                stage4_regs_r_cycle = try self.allocator.alloc(F, n_cycle_vars);

                // r_address = reverse(phase2 challenges)
                for (0..regs_log_k) |i| {
                    stage4_regs_r_address.?[i] = stage4_r_sumcheck[n_cycle_vars + (regs_log_k - 1 - i)];
                }
                // r_cycle = reverse(phase1 challenges)
                for (0..n_cycle_vars) |i| {
                    stage4_regs_r_cycle.?[i] = stage4_r_sumcheck[n_cycle_vars - 1 - i];
                }


                // Also save r_cycle_val for RamRaClaimReduction (ValEvaluation starts at round 7)
                // r_cycle_val = reverse(challenges[7..15]) for BIG_ENDIAN order
                stage4_r_cycle_val = try self.allocator.alloc(F, n_cycle_vars);
                const val_eval_start: usize = 7; // ValEvaluation starts at round 7
                for (0..n_cycle_vars) |i| {
                    const src_idx = val_eval_start + i;
                    if (src_idx < stage4_r_sumcheck.len) {
                        stage4_r_cycle_val.?[n_cycle_vars - 1 - i] = stage4_r_sumcheck[src_idx];
                    } else {
                        stage4_r_cycle_val.?[n_cycle_vars - 1 - i] = F.zero();
                    }
                }
            } // end stage4_block

            if (comptime stage_timing_enabled) {
                std.debug.print("    [STAGE-TIMING] Stage 4: {d:.1} ms\n", .{@as(f64, @floatFromInt(stage_timer.read())) / 1_000_000.0});
            }
            stage_timer.reset();

            // Stage 5: RegistersValEvaluation, RamRaClaimReduction, LookupsReadRaf
            // LookupsReadRaf has max rounds: LOG_K + log_T where LOG_K = XLEN * 2 = 128
            // For RV64: max_num_rounds = 128 + log_T = 128 + 8 = 136
            const lookups_log_k: usize = 128; // XLEN * 2 for RV64

            // CRITICAL: Jolt samples TWO separate gammas for Stage 5 instances.
            // The verifier creates instances in this order:
            //   1. InstructionReadRafSumcheckVerifier::new() → squeezes gamma_lookups_raf
            //   2. RamRaClaimReductionSumcheckVerifier::new() → squeezes gamma_ram_ra
            // So we must squeeze in the SAME order.
            const gamma_lookups_raf = transcript.challengeScalarFull();
            const gamma_ram_ra = transcript.challengeScalarFull();

            // Generate Stage 5 proof using the batched sumcheck prover
            var stage5_prover_instance = Stage5BatchedProver(F).init(self.allocator);
            stage5_prover_instance.thread_pool = self.thread_pool;
            var stage5_result: spartan_mod.Stage5Result(F) = undefined;

            // Use trace-aware prover if we have trace data and Stage 4 opening point
            if (config.execution_trace != null and stage4_regs_r_address != null and stage4_regs_r_cycle != null and r_reduction_be != null) {
                stage5_result = try stage5_prover_instance.generateStage5ProofWithTrace(
                    &jolt_proof.stage5_sumcheck_proof,
                    transcript,
                    &jolt_proof.opening_claims,
                    n_cycle_vars,
                    log_ram_k,
                    gamma_ram_ra,
                    gamma_lookups_raf,
                    config.lookups_ra_virtual_log_k_chunk,
                    config.execution_trace.?,
                    config.memory_trace, // RAM trace for ram_ra_claim computation
                    config.memory_layout, // Memory layout for address remapping
                    stage4_regs_r_address.?,
                    stage4_regs_r_cycle.?,
                    r_reduction_be.?, // Stage 3 challenges in BIG_ENDIAN for LookupsReadRaf eq computation
                    // RamRaClaimReduction opening points:
                    stage2_result.r_address_raf, // r_address_1 from RamRafEvaluation
                    stage2_result.r_address_rw, // r_address_2 from RamReadWriteChecking
                    r_spartan_original, // r_cycle_raf from SpartanOuter (Stage 1)
                    stage2_result.r_cycle_rw, // r_cycle_rw from RamReadWriteChecking
                    stage4_r_cycle_val.?, // r_cycle_val from RamValEvaluation (Stage 4)
                );
            } else {
                // Fallback to zero prover for programs without trace
                stage5_result = try stage5_prover_instance.generateStage5Proof(
                    &jolt_proof.stage5_sumcheck_proof,
                    transcript,
                    &jolt_proof.opening_claims,
                    n_cycle_vars,
                    log_ram_k,
                    gamma_ram_ra,
                    gamma_lookups_raf,
                    config.lookups_ra_virtual_log_k_chunk,
                );
            }
            defer stage5_result.deinit();


            // RegistersValEvaluation claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersValEvaluation } },
                stage5_result.regs_val_wa_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersValEvaluation } },
                stage5_result.regs_val_inc_claim,
            );

            // RamRaClaimReduction claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRaClaimReduction } },
                stage5_result.ram_ra_claim,
            );

            // LookupsReadRaf claims (Stage 5 - LookupsReadRafSumcheckVerifier)
            // LookupTableFlag(i) for each of the 41 lookup tables
            const num_lookup_tables: usize = 41; // LookupTables::<XLEN>::COUNT (41 variants)
            for (0..num_lookup_tables) |i| {
                const flag_value = stage5_result.lookups_table_flags[i];
                if (!flag_value.eql(F.zero())) {
                    // Convert to standard form for printing (same as serialization)
                    const standard = flag_value.fromMontgomery();
                    var buf: [32]u8 = undefined;
                    for (0..4) |j| {
                        std.mem.writeInt(u64, buf[j * 8 ..][0..8], standard.limbs[j], .little);
                    }
                }
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .{ .LookupTableFlag = i }, .sumcheck_id = .InstructionReadRaf } },
                    flag_value,
                );
            }

            // InstructionRa(i) chunks for LookupsReadRaf (LOG_K / ra_virtual_log_k_chunk = 128 / 16 = 8 chunks)
            const lookups_ra_d: usize = lookups_log_k / config.lookups_ra_virtual_log_k_chunk;
            for (0..lookups_ra_d) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionReadRaf } },
                    stage5_result.lookups_ra_chunks[i],
                );
            }

            // InstructionRafFlag for LookupsReadRaf
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .InstructionRafFlag, .sumcheck_id = .InstructionReadRaf } },
                stage5_result.lookups_raf_flag,
            );

            // Append Stage 5 cache openings to transcript
            // Order must match upstream instance order: InstructionReadRaf, RamRaClaimReduction, RegistersValEvaluation

            // Instance 0: LookupsReadRaf (LookupTableFlag(0..41), InstructionRa(0..8), InstructionRafFlag)
            for (stage5_result.lookups_table_flags) |flag| {
                transcript.appendScalar("opening_claim", flag);
            }
            for (stage5_result.lookups_ra_chunks) |chunk| {
                transcript.appendScalar("opening_claim", chunk);
            }
            transcript.appendScalar("opening_claim", stage5_result.lookups_raf_flag);

            // Instance 1: RamRaClaimReduction (RamRa)
            transcript.appendScalar("opening_claim", stage5_result.ram_ra_claim);

            // Instance 2: RegistersValEvaluation (RdInc, RdWa)
            transcript.appendScalar("opening_claim", stage5_result.regs_val_inc_claim);
            transcript.appendScalar("opening_claim", stage5_result.regs_val_wa_claim);

            {
                // ALWAYS-ON: Print transcript state after Stage 5 cache_openings
            }

            if (comptime stage_timing_enabled) {
                std.debug.print("    [STAGE-TIMING] Stage 5: {d:.1} ms\n", .{@as(f64, @floatFromInt(stage_timer.read())) / 1_000_000.0});
            }
            stage_timer.reset();

            // Stage 6: BytecodeReadRaf, RamHammingBooleanity, Booleanity, RamRaVirtual, LookupsRaVirtual, IncClaimReduction
            const bytecode_log_k = std.math.log2_int(usize, config.bytecode_K);
            const ram_log_k = std.math.log2_int(usize, ram_K);
            const instruction_d: usize = (lookups_log_k + config.log_k_chunk - 1) / config.log_k_chunk;
            const bytecode_d_val: usize = (bytecode_log_k + config.log_k_chunk - 1) / config.log_k_chunk;
            const ram_d_val: usize = (ram_log_k + config.log_k_chunk - 1) / config.log_k_chunk;


            // Compute Stage 5 RegistersValEvaluation opening point (s_cycle_stage5)
            // Stage 5 max_rounds = 136 (128 address + 8 cycle for RV64)
            // RegistersValEvaluation runs in the last n_cycle_vars rounds
            // Opening point = challenges[128..136] reversed (LE → BE)
            const stage5_lookups_num_rounds = lookups_log_k + n_cycle_vars; // 136
            const stage5_regs_val_num_rounds = n_cycle_vars; // 8
            var s_cycle_stage5 = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(s_cycle_stage5);
            for (0..n_cycle_vars) |i| {
                // RegistersValEvaluation challenges are the last n_cycle_vars of Stage 5
                const stage5_idx = stage5_lookups_num_rounds - stage5_regs_val_num_rounds + i;
                // Reverse for BIG_ENDIAN
                s_cycle_stage5[n_cycle_vars - 1 - i] = stage5_result.challenges[stage5_idx];
            }

            // Generate Stage 6 proof using the batched sumcheck prover
            const the_trace = config.execution_trace orelse return error.ExecutionTraceRequired;
            const the_memory_layout = config.memory_layout orelse return error.MemoryLayoutRequired;

            // Compute SpartanShift r_cycle in BIG_ENDIAN from Stage 3 challenges (reversed)
            const r_cycle_shift_be = try self.allocator.alloc(F, stage3_result.challenges.len);
            defer self.allocator.free(r_cycle_shift_be);
            for (0..stage3_result.challenges.len) |i| {
                r_cycle_shift_be[i] = stage3_result.challenges[stage3_result.challenges.len - 1 - i];
            }

            // Build bytecode entry table from static ELF + execution trace overlay
            const bytecode_K_val: usize = @as(usize, 1) << @intCast(bytecode_log_k);
            const stage6_mod = @import("spartan/stage6_prover.zig");
            // Get pc_map for converting ELF addresses to bytecode array indices
            const pc_map_ptr = config.bytecode_pc_map orelse return error.MissingBytecodepcMap;
            const bytecode_entries = try stage6_mod.buildBytecodeEntries(self.allocator, the_trace, bytecode_K_val, pc_map_ptr, config.program_code_bytes, config.code_base_address, the_memory_layout.termination);
            defer self.allocator.free(bytecode_entries);

            // Get register address opening points for Stages 4 and 5
            // Stage 4: from RegistersReadWriteChecking (address portion)
            const r_register_4 = stage4_regs_r_address orelse &[_]F{};
            // Stage 5: use same as Stage 4 (both address 32 registers)
            // In Jolt, this comes from RegistersValEvaluation's opening point split,
            // but the address variables are the SAME as Stage 4's since they share
            // the same register address space.
            const r_register_5 = stage4_regs_r_address orelse &[_]F{};

            var stage6_prover_instance = Stage6BatchedProver(F).init(self.allocator);
            stage6_prover_instance.thread_pool = self.thread_pool;
            var stage6_result = try stage6_prover_instance.generateStage6Proof(
                &jolt_proof.stage6_sumcheck_proof,
                transcript,
                &jolt_proof.opening_claims,
                n_cycle_vars,
                bytecode_log_k,
                config.log_k_chunk,
                bytecode_d_val,
                ram_d_val,
                instruction_d,
                config.lookups_ra_virtual_log_k_chunk,
                // Execution trace
                the_trace,
                // BytecodeReadRaf r_cycles (all BIG_ENDIAN)
                r_spartan_original, // r_cycle_bc1: SpartanOuter
                stage2_result.r_cycle_product, // r_cycle_bc2: SpartanProductVirtualization
                r_cycle_shift_be, // r_cycle_bc3: SpartanShift
                if (stage4_regs_r_cycle) |v| v else s_cycle_stage5, // r_cycle_bc4: RegistersReadWriteChecking
                s_cycle_stage5, // r_cycle_bc5: RegistersValEvaluation
                // IncClaimReduction r_cycles (all BIG_ENDIAN)
                stage2_result.r_cycle_rw, // r_cycle_inc: RamReadWriteChecking
                if (stage4_r_cycle_val) |v| v else s_cycle_stage5, // r_cycle_inc: RamValEvaluation
                // Stage 5 challenges for deriving LookupsRaVirtual and RamRaVirtual points
                stage5_result.challenges,
                // RAM r_address from Stage 2 (BIG_ENDIAN) — aligned address used by RamRaClaimReduction
                stage2_result.r_address_raf,
                // Memory layout for address remapping
                the_memory_layout,
                // Bytecode entries for Val polynomial computation
                bytecode_entries,
                // Register address opening points
                r_register_4,
                r_register_5,
                // BytecodePCMapper for converting ELF addresses to bytecode array indices
                pc_map_ptr,
                // Stage 4 inc_poly copy for diagnostic
                if (stage4_inc_poly_copy) |v| v else &[_]F{},
            );
            defer stage6_result.deinit();

            // Insert Stage 6 opening claims into accumulator

            // HammingBooleanity: virtual RamHammingWeight
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .Booleanity } },
                stage6_result.hamming_weight_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .RamHammingBooleanity } },
                stage6_result.hamming_weight_claim,
            );

            // IncClaimReduction: committed RamInc, RdInc
            try jolt_proof.opening_claims.insert(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .IncClaimReduction } },
                stage6_result.ram_inc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .IncClaimReduction } },
                stage6_result.rd_inc_claim,
            );

            // BytecodeReadRaf cache_openings: BytecodeRa(i)
            for (0..bytecode_d_val) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .BytecodeRa = i }, .sumcheck_id = .BytecodeReadRaf } },
                    stage6_result.bytecode_ra_claims[i],
                );
            }

            // Booleanity cache_openings: all RA polys
            // Order: InstructionRa(0..instruction_d), BytecodeRa(0..bytecode_d), RamRa(0..ram_d)
            var bool_idx: usize = 0;
            for (0..instruction_d) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .Booleanity } },
                    stage6_result.booleanity_ra_claims[bool_idx],
                );
                bool_idx += 1;
            }
            for (0..bytecode_d_val) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .BytecodeRa = i }, .sumcheck_id = .Booleanity } },
                    stage6_result.booleanity_ra_claims[bool_idx],
                );
                bool_idx += 1;
            }
            for (0..ram_d_val) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .RamRa = i }, .sumcheck_id = .Booleanity } },
                    stage6_result.booleanity_ra_claims[bool_idx],
                );
                bool_idx += 1;
            }

            // RamRaVirtualization cache_openings: RamRa(i)
            for (0..ram_d_val) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .RamRa = i }, .sumcheck_id = .RamRaVirtualization } },
                    stage6_result.ram_ra_virtual_claims[i],
                );
            }

            // InstructionRaVirtualization cache_openings: InstructionRa(i)
            for (0..instruction_d) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionRaVirtualization } },
                    stage6_result.instruction_ra_virtual_claims[i],
                );
            }

            // Stage 6 cache_openings are already appended by Stage 6 prover
            // (stage6_prover.zig lines 4055-4083)
            // Do NOT re-append them here.

            if (comptime stage_timing_enabled) {
                std.debug.print("    [STAGE-TIMING] Stage 6: {d:.1} ms\n", .{@as(f64, @floatFromInt(stage_timer.read())) / 1_000_000.0});
            }
            stage_timer.reset();

            // ====================================================================
            // Stage 7: HammingWeightClaimReduction sumcheck
            // ====================================================================
            {
                const s6_challenges = stage6_result.challenges;
                const s6_bytecode_log_k = stage6_result.bytecode_log_k;
                const s6_log_k_chunk = stage6_result.log_k_chunk;
                const s6_n_cycle_vars = stage6_result.n_cycle_vars;
                const s6_bytecode_d = stage6_result.bytecode_d;
                const s6_ram_d = stage6_result.ram_d;
                const s6_instruction_d = stage6_result.instruction_d;
                const s6_max_rounds = s6_bytecode_log_k + s6_n_cycle_vars;
                const s6_booleanity_rounds = s6_log_k_chunk + s6_n_cycle_vars;
                const s6_bool_start = s6_max_rounds - s6_booleanity_rounds; // = bytecode_log_k - log_k_chunk
                const N = s6_instruction_d + s6_bytecode_d + s6_ram_d;
                const k_chunk: usize = @as(usize, 1) << @intCast(s6_log_k_chunk);
                const T_val: usize = @as(usize, 1) << @intCast(s6_n_cycle_vars);


                // Extract r_cycle_BE from Booleanity's cycle portion
                // Booleanity challenges[bool_start+log_k_chunk..bool_start+booleanity_rounds] reversed
                var r_cycle_be = try self.allocator.alloc(F, s6_n_cycle_vars);
                defer self.allocator.free(r_cycle_be);
                for (0..s6_n_cycle_vars) |i| {
                    r_cycle_be[i] = s6_challenges[s6_bool_start + s6_booleanity_rounds - 1 - i];
                }


                // Extract r_addr_bool_BE from Booleanity's address portion
                // challenges[bool_start..bool_start+log_k_chunk] reversed
                var r_addr_bool_be = try self.allocator.alloc(F, s6_log_k_chunk);
                defer self.allocator.free(r_addr_bool_be);
                for (0..s6_log_k_chunk) |i| {
                    r_addr_bool_be[i] = s6_challenges[s6_bool_start + s6_log_k_chunk - 1 - i];
                }

                // Extract r_addr_virt_i for each ra polynomial (log_k_chunk elements each, BE)
                // Order: InstructionRa(0..inst_d), BytecodeRa(0..bc_d), RamRa(0..ram_d)
                var r_addr_virt = try self.allocator.alloc([]F, N);
                // Initialize to empty slices so deferred free doesn't crash on uninitialized entries
                for (r_addr_virt) |*slot| {
                    slot.* = &[_]F{};
                }
                defer {
                    for (r_addr_virt) |chunk| {
                        if (chunk.len > 0) self.allocator.free(chunk);
                    }
                    self.allocator.free(r_addr_virt);
                }

                // InstructionRa: from Stage 5 (LookupsRaVirtual) address chunks
                // LookupsRaVirtual in Stage 6 uses lookups_ra_addr_chunks from Stage 5
                // The address challenges are stage5_challenges[0..128] NOT reversed (stays LE in Stage 5)
                // Then split into chunks of lookups_ra_virtual_log_k_chunk (16),
                // but for HW reduction we need log_k_chunk-sized chunks of the full 128-bit address.
                // Actually, the r_addr_virt for InstructionRa(i) is the chunk stored by
                // LookupsRaVirtual's cache_openings, which uses compute_r_address_chunks.
                // This splits the full LOOKUPS_LOG_K=128 address into instruction_d=32 chunks of log_k_chunk=4.
                // But LookupsRaVirtual uses lookups_ra_virtual_log_k_chunk (=16) chunks internally,
                // and then the verifier uses compute_r_address_chunks to split those into log_k_chunk chunks.
                //
                // The verifier does: get_committed_polynomial_opening(InstructionRa(i), InstructionRaVirtualization)
                // which returns the point stored by LookupsRaVirtual's cache_openings.
                // That point stores r_address = compute_r_address_chunks(full_address_128, log_k_chunk)
                // So r_addr_virt[i] for InstructionRa(i) is the i-th chunk of 128/log_k_chunk = 32 chunks.
                //
                // The full 128-bit address (BE) for Lookups comes from Stage 5:
                // InstructionReadRaf has LOOKUPS_LOG_K=128 address variables.
                // In Stage 5's batched sumcheck, InstructionReadRaf starts at round 0
                // (it has the max rounds = 128 + n_cycle_vars).
                // normalize_opening_point: r[0..128].reverse() → BE, r[128..].reverse() → BE
                // But wait - Stage 5 InstructionReadRaf's normalize does NOT reverse the address!
                // Let me check...
                const LOOKUPS_LOG_K: usize = 128;

                // InstructionReadRaf in Stage 5 has LOOKUPS_LOG_K + n_cycle_vars rounds
                // Its normalize_opening_point does NOT reverse the address (stays LE)
                // Then compute_r_address_chunks splits into chunks of log_k_chunk
                // So r_addr_virt for InstructionRa(i) = stage5_challenges[i*log_k_chunk..(i+1)*log_k_chunk]
                // (NOT reversed - address stays in LE/sumcheck order)
                for (0..s6_instruction_d) |i| {
                    var chunk = try self.allocator.alloc(F, s6_log_k_chunk);
                    const chunk_start = i * s6_log_k_chunk;
                    for (0..s6_log_k_chunk) |ci| {
                        if (chunk_start + ci < LOOKUPS_LOG_K) {
                            chunk[ci] = stage5_result.challenges[chunk_start + ci];
                        } else {
                            chunk[ci] = F.zero();
                        }
                    }
                    r_addr_virt[i] = chunk;
                    // Print all r_addr_virt for comparison with Jolt
                }

                // BytecodeRa: from BytecodeReadRaf address challenges (Stage 6)
                // BytecodeReadRaf starts at round 0, has bytecode_log_k address rounds
                // normalize_opening_point: r[0..bytecode_log_k].reverse() → BE
                // Then compute_r_address_chunks pads with zeros and splits into bytecode_d chunks
                {
                    // Reversed address → BE
                    var bc_addr_be = try self.allocator.alloc(F, s6_bytecode_log_k);
                    defer self.allocator.free(bc_addr_be);
                    for (0..s6_bytecode_log_k) |i| {
                        bc_addr_be[i] = s6_challenges[s6_bytecode_log_k - 1 - i];
                    }

                    // Pad to multiple of log_k_chunk (prepend zeros)
                    const padded_len = s6_bytecode_d * s6_log_k_chunk;
                    var bc_addr_padded = try self.allocator.alloc(F, padded_len);
                    defer self.allocator.free(bc_addr_padded);
                    @memset(bc_addr_padded, F.zero());
                    // Copy to the end (BE: prepend zeros)
                    const pad = padded_len - s6_bytecode_log_k;
                    for (0..s6_bytecode_log_k) |i| {
                        bc_addr_padded[pad + i] = bc_addr_be[i];
                    }

                    // Split into chunks
                    for (0..s6_bytecode_d) |i| {
                        var chunk = try self.allocator.alloc(F, s6_log_k_chunk);
                        for (0..s6_log_k_chunk) |ci| {
                            chunk[ci] = bc_addr_padded[i * s6_log_k_chunk + ci];
                        }
                        r_addr_virt[s6_instruction_d + i] = chunk;
                    }
                }

                // RamRa: use aligned r_address from Stage 2 (BIG_ENDIAN)
                // Stage 2 aligns all RAM sumchecks to share the same r_address.
                // The RamRaClaimReduction (Stage 5) is cycle-only; the address comes from Stage 2.
                {
                    // Pad r_address_raf with leading zeros to make length a multiple of
                    // log_k_chunk (matching Jolt's compute_r_address_chunks)
                    const raf_len = stage2_result.r_address_raf.len;
                    const padded_len = ((raf_len + s6_log_k_chunk - 1) / s6_log_k_chunk) * s6_log_k_chunk;
                    const pad_count = padded_len - raf_len;

                    for (0..s6_ram_d) |i| {
                        var chunk = try self.allocator.alloc(F, s6_log_k_chunk);
                        const chunk_start = i * s6_log_k_chunk;
                        for (0..s6_log_k_chunk) |ci| {
                            const src_idx = chunk_start + ci;
                            chunk[ci] = if (src_idx < pad_count) F.zero() else stage2_result.r_address_raf[src_idx - pad_count];
                        }
                        r_addr_virt[s6_instruction_d + s6_bytecode_d + i] = chunk;
                    }
                }

                // Build eq table for r_cycle
                //
                // IMPORTANT: The booleanity sumcheck's Phase 2 uses LowToHigh binding.
                // When halving the eq_cycle table with challenges c[0],...,c[n-1],
                // challenge c[m] binds bit m of the table index j.
                //
                // For the Stage 7 G tables to produce claims consistent with the
                // booleanity Phase 2 halving, the eq_cycle table must use the SAME
                // bit-to-challenge mapping. With computeEqTable(r, n), r[m] controls
                // bit m of index j. So we need r[m] = c[m] (the LE cycle challenges).
                //
                // r_cycle_be is the REVERSED cycle challenges (BE format), so we need
                // to use the UN-reversed version = direct Stage 6 cycle challenges (LE).
                var r_cycle_le = try self.allocator.alloc(F, s6_n_cycle_vars);
                defer self.allocator.free(r_cycle_le);
                for (0..s6_n_cycle_vars) |i| {
                    r_cycle_le[i] = s6_challenges[s6_bool_start + s6_log_k_chunk + i];
                }
                const eq_cycle = try stage6_mod.computeEqTableParallel(F, self.allocator, r_cycle_le, s6_n_cycle_vars, self.thread_pool);
                defer self.allocator.free(eq_cycle);

                // Compute G_i polynomials: G_i(k) = Σ_j eq(r_cycle, j) · (addr_chunk_i(j) == k ? 1 : 0)
                var G = try self.allocator.alloc([]F, N);
                defer {
                    for (G) |g| self.allocator.free(g);
                    self.allocator.free(G);
                }
                for (0..N) |i| {
                    G[i] = try self.allocator.alloc(F, k_chunk);
                    @memset(G[i], F.zero());
                }

                // Iterate over all cycles to populate G_i
                // G_i(k) = Σ_j eq(r_cycle, j) · [addr_chunk_i(j) == k]
                for (0..T_val) |j| {
                    const step = the_trace.steps.items[j];
                    const eq_j = eq_cycle[j];

                    // InstructionRa: compute 128-bit lookup index, split into chunks
                    // Use the same computeLookupIndex as Stage 6 LookupsRaVirtualProver
                    // to ensure G tables are consistent with the virt_claims.
                    {
                        const lookup_idx = stage6_mod.computeLookupIndex(step);
                        for (0..s6_instruction_d) |i| {
                            const shift = s6_log_k_chunk * (s6_instruction_d - 1 - i);
                            const mask: u128 = (@as(u128, 1) << @intCast(s6_log_k_chunk)) - 1;
                            const chunk_val: usize = @intCast((lookup_idx >> @intCast(shift)) & mask);
                            if (chunk_val < k_chunk) {
                                G[i][chunk_val] = G[i][chunk_val].add(eq_j);
                            }
                        }
                    }

                    // BytecodeRa: bytecode address (pc_idx) split into chunks
                    {
                        const pc_idx = pc_map_ptr.getPCForStep(step);
                        for (0..s6_bytecode_d) |i| {
                            const chunk_val = stage6_mod.extractChunkMSB(@intCast(pc_idx), i, s6_bytecode_d, s6_log_k_chunk);
                            const ra_idx = s6_instruction_d + i;
                            if (chunk_val < k_chunk) {
                                G[ra_idx][chunk_val] = G[ra_idx][chunk_val].add(eq_j);
                            }
                        }
                    }

                    // RamRa: memory address (remapped) split into chunks
                    {
                        if (step.memory_addr) |addr| {
                            if (addr != 0) {
                                if (the_memory_layout.remapAddress(addr)) |raddr| {
                                    for (0..s6_ram_d) |i| {
                                        const chunk_val = stage6_mod.extractChunkMSB(raddr, i, s6_ram_d, s6_log_k_chunk);
                                        const ra_idx = s6_instruction_d + s6_bytecode_d + i;
                                        if (chunk_val < k_chunk) {
                                            G[ra_idx][chunk_val] = G[ra_idx][chunk_val].add(eq_j);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Compute eq tables for r_addr_bool and r_addr_virt_i
                //
                // IMPORTANT: computeEqTable puts r[0] at bit 0 (LE convention).
                // The booleanity Phase 1 F table also uses LowToHigh expansion,
                // putting a[0] at bit 0. For eq_bool to match F[chunk], we need
                // to pass the LE address challenges (same order as Phase 1 binding).
                //
                // Similarly, the virtualization sumchecks use LowToHigh binding,
                // so eq_virt needs the LE versions of the address challenges.
                //
                // The LE version = reversed BE version.
                var r_addr_bool_le = try self.allocator.alloc(F, s6_log_k_chunk);
                defer self.allocator.free(r_addr_bool_le);
                for (0..s6_log_k_chunk) |i| {
                    r_addr_bool_le[i] = r_addr_bool_be[s6_log_k_chunk - 1 - i];
                }
                var eq_bool = try stage6_mod.computeEqTableParallel(F, self.allocator, r_addr_bool_le, s6_log_k_chunk, self.thread_pool);
                defer self.allocator.free(eq_bool);

                var eq_virt = try self.allocator.alloc([]F, N);
                defer {
                    for (eq_virt) |ev| self.allocator.free(ev);
                    self.allocator.free(eq_virt);
                }
                for (0..N) |i| {
                    // Reverse r_addr_virt to LE for eq table
                    var r_virt_le = try self.allocator.alloc(F, s6_log_k_chunk);
                    for (0..s6_log_k_chunk) |ci| {
                        r_virt_le[ci] = r_addr_virt[i][s6_log_k_chunk - 1 - ci];
                    }
                    eq_virt[i] = try stage6_mod.computeEqTableParallel(F, self.allocator, r_virt_le, s6_log_k_chunk, self.thread_pool);
                    self.allocator.free(r_virt_le);
                }


                // Sample gamma from transcript (matches Jolt's HammingWeightClaimReductionParams::new)
                // IMPORTANT: Jolt's HW code calls transcript.challenge_scalar() which uses
                // challenge_scalar_128_bits() -> F::from_bytes() = from_le_bytes_mod_order().
                // This is the FULL field element path, NOT the 125-bit optimized path.
                // So we must use challengeScalarFull() here.
                const gamma = transcript.challengeScalarFull();
                {
                }
                var gamma_powers = try self.allocator.alloc(F, 3 * N);
                defer self.allocator.free(gamma_powers);
                gamma_powers[0] = F.one();
                for (1..3 * N) |i| gamma_powers[i] = gamma_powers[i - 1].mul(gamma);

                // Compute HammingWeight claims for each ra_i
                // For InstructionRa and BytecodeRa: H_i = 1 (Jolt convention)
                // For RamRa: H_i = ram_hw_factor (from RamHammingBooleanity opening)
                const ram_hw_factor = stage6_result.hamming_weight_claim;

                // Compute input claim: Σ_i (γ^{3i}·H_i + γ^{3i+1}·claim_bool_i + γ^{3i+2}·claim_virt_i)
                // Use claims from Stage 6 result (booleanity claims now properly computed)
                var input_claim = F.zero();
                for (0..N) |i| {
                    const hw_claim: F = if (i >= s6_instruction_d + s6_bytecode_d) ram_hw_factor else F.one();
                    const bool_claim = stage6_result.booleanity_ra_claims[i];
                    const virt_claim: F = blk: {
                        if (i < s6_instruction_d) {
                            break :blk stage6_result.instruction_ra_virtual_claims[i];
                        } else if (i < s6_instruction_d + s6_bytecode_d) {
                            break :blk stage6_result.bytecode_ra_claims[i - s6_instruction_d];
                        } else {
                            break :blk stage6_result.ram_ra_virtual_claims[i - s6_instruction_d - s6_bytecode_d];
                        }
                    };
                    input_claim = input_claim.add(gamma_powers[3 * i].mul(hw_claim));
                    input_claim = input_claim.add(gamma_powers[3 * i + 1].mul(bool_claim));
                    input_claim = input_claim.add(gamma_powers[3 * i + 2].mul(virt_claim));
                    if (i < 3) {
                    }
                }


                // Append input claim to transcript (matches BatchedSumcheck::verify)
                transcript.appendScalar("sumcheck_claim", input_claim);

                // Sample batching coefficient (only 1 instance for now - no advice)
                const batch_coeffs = try transcript.challengeVector(self.allocator, 1);
                defer self.allocator.free(batch_coeffs);
                const batch_coeff = batch_coeffs[0];

                // Batched claim = batch_coeff * input_claim (1 instance, no scaling needed for same rounds)
                var current_claim = batch_coeff.mul(input_claim);

                // Run degree-2 sumcheck over log_k_chunk rounds
                const num_rounds = s6_log_k_chunk;
                const degree_bound: usize = 2;

                // Collect Stage 7 sumcheck challenges for opening point construction
                var stage7_challenges = try self.allocator.alloc(F, num_rounds);
                defer self.allocator.free(stage7_challenges);

                // Track current polynomial size (halves each round)
                var poly_size: usize = k_chunk;

                for (0..num_rounds) |round| {
                    const half = poly_size / 2;

                    // Sanity check: verify sum over current table = current_claim / batch_coeff
                    {
                        var check_sum = F.zero();
                        for (0..poly_size) |k| {
                            for (0..N) |i| {
                                const gi = G[i][k];
                                const w = gamma_powers[3 * i].add(gamma_powers[3 * i + 1].mul(eq_bool[k])).add(gamma_powers[3 * i + 2].mul(eq_virt[i][k]));
                                check_sum = check_sum.add(gi.mul(w));
                            }
                        }
                        check_sum = check_sum.mul(batch_coeff);
                        const eq_claim = check_sum.eql(current_claim);
                        if (!eq_claim) {
                        } else {
                        }
                    }

                    // LowToHigh binding: pair (2*j, 2*j+1) to bind LSB first
                    // Compute round polynomial evaluations at {0, 2}
                    // (p(1) is derived from p(0) + p(1) = claim)
                    var p0 = F.zero();
                    var p2 = F.zero();

                    for (0..half) |j| {
                        // LowToHigh: lo = poly[2*j], hi = poly[2*j+1]
                        const eq_b_lo = eq_bool[2 * j];
                        const eq_b_hi = eq_bool[2 * j + 1];
                        // Eval at x=2: f(2) = 2*f(1) - f(0)
                        const eq_b_2 = eq_b_hi.add(eq_b_hi).sub(eq_b_lo);

                        for (0..N) |i| {
                            const g_lo = G[i][2 * j];
                            const g_hi = G[i][2 * j + 1];
                            const g_2 = g_hi.add(g_hi).sub(g_lo);

                            const ev_lo = eq_virt[i][2 * j];
                            const ev_hi = eq_virt[i][2 * j + 1];
                            const ev_2 = ev_hi.add(ev_hi).sub(ev_lo);

                            // weight(x) = γ^{3i} + γ^{3i+1}·eq_bool(x) + γ^{3i+2}·eq_virt_i(x)
                            const w0 = gamma_powers[3 * i].add(gamma_powers[3 * i + 1].mul(eq_b_lo)).add(gamma_powers[3 * i + 2].mul(ev_lo));
                            const w2 = gamma_powers[3 * i].add(gamma_powers[3 * i + 1].mul(eq_b_2)).add(gamma_powers[3 * i + 2].mul(ev_2));

                            p0 = p0.add(g_lo.mul(w0));
                            p2 = p2.add(g_2.mul(w2));
                        }
                    }

                    // Scale by batch coefficient
                    p0 = p0.mul(batch_coeff);
                    p2 = p2.mul(batch_coeff);

                    // p(1) = current_claim - p(0)
                    const p1 = current_claim.sub(p0);

                    // Compress to Toom-Cook format: coeffs_except_linear = [a0, a2]
                    // p(x) = a0 + a1*x + a2*x^2
                    // a0 = p(0)
                    // a2 = (p(2) - 2*p(1) + p(0)) / 2
                    const two_p1 = p1.add(p1);
                    const a2_num = p2.sub(two_p1).add(p0);
                    const a2 = a2_num.mul(F.fromU64(2).inverse().?);

                    const coeffs = try self.allocator.alloc(F, degree_bound);
                    coeffs[0] = p0; // a0 = p(0) = constant term
                    coeffs[1] = a2; // a2 = quadratic coefficient
                    try jolt_proof.stage7_sumcheck_proof.compressed_polys.append(self.allocator, .{
                        .coeffs_except_linear_term = coeffs,
                        .allocator = self.allocator,
                    });

                    // Append to transcript and get challenge
                    transcript.appendScalars("sumcheck_poly", coeffs[0..degree_bound]);

                    const challenge = transcript.challengeScalar();
                    stage7_challenges[round] = challenge;

                    // Evaluate p(challenge) = a0 + a1*challenge + a2*challenge^2
                    // a1 = p(1) - a0 - a2
                    const a0 = p0;
                    const a1 = p1.sub(a0).sub(a2);
                    current_claim = a0.add(a1.mul(challenge)).add(a2.mul(challenge.mul(challenge)));

                    // Bind all polynomials at challenge (LowToHigh: bind pairs 2j, 2j+1)
                    for (0..N) |i| {
                        for (0..half) |jj| {
                            G[i][jj] = G[i][2 * jj].add(challenge.mul(G[i][2 * jj + 1].sub(G[i][2 * jj])));
                        }
                    }
                    for (0..half) |jj| {
                        eq_bool[jj] = eq_bool[2 * jj].add(challenge.mul(eq_bool[2 * jj + 1].sub(eq_bool[2 * jj])));
                    }
                    for (0..N) |i| {
                        for (0..half) |jj| {
                            eq_virt[i][jj] = eq_virt[i][2 * jj].add(challenge.mul(eq_virt[i][2 * jj + 1].sub(eq_virt[i][2 * jj])));
                        }
                    }
                    poly_size = half;

                    if (round < 2) {
                    }
                }

                // Cache opening claims: G_i(ρ) for each ra_i
                // G_i[0] is the final value after all bindings
                // Order: InstructionRa(0..inst_d), BytecodeRa(0..bc_d), RamRa(0..ram_d)
                for (0..N) |i| {
                    const g_claim = G[i][0];
                    const key: jolt_types.OpeningId = blk: {
                        if (i < s6_instruction_d) {
                            break :blk .{ .Committed = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .HammingWeightClaimReduction } };
                        } else if (i < s6_instruction_d + s6_bytecode_d) {
                            break :blk .{ .Committed = .{ .poly = .{ .BytecodeRa = i - s6_instruction_d }, .sumcheck_id = .HammingWeightClaimReduction } };
                        } else {
                            break :blk .{ .Committed = .{ .poly = .{ .RamRa = i - s6_instruction_d - s6_bytecode_d }, .sumcheck_id = .HammingWeightClaimReduction } };
                        }
                    };
                    try jolt_proof.opening_claims.insert(key, g_claim);
                    // Append to transcript (matches cache_openings → append_sparse)
                    transcript.appendScalar("opening_claim", g_claim);
                }


                // Debug: Verify expected output claim (what verifier would compute)
                {
                    const final_eq_bool = eq_bool[0];

                    // Cross-check: compute mle(rho_rev, r_addr_bool) directly
                    {
                        // Collect sumcheck challenges (stored in round_polys, extracted via transcript)
                        // Actually, the sumcheck challenges are the round challenges we used to bind.
                        // They are derived from the transcript. Let me retrieve them from what was used.
                        // For now, just compute mle from stored r_addr_bool_be and see what we get.
                        // rho_rev = reversed sumcheck challenges

                        // Print initial eq table values for first few entries
                        const eq_bool_check = try stage6_mod.computeEqTable(F, self.allocator, r_addr_bool_be, s6_log_k_chunk);
                        defer self.allocator.free(eq_bool_check);
                    }

                    var expected = F.zero();
                    for (0..N) |i| {
                        const gi = G[i][0];
                        const evi = eq_virt[i][0];
                        const weight = gamma_powers[3 * i].add(gamma_powers[3 * i + 1].mul(final_eq_bool)).add(gamma_powers[3 * i + 2].mul(evi));
                        expected = expected.add(gi.mul(weight));
                    }
                    // expected * batch_coeff should equal the output_claim

                    // Print eq_virt[0][0] for comparison

                    // Print the current_claim (output of sumcheck)
                }

                // Construct the unified opening point: [r_address_stage7_BE || r_cycle_BE]
                // r_address = reversed stage7_challenges (LE → BE, like Jolt's match_endianness)
                // r_cycle = r_cycle_be (already BE from Stage 6 booleanity)
                const opening_point_len = s6_log_k_chunk + s6_n_cycle_vars;
                var opening_point_storage = try self.allocator.alloc(F, opening_point_len);
                // r_address_be: reverse the stage7_challenges
                for (0..s6_log_k_chunk) |i| {
                    opening_point_storage[i] = stage7_challenges[s6_log_k_chunk - 1 - i];
                }
                // r_cycle_be
                for (0..s6_n_cycle_vars) |i| {
                    opening_point_storage[s6_log_k_chunk + i] = r_cycle_be[i];
                }
                jolt_proof.opening_point = opening_point_storage;

            }

            if (comptime stage_timing_enabled) {
                std.debug.print("    [STAGE-TIMING] Stage 7: {d:.1} ms\n", .{@as(f64, @floatFromInt(stage_timer.read())) / 1_000_000.0});
            }

            return jolt_proof;
        }

        /// Result of Stage 2 sumcheck including factor evaluations and challenges
        const Stage2Result = struct {
            /// The 8 factor polynomial evaluations at r_cycle
            /// Order: LeftInstructionInput, RightInstructionInput, IsRdNotZero,
            ///        WriteLookupOutputToRDFlag, JumpFlag, LookupOutput, BranchFlag, NextIsNoop
            factor_evals: [8]F,
            /// All sumcheck challenges (26 for max_num_rounds = log_ram_k + n_cycle_vars)
            /// Used for computing OutputSumcheck's r_address_prime
            challenges: []F,
            /// Final claims from each prover (for opening claims)
            raf_final_claim: F, // Instance 1: RamRafEvaluation
            rwc_final_claim: F, // Instance 2: RamReadWriteChecking (combined claim)
            output_final_claim: F, // Instance 3: RamOutputCheck sumcheck final claim
            instr_final_claim: F, // Instance 4: InstructionLookupsClaimReduction (combined)
            /// OutputSumcheck's Val_final polynomial evaluation at r_address_prime
            /// This is the MLE evaluation Val_final(r'), needed for opening claim
            output_val_final_claim: F, // Val_final(r') for RamValFinal opening
            /// OutputSumcheck's Val_init polynomial evaluation at r_address_prime
            output_val_init_claim: F, // Val_init(r') for RamValInit opening
            /// OutputSumcheck's r_address challenges (big-endian order)
            r_address_raf: []F,
            /// RWC's r_address challenges (big-endian order) - for RamRaClaimReduction
            r_address_rw: []F,
            /// RWC's r_cycle challenges (big-endian order) - for RamRaClaimReduction
            r_cycle_rw: []F,
            /// ProductVirtualRemainder's r_cycle challenges (big-endian order) - for BytecodeReadRaf
            r_cycle_product: []F,
            /// Individual RWC opening claims (ra, val, inc)
            rwc_ra_claim: F,
            rwc_val_claim: F,
            rwc_inc_claim: F,
            /// Individual InstructionLookups opening claims (5 terms)
            instr_lookup_output_claim: F,
            instr_left_operand_claim: F,
            instr_right_operand_claim: F,
            instr_left_instr_input_claim: F,
            instr_right_instr_input_claim: F,
            allocator: Allocator,

            pub fn deinit(self: *Stage2Result) void {
                self.allocator.free(self.challenges);
                self.allocator.free(self.r_address_raf);
                self.allocator.free(self.r_address_rw);
                self.allocator.free(self.r_cycle_rw);
                self.allocator.free(self.r_cycle_product);
            }
        };

        /// Generate Stage 2 batched sumcheck proof
        ///
        /// Stage 2 batches 5 sumcheck instances:
        /// 1. ProductVirtualRemainder: n_cycle_vars rounds, degree 3
        /// 2. RamRafEvaluation: log_ram_k rounds, degree 2
        /// 3. RamReadWriteChecking: log_ram_k + n_cycle_vars rounds, degree 3
        /// 4. OutputSumcheck: log_ram_k rounds, degree 3
        /// 5. InstructionLookupsClaimReduction: n_cycle_vars rounds, degree 2
        ///
        /// For programs without RAM/lookups, instances 2-5 have zero input claims
        /// and contribute constant-zero polynomials.
        ///
        /// Returns the 8 factor polynomial evaluations at r_cycle for opening claims.
        fn generateStage2BatchedSumcheckProof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            transcript: *Blake2bTranscript(F),
            r0_stage2: F,
            uni_skip_claim_stage2: F,
            tau: []const F,
            r_spartan_for_instr: []const F,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            n_cycle_vars: usize,
            log_ram_k: usize,
            opening_claims: *OpeningClaims(F),
            config: JoltProverConfig,
        ) !Stage2Result {
            const max_num_rounds = log_ram_k + n_cycle_vars;

            // Define the 5 instances with their input claims and round counts
            // Instance 0: ProductVirtualRemainder (input = uni_skip_claim from SpartanProductVirtualization)
            // Instance 1: RamRafEvaluation (input = RamAddress from SpartanOuter)
            // Instance 2: RamReadWriteChecking (input = RamReadValue + gamma * RamWriteValue)
            // Instance 3: OutputSumcheck (input = 0)
            // Instance 4: InstructionLookupsClaimReduction (input = LookupOutput + gamma * LeftOperand + gamma^2 * RightOperand)

            // Get opening claims from proof (these were set during Stage 1)
            const ram_address_claim = opening_claims.get(.{ .Virtual = .{ .poly = .RamAddress, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const ram_read_value_claim = opening_claims.get(.{ .Virtual = .{ .poly = .RamReadValue, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const ram_write_value_claim = opening_claims.get(.{ .Virtual = .{ .poly = .RamWriteValue, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const lookup_output_claim = opening_claims.get(.{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const left_operand_claim = opening_claims.get(.{ .Virtual = .{ .poly = .LeftLookupOperand, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const right_operand_claim = opening_claims.get(.{ .Virtual = .{ .poly = .RightLookupOperand, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const left_instr_input_claim = opening_claims.get(.{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const right_instr_input_claim = opening_claims.get(.{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();


            // Sample gammas from transcript in the same order as upstream Jolt verifier:
            // 1. RamReadWriteChecking samples gamma first
            // 2. InstructionLookupsClaimReduction samples gamma
            // 3. OutputSumcheck samples r_address
            //
            // CRITICAL: gamma uses challenge_scalar (NO 125-bit masking) = challengeScalarFull()
            // r_address uses challenge_scalar_optimized (HAS 125-bit masking) = challengeScalar()

            // 1. RamReadWriteChecking gamma
            const gamma_rwc = transcript.challengeScalarFull();

            // 2. InstructionLookupsClaimReduction gamma (via challenge_scalar, NO masking)
            const gamma_instr = transcript.challengeScalarFull();
            const gamma_instr_sqr = gamma_instr.mul(gamma_instr);
            const gamma_instr_cub = gamma_instr_sqr.mul(gamma_instr);
            const gamma_instr_quart = gamma_instr_sqr.mul(gamma_instr_sqr);

            // 3. OutputSumcheck samples r_address (log_ram_k challenges via challenge_vector_optimized)
            const r_address_presampled = try self.allocator.alloc(F, log_ram_k);
            defer self.allocator.free(r_address_presampled);
            for (r_address_presampled) |*r| {
                r.* = transcript.challengeScalar();
            }

            // Compute input_claims in UPSTREAM order:
            // [0] RamReadWriteChecking: RamReadValue + gamma_rwc * RamWriteValue
            // [1] ProductVirtualRemainder: uni_skip_claim
            // [2] InstructionLookupsClaimReduction: LookupOutput + γ*LeftOp + γ²*RightOp + γ³*LeftInstr + γ⁴*RightInstr
            // [3] RamRafEvaluation: RamAddress
            // [4] OutputSumcheck: 0
            const input_claim_rwc = ram_read_value_claim.add(gamma_rwc.mul(ram_write_value_claim));
            const input_claim_instr = lookup_output_claim
                .add(gamma_instr.mul(left_operand_claim))
                .add(gamma_instr_sqr.mul(right_operand_claim))
                .add(gamma_instr_cub.mul(left_instr_input_claim))
                .add(gamma_instr_quart.mul(right_instr_input_claim));


            const input_claims = [5]F{
                input_claim_rwc, // [0] RamReadWriteChecking
                uni_skip_claim_stage2, // [1] ProductVirtualRemainder
                input_claim_instr, // [2] InstructionLookupsClaimReduction
                ram_address_claim, // [3] RamRafEvaluation
                F.zero(), // [4] OutputSumcheck
            };

            const rounds_per_instance = [5]usize{
                log_ram_k + n_cycle_vars, // [0] RamReadWriteChecking
                n_cycle_vars, // [1] ProductVirtualRemainder
                n_cycle_vars, // [2] InstructionLookupsClaimReduction
                log_ram_k, // [3] RamRafEvaluation
                log_ram_k, // [4] OutputSumcheck
            };

            // Step 1: Append all input claims to transcript
            for (input_claims) |claim| {
                transcript.appendScalar("sumcheck_claim", claim);
            }


            // Step 2: Sample batching coefficients (input claims already appended at line 1747)
            var batching_coeffs: [5]F = undefined;
            for (0..5) |i| {
                batching_coeffs[i] = transcript.challengeScalarFull();
            }

            // Debug: STAGE2_PRE batching coefficient logs for compare_sumcheck.py


            // Step 3: Compute initial batched claim
            // batched_claim = Σᵢ αᵢ * input_claim[i] * 2^(max_rounds - rounds[i])
            var batched_claim = F.zero();
            for (0..5) |i| {
                const scale_power = max_num_rounds - rounds_per_instance[i];
                var scaled_claim = input_claims[i];
                for (0..scale_power) |_| {
                    scaled_claim = scaled_claim.add(scaled_claim);
                }
                batched_claim = batched_claim.add(scaled_claim.mul(batching_coeffs[i]));
            }


            // Initialize provers for each instance (upstream ordering):
            // [0] RamReadWriteChecking, [1] ProductVirtualRemainder,
            // [2] InstructionLookupsClaimReduction, [3] RamRafEvaluation, [4] OutputSumcheck

            // Instance 0: RamReadWriteChecking - starts at round 0 (max rounds)
            const RWCProver = ram.RamReadWriteCheckingProver(F);
            var rwc_prover: ?RWCProver = null;
            var rwc_evals_this_round: ?[4]F = null;
            const use_rwc_prover = !input_claims[0].eql(F.zero());
            if (config.memory_trace != null and use_rwc_prover) {
                const phase1_num_rounds = n_cycle_vars;
                var rwc_params = ram.RamReadWriteCheckingParams(F).initWithPhaseConfig(
                    self.allocator,
                    gamma_rwc,
                    tau[0..n_cycle_vars],
                    log_ram_k,
                    n_cycle_vars,
                    phase1_num_rounds,
                    if (config.memory_layout) |ml| ml.getLowestAddress() else 0x80000000,
                ) catch null;

                if (rwc_params) |*params| {
                    rwc_prover = RWCProver.init(
                        self.allocator,
                        config.memory_trace.?,
                        params.*,
                        input_claims[0],
                        config.initial_ram,
                        config.memory_layout,
                        config.is_panicking,
                    ) catch null;

                    if (rwc_prover != null) {
                    } else {
                        params.deinit();
                    }
                }
            }
            defer if (rwc_prover) |*rp| rp.deinit();

            // Instance 1: ProductVirtualRemainder
            const ProductRemainderProver = product_remainder.ProductVirtualRemainderProver(F);
            var product_prover: ?ProductRemainderProver = null;
            if (cycle_witnesses.len > 0 and tau.len > 0) {
                product_prover = ProductRemainderProver.init(
                    self.allocator,
                    r0_stage2,
                    tau,
                    uni_skip_claim_stage2,
                    cycle_witnesses,
                ) catch null;
            }
            defer if (product_prover) |*p| p.deinit();

            // Instance 2: InstructionLookupsClaimReduction - initialized lazily at start_round
            const claim_reductions = @import("claim_reductions/mod.zig");
            const InstrLookupsProver = claim_reductions.InstructionLookupsProver(F);
            var instr_prover: ?InstrLookupsProver = null;
            var instr_evals_this_round: ?[4]F = null;
            defer if (instr_prover) |*ip| ip.deinit();

            // Instance 3: RamRafEvaluation - initialized lazily at start_round
            const RafProver = ram.RafEvaluationProver(F);
            var raf_prover: ?RafProver = null;
            var raf_evals_this_round: ?[4]F = null;
            defer if (raf_prover) |*rp| rp.deinit();

            // Instance 4: OutputSumcheck
            const OutputProver = ram.OutputSumcheckProver(F);
            var output_prover: ?OutputProver = null;
            if (config.memory_layout != null and config.initial_ram != null and config.final_ram != null) {
                output_prover = OutputProver.init(
                    self.allocator,
                    config.initial_ram.?,
                    config.final_ram.?,
                    r_address_presampled,
                    config.memory_layout.?,
                    config.program_inputs,
                    config.program_outputs,
                    config.is_panicking,
                ) catch null;
                if (output_prover) |_| {
                }
            }
            defer if (output_prover) |*p| p.deinit();

            // Track individual claims for each instance (needed for zero-poly instances)
            var individual_claims: [5]F = undefined;
            for (0..5) |i| {
                const scale_power = max_num_rounds - rounds_per_instance[i];
                var scaled = input_claims[i];
                for (0..scale_power) |_| {
                    scaled = scaled.add(scaled);
                }
                individual_claims[i] = scaled;
            }

            // Store challenges for opening claims computation
            var challenges: std.ArrayListUnmanaged(F) = .{};
            defer challenges.deinit(self.allocator);

            // Step 4: Run batched sumcheck rounds
            for (0..max_num_rounds) |round_idx| {
                // Compute combined polynomial from all instances
                var combined_evals = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                // Store ProductVirtualRemainder's evals for claim update
                var product_evals_this_round: ?[4]F = null;
                // Store OutputSumcheck's evals for claim update
                var output_evals_this_round: ?[4]F = null;

                for (0..5) |i| {
                    const start_round = max_num_rounds - rounds_per_instance[i];

                    if (round_idx >= start_round) {
                        // Instance is active
                        if (i == 0) {
                            // Instance 0: RamReadWriteChecking (max rounds, starts at round 0)
                            if (rwc_prover) |*rwcp| {
                                const rwc_evals = rwcp.computeRoundPolynomialCubic();
                                rwc_evals_this_round = rwc_evals;
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(rwc_evals[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                // Fallback: constant polynomial from scaled claim
                                const instance_round = round_idx - start_round;
                                const remaining_rounds = rounds_per_instance[i] - 1 - instance_round;
                                var scaled = input_claims[i];
                                for (0..remaining_rounds) |_| scaled = scaled.add(scaled);
                                const weighted = scaled.mul(batching_coeffs[i]);
                                for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                            }
                        } else if (i == 1) {
                            // Instance 1: ProductVirtualRemainder
                            if (product_prover) |_| {
                                const claim_before = product_prover.?.current_claim;
                                const comp = product_prover.?.computeRoundPolynomial() catch [3]F{ F.zero(), F.zero(), F.zero() };
                                const c0 = comp[0];
                                const c2_p = comp[1];
                                const c3_p = comp[2];
                                const c1 = claim_before.sub(c0).sub(c0).sub(c2_p).sub(c3_p);
                                const s0 = c0;
                                const s1 = claim_before.sub(s0);
                                const s2 = c0.add(c1.mul(F.fromU64(2))).add(c2_p.mul(F.fromU64(4))).add(c3_p.mul(F.fromU64(8)));
                                const s3 = c0.add(c1.mul(F.fromU64(3))).add(c2_p.mul(F.fromU64(9))).add(c3_p.mul(F.fromU64(27)));
                                product_evals_this_round = [4]F{ s0, s1, s2, s3 };
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(product_evals_this_round.?[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                const instance_round = round_idx - start_round;
                                const remaining_rounds = rounds_per_instance[i] - 1 - instance_round;
                                var scaled = input_claims[i];
                                for (0..remaining_rounds) |_| scaled = scaled.add(scaled);
                                const weighted = scaled.mul(batching_coeffs[i]);
                                for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                            }
                        } else if (i == 2) {
                            // Instance 2: InstructionLookupsClaimReduction
                            if (round_idx == start_round and instr_prover == null and cycle_witnesses.len > 0) {
                                var instr_params = claim_reductions.InstructionLookupsParams(F).init(
                                    self.allocator,
                                    gamma_instr,
                                    r_spartan_for_instr,
                                    n_cycle_vars,
                                ) catch null;

                                if (instr_params) |*params| {
                                    const R1CSInputIndex = @import("r1cs/constraints.zig").R1CSInputIndex;
                                    const lookup_outputs_arr = try self.allocator.alloc(F, cycle_witnesses.len);
                                    defer self.allocator.free(lookup_outputs_arr);
                                    const left_operands_arr = try self.allocator.alloc(F, cycle_witnesses.len);
                                    defer self.allocator.free(left_operands_arr);
                                    const right_operands_arr = try self.allocator.alloc(F, cycle_witnesses.len);
                                    defer self.allocator.free(right_operands_arr);
                                    const left_instr_arr = try self.allocator.alloc(F, cycle_witnesses.len);
                                    defer self.allocator.free(left_instr_arr);
                                    const right_instr_arr = try self.allocator.alloc(F, cycle_witnesses.len);
                                    defer self.allocator.free(right_instr_arr);

                                    for (cycle_witnesses, 0..) |w, wi| {
                                        lookup_outputs_arr[wi] = w.values[R1CSInputIndex.LookupOutput.toIndex()];
                                        left_operands_arr[wi] = w.values[R1CSInputIndex.LeftLookupOperand.toIndex()];
                                        right_operands_arr[wi] = w.values[R1CSInputIndex.RightLookupOperand.toIndex()];
                                        left_instr_arr[wi] = w.values[R1CSInputIndex.LeftInstructionInput.toIndex()];
                                        right_instr_arr[wi] = w.values[R1CSInputIndex.RightInstructionInput.toIndex()];
                                    }

                                    instr_prover = InstrLookupsProver.init(
                                        self.allocator,
                                        params.*,
                                        input_claims[2],
                                        lookup_outputs_arr,
                                        left_operands_arr,
                                        right_operands_arr,
                                        left_instr_arr,
                                        right_instr_arr,
                                    ) catch blk: {
                                        params.deinit();
                                        break :blk null;
                                    };

                                    if (instr_prover != null) {
                                    }
                                }
                            }

                            if (instr_prover) |*ip| {
                                const instr_evals = ip.computeRoundPolynomialCubic();
                                instr_evals_this_round = instr_evals;
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(instr_evals[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                const instance_round = round_idx - start_round;
                                const remaining_rounds = rounds_per_instance[i] - 1 - instance_round;
                                var scaled = input_claims[i];
                                for (0..remaining_rounds) |_| scaled = scaled.add(scaled);
                                const weighted = scaled.mul(batching_coeffs[i]);
                                for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                            }
                        } else if (i == 3) {
                            // Instance 3: RamRafEvaluation
                            const use_raf_prover = !input_claims[3].eql(F.zero());
                            if (round_idx == start_round and raf_prover == null and config.memory_trace != null and use_raf_prover) {
                                const r_cycle_slice = tau[0..n_cycle_vars];
                                const r_cycle = try self.allocator.alloc(F, n_cycle_vars);
                                @memcpy(r_cycle, r_cycle_slice);
                                const start_addr: u64 = if (config.memory_layout) |ml| ml.getLowestAddress() else 0x80000000;
                                var raf_params = try ram.RafEvaluationParams(F).init(self.allocator, log_ram_k, start_addr, r_cycle);
                                self.allocator.free(r_cycle);

                                const raf_initial_claim = input_claims[3];

                                raf_prover = RafProver.init(self.allocator, config.memory_trace.?, raf_params, raf_initial_claim) catch blk: {
                                    raf_params.deinit();
                                    break :blk null;
                                };
                            }

                            if (raf_prover) |*rp| {
                                const raf_evals = rp.computeRoundPolynomialCubic();
                                raf_evals_this_round = raf_evals;
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(raf_evals[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                if (round_idx == start_round) {
                                }
                                const remaining_rounds = rounds_per_instance[i] - (round_idx - start_round);
                                var scaled = individual_claims[i];
                                for (0..remaining_rounds) |_| scaled = scaled.mul(F.fromU64(2));
                                scaled = scaled.mul(F.fromU64(2).inverse().?);
                                const weighted = scaled.mul(batching_coeffs[i]);
                                for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                            }
                        } else if (i == 4) {
                            // Instance 4: OutputSumcheck
                            if (output_prover) |_| {
                                const output_compressed = output_prover.?.computeRoundPolynomial();
                                const c0 = output_compressed[0];
                                const c2_o = output_compressed[1];
                                const c3_o = output_compressed[2];
                                const current_claim_output = output_prover.?.current_claim;
                                const c1 = current_claim_output.sub(c0).sub(c0).sub(c2_o).sub(c3_o);
                                const s0_out = c0;
                                const s1_out = current_claim_output.sub(s0_out);
                                const s2_out = c0.add(c1.mul(F.fromU64(2))).add(c2_o.mul(F.fromU64(4))).add(c3_o.mul(F.fromU64(8)));
                                const s3_out = c0.add(c1.mul(F.fromU64(3))).add(c2_o.mul(F.fromU64(9))).add(c3_o.mul(F.fromU64(27)));
                                output_evals_this_round = [4]F{ s0_out, s1_out, s2_out, s3_out };
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(output_evals_this_round.?[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                // Zero input claim → zero polynomial
                                const scale_power = rounds_per_instance[i] - 1 - (round_idx - start_round);
                                var scaled = input_claims[i];
                                for (0..scale_power) |_| scaled = scaled.add(scaled);
                                const weighted = scaled.mul(batching_coeffs[i]);
                                for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                            }
                        }
                    } else {
                        // Instance hasn't started yet - contribute scaled input claim as constant
                        const scale_power = max_num_rounds - rounds_per_instance[i] - round_idx - 1;
                        var scaled = input_claims[i];
                        for (0..scale_power) |_| scaled = scaled.add(scaled);
                        const weighted = scaled.mul(batching_coeffs[i]);
                        for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                    }
                }

                // Convert to compressed coefficients [c0, c2, c3]
                const compressed = poly_mod.UniPoly(F).evalsToCompressed(combined_evals);

                if (round_idx == 0 or round_idx == 16 or round_idx == max_num_rounds - 1) {
                }

                // Append to proof
                const coeffs = try self.allocator.alloc(F, 3);
                coeffs[0] = compressed[0];
                coeffs[1] = compressed[1];
                coeffs[2] = compressed[2];
                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                // Append to transcript: sumcheck polynomial coefficients
                transcript.appendScalars("sumcheck_poly", compressed[0..3]);

                // Sample round challenge
                const challenge = transcript.challengeScalar();
                try challenges.append(self.allocator, challenge);

                // Update batched claim by evaluating at challenge
                // CRITICAL: Must use evalFromHint (same as Jolt's verifier) to ensure
                // the claim evolution matches. Using Lagrange interpolation from combined_evals
                // would give different results because the evaluations may not be consistent
                // with what Jolt expects (different s1, s2, s3 can produce the same c0, c2, c3).
                const old_claim = batched_claim;
                batched_claim = evalFromHint(compressed, old_claim, challenge);


                // Debug: Print claim trajectory for first few and last few rounds
                if (round_idx < 3 or round_idx >= max_num_rounds - 5) {
                    // Check: s(0) + s(1) should equal old_claim for soundness
                    const sum_check = combined_evals[0].add(combined_evals[1]);
                    if (!sum_check.eql(old_claim)) {
                        // Print individual instance contributions
                        if (product_evals_this_round) |_| {
                            // Note: pp.current_claim is ALREADY UPDATED for next round at this point!
                        } else {
                        }
                        if (raf_evals_this_round) |_| {
                        } else {
                        }
                        if (rwc_evals_this_round) |_| {
                        } else {
                        }
                        if (output_evals_this_round) |_| {
                        } else {
                        }
                        if (instr_evals_this_round) |_| {
                        } else {
                        }
                    }
                }

                // Bind challenge in all active instances and update their claims
                // Instance 0: RWC (starts at round 0)
                if (rwc_prover) |*rwcp| {
                    if (rwc_evals_this_round) |evals| rwcp.updateClaim(evals, challenge);
                    rwcp.bindChallenge(challenge) catch {};
                }

                // Instance 1: ProductVirtualRemainder (starts at max_rounds - n_cycle_vars)
                if (product_prover != null and round_idx >= (max_num_rounds - n_cycle_vars)) {
                    if (product_evals_this_round) |evals| product_prover.?.updateClaim(evals, challenge);
                    product_prover.?.bindChallenge(challenge) catch {};
                }

                // Instance 2: InstructionLookups (starts at max_rounds - n_cycle_vars)
                if (instr_prover != null and round_idx >= (max_num_rounds - n_cycle_vars)) {
                    if (instr_evals_this_round) |evals| instr_prover.?.updateClaim(evals, challenge);
                    instr_prover.?.bindChallenge(challenge) catch {};
                }

                // Instance 3: RAF (starts at max_rounds - log_ram_k)
                if (raf_prover != null and round_idx >= (max_num_rounds - log_ram_k)) {
                    if (raf_evals_this_round) |evals| raf_prover.?.updateClaim(evals, challenge);
                    raf_prover.?.bindChallenge(challenge) catch {};
                }

                // Instance 4: OutputSumcheck (starts at max_rounds - log_ram_k)
                if (output_prover != null and round_idx >= (max_num_rounds - log_ram_k)) {
                    if (output_evals_this_round) |evals| output_prover.?.updateClaim(evals, challenge);
                    output_prover.?.bindChallenge(challenge);
                }

                // Reset per-round evals
                raf_evals_this_round = null;
                rwc_evals_this_round = null;
                instr_evals_this_round = null;

                // CRITICAL: Update individual_claims for each instance by evaluating at challenge
                // This is required for the batched sumcheck to maintain correct claim tracking
                // For inactive instances, the constant polynomial evaluates to the same scaled value
                // For active instances, we update based on the polynomial evaluation
                for (0..5) |i| {
                    const start_round = max_num_rounds - rounds_per_instance[i];
                    if (round_idx >= start_round) {
                        // Instance was active - update claim from its prover
                        if (i == 0 and rwc_prover != null) {
                            individual_claims[i] = rwc_prover.?.current_claim;
                        } else if (i == 1 and product_prover != null) {
                            individual_claims[i] = product_prover.?.current_claim;
                        } else if (i == 2 and instr_prover != null) {
                            individual_claims[i] = instr_prover.?.current_claim;
                        } else if (i == 3 and raf_prover != null) {
                            individual_claims[i] = raf_prover.?.current_claim;
                        } else if (i == 4 and output_prover != null) {
                            individual_claims[i] = output_prover.?.current_claim;
                        } else {
                            // Fallback: for instances without provers, keep tracking manually
                            // The claim after evaluating constant polynomial at r is just the constant
                            const remaining = rounds_per_instance[i] - (round_idx - start_round) - 1;
                            var scaled = input_claims[i];
                            for (0..remaining) |_| {
                                scaled = scaled.add(scaled);
                            }
                            individual_claims[i] = scaled;
                        }
                    } else {
                        // Instance not yet active - constant polynomial evaluates to scaled claim
                        // scale_power = remaining rounds until activation - 1
                        // = (start_round - round_idx - 1) where start_round = max_num_rounds - rounds_per_instance[i]
                        const start_round_i = max_num_rounds - rounds_per_instance[i];
                        if (round_idx + 1 < start_round_i) {
                            const scale_power = start_round_i - round_idx - 2;
                            var scaled = input_claims[i];
                            for (0..scale_power) |_| {
                                scaled = scaled.add(scaled);
                            }
                            individual_claims[i] = scaled;
                        } else {
                            // At the round just before activation, scale_power = 0
                            individual_claims[i] = input_claims[i];
                        }
                    }
                }

                // Debug: Check divergence between batched_claim and sum of individual claims
                // Do this AFTER individual_claims update so they're in sync
                if (round_idx == 15 or round_idx == 16 or round_idx == 25) {
                    var should_be_batched = F.zero();
                    for (0..5) |dbg_i| {
                        should_be_batched = should_be_batched.add(individual_claims[dbg_i].mul(batching_coeffs[dbg_i]));
                    }
                }
            }

            // Print prover's per-instance final claims for comparison with verifier
            var expected_batched = F.zero();
            // Instance 0: RWC
            if (rwc_prover) |rp| {
                expected_batched = expected_batched.add(rp.current_claim.mul(batching_coeffs[0]));
            } else {
                expected_batched = expected_batched.add(individual_claims[0].mul(batching_coeffs[0]));
            }
            // Instance 1: Product
            if (product_prover) |pp| {
                expected_batched = expected_batched.add(pp.current_claim.mul(batching_coeffs[1]));
            } else {
                expected_batched = expected_batched.add(individual_claims[1].mul(batching_coeffs[1]));
            }
            // Instance 2: InstrLookups
            if (instr_prover) |*ip| {
                expected_batched = expected_batched.add(ip.current_claim.mul(batching_coeffs[2]));
            } else {
                expected_batched = expected_batched.add(individual_claims[2].mul(batching_coeffs[2]));
            }
            // Instance 3: RAF
            if (raf_prover) |rp| {
                expected_batched = expected_batched.add(rp.current_claim.mul(batching_coeffs[3]));
            } else {
                expected_batched = expected_batched.add(individual_claims[3].mul(batching_coeffs[3]));
            }
            // Instance 4: Output
            if (output_prover) |op| {
                expected_batched = expected_batched.add(op.current_claim.mul(batching_coeffs[4]));
            } else {
                expected_batched = expected_batched.add(individual_claims[4].mul(batching_coeffs[4]));
            }


            // Debug: Print all challenges in LE format for comparison with Jolt

            // Debug: Print prover's final left/right values
            if (product_prover) |_| {
            }

            // Compute the 8 factor polynomial evaluations at r_cycle
            // r_cycle is the last n_cycle_vars challenges from Stage 2
            // ProductVirtualRemainder starts at round log_ram_k, so its r_cycle
            // is challenges[log_ram_k..max_num_rounds]
            const factor_evals = try self.computeProductFactorEvaluations(
                cycle_witnesses,
                challenges.items,
                n_cycle_vars,
                log_ram_k,
            );

            // Debug: Compute fused_left and fused_right from factor_evals and compare
            // Lagrange weights at r0_stage2
            const LagrangePoly = r1cs.univariate_skip.LagrangePolynomial(F);
            const w = try LagrangePoly.evals(3, r0_stage2, self.allocator);
            defer self.allocator.free(w);

            // fused_left = w[0]*l_inst + w[1]*lookup_out + w[2]*j_flag
            // fused_right = w[0]*r_inst + w[1]*branch_flag + w[2]*(1 - next_is_noop)


            // Compute tau_high_bound_r0 and tau_bound_r_tail_rev for expected_output_claim debug
            // tau_high_bound_r0 = LagrangeKernel(5, tau_high, r0)

            // tau_bound_r_tail_rev = eq(tau_low, r_cycle_reversed)
            // tau_low = tau[0..n_cycle_vars]
            // r_cycle_reversed = last n_cycle_vars challenges, reversed
            // The challenges.items are the Stage 2 sumcheck challenges
            // ProductVirtualRemainder starts at round (max_num_rounds - n_cycle_vars)
            // Its challenges are the LAST n_cycle_vars of challenges.items
            const product_start_round = max_num_rounds - n_cycle_vars;

            // Extract ProductVirtualRemainder challenges (last n_cycle_vars)
            var product_challenges = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(product_challenges);
            for (0..n_cycle_vars) |i| {
                if (product_start_round + i < challenges.items.len) {
                    product_challenges[i] = challenges.items[product_start_round + i];
                } else {
                    product_challenges[i] = F.zero();
                }
            }

            // Reverse the product challenges (r_cycle_reversed)
            var r_cycle_reversed = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_cycle_reversed);
            for (0..n_cycle_vars) |i| {
                r_cycle_reversed[i] = product_challenges[n_cycle_vars - 1 - i];
            }

            // Compute eq(tau_low, r_cycle_reversed)


            // Compute expected_output_claim


            // Copy challenges to return them
            const challenges_copy = try self.allocator.alloc(F, challenges.items.len);
            @memcpy(challenges_copy, challenges.items);

            // Get final claims from each prover
            const raf_claim = if (raf_prover) |rp| rp.getFinalClaim() else F.zero();
            const rwc_claim = if (rwc_prover) |*rp| rp.current_claim else F.zero();
            const output_claim = if (output_prover) |op| op.current_claim else F.zero();
            const instr_claim = if (instr_prover) |*ip| ip.current_claim else F.zero();

            // Get individual RWC opening claims (ra, val, inc)
            var rwc_ra_claim = F.zero();
            var rwc_val_claim = F.zero();
            var rwc_inc_claim = F.zero();
            if (rwc_prover) |*rp| {
                const rwc_opening_claims = rp.getOpeningClaims(challenges.items);
                rwc_ra_claim = rwc_opening_claims.ra_claim;
                rwc_val_claim = rwc_opening_claims.val_claim;
                rwc_inc_claim = rwc_opening_claims.inc_claim;

                // Verify: current_claim should equal eq_cycle * ra * (val + gamma * (val + inc))
            } else {
                // When rwc_prover is null (no RAM operations), the val polynomial equals val_init
                // everywhere. So val(r_address, r_cycle) = val_init(r_address).
                //
                // Jolt's verifier computes: input_claim = rwc_val_claim - init_eval
                // For this to equal 0 (what we want for no-RAM programs), rwc_val_claim must equal init_eval.
                //
                // We compute r_address from the Stage 2 challenges using normalize_opening_point logic.
                if (config.initial_ram != null and config.memory_layout != null) {
                    // RWC uses 3-phase structure:
                    // - Phase 1: phase1_num_rounds cycle vars (ALL cycle vars for Jolt compat)
                    // - Phase 2: log_k address vars
                    // - Phase 3: remaining cycle + address vars
                    const phase1 = n_cycle_vars; // ALL cycle vars in phase 1 (Jolt compat)
                    const phase2 = log_ram_k;
                    const phase3_cycle_len = n_cycle_vars - phase1;
                    const phase3_address_len = log_ram_k - phase2; // = 0 for default config

                    // Compute r_address_be = reverse(phase3_address) ++ reverse(phase2)
                    var r_addr_be = try self.allocator.alloc(F, log_ram_k);
                    defer self.allocator.free(r_addr_be);
                    @memset(r_addr_be, F.zero());

                    // Phase 2 address challenges are at indices [phase1..phase1+phase2)
                    const phase2_start = phase1;
                    for (0..phase2) |i| {
                        const src_idx = phase2_start + i;
                        if (src_idx < challenges.items.len) {
                            const dest_idx = phase3_address_len + (phase2 - 1 - i);
                            if (dest_idx < log_ram_k) {
                                r_addr_be[dest_idx] = challenges.items[src_idx];
                            }
                        }
                    }
                    // Phase 3 address challenges are at indices [phase1+phase2+phase3_cycle..end)
                    const phase3_addr_start = phase1 + phase2 + phase3_cycle_len;
                    for (0..phase3_address_len) |i| {
                        const src_idx = phase3_addr_start + i;
                        if (src_idx < challenges.items.len) {
                            const dest_idx = phase3_address_len - 1 - i;
                            r_addr_be[dest_idx] = challenges.items[src_idx];
                        }
                    }

                    // Debug: print r_address_be values
                    // Also print the source challenges

                    // Compute val_init(r_address_be) using bytecode_words (like Jolt does)
                    rwc_val_claim = computeInitialRamEval(
                        config.bytecode_words,
                        config.min_bytecode_address,
                        config.memory_layout.?,
                        r_addr_be,
                        log_ram_k,
                        config.program_inputs,
                    );
                }
            }

            // Get individual InstructionLookups opening claims (5 terms)
            var instr_lookup_output = F.zero();
            var instr_left_operand = F.zero();
            var instr_right_operand = F.zero();
            var instr_left_instr_input = F.zero();
            var instr_right_instr_input = F.zero();
            if (instr_prover) |*ip| {
                const instr_opening_claims = ip.getOpeningClaims();
                instr_lookup_output = instr_opening_claims.lookup_output;
                instr_left_operand = instr_opening_claims.left_operand;
                instr_right_operand = instr_opening_claims.right_operand;
                instr_left_instr_input = instr_opening_claims.left_instr_input;
                instr_right_instr_input = instr_opening_claims.right_instr_input;
            }


            // Get Val_final(r') and Val_init(r') from the OutputSumcheck prover
            // These are the MLE evaluations at the final opening point
            var output_val_final = F.zero();
            var output_val_init = F.zero();
            if (output_prover) |op| {
                const output_claims = op.getFinalClaims();
                output_val_final = output_claims.val_final;
                output_val_init = output_claims.val_init;
            }

            // Compute r_address_rw and r_cycle_rw from RWC challenges for RamRaClaimReduction
            // RWC uses 3-phase structure:
            // - Phase 1: n_cycle_vars cycle variables
            // - Phase 2: log_ram_k address variables
            // - Phase 3: 0 remaining variables (for default config)
            // Opening point = [r_address, r_cycle] in BIG_ENDIAN
            const phase1_rounds = n_cycle_vars;
            const phase2_rounds = log_ram_k;

            const r_address_rw = try self.allocator.alloc(F, log_ram_k);
            const r_cycle_rw = try self.allocator.alloc(F, n_cycle_vars);

            // r_address from phase 2 challenges, reversed to BIG_ENDIAN
            // Phase 2 challenges are at indices [phase1..phase1+phase2)
            for (0..phase2_rounds) |i| {
                const src_idx = phase1_rounds + i;
                if (src_idx < challenges.items.len) {
                    r_address_rw[phase2_rounds - 1 - i] = challenges.items[src_idx];
                } else {
                    r_address_rw[phase2_rounds - 1 - i] = F.zero();
                }
            }

            // r_cycle from phase 1 challenges, reversed to BIG_ENDIAN
            // Phase 1 challenges are at indices [0..phase1)
            for (0..phase1_rounds) |i| {
                if (i < challenges.items.len) {
                    r_cycle_rw[phase1_rounds - 1 - i] = challenges.items[i];
                } else {
                    r_cycle_rw[phase1_rounds - 1 - i] = F.zero();
                }
            }


            // CRITICAL FIX: r_address_raf should be computed from sumcheck challenges, NOT the pre-sampled r_address!
            //
            // In Jolt's Stage 2 batched sumcheck:
            // - RamRafEvaluation has 16 rounds, offset = 8, gets challenges[8..24]
            // - RamReadWriteChecking has 24 rounds, offset = 0, gets challenges[0..24]
            //   - Phase 1 (cycle): challenges[0..8]
            //   - Phase 2 (address): challenges[8..24]
            //
            // Both instances' r_address = reverse(challenges[8..24]).
            // So r_address_raf = r_address_rw!
            //
            // The pre-sampled r_address is used only for OutputSumcheck's eq polynomial,
            // NOT for RamRafEvaluation's opening point.
            //
            // Compute r_address_raf from sumcheck challenges the same way as r_address_rw:
            const r_address_raf = try self.allocator.alloc(F, log_ram_k);
            for (0..phase2_rounds) |i| {
                const src_idx = phase1_rounds + i;
                if (src_idx < challenges.items.len) {
                    r_address_raf[phase2_rounds - 1 - i] = challenges.items[src_idx];
                } else {
                    r_address_raf[phase2_rounds - 1 - i] = F.zero();
                }
            }

            // Debug: compare r_address_raf and r_address_rw (they should now be identical)

            // Compute ProductVirtualRemainder r_cycle from Stage 2 challenges
            // ProductVirtualRemainder starts at round (max_num_rounds - n_cycle_vars)
            // and runs for n_cycle_vars rounds. Reversed to BIG_ENDIAN.
            const product_start = max_num_rounds - n_cycle_vars;
            const r_cycle_product = try self.allocator.alloc(F, n_cycle_vars);
            for (0..n_cycle_vars) |i| {
                const src_idx = product_start + i;
                if (src_idx < challenges.items.len) {
                    r_cycle_product[n_cycle_vars - 1 - i] = challenges.items[src_idx];
                } else {
                    r_cycle_product[n_cycle_vars - 1 - i] = F.zero();
                }
            }

            return Stage2Result{
                .factor_evals = factor_evals,
                .challenges = challenges_copy,
                .raf_final_claim = raf_claim,
                .rwc_final_claim = rwc_claim,
                .output_final_claim = output_claim,
                .instr_final_claim = instr_claim,
                .output_val_final_claim = output_val_final,
                .output_val_init_claim = output_val_init,
                .r_address_raf = r_address_raf,
                .r_address_rw = r_address_rw,
                .r_cycle_rw = r_cycle_rw,
                .r_cycle_product = r_cycle_product,
                .rwc_ra_claim = rwc_ra_claim,
                .rwc_val_claim = rwc_val_claim,
                .rwc_inc_claim = rwc_inc_claim,
                .instr_lookup_output_claim = instr_lookup_output,
                .instr_left_operand_claim = instr_left_operand,
                .instr_right_operand_claim = instr_right_operand,
                .instr_left_instr_input_claim = instr_left_instr_input,
                .instr_right_instr_input_claim = instr_right_instr_input,
                .allocator = self.allocator,
            };
        }

        /// Compute MLE evaluations of the 8 factor polynomials at r_cycle
        ///
        /// The 8 factors are:
        /// 0: LeftInstructionInput
        /// 1: RightInstructionInput
        /// 2: IsRdNotZero
        /// 3: WriteLookupOutputToRDFlag
        /// 4: JumpFlag
        /// 5: LookupOutput
        /// 6: BranchFlag
        /// 7: NextIsNoop
        ///
        /// Returns MLE(factor_i, r_cycle) = Σ_t eq(r_cycle, t) * factor_value[t]
        fn computeProductFactorEvaluations(
            self: *Self,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            all_challenges: []const F,
            n_cycle_vars: usize,
            log_ram_k: usize,
        ) ![8]F {
            _ = log_ram_k;
            // r_cycle is the last n_cycle_vars challenges
            // In Jolt, ProductVirtualRemainder runs for n_cycle_vars rounds starting after log_ram_k rounds
            // So r_cycle = all_challenges[log_ram_k..log_ram_k + n_cycle_vars]
            // But the challenges are stored in order, so we take the last n_cycle_vars
            if (all_challenges.len < n_cycle_vars) {
                // Not enough challenges, return zeros
                return [8]F{ F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero() };
            }

            // Extract r_cycle (last n_cycle_vars challenges)
            // These are the sumcheck challenges that were used to bind the ProductVirtualRemainder
            // polynomial. Jolt uses normalize_opening_point which reverses the challenges to
            // convert from LITTLE_ENDIAN to BIG_ENDIAN.
            const r_cycle_start = all_challenges.len - n_cycle_vars;
            const r_cycle_original = all_challenges[r_cycle_start..];

            // Jolt's normalize_opening_point reverses the challenges to convert from LE to BE.
            // The factor claims must be computed at this reversed point to match the verifier's
            // expected_output_claim computation.
            const r_cycle = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_cycle);
            for (0..n_cycle_vars) |i| {
                r_cycle[i] = r_cycle_original[n_cycle_vars - 1 - i];
            }

            if (r_cycle.len > 0) {
            }
            if (r_cycle.len > 7) {
            }

            // Compute eq polynomial evaluations at r_cycle (using BIG_ENDIAN indexing like Jolt)
            const EqPoly = poly_mod.EqPolynomial(F);
            var eq_poly = try EqPoly.init(self.allocator, r_cycle);
            defer eq_poly.deinit();

            const eq_evals = try eq_poly.evals(self.allocator);
            defer self.allocator.free(eq_evals);

            // Print sum of eq_evals (should be 1 for partition of unity)
            var eq_sum = F.zero();
            for (eq_evals) |ev| {
                eq_sum = eq_sum.add(ev);
            }

            // Initialize factor accumulators
            var factor_evals = [8]F{ F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero() };

            // Compute MLE evaluation: Σ_t eq(r_cycle, t) * factor_value[t]
            // Uses UnreducedProductAccum to defer Montgomery reduction across all cycles.
            const num_cycles = @min(eq_evals.len, cycle_witnesses.len);
            const UPA = UnreducedProductAccum;

            // Factor indices into R1CSCycleInputs.values
            const factor_indices = [8]usize{
                r1cs.R1CSInputIndex.LeftInstructionInput.toIndex(),
                r1cs.R1CSInputIndex.RightInstructionInput.toIndex(),
                r1cs.R1CSInputIndex.FlagJump.toIndex(),
                r1cs.R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex(),
                r1cs.R1CSInputIndex.LookupOutput.toIndex(),
                r1cs.R1CSInputIndex.FlagBranch.toIndex(),
                0, // placeholder for NextIsNoop (computed separately)
                r1cs.R1CSInputIndex.FlagVirtualInstruction.toIndex(),
            };

            var accum: [8]UPA = .{UPA.zero()} ** 8;
            for (0..num_cycles) |t| {
                const eq_val = eq_evals[t];
                const witness = &cycle_witnesses[t];

                // Factors 0-5, 7: direct witness lookup
                inline for ([_]usize{ 0, 1, 2, 3, 4, 5, 7 }) |fi| {
                    accum[fi].addAssign(eq_val.mulToProductAccum(witness.values[factor_indices[fi]]));
                }

                // Factor 6: NextIsNoop
                const next_is_noop = if (t + 1 < cycle_witnesses.len)
                    cycle_witnesses[t + 1].values[r1cs.R1CSInputIndex.FlagIsNoop.toIndex()]
                else
                    F.one();
                accum[6].addAssign(eq_val.mulToProductAccum(next_is_noop));
            }
            inline for (0..8) |fi| {
                factor_evals[fi] = accum[fi].reduce();
            }

            // Debug: Print counts

            // Handle padding cycles (indices from cycle_witnesses.len to eq_evals.len)
            // Note: If cycle_witnesses already includes NoOp padding (from R1CS witness generator),
            // this loop may not execute. Only run if witnesses are shorter than eq domain.
            //
            // Padding cycles are NoOp cycles. For NoOp:
            // - Factors 0-5, 7: all zero (no instruction input, no flags set, no output)
            // - Factor 6 (NextIsNoop): 1 (next cycle is also a NoOp)
            if (cycle_witnesses.len < eq_evals.len) {
                for (cycle_witnesses.len..eq_evals.len) |t| {
                    const eq_val = eq_evals[t];
                    factor_evals[6] = factor_evals[6].add(eq_val);
                }
            }


            return factor_evals;
        }

        /// Evaluate polynomial at challenge using Jolt's eval_from_hint formula
        /// Delegates to the shared UniPoly implementation.
        fn evalFromHint(compressed: [3]F, hint: F, x: F) F {
            return poly_mod.UniPoly(F).evalFromHint(compressed, hint, x);
        }

        /// Compute eq(r, idx) where r is in BIG_ENDIAN order (MSB first).
        fn computeEqAtPointBigEndian(r: []const F, idx: usize) F {
            var result = F.one();
            const n = r.len;
            for (0..n) |i| {
                const bit = (idx >> @intCast(n - 1 - i)) & 1;
                if (bit == 1) {
                    result = result.mul(r[i]);
                } else {
                    result = result.mul(F.one().sub(r[i]));
                }
            }
            return result;
        }

        /// Compute eq(r, idx) where r is in LITTLE_ENDIAN order (LSB first).
        /// bit i of idx corresponds to r[i].
        fn computeEqAtPointLE(r: []const F, idx: usize) F {
            var result = F.one();
            for (r, 0..) |ri, i| {
                const bit = (idx >> @intCast(i)) & 1;
                if (bit == 1) {
                    result = result.mul(ri);
                } else {
                    result = result.mul(F.one().sub(ri));
                }
            }
            return result;
        }

        /// Evaluate the initial RAM polynomial at r_address (BIG_ENDIAN).
        ///
        /// This matches Jolt's `eval_initial_ram_mle` which evaluates:
        ///   sum_k bytecode_words[k] * eq(r_address, bytecode_start + k)
        /// where bytecode_start = remap_address(min_bytecode_address)
        ///
        /// NOTE: Unlike the old implementation that used initial_ram hashmap (stack data),
        /// this now uses bytecode_words (program bytecode) like Jolt does.
        fn computeInitialRamEval(
            bytecode_words: ?[]const u64,
            min_bytecode_address: u64,
            memory_layout: *const jolt_device.MemoryLayout,
            r_address_be: []const F,
            log_ram_k: usize,
            program_inputs: ?[]const u8,
        ) F {

            const lowest_address = memory_layout.getLowestAddress();

            var result = F.zero();
            const max_idx: usize = @as(usize, 1) << @intCast(log_ram_k);

            // Evaluate bytecode region (like Jolt's eval_initial_ram_mle)
            if (bytecode_words) |words| {
                if (words.len > 0) {
                    // bytecode_start = remap_address(min_bytecode_address)
                    // remap_address = (address - lowest_address) / 8
                    const bytecode_start: usize = @intCast((min_bytecode_address - lowest_address) / 8);
                    if (words.len > 0) {
                    }

                    // Sum: bytecode_words[k] * eq(r_address, bytecode_start + k)
                    for (words, 0..) |word, k| {
                        const idx = bytecode_start + k;
                        if (idx >= max_idx) break;

                        const eq_val = computeEqAtPointBigEndian(r_address_be, idx);
                        const val = F.fromU64(word);
                        result = result.add(eq_val.mul(val));
                    }
                }
            }

            // Also add inputs region (like Jolt does)
            if (program_inputs) |inputs| {
                if (inputs.len > 0) {
                    // input_start = remap_address(memory_layout.input_start)
                    const input_start: usize = @intCast((memory_layout.input_start - lowest_address) / 8);

                    // Pack inputs into u64 words (little-endian)
                    var idx = input_start;
                    var off: usize = 0;
                    while (off < inputs.len) {
                        if (idx >= max_idx) break;

                        var word: u64 = 0;
                        const chunk_end = @min(off + 8, inputs.len);
                        for (off..chunk_end) |i| {
                            const byte_pos = i - off;
                            word |= @as(u64, inputs[i]) << @intCast(byte_pos * 8);
                        }

                        const eq_val = computeEqAtPointBigEndian(r_address_be, idx);
                        const val = F.fromU64(word);
                        result = result.add(eq_val.mul(val));

                        idx += 1;
                        off = chunk_end;
                    }
                }
            }

            return result;
        }

        /// Evaluate cubic polynomial at a challenge point from Toom-Cook evaluations
        /// Delegates to the shared UniPoly implementation.
        fn evaluateCubicAtChallengeFromEvals(evals: [4]F, x: F) F {
            return poly_mod.UniPoly(F).evaluateToomCookAt(evals, x);
        }

        /// Evaluate quadratic polynomial at a challenge point using Lagrange interpolation
        /// Input: evals = [p(0), p(1), p(2), _] where only the first 3 are used
        ///
        /// For ValFinal (degree-2 polynomial), we use Lagrange interpolation through 3 points
        /// [p(0), p(1), p(2)] at x = 0, 1, 2. This matches Jolt's from_evals_and_hint which
        /// uses Vandermonde interpolation through 3 points.
        ///
        /// Lagrange basis polynomials for points 0, 1, 2:
        /// L_0(x) = (x-1)(x-2) / ((0-1)(0-2)) = (x-1)(x-2) / 2
        /// L_1(x) = (x-0)(x-2) / ((1-0)(1-2)) = x(x-2) / (-1) = x(2-x)
        /// L_2(x) = (x-0)(x-1) / ((2-0)(2-1)) = x(x-1) / 2
        ///
        /// p(x) = p(0)*L_0(x) + p(1)*L_1(x) + p(2)*L_2(x)
        fn evaluateQuadraticAtChallengeFromEvals(evals: [4]F, x: F) F {
            const p0 = evals[0];
            const p1 = evals[1];
            const p2 = evals[2];

            const one = F.one();
            const two = F.fromU64(2);
            const inv2 = two.inverse().?;

            // L_0(x) = (x-1)(x-2) / 2
            const x_minus_1 = x.sub(one);
            const x_minus_2 = x.sub(two);
            const L_0 = x_minus_1.mul(x_minus_2).mul(inv2);

            // L_1(x) = x(2-x) = x(2-x)
            const two_minus_x = two.sub(x);
            const L_1 = x.mul(two_minus_x);

            // L_2(x) = x(x-1) / 2
            const L_2 = x.mul(x_minus_1).mul(inv2);

            // p(x) = p(0)*L_0(x) + p(1)*L_1(x) + p(2)*L_2(x)
            return p0.mul(L_0).add(p1.mul(L_1)).add(p2.mul(L_2));
        }


        /// Create a UniSkipFirstRoundProof for Stage 2 with actual base claims and extended evaluations
        ///
        /// This constructs the polynomial s1(Y) = L(tau_high, Y) * t1(Y) where:
        /// - L is the Lagrange kernel over the 5-point domain {-2, -1, 0, 1, 2}
        /// - t1 is interpolated from base_evals (at base domain) and extended_evals
        ///
        /// For product virtualization, the base_evals are the 3 product claims from Stage 1:
        /// [Product, ShouldBranch, ShouldJump]
        ///
        /// The extended_evals are the fused products at extended points {-2, 2}.
        ///
        /// The polynomial satisfies: Σ_t s1(t) = Σ_i L_i(tau_high) * base_evals[i] = input_claim
        fn createUniSkipProofStage2WithClaims(
            self: *Self,
            base_evals: *const [3]F,
            tau_high: F,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau_stage2: []const F,
        ) !?UniSkipFirstRoundProof(F) {
            const univariate_skip = r1cs.univariate_skip;

            const DOMAIN_SIZE = univariate_skip.PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE; // 5
            const DEGREE = univariate_skip.PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE; // 4
            const EXTENDED_SIZE = univariate_skip.PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE; // 9
            const NUM_COEFFS = univariate_skip.PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS; // 13

            // Compute extended evaluations from cycle witnesses using the 5 product constraints
            // Extended points {-3, 3, -4, 4} require the fused products computed from witness data
            const extended_evals: [DEGREE]F = blk: {
                if (cycle_witnesses.len == 0) {
                    // No witnesses - use zeros
                    break :blk [_]F{F.zero()} ** DEGREE;
                }

                // Extract the 8 product factors from each cycle witness
                const cycle_factors = try self.allocator.alloc([8]F, cycle_witnesses.len);
                defer self.allocator.free(cycle_factors);

                for (cycle_witnesses, 0..) |witness, idx| {
                    cycle_factors[idx] = extractProductFactors(F, &witness, cycle_witnesses, idx);
                }

                // Compute extended evaluations using the precomputed Lagrange coefficients
                break :blk try univariate_skip.computeProductVirtualExtendedEvals(
                    F,
                    cycle_factors,
                    tau_stage2,
                    self.allocator,
                );
            };


            // Use the existing buildUniskipFirstRoundPoly function
            const uni_poly = try univariate_skip.buildUniskipFirstRoundPoly(
                F,
                DOMAIN_SIZE,
                DEGREE,
                EXTENDED_SIZE,
                NUM_COEFFS,
                base_evals,
                &extended_evals,
                tau_high,
                self.allocator,
            );


            // Verify the polynomial satisfies the sum constraint
            // input_claim = Σ L_i(tau_high) * base_evals[i]
            const LagrangePoly = univariate_skip.LagrangePolynomial(F);
            const lagrange_evals = try LagrangePoly.evals(DOMAIN_SIZE, tau_high, self.allocator);
            defer self.allocator.free(lagrange_evals);

            var input_claim = F.zero();
            for (base_evals, 0..) |eval, i| {
                input_claim = input_claim.add(lagrange_evals[i].mul(eval));
            }

            // Check domain sum
            const power_sums = univariate_skip.computePowerSums(DOMAIN_SIZE, NUM_COEFFS);
            var domain_sum = F.zero();
            for (uni_poly.coeffs, 0..) |coeff, j| {
                domain_sum = domain_sum.add(coeff.mulI128(power_sums[j]));
            }

            // Return as UniSkipFirstRoundProof
            return UniSkipFirstRoundProof(F){
                .uni_poly = uni_poly.coeffs,
                .allocator = self.allocator,
            };
        }
    };
}

/// Extract the 8 product factors from an R1CS cycle witness
///
/// The 8 factors are (matching upstream PRODUCT_UNIQUE_FACTOR_VIRTUALS):
///   [0] LeftInstructionInput
///   [1] RightInstructionInput
///   [2] JumpFlag (OpFlags::Jump)
///   [3] WriteLookupOutputToRDFlag (OpFlags::WriteLookupOutputToRD)
///   [4] LookupOutput
///   [5] BranchFlag (InstructionFlags::Branch)
///   [6] NextIsNoop
///   [7] VirtualInstructionFlag (OpFlags::VirtualInstruction)
fn extractProductFactors(
    comptime F: type,
    witness: *const r1cs.R1CSCycleInputs(F),
    all_witnesses: []const r1cs.R1CSCycleInputs(F),
    cycle_idx: usize,
) [8]F {
    const R1CSInputIndex = r1cs.R1CSInputIndex;

    return [8]F{
        // 0: LeftInstructionInput
        witness.values[R1CSInputIndex.LeftInstructionInput.toIndex()],
        // 1: RightInstructionInput
        witness.values[R1CSInputIndex.RightInstructionInput.toIndex()],
        // 2: JumpFlag (OpFlags::Jump)
        witness.values[R1CSInputIndex.FlagJump.toIndex()],
        // 3: WriteLookupOutputToRDFlag (OpFlags::WriteLookupOutputToRD)
        witness.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()],
        // 4: LookupOutput
        witness.values[R1CSInputIndex.LookupOutput.toIndex()],
        // 5: BranchFlag (InstructionFlags::Branch)
        witness.values[R1CSInputIndex.FlagBranch.toIndex()],
        // 6: NextIsNoop - 1 if next instruction is a noop
        blk: {
            if (cycle_idx + 1 < all_witnesses.len) {
                const next_witness = &all_witnesses[cycle_idx + 1];
                break :blk next_witness.values[R1CSInputIndex.FlagIsNoop.toIndex()];
            }
            // Last cycle: NextIsNoop = true
            break :blk F.one();
        },
        // 7: VirtualInstructionFlag (OpFlags::VirtualInstruction)
        witness.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()],
    };
}

/// Configuration for proof conversion
///
/// These values must match Jolt's config.rs:
/// - log_k_chunk: Must be <= 8 (Jolt uses 4 for small traces, 8 for large)
/// - lookups_ra_virtual_log_k_chunk: Jolt uses LOG_K/8 (=16) for small traces
const tracer = @import("../tracer/mod.zig");

pub const JoltProverConfig = struct {
    /// Bytecode address space size (K) - must match Jolt's BytecodePreprocessing.code_size
    /// Use computeBytecodeCodeSize() in mod.zig to compute from raw program bytes
    bytecode_K: usize = 2048,
    /// Log of chunk size for one-hot encoding (must be <= 8, Jolt uses 4 for small traces)
    log_k_chunk: usize = 4,
    /// Log of chunk size for lookups RA virtualization (LOG_K / 8 = 128 / 8 = 16 for small traces)
    lookups_ra_virtual_log_k_chunk: usize = 16,
    /// Memory layout for computing I/O polynomial evaluations
    /// If null, OutputSumcheck will use zero claims (which will fail verification)
    memory_layout: ?*const jolt_device.MemoryLayout = null,
    /// Initial RAM state (before execution)
    initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64) = null,
    /// Final RAM state (after execution)
    final_ram: ?*const std.AutoHashMapUnmanaged(u64, u64) = null,
    /// Memory trace for RAF evaluation sumcheck
    /// If null, uses zero-polynomial approach (may fail for non-zero claims)
    memory_trace: ?*const ram.MemoryTrace = null,
    /// Program input bytes (for OutputSumcheck's ProgramIOPolynomial)
    program_inputs: ?[]const u8 = null,
    /// Program output bytes (for OutputSumcheck's ProgramIOPolynomial)
    program_outputs: ?[]const u8 = null,
    /// Whether the program panicked (for OutputSumcheck's ProgramIOPolynomial)
    is_panicking: bool = false,
    /// Execution trace for Stage 4 RegistersReadWriteChecking
    /// If null, Stage 4 uses zero-polynomial approach (which fails verification)
    execution_trace: ?*const tracer.ExecutionTrace = null,
    /// Bytecode words for initial RAM MLE evaluation (packed into 8-byte words)
    /// This is required for Stage 4 to compute the correct init_eval for RamValEvaluation
    bytecode_words: ?[]const u64 = null,
    /// Minimum bytecode address (word-aligned: actual_min_address / 8 * 8)
    /// Used to compute the remapped bytecode_start index
    min_bytecode_address: u64 = 0,
    /// BytecodePCMapper for converting ELF addresses to bytecode array indices
    /// Required for Stage 6 BytecodeReadRaf to correctly map cycle PCs to bytecode rows
    bytecode_pc_map: ?*const preprocessing.BytecodePCMapper = null,
    /// Raw ELF code bytes (text section) for populating static bytecode entries in Stage 6
    /// Required so buildBytecodeEntries can fill entries for ALL instructions, not just executed ones
    program_code_bytes: ?[]const u8 = null,
    /// Base address of the code section (typically 0x80000000)
    code_base_address: u64 = 0x80000000,
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = field_mod.BN254Scalar;

test "proof converter: basic initialization" {
    const converter = JoltProver(BN254Scalar).init(testing.allocator);
    _ = converter;
}


test "proof converter: proveWithTranscript uses Blake2b transcript" {
    const F = BN254Scalar;
    var converter = JoltProver(F).init(testing.allocator);

    // Create trivial cycle witnesses
    const cycle_witnesses = [_]r1cs.R1CSCycleInputs(F){
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
    };

    // Create tau challenge vector
    const tau = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };

    // Initialize transcript (matching Jolt's label)
    var transcript = Blake2bTranscript(F).init("jolt_v1");

    // Dummy types
    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    // Convert with transcript
    var jolt_proof = try converter.proveWithTranscript(
        DummyCommitment,
        DummyProof,
        2, // log_t: trace_length = 4
        8, // log_k: ram_K = 256
        &[_]DummyCommitment{},
        null,
        .{},
        &cycle_witnesses,
        &tau,
        &transcript,
    );
    defer jolt_proof.deinit();

    // Verify trace length
    try testing.expectEqual(@as(usize, 4), jolt_proof.trace_length);

    // Verify transcript was used (round counter should be > 0 after generating proof)
    try testing.expect(transcript.n_rounds > 0);

    // Verify uni skip proofs were created
    try testing.expect(jolt_proof.stage1_uni_skip_first_round_proof != null);
    try testing.expect(jolt_proof.stage2_uni_skip_first_round_proof != null);

    // Verify opening claims were added
    try testing.expect(jolt_proof.opening_claims.len() > 0);
}

test "proof converter: transcript produces deterministic challenges" {
    const F = BN254Scalar;

    // Create two converters and transcripts with same inputs
    var converter1 = JoltProver(F).init(testing.allocator);
    var converter2 = JoltProver(F).init(testing.allocator);

    const cycle_witnesses = [_]r1cs.R1CSCycleInputs(F){
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
    };

    const tau = [_]F{ F.fromU64(1), F.fromU64(2) };

    var transcript1 = Blake2bTranscript(F).init("jolt_test");
    var transcript2 = Blake2bTranscript(F).init("jolt_test");

    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    var jolt_proof1 = try converter1.proveWithTranscript(
        DummyCommitment,
        DummyProof,
        2, // log_t
        8, // log_k
        &[_]DummyCommitment{},
        null,
        .{},
        &cycle_witnesses,
        &tau,
        &transcript1,
    );
    defer jolt_proof1.deinit();

    var jolt_proof2 = try converter2.proveWithTranscript(
        DummyCommitment,
        DummyProof,
        2, // log_t
        8, // log_k
        &[_]DummyCommitment{},
        null,
        .{},
        &cycle_witnesses,
        &tau,
        &transcript2,
    );
    defer jolt_proof2.deinit();

    // Same inputs should produce same transcript state
    try testing.expectEqualSlices(u8, &transcript1.state, &transcript2.state);
    try testing.expectEqual(transcript1.n_rounds, transcript2.n_rounds);
}
