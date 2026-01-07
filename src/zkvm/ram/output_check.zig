//! RAM Output Sumcheck
//!
//! This module implements the output sumcheck protocol that proves the relation:
//!   Σ_k eq(r_address, k) ⋅ io_mask(k) ⋅ (val_final(k) − val_io(k)) = 0
//!
//! Where:
//! - r_address is a random address challenge vector
//! - io_mask(k) = 1 if k is in the I/O region of memory, 0 otherwise
//! - val_final(k) is the final memory value at address k
//! - val_io(k) is the publicly claimed output value at address k
//!
//! This proves that the final RAM state matches the expected I/O in the I/O region.
//!
//! Reference: jolt-core/src/zkvm/ram/output_check.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const poly_mod = @import("../../poly/mod.zig");
const jolt_device = @import("../jolt_device.zig");
const constants = @import("../../common/constants.zig");

/// Degree bound of the sumcheck round polynomials
/// eq * io_mask * (val_final - val_io) has degree 3 in the current variable
const OUTPUT_SUMCHECK_DEGREE_BOUND: usize = 3;

/// Parameters for output sumcheck
pub fn OutputSumcheckParams(comptime F: type) type {
    return struct {
        const Self = @This();

        /// K = 2^log_K addresses
        K: usize,
        log_K: usize,
        /// Random address challenge
        r_address: []const F,
        /// Memory layout
        memory_layout: *const jolt_device.MemoryLayout,
        /// Allocator
        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            log_K: usize,
            r_address: []const F,
            memory_layout: *const jolt_device.MemoryLayout,
        ) !Self {
            const r_copy = try allocator.alloc(F, r_address.len);
            @memcpy(r_copy, r_address);

            return Self{
                .K = @as(usize, 1) << @intCast(log_K),
                .log_K = log_K,
                .r_address = r_copy,
                .memory_layout = memory_layout,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(@constCast(self.r_address));
        }

        pub fn numRounds(self: *const Self) usize {
            return self.log_K;
        }

        pub fn degreeBound() usize {
            return OUTPUT_SUMCHECK_DEGREE_BOUND;
        }

        /// Input claim is always zero (this is a zero-check)
        pub fn inputClaim() F {
            return F.zero();
        }
    };
}

/// Output sumcheck prover
pub fn OutputSumcheckProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// val_init[k] = initial RAM value at address k
        val_init: []F,
        /// val_final[k] = final RAM value at address k
        val_final: []F,
        /// val_io[k] = expected I/O value at address k (= val_final[k] if k in IO region)
        val_io: []F,
        /// io_mask[k] = 1 if k in IO region, 0 otherwise
        io_mask: []F,
        /// EQ polynomial evals: eq_r_address[k] = eq(r_address, k)
        eq_r_address: []F,
        /// Number of variables (= log_K)
        num_vars: usize,
        /// Current size (halves each round)
        current_size: usize,
        /// Current claim
        current_claim: F,
        /// Allocator
        allocator: Allocator,

        /// Initialize from RAM states and memory layout
        ///
        /// Parameters:
        /// - initial_ram: Initial RAM state as sparse map (address -> value)
        /// - final_ram: Final RAM state as sparse map (address -> value)
        /// - r_address: Random address challenges
        /// - memory_layout: Memory layout defining IO region
        pub fn init(
            allocator: Allocator,
            initial_ram: *const std.AutoHashMapUnmanaged(u64, u64),
            final_ram: *const std.AutoHashMapUnmanaged(u64, u64),
            r_address: []const F,
            memory_layout: *const jolt_device.MemoryLayout,
        ) !Self {
            const log_K = r_address.len;
            const K: usize = @as(usize, 1) << @intCast(log_K);

            // Allocate arrays
            const val_init = try allocator.alloc(F, K);
            const val_final = try allocator.alloc(F, K);
            const val_io = try allocator.alloc(F, K);
            const io_mask = try allocator.alloc(F, K);
            const eq_r_address = try allocator.alloc(F, K);

            // Initialize val_init and val_final from sparse maps
            var non_zero_count: usize = 0;
            var io_region_values: usize = 0;
            for (val_init, val_final, 0..) |*vi, *vf, k| {
                // Convert index k to address
                const address = indexToAddress(k, memory_layout);

                // Look up values (default 0)
                vi.* = if (initial_ram.get(address)) |v| F.fromU64(v) else F.zero();
                vf.* = if (final_ram.get(address)) |v| blk: {
                    non_zero_count += 1;
                    if (k >= 1024 and k < 4096) { // IO region
                        io_region_values += 1;
                        std.debug.print("[ZOLT] OutputSumcheck: IO region k={}, addr=0x{X:0>8}, val={}\n", .{ k, address, v });
                    }
                    break :blk F.fromU64(v);
                } else F.zero();
            }
            std.debug.print("[ZOLT] OutputSumcheck: non_zero_count={}, io_region_values={}, K={}\n", .{ non_zero_count, io_region_values, K });

            // Compute IO region bounds
            const io_start = remapAddress(memory_layout.input_start, memory_layout) orelse 0;
            const io_end = remapAddress(constants.RAM_START_ADDRESS, memory_layout) orelse K;
            std.debug.print("[ZOLT] OutputSumcheck: io_start={}, io_end={}\n", .{ io_start, io_end });

            // Initialize io_mask and val_io
            for (io_mask, val_io, 0..) |*mask, *vio, k| {
                if (k >= io_start and k < io_end) {
                    mask.* = F.one();
                    vio.* = val_final[k]; // Copy from final state
                } else {
                    mask.* = F.zero();
                    vio.* = F.zero();
                }
            }

            // Set termination bit if not panicking
            // For a correctly terminating program, termination addr should have value 1
            const termination_index = remapAddress(memory_layout.termination, memory_layout) orelse 0;
            std.debug.print("[ZOLT] OutputSumcheck: termination_index={}, in IO={}\n", .{ termination_index, termination_index >= io_start and termination_index < io_end });
            if (termination_index < K and termination_index >= io_start and termination_index < io_end) {
                val_io[termination_index] = F.one();
                std.debug.print("[ZOLT] OutputSumcheck: set val_io[{}] = 1 (termination)\n", .{termination_index});
            }

            // Compute EQ polynomial evaluations
            computeEqEvals(F, eq_r_address, r_address);

            return Self{
                .val_init = val_init,
                .val_final = val_final,
                .val_io = val_io,
                .io_mask = io_mask,
                .eq_r_address = eq_r_address,
                .num_vars = log_K,
                .current_size = K,
                .current_claim = F.zero(), // Input claim is 0
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.val_init);
            self.allocator.free(self.val_final);
            self.allocator.free(self.val_io);
            self.allocator.free(self.io_mask);
            self.allocator.free(self.eq_r_address);
        }

        /// Compute round polynomial and return compressed coefficients [c0, c2, c3]
        ///
        /// The round polynomial is:
        ///   s(X) = Σ_{k with current var = X} eq(r, k) * io_mask(k) * (vf(k) - vio(k))
        ///
        /// This is degree 3 in X.
        pub fn computeRoundPolynomial(self: *Self) [3]F {
            const half = self.current_size / 2;

            // Evaluate s(0), s(1), s(2), s(3)
            var s0 = F.zero();
            var s1 = F.zero();
            var s2 = F.zero();
            var s3 = F.zero();

            // For each pair (2g, 2g+1)
            for (0..half) |g| {
                const idx0 = 2 * g;
                const idx1 = 2 * g + 1;

                // Get values at X=0 and X=1
                const eq0 = self.eq_r_address[idx0];
                const eq1 = self.eq_r_address[idx1];
                const io0 = self.io_mask[idx0];
                const io1 = self.io_mask[idx1];
                const vf0 = self.val_final[idx0];
                const vf1 = self.val_final[idx1];
                const vio0 = self.val_io[idx0];
                const vio1 = self.val_io[idx1];

                // v0 = vf0 - vio0, v1 = vf1 - vio1
                const v0 = vf0.sub(vio0);
                const v1 = vf1.sub(vio1);

                // p(X) = eq(X) * io(X) * v(X) where all are linear in X
                // eq(X) = eq0 + (eq1 - eq0) * X = eq0 + deq * X
                // io(X) = io0 + (io1 - io0) * X = io0 + dio * X
                // v(X) = v0 + (v1 - v0) * X = v0 + dv * X
                const deq = eq1.sub(eq0);
                const dio = io1.sub(io0);
                const dv = v1.sub(v0);

                // p(0) = eq0 * io0 * v0
                const p0 = eq0.mul(io0).mul(v0);

                // p(1) = eq1 * io1 * v1
                const p1 = eq1.mul(io1).mul(v1);

                // p(2) = (eq0 + 2*deq) * (io0 + 2*dio) * (v0 + 2*dv)
                const eq2 = eq0.add(deq).add(deq);
                const io2 = io0.add(dio).add(dio);
                const v2 = v0.add(dv).add(dv);
                const p2 = eq2.mul(io2).mul(v2);

                // p(3) = (eq0 + 3*deq) * (io0 + 3*dio) * (v0 + 3*dv)
                const eq3 = eq2.add(deq);
                const io3 = io2.add(dio);
                const v3 = v2.add(dv);
                const p3 = eq3.mul(io3).mul(v3);

                s0 = s0.add(p0);
                s1 = s1.add(p1);
                s2 = s2.add(p2);
                s3 = s3.add(p3);
            }

            // Convert evaluations to compressed coefficients [c0, c2, c3]
            return poly_mod.UniPoly(F).evalsToCompressed([4]F{ s0, s1, s2, s3 });
        }

        /// Bind challenge and update polynomials for next round
        pub fn bindChallenge(self: *Self, r: F) void {
            const half = self.current_size / 2;

            // Bind all polynomials
            for (0..half) |g| {
                const idx0 = 2 * g;
                const idx1 = 2 * g + 1;

                // eq[g] = eq0 + r * (eq1 - eq0)
                self.eq_r_address[g] = self.eq_r_address[idx0].add(
                    r.mul(self.eq_r_address[idx1].sub(self.eq_r_address[idx0])),
                );

                // Same for other polynomials
                self.io_mask[g] = self.io_mask[idx0].add(
                    r.mul(self.io_mask[idx1].sub(self.io_mask[idx0])),
                );
                self.val_final[g] = self.val_final[idx0].add(
                    r.mul(self.val_final[idx1].sub(self.val_final[idx0])),
                );
                self.val_io[g] = self.val_io[idx0].add(
                    r.mul(self.val_io[idx1].sub(self.val_io[idx0])),
                );
                self.val_init[g] = self.val_init[idx0].add(
                    r.mul(self.val_init[idx1].sub(self.val_init[idx0])),
                );
            }

            self.current_size = half;
        }

        /// Update claim from evaluations at challenge
        pub fn updateClaim(self: *Self, evals: [4]F, challenge: F) void {
            // Evaluate cubic at challenge: c0 + c1*r + c2*r^2 + c3*r^3
            // Use Horner's method
            const r = challenge;
            const r2 = r.mul(r);
            const r3 = r2.mul(r);

            // First recover c1 from evals
            // s(0) = c0, s(1) = c0 + c1 + c2 + c3
            // c1 = s(1) - s(0) - c2 - c3
            const c0 = evals[0];
            const c2 = lagrangeC2(evals);
            const c3 = lagrangeC3(evals);
            const c1 = evals[1].sub(c0).sub(c2).sub(c3);

            self.current_claim = c0.add(c1.mul(r)).add(c2.mul(r2)).add(c3.mul(r3));
        }

        /// Get final claim values
        pub fn getFinalClaims(self: *const Self) struct { val_final: F, val_init: F } {
            return .{
                .val_final = self.val_final[0],
                .val_init = self.val_init[0],
            };
        }

        // Helper: compute c2 from evaluations using Lagrange
        fn lagrangeC2(evals: [4]F) F {
            // c2 = (s(0) - 2*s(1) + s(2)) / 2
            const two = F.fromU64(2);
            const half = two.inverse() orelse F.zero();
            return evals[0].sub(evals[1].mul(two)).add(evals[2]).mul(half);
        }

        // Helper: compute c3 from evaluations
        fn lagrangeC3(evals: [4]F) F {
            // c3 = (-s(0) + 3*s(1) - 3*s(2) + s(3)) / 6
            const six = F.fromU64(6);
            const three = F.fromU64(3);
            const sixth = six.inverse() orelse F.zero();
            return F.zero()
                .sub(evals[0])
                .add(evals[1].mul(three))
                .sub(evals[2].mul(three))
                .add(evals[3])
                .mul(sixth);
        }
    };
}

/// Convert index k to memory address
/// Index k maps to the address at lowest_address + k * 8 (word-aligned)
fn indexToAddress(k: usize, memory_layout: *const jolt_device.MemoryLayout) u64 {
    const lowest = memory_layout.getLowestAddress();
    return lowest + @as(u64, @intCast(k)) * 8;
}

/// Remap address to index
fn remapAddress(address: u64, memory_layout: *const jolt_device.MemoryLayout) ?usize {
    // Simplified remapping - should match jolt_device.remapAddress
    const lowest = memory_layout.getLowestAddress();
    if (address < lowest) return null;
    const offset = address - lowest;
    if (offset % 8 != 0) return null;
    return @as(usize, @intCast(offset / 8));
}

/// Compute EQ polynomial evaluations using LowToHigh (LSB-first) ordering
/// eq_evals[k] = eq(r, k) where k's bit 0 = x_0 (first variable to bind)
///
/// This matches Jolt's LowToHigh binding order:
/// - Variable 0 (r[0]) corresponds to bit 0 (LSB) of index k
/// - Variable n-1 (r[n-1]) corresponds to bit n-1 (MSB) of index k
fn computeEqEvals(comptime F: type, eq_evals: []F, r: []const F) void {
    const n = r.len;
    const size = eq_evals.len;

    // eq(r, x) = Π_i (r_i * x_i + (1-r_i) * (1-x_i))
    //          = Π_i ((1-r_i) + (2*r_i - 1) * x_i)

    // Start with all 1s
    for (eq_evals) |*e| {
        e.* = F.one();
    }

    // For each variable i (from 0 to n-1, matching LowToHigh binding)
    // Variable i corresponds to bit i of the index
    for (0..n) |i| {
        const r_i = r[i];
        const one_minus_r = F.one().sub(r_i);
        // Stride for variable i: indices differ by 2^i in bit i
        const stride = @as(usize, 1) << @intCast(i);

        var k: usize = 0;
        while (k < size) : (k += 2 * stride) {
            for (0..stride) |j| {
                const idx0 = k + j;
                const idx1 = k + j + stride;
                if (idx1 < size) {
                    // idx0 has bit i = 0, idx1 has bit i = 1
                    const val = eq_evals[idx0];
                    eq_evals[idx0] = val.mul(one_minus_r);
                    eq_evals[idx1] = val.mul(r_i);
                }
            }
        }
    }
}

// Tests
const testing = std.testing;

test "output_sumcheck: basic init" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = testing.allocator;

    var initial_ram = std.AutoHashMapUnmanaged(u64, u64){};
    defer initial_ram.deinit(allocator);

    var final_ram = std.AutoHashMapUnmanaged(u64, u64){};
    defer final_ram.deinit(allocator);

    // Set up a simple memory layout
    var memory_layout = jolt_device.MemoryLayout{
        .max_input_size = 1024,
        .max_output_size = 1024,
        .max_trusted_advice_size = 1024,
        .max_untrusted_advice_size = 1024,
        .input_start = 0x7fff8000,
        .output_start = 0x7fff9000,
        .panic = 0x7fffb000,
        .termination = 0x7fffc008,
        .trusted_advice_start = 0x7fff8000,
        .untrusted_advice_start = 0x7fff9000,
    };

    // Use small log_K for testing
    const log_K: usize = 4;
    const r_address = try allocator.alloc(F, log_K);
    defer allocator.free(r_address);
    for (r_address) |*r| {
        r.* = F.fromU64(12345);
    }

    var prover = try OutputSumcheckProver(F).init(
        allocator,
        &initial_ram,
        &final_ram,
        r_address,
        &memory_layout,
    );
    defer prover.deinit();

    // For empty RAM, the polynomial should be all zeros
    const compressed = prover.computeRoundPolynomial();
    try testing.expectEqual(F.zero(), compressed[0]);
}
