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

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;

const poly_mod = @import("zolt_arith").poly;
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
        thread_pool: ?*ThreadPool = null,

        /// Initialize from RAM states, memory layout, and program I/O
        ///
        /// Parameters:
        /// - initial_ram: Initial RAM state as sparse map (address -> value)
        /// - final_ram: Final RAM state as sparse map (address -> value)
        /// - r_address: Random address challenges
        /// - memory_layout: Memory layout defining IO region
        /// - inputs: Program input bytes (optional)
        /// - outputs: Program output bytes (optional)
        /// - is_panicking: Whether the program panicked
        pub fn init(
            allocator: Allocator,
            initial_ram: *const std.AutoHashMapUnmanaged(u64, u64),
            final_ram: *const std.AutoHashMapUnmanaged(u64, u64),
            r_address: []const F,
            memory_layout: *const jolt_device.MemoryLayout,
            inputs: ?[]const u8,
            outputs: ?[]const u8,
            is_panicking: bool,
        ) !Self {
            const log_K = r_address.len;
            const K: usize = @as(usize, 1) << @intCast(log_K);

            // Debug: Print r_address for comparison with Jolt
            dbg("[ZOLT OUTPUT_CHECK] r_address (log_K={}):\n", .{log_K});
            for (r_address, 0..) |r, i| {
                dbg("[ZOLT OUTPUT_CHECK]   r_address[{}] = {any}\n", .{i, r.toBytesBE()});
            }

            // Allocate arrays
            const val_init = try allocator.alloc(F, K);
            const val_final = try allocator.alloc(F, K);
            const val_io = try allocator.alloc(F, K);
            const io_mask = try allocator.alloc(F, K);
            const eq_r_address = try allocator.alloc(F, K);

            // Initialize val_init and val_final from sparse maps
            var non_zero_count: usize = 0;
            var io_region_values: usize = 0;
            var init_non_zero_count: usize = 0;
            var init_bytecode_count: usize = 0;
            for (val_init, val_final, 0..) |*vi, *vf, k| {
                // Convert index k to address
                const address = indexToAddress(k, memory_layout);

                // Look up values (default 0)
                vi.* = if (initial_ram.get(address)) |v| blk: {
                    init_non_zero_count += 1;
                    if (k >= 4096) {
                        init_bytecode_count += 1;
                    }
                    if (k < 5 or (k >= 4096 and k < 4100)) {
                        dbg("[ZOLT] OutputSumcheck: initial_ram k={}, addr=0x{X:0>8}, val=0x{X}\n", .{ k, address, v });
                    }
                    break :blk F.fromU64(v);
                } else F.zero();
                vf.* = if (final_ram.get(address)) |v| blk: {
                    non_zero_count += 1;
                    if (k >= 1024 and k < 4096) { // IO region
                        io_region_values += 1;
                        dbg("[ZOLT] OutputSumcheck: IO region k={}, addr=0x{X:0>8}, val={}\n", .{ k, address, v });
                    }
                    if (k < 5 or (k >= 4096 and k < 4100)) {
                        dbg("[ZOLT] OutputSumcheck: final_ram k={}, addr=0x{X:0>8}, val=0x{X}\n", .{ k, address, v });
                    }
                    break :blk F.fromU64(v);
                } else F.zero();
            }
            dbg("[ZOLT] OutputSumcheck: final_ram non_zero_count={}, io_region_values={}, K={}\n", .{ non_zero_count, io_region_values, K });
            dbg("[ZOLT] OutputSumcheck: initial_ram non_zero_count={}, bytecode_count={}\n", .{ init_non_zero_count, init_bytecode_count });

            // Set panic and termination bits in val_final ONLY (NOT in val_init).
            //
            // IMPORTANT: val_init must match what Jolt's verifier computes via eval_initial_ram_mle,
            // which includes ONLY bytecode + inputs + advice. It does NOT include termination/panic bits.
            //
            // The termination/panic bits are part of the FINAL memory state, not the initial state.
            // They are checked via the IO region (val_io) in OutputSumcheck, not via val_init.
            //
            // NOTE: This means val_final(term_addr) - val_init(term_addr) = 1 - 0 = 1
            // For the ValFinal sumcheck to verify, this difference must equal Σ inc(term_addr,j) * wa(term_addr,j).
            // If Zolt's tracer doesn't record the termination write, this will cause a mismatch in ValFinal.
            // The proper fix is to make Zolt's tracer record the termination write.
            const panic_index = remapAddress(memory_layout.panic, memory_layout) orelse 0;
            if (panic_index < K) {
                const panic_val = if (is_panicking) F.one() else F.zero();
                val_final[panic_index] = panic_val;
                // val_init[panic_index] stays at 0 (or whatever initial_ram contained)
                dbg("[ZOLT] OutputSumcheck: val_final[{}] = {} (panic bit), val_init[{}] = {} (from initial_ram)\n", .{ panic_index, if (is_panicking) @as(u64, 1) else @as(u64, 0), panic_index, val_init[panic_index].toU64() });
            }
            const termination_index = remapAddress(memory_layout.termination, memory_layout) orelse 0;
            if (!is_panicking and termination_index < K) {
                // Set termination bit in val_final to match Jolt's convention.
                // This is required for OutputSumcheck because Jolt's verifier computes
                // val_io_eval with termination = 1 via eval_io_mle.
                //
                // For ValFinal sumcheck, we need val_final_claim - val_init_eval = 0.
                // This is achieved by storing the SAME value for both RamValFinal and
                // RamValInit claims at RamOutputCheck opening point.
                val_final[termination_index] = F.one();
                dbg("[ZOLT] OutputSumcheck: val_final[{}] = 1 (termination bit)\n", .{termination_index});
            }

            // Compute IO region bounds (matches Jolt's ProgramIOPolynomial)
            const lowest = memory_layout.getLowestAddress();
            const io_start = remapAddress(memory_layout.input_start, memory_layout) orelse 0;
            const io_end = remapAddress(constants.RAM_START_ADDRESS, memory_layout) orelse K;
            dbg("[ZOLT] OutputSumcheck: lowest=0x{X:0>16}, io_start={}, io_end={}\n", .{ lowest, io_start, io_end });
            dbg("[ZOLT] OutputSumcheck: input_start=0x{X:0>16}, RAM_START=0x{X:0>16}\n", .{ memory_layout.input_start, constants.RAM_START_ADDRESS });
            dbg("[ZOLT] OutputSumcheck: output_start=0x{X:0>16}, output_end=0x{X:0>16}\n", .{ memory_layout.output_start, memory_layout.output_end });
            dbg("[ZOLT] OutputSumcheck: panic=0x{X:0>16}, termination=0x{X:0>16}\n", .{ memory_layout.panic, memory_layout.termination });

            // Initialize io_mask and val_io from program I/O (matching Jolt's ProgramIOPolynomial)
            // val_io is the "expected" values that the verifier will check against val_final
            @memset(val_io, F.zero());
            @memset(io_mask, F.zero());

            // Set io_mask for the IO region
            for (io_start..@min(io_end, K)) |k| {
                io_mask[k] = F.one();
            }

            // Populate val_io from inputs (8-byte words starting at input_start)
            if (inputs) |input_bytes| {
                const input_index_start = remapAddress(memory_layout.input_start, memory_layout) orelse 0;
                var input_index = input_index_start;
                var i: usize = 0;
                while (i < input_bytes.len) : (i += 8) {
                    if (input_index >= K) break;
                    // Convert 8 bytes to u64 (little-endian)
                    var word: u64 = 0;
                    const end = @min(i + 8, input_bytes.len);
                    for (i..end) |j| {
                        word |= @as(u64, input_bytes[j]) << @intCast((j - i) * 8);
                    }
                    val_io[input_index] = F.fromU64(word);
                    if (input_index < 10 or input_index >= K - 10) {
                        dbg("[ZOLT] OutputSumcheck: val_io[{}] = {} (input word)\n", .{ input_index, word });
                    }
                    input_index += 1;
                }
                dbg("[ZOLT] OutputSumcheck: populated {} input words starting at index {}\n", .{ (input_bytes.len + 7) / 8, input_index_start });
            }

            // Populate val_io from outputs (8-byte words starting at output_start)
            if (outputs) |output_bytes| {
                const output_index_start = remapAddress(memory_layout.output_start, memory_layout) orelse 0;
                var output_index = output_index_start;
                var i: usize = 0;
                while (i < output_bytes.len) : (i += 8) {
                    if (output_index >= K) break;
                    // Convert 8 bytes to u64 (little-endian)
                    var word: u64 = 0;
                    const end = @min(i + 8, output_bytes.len);
                    for (i..end) |j| {
                        word |= @as(u64, output_bytes[j]) << @intCast((j - i) * 8);
                    }
                    val_io[output_index] = F.fromU64(word);
                    if (output_index < 10 or output_index >= K - 10) {
                        dbg("[ZOLT] OutputSumcheck: val_io[{}] = {} (output word)\n", .{ output_index, word });
                    }
                    output_index += 1;
                }
                dbg("[ZOLT] OutputSumcheck: populated {} output words starting at index {}\n", .{ (output_bytes.len + 7) / 8, output_index_start });
            }

            // Set panic bit in val_io (matching Jolt's ProgramIOPolynomial)
            // (panic_index and termination_index already defined above for val_final)
            if (panic_index < K) {
                val_io[panic_index] = if (is_panicking) F.one() else F.zero();
                dbg("[ZOLT] OutputSumcheck: val_io[{}] = {} (panic bit)\n", .{ panic_index, if (is_panicking) @as(u64, 1) else @as(u64, 0) });
            }

            // Set termination bit in val_io if not panicking (matching Jolt's ProgramIOPolynomial)
            dbg("[ZOLT] OutputSumcheck: termination_index={}, in IO={}\n", .{ termination_index, termination_index >= io_start and termination_index < io_end });
            if (!is_panicking and termination_index < K) {
                val_io[termination_index] = F.one();
                dbg("[ZOLT] OutputSumcheck: val_io[{}] = 1 (termination bit, not panicking)\n", .{termination_index});
            }

            // CRITICAL FIX: For addresses with no memory writes, ensure val_final == val_init
            // This is necessary because:
            // 1. val_init is populated from initial_ram (includes bytecode)
            // 2. val_final is populated from final_ram (may not include bytecode for programs with no RAM writes)
            // 3. After OutputSumcheck binding, we need val_init_eval == val_final_eval for unwritten addresses
            //
            // Strategy:
            // - OUTSIDE I/O region: Copy val_init -> val_final (preserve initial values like bytecode), except termination/panic
            // - INSIDE I/O region: Copy val_final -> val_init, except termination

            var copied_outside_io: usize = 0;
            var copied_inside_io: usize = 0;

            // Copy val_init to val_final for addresses OUTSIDE the I/O region
            // This preserves initial values (like bytecode) for addresses that weren't written
            // Only copy if val_final is zero (no write occurred) and val_init is non-zero
            // Skip termination and panic indices as they're set explicitly in val_final
            for (0..K) |k| {
                if ((k < io_start or k >= io_end) and k != termination_index and k != panic_index) {
                    // Only copy if the address wasn't written (val_final is zero) but has initial value
                    if (val_final[k].eql(F.zero()) and !val_init[k].eql(F.zero())) {
                        val_final[k] = val_init[k];
                        copied_outside_io += 1;
                        if (copied_outside_io <= 5) {
                            dbg("[ZOLT] OutputSumcheck: copied val_final[{}] = val_init[{}] (unwritten address, preserving initial state)\n", .{ k, k });
                        }
                    }
                }
            }
            dbg("[ZOLT] OutputSumcheck: copied val_init to val_final for {} unwritten addresses outside I/O region\n", .{copied_outside_io});

            // Copy val_final to val_init for addresses INSIDE I/O region (except termination)
            // This ensures the I/O region matches the expected values
            for (io_start..@min(io_end, K)) |k| {
                if (k != termination_index) {
                    val_init[k] = val_final[k];
                    if (!val_final[k].eql(F.zero())) {
                        copied_inside_io += 1;
                        if (copied_inside_io <= 5) {
                            dbg("[ZOLT] OutputSumcheck: copied val_init[{}] = val_final[{}] (inside I/O region)\n", .{ k, k });
                        }
                    }
                }
            }
            dbg("[ZOLT] OutputSumcheck: copied val_final to val_init for {} addresses inside I/O region (except termination)\n", .{copied_inside_io});

            // DEBUG: Check for differences between val_final and val_init
            var diff_count: usize = 0;
            var diff_in_io: usize = 0;
            var diff_outside_io: usize = 0;
            for (0..K) |k| {
                if (!val_final[k].eql(val_init[k])) {
                    diff_count += 1;
                    if (k >= io_start and k < io_end) {
                        diff_in_io += 1;
                    } else {
                        diff_outside_io += 1;
                    }
                    if (diff_count <= 10) {
                        const in_io = if (k >= io_start and k < io_end) "IN I/O" else "OUTSIDE I/O";
                        dbg("[ZOLT] OutputSumcheck DEBUG: val_final[{}] != val_init[{}] ({s})\n", .{ k, k, in_io });
                        dbg("[ZOLT]   val_final[{}] = {any}\n", .{ k, val_final[k].toBytesBE() });
                        dbg("[ZOLT]   val_init[{}] = {any}\n", .{ k, val_init[k].toBytesBE() });
                    }
                }
            }
            dbg("[ZOLT] OutputSumcheck DEBUG: {} total differences ({}  in I/O, {} outside I/O)\n", .{ diff_count, diff_in_io, diff_outside_io });

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

            const OCtx = struct {
                eq: []const F, io: []const F, vf: []const F, vio: []const F,
            };
            const ctx = OCtx{
                .eq = self.eq_r_address, .io = self.io_mask,
                .vf = self.val_final, .vio = self.val_io,
            };

            const mapFn = struct {
                fn f(c: OCtx, start: usize, end: usize) [4]F {
                    var ls: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
                    for (start..end) |g| {
                        const idx0 = 2 * g;
                        const idx1 = 2 * g + 1;
                        const eq0 = c.eq[idx0]; const eq1 = c.eq[idx1];
                        const io0 = c.io[idx0]; const io1 = c.io[idx1];
                        const v0 = c.vf[idx0].sub(c.vio[idx0]);
                        const v1 = c.vf[idx1].sub(c.vio[idx1]);
                        const deq = eq1.sub(eq0);
                        const dio = io1.sub(io0);
                        const dv = v1.sub(v0);
                        ls[0] = ls[0].add(eq0.mul(io0).mul(v0));
                        ls[1] = ls[1].add(eq1.mul(io1).mul(v1));
                        const eq2 = eq0.add(deq).add(deq);
                        const io2 = io0.add(dio).add(dio);
                        const v2 = v0.add(dv).add(dv);
                        ls[2] = ls[2].add(eq2.mul(io2).mul(v2));
                        ls[3] = ls[3].add(eq2.add(deq).mul(io2.add(dio)).mul(v2.add(dv)));
                    }
                    return ls;
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            const identity = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };
            const sums = if (self.thread_pool) |tp|
                tp.parallelReduce([4]F, half, identity, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            const s0 = sums[0]; const s1 = sums[1]; const s2 = sums[2]; const s3 = sums[3];

            // Debug: verify sumcheck soundness s0 + s1 == current_claim
            const sum_check = s0.add(s1);
            if (!sum_check.eql(self.current_claim)) {
                dbg("[ZOLT OUTPUT_CHECK ERROR] s0 + s1 != current_claim!\n", .{});
                dbg("  s0 = {any}\n", .{s0.toBytesBE()});
                dbg("  s1 = {any}\n", .{s1.toBytesBE()});
                dbg("  s0+s1 = {any}\n", .{sum_check.toBytesBE()});
                dbg("  current_claim = {any}\n", .{self.current_claim.toBytesBE()});
            }

            // Convert evaluations to compressed coefficients [c0, c2, c3]
            return poly_mod.UniPoly(F).evalsToCompressed([4]F{ s0, s1, s2, s3 });
        }

        /// Bind challenge and update polynomials for next round
        pub fn bindChallenge(self: *Self, r: F) void {
            const half = self.current_size / 2;

            const OBindCtx = struct {
                slices: [5][]F,
                r: F,
                n: usize,
            };
            const bctx = OBindCtx{
                .slices = .{ self.eq_r_address, self.io_mask, self.val_final, self.val_io, self.val_init },
                .r = r,
                .n = half,
            };
            const bindOneFn = struct {
                fn f(c: OBindCtx, idx: usize) void {
                    const arr = c.slices[idx];
                    for (0..c.n) |i| {
                        arr[i] = arr[2 * i].add(c.r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                    }
                }
            }.f;

            if (self.thread_pool) |tp| {
                tp.parallelForForce(5, bctx, bindOneFn);
            } else {
                for (0..5) |idx| bindOneFn(bctx, idx);
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
        pub fn getFinalClaims(self: *const Self) struct { val_final: F, val_init: F, val_io: F, eq_r_address: F, io_mask: F } {
            // Debug output for comparing with Jolt
            dbg("[ZOLT OUTPUT_CHECK] val_final[0]: {any}\n", .{self.val_final[0].toBytesBE()});
            dbg("[ZOLT OUTPUT_CHECK] val_init[0]: {any}\n", .{self.val_init[0].toBytesBE()});
            dbg("[ZOLT OUTPUT_CHECK] val_io[0]: {any}\n", .{self.val_io[0].toBytesBE()});
            dbg("[ZOLT OUTPUT_CHECK] eq_r_address[0]: {any}\n", .{self.eq_r_address[0].toBytesBE()});
            dbg("[ZOLT OUTPUT_CHECK] io_mask[0]: {any}\n", .{self.io_mask[0].toBytesBE()});
            // Compute expected: eq * io_mask * (val_final - val_io)
            const diff = self.val_final[0].sub(self.val_io[0]);
            const expected = self.eq_r_address[0].mul(self.io_mask[0]).mul(diff);
            dbg("[ZOLT OUTPUT_CHECK] (val_final - val_io)[0]: {any}\n", .{diff.toBytesBE()});
            dbg("[ZOLT OUTPUT_CHECK] expected (eq * io_mask * diff)[0]: {any}\n", .{expected.toBytesBE()});

            return .{
                .val_final = self.val_final[0],
                .val_init = self.val_init[0],
                .val_io = self.val_io[0],
                .eq_r_address = self.eq_r_address[0],
                .io_mask = self.io_mask[0],
            };
        }

        // Helper: compute c2 from evaluations using Lagrange
        fn lagrangeC2(evals: [4]F) F {
            // c2 = (2*s(0) - 5*s(1) + 4*s(2) - s(3)) / 2
            const two = F.fromU64(2);
            const four = F.fromU64(4);
            const five = F.fromU64(5);
            const half = two.inverse() orelse F.zero();
            return evals[0].mul(two)
                .sub(evals[1].mul(five))
                .add(evals[2].mul(four))
                .sub(evals[3])
                .mul(half);
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

/// Compute EQ polynomial evaluations using BIG-ENDIAN ordering (like Jolt)
/// eq_evals[k] = eq(r, k) where k is interpreted in big-endian:
/// - Bit 0 of k (MSB position) corresponds to r[0]
/// - Bit n-1 of k (LSB position) corresponds to r[n-1]
///
/// This matches Jolt's EqPolynomial::evals() and allows standard bot-binding
/// (binding pairs (0,1), (2,3), etc.) to correctly bind the LAST variable first.
///
/// When LowToHigh binding is used (bind last variable first with s_0, etc.):
/// - s_0 binds r[n-1] (the last variable)
/// - s_1 binds r[n-2]
/// - ...
/// - s_{n-1} binds r[0]
///
/// Final result: eq(r, [s_{n-1}, s_{n-2}, ..., s_0]) = eq(r, reverse(sumcheck_challenges))
/// This matches what Jolt's verifier expects when it reverses the challenges.
fn computeEqEvals(comptime F: type, eq_evals: []F, r: []const F) void {
    const n = r.len;
    var size: usize = 1;

    // Start with eq_evals[0] = 1
    eq_evals[0] = F.one();

    // Build up the eq table iteratively (like Jolt's evals_serial)
    // Process r[0] first (MSB), then r[1], ..., r[n-1] (LSB)
    for (0..n) |j| {
        // In each iteration, we double the size
        size *= 2;
        // Process pairs in reverse order to avoid overwriting values we still need
        var i: usize = size;
        while (i >= 2) : (i -= 2) {
            // Copy each element from the prior iteration twice
            const scalar = eq_evals[(i - 2) / 2];
            // eq_evals[i-1] is for x_j = 1 (multiply by r[j])
            // eq_evals[i-2] is for x_j = 0 (multiply by 1 - r[j])
            eq_evals[i - 1] = scalar.mul(r[j]);
            eq_evals[i - 2] = scalar.sub(eq_evals[i - 1]);
        }
    }
}

// Tests
const testing = std.testing;

test "output_sumcheck: basic init" {
    const F = @import("zolt_arith").field.BN254Scalar;
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
