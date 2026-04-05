//! RamRaClaimReduction Prover (Instance 1 of Stage 5 Batched Sumcheck)
//!
//! Extracted from stage5_prover.zig to encapsulate the RamRaClaimReduction
//! sumcheck logic. This instance proves:
//!   Sum_{k,c} eq_combined(k,c) * ra(k,c) = input_claim
//! where ra(k,c) = 1 iff there's a RAM access at (address=k, cycle=c).
//!
//! The sumcheck has 2 phases (cycle-only binding, no PhaseAddress):
//!   - PhaseCycle1: first half of cycle rounds using prefix-suffix P*Q decomposition
//!   - PhaseCycle2: second half of cycle rounds using H' * eq_hi dense sumcheck

const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;

const poly_mod = @import("zolt_arith").poly;
const UniPoly = poly_mod.UniPoly;

const stage5_inst = @import("stage5_instances.zig");
const ram = @import("../ram/mod.zig");
const jolt_device = @import("../jolt_device.zig");

// Re-export computeEqAtPoint from stage5_prover (shared helper)
const computeEqAtPoint = stage5_inst.computeEqAtPoint;

/// RamRaClaimReduction prover for Instance 1 of Stage 5.
///
/// Encapsulates all state and methods for the cycle-only RamRaClaimReduction
/// sumcheck, using prefix-suffix (P*Q) decomposition for the first half of
/// cycle rounds and H'*eq_hi for the second half.
pub fn RamRaClaimReductionProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const Inst = stage5_inst.Helpers(F);

        // ---- Allocator / thread pool / GPU ----
        allocator: Allocator,
        thread_pool: ?*ThreadPool,
        gpu_ops: ?*GpuPolyOps,

        // ---- Input claim components ----
        gamma: F,
        gamma2: F,
        ram_ra_input: F,
        current_claim: F,

        // ---- Sparse access arrays ----
        ram_access_count: usize,
        ram_addresses: []u64,
        ram_cycles: []u64,
        ram_G_A: []F,
        ram_G_B: []F,
        eq_raf_access: []F,
        eq_rw_access: []F,
        eq_val_access: []F,

        // ---- Full-size eq polynomial tables ----
        G_A_full: []F,
        G_B_full: []F,
        B_1: []F,
        B_2: []F,

        // ---- Per-access eq cycle tracking ----
        eq_cycle_bound: []F,
        eq_raf_bound: F,
        eq_rw_bound: F,
        eq_val_bound: F,
        eq_raf_remaining: []F,
        eq_rw_remaining: []F,
        eq_val_remaining: []F,

        // ---- P*Q prefix-suffix decomposition ----
        prefix_n_vars: usize,
        suffix_n_vars: usize,
        prefix_size: usize,
        suffix_size: usize,
        P_raf: []F,
        P_rw: []F,
        P_val: []F,
        Q_raf: []F,
        Q_rw: []F,
        Q_val: []F,
        eq_raf_hi: []F,
        eq_rw_hi: []F,
        eq_val_hi: []F,

        // ---- PhaseCycle2 state ----
        H_prime: []F,
        cycle_challenges: []F,
        phase_cycle_q_initialized: bool,
        phase_cycle2_initialized: bool,
        scale_raf: F,
        scale_rw: F,
        scale_val: F,

        // ---- Cached evaluations for claim update ----
        inst1_eval_0: F,
        inst1_eval_1: F,
        inst1_eval_2: F,

        // ---- Configuration ----
        n_cycle_vars: usize,
        log_ram_k: usize,
        K: usize,

        // ---- Opening point slices (borrowed, not owned) ----
        r_cycle_raf: []const F,
        r_cycle_rw: []const F,
        r_cycle_val: []const F,

        /// Initialize the RamRaClaimReduction prover from trace data and opening points.
        pub fn init(
            allocator: Allocator,
            thread_pool: ?*ThreadPool,
            gpu_ops: ?*GpuPolyOps,
            // Gamma and input claims
            gamma: F,
            claim_raf: F,
            claim_rw: F,
            claim_val: F,
            // Configuration
            n_cycle_vars: usize,
            log_ram_k: usize,
            // Trace data
            memory_trace: ?*const ram.MemoryTrace,
            memory_layout: ?*const jolt_device.MemoryLayout,
            // Opening points (BIG_ENDIAN)
            r_address_raf: []const F,
            r_address_rw: []const F,
            r_cycle_raf: []const F,
            r_cycle_rw: []const F,
            r_cycle_val: []const F,
        ) !Self {
            const Inst_local = stage5_inst.Helpers(F);

            // Compute input claim: claim_raf + gamma*claim_rw + gamma^2*claim_val
            const gamma2 = gamma.mul(gamma);
            const ram_ra_input = claim_raf
                .add(gamma.mul(claim_rw))
                .add(gamma2.mul(claim_val));

            if (comptime debug_verbose) {
                dbg("[STAGE5 RAM_RA] RamRaClaimReduction input components:\n", .{});
                dbg("  claim_raf (RamRafEvaluation) = {any}\n", .{claim_raf.toBytesBE()[16..32].*});
                dbg("  claim_rw (RamReadWriteChecking) = {any}\n", .{claim_rw.toBytesBE()[16..32].*});
                dbg("  claim_val (RamValCheck) = {any}\n", .{claim_val.toBytesBE()[16..32].*});
                dbg("  gamma = {any}\n", .{gamma.toBytesBE()[16..32].*});
            }

            const K = @as(usize, 1) << @intCast(log_ram_k);
            const ram_access_count = if (memory_trace) |mt| mt.accesses.items.len else 0;

            if (comptime debug_verbose) {
                dbg("[STAGE5 RAM_RA] Initializing with {} RAM accesses\n", .{ram_access_count});
            }

            // Allocate sparse access arrays
            var ram_addresses = try allocator.alloc(u64, ram_access_count);
            errdefer allocator.free(ram_addresses);
            var ram_cycles = try allocator.alloc(u64, ram_access_count);
            errdefer allocator.free(ram_cycles);
            var ram_G_A = try allocator.alloc(F, ram_access_count);
            errdefer allocator.free(ram_G_A);
            var ram_G_B = try allocator.alloc(F, ram_access_count);
            errdefer allocator.free(ram_G_B);
            var eq_raf_access = try allocator.alloc(F, ram_access_count);
            errdefer allocator.free(eq_raf_access);
            var eq_rw_access = try allocator.alloc(F, ram_access_count);
            errdefer allocator.free(eq_rw_access);
            var eq_val_access = try allocator.alloc(F, ram_access_count);
            errdefer allocator.free(eq_val_access);

            // Precompute per-access eq(r_cycle_*, cycle) values
            if (memory_trace) |mt| {
                for (mt.accesses.items, 0..) |access, i| {
                    const cycle = access.timestamp;
                    eq_raf_access[i] = computeEqAtPoint(F, r_cycle_raf, cycle);
                    eq_rw_access[i] = computeEqAtPoint(F, r_cycle_rw, cycle);
                    eq_val_access[i] = computeEqAtPoint(F, r_cycle_val, cycle);
                }
            }

            // Precompute G_A and G_B, remap addresses
            if (memory_trace) |mt| {
                for (mt.accesses.items, 0..) |access, i| {
                    const remapped_addr: u64 = if (memory_layout) |ml|
                        ml.remapAddress(access.address) orelse 0
                    else
                        access.address & (@as(u64, K) - 1);

                    ram_addresses[i] = remapped_addr;
                    ram_cycles[i] = access.timestamp;

                    const eq_raf_c = eq_raf_access[i];
                    const eq_rw_c = eq_rw_access[i];
                    const eq_val_c = eq_val_access[i];

                    // G_A = eq_raf + gamma * eq_val
                    // G_B = eq_rw + gamma * eq_val
                    ram_G_A[i] = eq_raf_c.add(gamma.mul(eq_val_c));
                    ram_G_B[i] = eq_rw_c.add(gamma.mul(eq_val_c));

                    if (comptime debug_verbose) {
                        dbg("[STAGE5 RAM_RA] Access {}: raw_addr=0x{x}, remapped_addr={}, cycle={}\n", .{ i, access.address, remapped_addr, access.timestamp });
                        dbg("  eq_raf_c={any}, eq_rw_c={any}, eq_val_c={any}\n", .{
                            eq_raf_c.toBytesBE()[16..32].*,
                            eq_rw_c.toBytesBE()[16..32].*,
                            eq_val_c.toBytesBE()[16..32].*,
                        });
                        dbg("  G_A={any}, G_B={any}\n", .{
                            ram_G_A[i].toBytesBE()[16..32].*,
                            ram_G_B[i].toBytesBE()[16..32].*,
                        });
                    }
                }
            }

            // Full-size G_A and G_B arrays
            var G_A_full = try allocator.alloc(F, K);
            errdefer allocator.free(G_A_full);
            var G_B_full = try allocator.alloc(F, K);
            errdefer allocator.free(G_B_full);
            @memset(G_A_full, F.zero());
            @memset(G_B_full, F.zero());
            for (0..ram_access_count) |i| {
                const addr_usize: usize = @intCast(ram_addresses[i]);
                G_A_full[addr_usize] = ram_G_A[i];
                G_B_full[addr_usize] = ram_G_B[i];
            }
            if (comptime debug_verbose) {
                dbg("[STAGE5 RAM_RA] Created full-size G_A/G_B arrays (K={}), {} non-zero entries\n", .{ K, ram_access_count });
            }

            // Initialize B_1 and B_2 eq polynomials for address evaluation
            var B_1 = try allocator.alloc(F, K);
            errdefer allocator.free(B_1);
            var B_2 = try allocator.alloc(F, K);
            errdefer allocator.free(B_2);
            Inst_local.buildFullEqTable(r_address_raf, B_1[0..K], thread_pool);
            Inst_local.buildFullEqTable(r_address_rw, B_2[0..K], thread_pool);

            // Debug: print B_1 and B_2 for first few addresses
            if (comptime debug_verbose) {
                dbg("[STAGE5 RAM_RA] B_1/B_2 eq polynomials (first 4 and last 4 of {}):\n", .{K});
            }
            for (0..@min(4, K)) |k| {
                if (comptime debug_verbose) {
                    dbg("  B_1[{}]={any}, B_2[{}]={any}\n", .{
                        k, B_1[k].toBytesBE()[16..32].*, k, B_2[k].toBytesBE()[16..32].*,
                    });
                }
            }
            if (K > 8) {
                for (K - 4..K) |k| {
                    if (comptime debug_verbose) {
                        dbg("  B_1[{}]={any}, B_2[{}]={any}\n", .{
                            k, B_1[k].toBytesBE()[16..32].*, k, B_2[k].toBytesBE()[16..32].*,
                        });
                    }
                }
            }

            // Per-access eq_cycle_bound
            var eq_cycle_bound = try allocator.alloc(F, ram_access_count);
            errdefer allocator.free(eq_cycle_bound);
            for (0..ram_access_count) |i| {
                eq_cycle_bound[i] = F.one();
            }

            // Remaining eq factors (per-access)
            var eq_raf_remaining = try allocator.alloc(F, ram_access_count);
            errdefer allocator.free(eq_raf_remaining);
            var eq_rw_remaining = try allocator.alloc(F, ram_access_count);
            errdefer allocator.free(eq_rw_remaining);
            var eq_val_remaining = try allocator.alloc(F, ram_access_count);
            errdefer allocator.free(eq_val_remaining);
            for (0..ram_access_count) |i| {
                eq_raf_remaining[i] = eq_raf_access[i];
                eq_rw_remaining[i] = eq_rw_access[i];
                eq_val_remaining[i] = eq_val_access[i];
            }

            // P*Q prefix-suffix decomposition
            const prefix_n_vars = n_cycle_vars / 2;
            const suffix_n_vars = n_cycle_vars - prefix_n_vars;
            const prefix_size = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_size = @as(usize, 1) << @intCast(suffix_n_vars);

            if (comptime debug_verbose) {
                dbg("[STAGE5 PQ] PhaseCycle P*Q setup: n_cycle={}, prefix={}, suffix={}\n", .{
                    n_cycle_vars, prefix_n_vars, suffix_n_vars,
                });
                dbg("[STAGE5 PQ] prefix_size={}, suffix_size={}\n", .{ prefix_size, suffix_size });
            }

            // P arrays: eq evaluations for prefix (low) bits
            var P_raf = try allocator.alloc(F, prefix_size);
            errdefer allocator.free(P_raf);
            var P_rw = try allocator.alloc(F, prefix_size);
            errdefer allocator.free(P_rw);
            var P_val = try allocator.alloc(F, prefix_size);
            errdefer allocator.free(P_val);

            // Q arrays: suffix-weighted sums
            const Q_raf = try allocator.alloc(F, prefix_size);
            errdefer allocator.free(Q_raf);
            const Q_rw = try allocator.alloc(F, prefix_size);
            errdefer allocator.free(Q_rw);
            const Q_val = try allocator.alloc(F, prefix_size);
            errdefer allocator.free(Q_val);

            // Precompute eq evaluations for suffix (high) bits
            var eq_raf_hi = try allocator.alloc(F, suffix_size);
            errdefer allocator.free(eq_raf_hi);
            var eq_rw_hi = try allocator.alloc(F, suffix_size);
            errdefer allocator.free(eq_rw_hi);
            var eq_val_hi = try allocator.alloc(F, suffix_size);
            errdefer allocator.free(eq_val_hi);

            for (0..suffix_size) |c_hi| {
                eq_raf_hi[c_hi] = computeEqAtPoint(F, r_cycle_raf[0..suffix_n_vars], c_hi);
                eq_rw_hi[c_hi] = computeEqAtPoint(F, r_cycle_rw[0..suffix_n_vars], c_hi);
                eq_val_hi[c_hi] = computeEqAtPoint(F, r_cycle_val[0..suffix_n_vars], c_hi);
            }

            // Initialize P arrays from prefix bits
            for (0..prefix_size) |c_lo| {
                P_raf[c_lo] = computeEqAtPoint(F, r_cycle_raf[suffix_n_vars..], c_lo);
                P_rw[c_lo] = computeEqAtPoint(F, r_cycle_rw[suffix_n_vars..], c_lo);
                P_val[c_lo] = computeEqAtPoint(F, r_cycle_val[suffix_n_vars..], c_lo);
            }

            // Q arrays initialized lazily at first PhaseCycle round
            @memset(Q_raf, F.zero());
            @memset(Q_rw, F.zero());
            @memset(Q_val, F.zero());

            // H_prime for PhaseCycle2
            const H_prime = try allocator.alloc(F, suffix_size);
            errdefer allocator.free(H_prime);
            @memset(H_prime, F.zero());

            // Cycle challenges tracking
            const cycle_challenges = try allocator.alloc(F, n_cycle_vars);
            errdefer allocator.free(cycle_challenges);
            @memset(cycle_challenges, F.zero());

            return Self{
                .allocator = allocator,
                .thread_pool = thread_pool,
                .gpu_ops = gpu_ops,
                .gamma = gamma,
                .gamma2 = gamma2,
                .ram_ra_input = ram_ra_input,
                .current_claim = undefined, // Set by setScaledClaim after scaling
                .ram_access_count = ram_access_count,
                .ram_addresses = ram_addresses,
                .ram_cycles = ram_cycles,
                .ram_G_A = ram_G_A,
                .ram_G_B = ram_G_B,
                .eq_raf_access = eq_raf_access,
                .eq_rw_access = eq_rw_access,
                .eq_val_access = eq_val_access,
                .G_A_full = G_A_full,
                .G_B_full = G_B_full,
                .B_1 = B_1,
                .B_2 = B_2,
                .eq_cycle_bound = eq_cycle_bound,
                .eq_raf_bound = F.one(),
                .eq_rw_bound = F.one(),
                .eq_val_bound = F.one(),
                .eq_raf_remaining = eq_raf_remaining,
                .eq_rw_remaining = eq_rw_remaining,
                .eq_val_remaining = eq_val_remaining,
                .prefix_n_vars = prefix_n_vars,
                .suffix_n_vars = suffix_n_vars,
                .prefix_size = prefix_size,
                .suffix_size = suffix_size,
                .P_raf = P_raf,
                .P_rw = P_rw,
                .P_val = P_val,
                .Q_raf = Q_raf,
                .Q_rw = Q_rw,
                .Q_val = Q_val,
                .eq_raf_hi = eq_raf_hi,
                .eq_rw_hi = eq_rw_hi,
                .eq_val_hi = eq_val_hi,
                .H_prime = H_prime,
                .cycle_challenges = cycle_challenges,
                .phase_cycle_q_initialized = false,
                .phase_cycle2_initialized = false,
                .scale_raf = F.one(),
                .scale_rw = F.one(),
                .scale_val = F.one(),
                .inst1_eval_0 = F.zero(),
                .inst1_eval_1 = F.zero(),
                .inst1_eval_2 = F.zero(),
                .n_cycle_vars = n_cycle_vars,
                .log_ram_k = log_ram_k,
                .K = K,
                .r_cycle_raf = r_cycle_raf,
                .r_cycle_rw = r_cycle_rw,
                .r_cycle_val = r_cycle_val,
            };
        }

        /// Set the scaled initial claim (called after scaling by 2^(max_rounds - instance_rounds)).
        pub fn setScaledClaim(self: *Self, scaled_claim: F) void {
            self.current_claim = scaled_claim;
        }

        /// Deinitialize all owned memory.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.ram_addresses);
            self.allocator.free(self.ram_cycles);
            self.allocator.free(self.ram_G_A);
            self.allocator.free(self.ram_G_B);
            self.allocator.free(self.eq_raf_access);
            self.allocator.free(self.eq_rw_access);
            self.allocator.free(self.eq_val_access);
            self.allocator.free(self.G_A_full);
            self.allocator.free(self.G_B_full);
            self.allocator.free(self.B_1);
            self.allocator.free(self.B_2);
            self.allocator.free(self.eq_cycle_bound);
            self.allocator.free(self.eq_raf_remaining);
            self.allocator.free(self.eq_rw_remaining);
            self.allocator.free(self.eq_val_remaining);
            self.allocator.free(self.P_raf);
            self.allocator.free(self.P_rw);
            self.allocator.free(self.P_val);
            self.allocator.free(self.Q_raf);
            self.allocator.free(self.Q_rw);
            self.allocator.free(self.Q_val);
            self.allocator.free(self.eq_raf_hi);
            self.allocator.free(self.eq_rw_hi);
            self.allocator.free(self.eq_val_hi);
            self.allocator.free(self.H_prime);
            self.allocator.free(self.cycle_challenges);
        }

        /// Compute the round polynomial evaluations [eval_0, eval_1, eval_2, 0] for a
        /// cycle round of the RamRaClaimReduction sumcheck.
        ///
        /// `ram_ra_round` is the instance-local round index (0 to n_cycle_vars-1).
        /// Returns [4]F in Toom-Cook format: [p(0), p(1), p(2), p_inf].
        pub fn computeRoundPoly(self: *Self, ram_ra_round: usize) [4]F {
            // All ram_ra rounds are cycle rounds (upstream cycle-only reduction)
            const cycle_round = ram_ra_round;

            // Initialize Q arrays at the start of PhaseCycle (round 0)
            if (!self.phase_cycle_q_initialized) {
                self.phase_cycle_q_initialized = true;

                if (comptime debug_verbose) {
                    dbg("[STAGE5 PQ] prefix_n_vars={}, suffix_n_vars={}\n", .{
                        self.prefix_n_vars, self.suffix_n_vars,
                    });
                }

                // Compute Q arrays: Q_x[c_lo] = Sum_{c_hi} H[c_lo,c_hi] * eq_x_hi(c_hi)
                @memset(self.Q_raf, F.zero());
                @memset(self.Q_rw, F.zero());
                @memset(self.Q_val, F.zero());

                for (0..self.ram_access_count) |access_idx| {
                    const cycle = self.ram_cycles[access_idx];
                    const cycle_usize: usize = @intCast(cycle);
                    const addr = self.ram_addresses[access_idx];
                    const addr_usize: usize = @intCast(addr);

                    // H[c] = B_1[address[c]] = eq(r_address_raf, addr)
                    const H_c = self.B_1[addr_usize];

                    const c_lo = cycle_usize & (self.prefix_size - 1);
                    const c_hi = cycle_usize >> @intCast(self.prefix_n_vars);

                    self.Q_raf[c_lo] = self.Q_raf[c_lo].add(H_c.mul(self.eq_raf_hi[c_hi]));
                    self.Q_rw[c_lo] = self.Q_rw[c_lo].add(H_c.mul(self.eq_rw_hi[c_hi]));
                    self.Q_val[c_lo] = self.Q_val[c_lo].add(H_c.mul(self.eq_val_hi[c_hi]));
                }

                if (comptime debug_verbose) {
                    dbg("[STAGE5 PQ] Q arrays initialized with {} accesses\n", .{self.ram_access_count});
                }
            }

            // Upstream coefficients: 1, gamma, gamma^2
            const coeff_raf = F.one();
            const coeff_rw = self.gamma;
            const coeff_val = self.gamma2;

            const current_P_len = self.prefix_size >> @intCast(cycle_round);
            const half_len = current_P_len / 2;

            if (cycle_round < self.prefix_n_vars and half_len > 0) {
                // PhaseCycle1: Bind prefix bits using P*Q decomposition
                if (comptime debug_verbose) {
                    dbg("[INST1 HINT DEBUG] Round (inst-local {}): ram_ra_current_claim at poly start = {x}\n", .{
                        ram_ra_round,
                        self.current_claim.toBytesBE()[16..32].*,
                    });
                }

                var eval_1 = F.zero();

                for (0..half_len) |j| {
                    const p_raf_1 = self.P_raf[2 * j + 1];
                    const p_rw_1 = self.P_rw[2 * j + 1];
                    const p_val_1 = self.P_val[2 * j + 1];

                    const q_raf_1 = self.Q_raf[2 * j + 1];
                    const q_rw_1 = self.Q_rw[2 * j + 1];
                    const q_val_1 = self.Q_val[2 * j + 1];

                    const contrib_1 = coeff_raf.mul(p_raf_1.mul(q_raf_1))
                        .add(coeff_rw.mul(p_rw_1.mul(q_rw_1)))
                        .add(coeff_val.mul(p_val_1.mul(q_val_1)));
                    eval_1 = eval_1.add(contrib_1);
                }

                // HINT MECHANISM: derive eval_0 from claim
                const eval_0 = self.current_claim.sub(eval_1);

                // Compute eval_2 = p(2) for degree-2 polynomial
                var eval_2 = F.zero();
                for (0..half_len) |j| {
                    const p_raf_at_2 = self.P_raf[2 * j + 1].add(self.P_raf[2 * j + 1]).sub(self.P_raf[2 * j]);
                    const q_raf_at_2 = self.Q_raf[2 * j + 1].add(self.Q_raf[2 * j + 1]).sub(self.Q_raf[2 * j]);
                    const p_rw_at_2 = self.P_rw[2 * j + 1].add(self.P_rw[2 * j + 1]).sub(self.P_rw[2 * j]);
                    const q_rw_at_2 = self.Q_rw[2 * j + 1].add(self.Q_rw[2 * j + 1]).sub(self.Q_rw[2 * j]);
                    const p_val_at_2 = self.P_val[2 * j + 1].add(self.P_val[2 * j + 1]).sub(self.P_val[2 * j]);
                    const q_val_at_2 = self.Q_val[2 * j + 1].add(self.Q_val[2 * j + 1]).sub(self.Q_val[2 * j]);

                    const contrib_2 = coeff_raf.mul(p_raf_at_2.mul(q_raf_at_2))
                        .add(coeff_rw.mul(p_rw_at_2.mul(q_rw_at_2)))
                        .add(coeff_val.mul(p_val_at_2.mul(q_val_at_2)));
                    eval_2 = eval_2.add(contrib_2);
                }

                self.inst1_eval_0 = eval_0;
                self.inst1_eval_1 = eval_1;
                self.inst1_eval_2 = eval_2;

                if (comptime debug_verbose) {
                    dbg("[STAGE5 RAM_RA] PhaseCycle1 round {}: eval_0={x}, eval_1={x}, eval_2={x}\n", .{
                        cycle_round,
                        eval_0.toBytesBE()[16..32].*,
                        eval_1.toBytesBE()[16..32].*,
                        eval_2.toBytesBE()[16..32].*,
                    });
                }

                return .{ eval_0, eval_1, eval_2, F.zero() };
            } else {
                // PhaseCycle2: After prefix rounds, use H' * eq_hi for suffix
                const suffix_round = cycle_round - self.prefix_n_vars;

                // Initialize H_prime at the start of PhaseCycle2
                if (!self.phase_cycle2_initialized) {
                    self.phase_cycle2_initialized = true;

                    // Compute eq_prefix[c_lo] = eq(r_cycle_prefix_challenges, c_lo)
                    var eq_prefix = self.allocator.alloc(F, self.prefix_size) catch unreachable;
                    defer self.allocator.free(eq_prefix);

                    for (0..self.prefix_size) |c_lo| {
                        var eq_val_local = F.one();
                        var c_lo_var = c_lo;
                        for (0..self.prefix_n_vars) |j_var| {
                            const bit: u1 = @truncate(c_lo_var);
                            c_lo_var >>= 1;
                            const r_j = self.cycle_challenges[j_var];
                            const eq_bit_val = if (bit == 0) F.one().sub(r_j) else r_j;
                            eq_val_local = eq_val_local.mul(eq_bit_val);
                        }
                        eq_prefix[c_lo] = eq_val_local;
                    }

                    // Compute H_prime[c_hi] = Sum_{c_lo} H[c_lo,c_hi] * eq_prefix[c_lo]
                    @memset(self.H_prime, F.zero());
                    for (0..self.ram_access_count) |access_idx| {
                        const cycle = self.ram_cycles[access_idx];
                        const cycle_usize: usize = @intCast(cycle);
                        const addr = self.ram_addresses[access_idx];
                        const addr_usize: usize = @intCast(addr);

                        const H_c = self.B_1[addr_usize];
                        const c_lo = cycle_usize & (self.prefix_size - 1);
                        const c_hi = cycle_usize >> @intCast(self.prefix_n_vars);

                        self.H_prime[c_hi] = self.H_prime[c_hi].add(H_c.mul(eq_prefix[c_lo]));
                    }

                    // Compute scaling factors: eq(r_cycle_x_lo, r_cycle_prefix_reduced)
                    self.scale_raf = F.one();
                    self.scale_rw = F.one();
                    self.scale_val = F.one();
                    for (0..self.prefix_n_vars) |j_var| {
                        const r_raf_j = self.r_cycle_raf[self.suffix_n_vars + self.prefix_n_vars - 1 - j_var];
                        const r_rw_j = self.r_cycle_rw[self.suffix_n_vars + self.prefix_n_vars - 1 - j_var];
                        const r_val_j = self.r_cycle_val[self.suffix_n_vars + self.prefix_n_vars - 1 - j_var];
                        const chal_j = self.cycle_challenges[j_var];
                        self.scale_raf = self.scale_raf.mul(F.one().sub(r_raf_j).mul(F.one().sub(chal_j)).add(r_raf_j.mul(chal_j)));
                        self.scale_rw = self.scale_rw.mul(F.one().sub(r_rw_j).mul(F.one().sub(chal_j)).add(r_rw_j.mul(chal_j)));
                        self.scale_val = self.scale_val.mul(F.one().sub(r_val_j).mul(F.one().sub(chal_j)).add(r_val_j.mul(chal_j)));
                    }

                    if (comptime debug_verbose) {
                        dbg("[STAGE5 RAM_RA] PhaseCycle2 starting at suffix_round={}\n", .{suffix_round});
                    }
                }

                // Compute polynomial using H_prime * eq_hi products
                const current_len = self.suffix_size >> @intCast(suffix_round);
                const half_len_suffix = current_len / 2;

                const coeff_raf_scaled = self.scale_raf;
                const coeff_rw_scaled = self.gamma.mul(self.scale_rw);
                const coeff_val_scaled = self.gamma2.mul(self.scale_val);

                var eval_1 = F.zero();

                for (0..half_len_suffix) |j| {
                    const h_1 = self.H_prime[2 * j + 1];
                    const eq_raf_1 = self.eq_raf_hi[2 * j + 1];
                    const eq_rw_1 = self.eq_rw_hi[2 * j + 1];
                    const eq_val_1 = self.eq_val_hi[2 * j + 1];

                    const contrib_1 = h_1.mul(
                        coeff_raf_scaled.mul(eq_raf_1)
                            .add(coeff_rw_scaled.mul(eq_rw_1))
                            .add(coeff_val_scaled.mul(eq_val_1)),
                    );
                    eval_1 = eval_1.add(contrib_1);
                }

                const eval_0 = self.current_claim.sub(eval_1);

                var eval_2 = F.zero();
                for (0..half_len_suffix) |j| {
                    const h_at_2 = self.H_prime[2 * j + 1].add(self.H_prime[2 * j + 1]).sub(self.H_prime[2 * j]);
                    const eq_raf_at_2 = self.eq_raf_hi[2 * j + 1].add(self.eq_raf_hi[2 * j + 1]).sub(self.eq_raf_hi[2 * j]);
                    const eq_rw_at_2 = self.eq_rw_hi[2 * j + 1].add(self.eq_rw_hi[2 * j + 1]).sub(self.eq_rw_hi[2 * j]);
                    const eq_val_at_2 = self.eq_val_hi[2 * j + 1].add(self.eq_val_hi[2 * j + 1]).sub(self.eq_val_hi[2 * j]);

                    const contrib_2 = h_at_2.mul(
                        coeff_raf_scaled.mul(eq_raf_at_2)
                            .add(coeff_rw_scaled.mul(eq_rw_at_2))
                            .add(coeff_val_scaled.mul(eq_val_at_2)),
                    );
                    eval_2 = eval_2.add(contrib_2);
                }

                self.inst1_eval_0 = eval_0;
                self.inst1_eval_1 = eval_1;
                self.inst1_eval_2 = eval_2;

                if (comptime debug_verbose) {
                    dbg("[STAGE5 RAM_RA] PhaseCycle2 round {}: eval_0={x}, eval_1={x}, eval_2={x}\n", .{
                        cycle_round,
                        eval_0.toBytesBE()[16..32].*,
                        eval_1.toBytesBE()[16..32].*,
                        eval_2.toBytesBE()[16..32].*,
                    });
                }

                return .{ eval_0, eval_1, eval_2, F.zero() };
            }
        }

        /// Bind the challenge for a cycle round. Updates P/Q or H_prime/eq_hi arrays,
        /// eq bound/remaining tracking, and the current claim.
        ///
        /// `ram_ra_round` is the instance-local round index (0 to n_cycle_vars-1).
        /// `challenge` is the sumcheck challenge for this round.
        pub fn bindChallenge(self: *Self, ram_ra_round: usize, challenge: F) void {
            const cycle_round = ram_ra_round;
            const one_minus_r = F.one().sub(challenge);

            // Get r_cycle_* values for this bit (LowToHigh order)
            const r_raf_bit = self.r_cycle_raf[self.n_cycle_vars - 1 - cycle_round];
            const r_rw_bit = self.r_cycle_rw[self.n_cycle_vars - 1 - cycle_round];
            const r_val_bit = self.r_cycle_val[self.n_cycle_vars - 1 - cycle_round];

            // Update eq_*_bound with eq_bit(r_*[m], r_m)
            const eq_raf_update = F.one().sub(r_raf_bit).sub(challenge).add(r_raf_bit.mul(challenge).add(r_raf_bit.mul(challenge)));
            const eq_rw_update = F.one().sub(r_rw_bit).sub(challenge).add(r_rw_bit.mul(challenge).add(r_rw_bit.mul(challenge)));
            const eq_val_update = F.one().sub(r_val_bit).sub(challenge).add(r_val_bit.mul(challenge).add(r_val_bit.mul(challenge)));
            self.eq_raf_bound = self.eq_raf_bound.mul(eq_raf_update);
            self.eq_rw_bound = self.eq_rw_bound.mul(eq_rw_update);
            self.eq_val_bound = self.eq_val_bound.mul(eq_val_update);

            // Precompute the 6 inverses
            const one_minus_r_raf = F.one().sub(r_raf_bit);
            const one_minus_r_rw = F.one().sub(r_rw_bit);
            const one_minus_r_val = F.one().sub(r_val_bit);
            const inv_r_raf = if (!r_raf_bit.eql(F.zero())) r_raf_bit.inverse().? else F.zero();
            const inv_1mr_raf = if (!one_minus_r_raf.eql(F.zero())) one_minus_r_raf.inverse().? else F.zero();
            const inv_r_rw = if (!r_rw_bit.eql(F.zero())) r_rw_bit.inverse().? else F.zero();
            const inv_1mr_rw = if (!one_minus_r_rw.eql(F.zero())) one_minus_r_rw.inverse().? else F.zero();
            const inv_r_val = if (!r_val_bit.eql(F.zero())) r_val_bit.inverse().? else F.zero();
            const inv_1mr_val = if (!one_minus_r_val.eql(F.zero())) one_minus_r_val.inverse().? else F.zero();

            for (0..self.ram_access_count) |access_idx| {
                const cycle = self.ram_cycles[access_idx];
                const cycle_usize: usize = @intCast(cycle);
                const c_m: u1 = @truncate(cycle_usize >> @intCast(cycle_round));
                const factor = if (c_m == 0) one_minus_r else challenge;
                self.eq_cycle_bound[access_idx] = self.eq_cycle_bound[access_idx].mul(factor);

                const inv_raf = if (c_m == 0) inv_1mr_raf else inv_r_raf;
                const inv_rw = if (c_m == 0) inv_1mr_rw else inv_r_rw;
                const inv_val = if (c_m == 0) inv_1mr_val else inv_r_val;

                if (!inv_raf.eql(F.zero())) {
                    self.eq_raf_remaining[access_idx] = self.eq_raf_remaining[access_idx].mul(inv_raf);
                }
                if (!inv_rw.eql(F.zero())) {
                    self.eq_rw_remaining[access_idx] = self.eq_rw_remaining[access_idx].mul(inv_rw);
                }
                if (!inv_val.eql(F.zero())) {
                    self.eq_val_remaining[access_idx] = self.eq_val_remaining[access_idx].mul(inv_val);
                }
            }

            if (comptime debug_verbose) {
                dbg("[STAGE5 CYCLE BIND] cycle_round={}, challenge={x}\n", .{
                    cycle_round,
                    challenge.toBytesBE()[16..32].*,
                });
                dbg("  eq_raf_bound={x}\n", .{self.eq_raf_bound.toBytesBE()[16..32].*});
            }
            if (self.ram_access_count > 0) {
                if (comptime debug_verbose) {
                    dbg("  eq_cycle_bound[0]={x}\n", .{self.eq_cycle_bound[0].toBytesBE()[16..32].*});
                }
            }

            // Store cycle challenge for PhaseCycle2 eq_prefix computation
            self.cycle_challenges[cycle_round] = challenge;

            if (cycle_round < self.prefix_n_vars) {
                // PhaseCycle1: bind P and Q polynomials
                const current_len = self.prefix_size >> @intCast(cycle_round);
                const half_len = current_len / 2;

                // Bind P and Q arrays: X'[j] = (1-r)*X[2j] + r*X[2j+1]
                if (self.gpu_ops) |gpu| {
                    if (half_len >= 16384) {
                        const pq_arrays = [_][]F{ self.P_raf, self.P_rw, self.P_val, self.Q_raf, self.Q_rw, self.Q_val };
                        for (pq_arrays) |arr| {
                            gpu.polyBindLow(arr[0 .. half_len * 2], challenge, arr[0..half_len]) catch {
                                for (0..half_len) |j| {
                                    const lo = arr[2 * j];
                                    arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(challenge.limbs));
                                }
                            };
                        }
                    } else {
                        for (0..half_len) |j| {
                            self.P_raf[j] = self.P_raf[2 * j].add(self.P_raf[2 * j + 1].sub(self.P_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                            self.P_rw[j] = self.P_rw[2 * j].add(self.P_rw[2 * j + 1].sub(self.P_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                            self.P_val[j] = self.P_val[2 * j].add(self.P_val[2 * j + 1].sub(self.P_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                        }
                        for (0..half_len) |j| {
                            self.Q_raf[j] = self.Q_raf[2 * j].add(self.Q_raf[2 * j + 1].sub(self.Q_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                            self.Q_rw[j] = self.Q_rw[2 * j].add(self.Q_rw[2 * j + 1].sub(self.Q_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                            self.Q_val[j] = self.Q_val[2 * j].add(self.Q_val[2 * j + 1].sub(self.Q_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                        }
                    }
                } else if (self.thread_pool) |tp| {
                    const BindCtx = struct {
                        p_raf: []F,
                        p_rw: []F,
                        p_val: []F,
                        q_raf: []F,
                        q_rw: []F,
                        q_val: []F,
                        chal_limbs: [4]u64,
                        h: usize,
                    };
                    const bctx = BindCtx{ .p_raf = self.P_raf, .p_rw = self.P_rw, .p_val = self.P_val, .q_raf = self.Q_raf, .q_rw = self.Q_rw, .q_val = self.Q_val, .chal_limbs = challenge.limbs, .h = half_len };
                    tp.parallelForForce(6, bctx, struct {
                        fn f(c: BindCtx, arr_idx: usize) void {
                            const arr = switch (arr_idx) {
                                0 => c.p_raf,
                                1 => c.p_rw,
                                2 => c.p_val,
                                3 => c.q_raf,
                                4 => c.q_rw,
                                5 => c.q_val,
                                else => unreachable,
                            };
                            for (0..c.h) |j| {
                                const lo = arr[2 * j];
                                arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(c.chal_limbs));
                            }
                        }
                    }.f);
                } else {
                    for (0..half_len) |j| {
                        self.P_raf[j] = self.P_raf[2 * j].add(self.P_raf[2 * j + 1].sub(self.P_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                        self.P_rw[j] = self.P_rw[2 * j].add(self.P_rw[2 * j + 1].sub(self.P_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                        self.P_val[j] = self.P_val[2 * j].add(self.P_val[2 * j + 1].sub(self.P_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                    }
                    for (0..half_len) |j| {
                        self.Q_raf[j] = self.Q_raf[2 * j].add(self.Q_raf[2 * j + 1].sub(self.Q_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                        self.Q_rw[j] = self.Q_rw[2 * j].add(self.Q_rw[2 * j + 1].sub(self.Q_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                        self.Q_val[j] = self.Q_val[2 * j].add(self.Q_val[2 * j + 1].sub(self.Q_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                    }
                }

                if (comptime debug_verbose) {
                    dbg("[STAGE5 CYCLE BIND] Bound P/Q arrays: half_len={}\n", .{
                        half_len,
                    });
                }
            } else {
                // PhaseCycle2: bind H_prime and eq_hi arrays
                const suffix_round = cycle_round - self.prefix_n_vars;
                const current_len = self.suffix_size >> @intCast(suffix_round);
                const half_len = current_len / 2;

                if (self.gpu_ops) |gpu| {
                    if (half_len >= 16384) {
                        const heq_arrays = [_][]F{ self.H_prime, self.eq_raf_hi, self.eq_rw_hi, self.eq_val_hi };
                        for (heq_arrays) |arr| {
                            gpu.polyBindLow(arr[0 .. half_len * 2], challenge, arr[0..half_len]) catch {
                                for (0..half_len) |j| {
                                    const lo = arr[2 * j];
                                    arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(challenge.limbs));
                                }
                            };
                        }
                    } else {
                        for (0..half_len) |j| {
                            self.H_prime[j] = self.H_prime[2 * j].add(self.H_prime[2 * j + 1].sub(self.H_prime[2 * j]).mulHiBigIntU128(challenge.limbs));
                        }
                        for (0..half_len) |j| {
                            self.eq_raf_hi[j] = self.eq_raf_hi[2 * j].add(self.eq_raf_hi[2 * j + 1].sub(self.eq_raf_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                            self.eq_rw_hi[j] = self.eq_rw_hi[2 * j].add(self.eq_rw_hi[2 * j + 1].sub(self.eq_rw_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                            self.eq_val_hi[j] = self.eq_val_hi[2 * j].add(self.eq_val_hi[2 * j + 1].sub(self.eq_val_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                        }
                    }
                } else if (self.thread_pool) |tp| {
                    const BindCtx2 = struct {
                        h_prime: []F,
                        eq_raf: []F,
                        eq_rw: []F,
                        eq_val: []F,
                        chal_limbs: [4]u64,
                        h: usize,
                    };
                    const bctx2 = BindCtx2{ .h_prime = self.H_prime, .eq_raf = self.eq_raf_hi, .eq_rw = self.eq_rw_hi, .eq_val = self.eq_val_hi, .chal_limbs = challenge.limbs, .h = half_len };
                    tp.parallelForForce(4, bctx2, struct {
                        fn f(c: BindCtx2, arr_idx: usize) void {
                            const arr = switch (arr_idx) {
                                0 => c.h_prime,
                                1 => c.eq_raf,
                                2 => c.eq_rw,
                                3 => c.eq_val,
                                else => unreachable,
                            };
                            for (0..c.h) |j| {
                                const lo = arr[2 * j];
                                arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(c.chal_limbs));
                            }
                        }
                    }.f);
                } else {
                    for (0..half_len) |j| {
                        self.H_prime[j] = self.H_prime[2 * j].add(self.H_prime[2 * j + 1].sub(self.H_prime[2 * j]).mulHiBigIntU128(challenge.limbs));
                    }
                    for (0..half_len) |j| {
                        self.eq_raf_hi[j] = self.eq_raf_hi[2 * j].add(self.eq_raf_hi[2 * j + 1].sub(self.eq_raf_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                        self.eq_rw_hi[j] = self.eq_rw_hi[2 * j].add(self.eq_rw_hi[2 * j + 1].sub(self.eq_rw_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                        self.eq_val_hi[j] = self.eq_val_hi[2 * j].add(self.eq_val_hi[2 * j + 1].sub(self.eq_val_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                    }
                }

                if (comptime debug_verbose) {
                    dbg("[STAGE5 RAM_RA] Bound PhaseCycle2 round {} (suffix {}): challenge={x}, new_len={}\n", .{
                        cycle_round,
                        suffix_round,
                        challenge.toBytesBE()[16..32].*,
                        half_len,
                    });
                }
                if (half_len > 0) {
                    if (comptime debug_verbose) {
                        dbg("  H_prime[0]={x}, eq_raf_hi[0]={x}\n", .{
                            self.H_prime[0].toBytesBE()[16..32].*,
                            self.eq_raf_hi[0].toBytesBE()[16..32].*,
                        });
                    }
                }
            }

            // Update Instance 1 claim: p(r) = c0 + r*c1 + r^2*c2
            const c2_inst1 = self.inst1_eval_2.sub(self.inst1_eval_1).sub(self.inst1_eval_1).add(self.inst1_eval_0).mul(UniPoly(F).INV2);
            const c1_inst1 = self.inst1_eval_1.sub(self.inst1_eval_0).sub(c2_inst1);
            const c0_inst1 = self.inst1_eval_0;
            const r2 = challenge.mul(challenge);
            self.current_claim = c0_inst1.add(c1_inst1.mulHiBigIntU128(challenge.limbs)).add(c2_inst1.mul(r2));

            if (comptime debug_verbose) {
                dbg("[STAGE5 INST1 CLAIM] new_claim={x}\n", .{
                    self.current_claim.toBytesBE()[16..32].*,
                });
            }
        }

        /// Get the final ram_ra_claim after all cycle rounds are bound.
        /// This is H_prime[0], the final evaluation of the ra polynomial.
        pub fn finalClaim(self: *const Self) F {
            return self.H_prime[0];
        }
    };
}
