//! Stage 3 InstructionInput Prover
//!
//! InstructionInputSumcheck proves operand computation (degree 3) using GruenSplitEq.
//! No prefix-suffix optimization — uses direct evaluation with factored eq.
//!
//! Formula:
//!   (eq(r, r_stage1) + gamma^2 * eq(r, r_stage2)) * (right + gamma * left)
//!   where left = left_is_rs1 * rs1 + left_is_pc * pc
//!         right = right_is_rs2 * rs2 + right_is_imm * imm

const std = @import("std");

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;
const poly_mod = @import("zolt_arith").poly;
const field_mod = @import("zolt_arith").field;
const RawR1CSInputs = @import("../r1cs/evaluators.zig").RawR1CSInputs;

pub fn InstructionInputProver(comptime F: type) type {
    return struct {
        const Self = @This();

        // Witness MLEs (double-buffered: main + scratch for parallel bind)
        left_is_rs1: []F,
        rs1_value: []F,
        left_is_pc: []F,
        unexpanded_pc: []F,
        right_is_rs2: []F,
        rs2_value: []F,
        right_is_imm: []F,
        imm: []F,

        // Scratch buffers for double-buffer parallel bind
        scratch: [8][]F,

        // Gruen split eq polynomial (factored eq at r_cycle_stage_2)
        gruen_eq: poly_mod.GruenSplitEqPolynomial(F),

        gamma: F,

        current_size: usize,
        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            raw_inputs: []const RawR1CSInputs,
            trace_len: usize,
            r_outer: []const F,
            r_product: []const F,
            gamma: F,
            thread_pool: ?*ThreadPool,
        ) !Self {
            // Allocate MLEs
            const left_is_rs1 = try allocator.alloc(F, trace_len);
            const rs1_value = try allocator.alloc(F, trace_len);
            const left_is_pc = try allocator.alloc(F, trace_len);
            const unexpanded_pc = try allocator.alloc(F, trace_len);
            const right_is_rs2 = try allocator.alloc(F, trace_len);
            const rs2_value = try allocator.alloc(F, trace_len);
            const right_is_imm = try allocator.alloc(F, trace_len);
            const imm = try allocator.alloc(F, trace_len);

            // Fill from RawR1CSInputs for better cache locality (~100 bytes vs 1344 bytes per entry)
            // Booleans use F.one()/F.zero(), u64s use F.fromU64(), Imm via toFieldValue (i128)
            const InstrFillCtx = struct {
                raw_inputs_ptr: []const RawR1CSInputs,
                left_is_rs1: []F,
                rs1_value_arr: []F,
                left_is_pc: []F,
                unexpanded_pc: []F,
                right_is_rs2: []F,
                rs2_value_arr: []F,
                right_is_imm: []F,
                imm_arr: []F,
                raw_len: usize,
            };
            const fill_ctx = InstrFillCtx{
                .raw_inputs_ptr = raw_inputs,
                .left_is_rs1 = left_is_rs1,
                .rs1_value_arr = rs1_value,
                .left_is_pc = left_is_pc,
                .unexpanded_pc = unexpanded_pc,
                .right_is_rs2 = right_is_rs2,
                .rs2_value_arr = rs2_value,
                .right_is_imm = right_is_imm,
                .imm_arr = imm,
                .raw_len = raw_inputs.len,
            };
            const fillWorker = struct {
                fn f(c: InstrFillCtx, i: usize) void {
                    if (i < c.raw_len) {
                        const raw = &c.raw_inputs_ptr[i];
                        // Booleans: conditional F.one()/F.zero() (no Montgomery mul)
                        c.left_is_rs1[i] = if (raw.bool_flags[21]) F.one() else F.zero();
                        c.left_is_pc[i] = if (raw.bool_flags[22]) F.one() else F.zero();
                        c.right_is_rs2[i] = if (raw.bool_flags[23]) F.one() else F.zero();
                        c.right_is_imm[i] = if (raw.bool_flags[24]) F.one() else F.zero();
                        // u64 values
                        c.rs1_value_arr[i] = F.fromU64(raw.u64_values[4]); // Rs1Value
                        c.unexpanded_pc[i] = F.fromU64(raw.u64_values[2]); // UnexpandedPC
                        c.rs2_value_arr[i] = F.fromU64(raw.u64_values[5]); // Rs2Value
                        // Imm is i128 — on-the-fly Montgomery encoding via toFieldValue
                        c.imm_arr[i] = raw.toFieldValue(F, .Imm);
                    } else {
                        c.left_is_rs1[i] = F.zero();
                        c.rs1_value_arr[i] = F.zero();
                        c.left_is_pc[i] = F.zero();
                        c.unexpanded_pc[i] = F.zero();
                        c.right_is_rs2[i] = F.zero();
                        c.rs2_value_arr[i] = F.zero();
                        c.right_is_imm[i] = F.zero();
                        c.imm_arr[i] = F.zero();
                    }
                }
            }.f;

            if (thread_pool) |tp| {
                tp.parallelForForce(trace_len, fill_ctx, fillWorker);
            } else {
                for (0..trace_len) |i| fillWorker(fill_ctx, i);
            }

            // Initialize Gruen split eq polynomial at r_product (= r_cycle_stage_2)
            _ = r_outer; // Not used for InstructionInput
            const gruen_eq = try poly_mod.GruenSplitEqPolynomial(F).init(allocator, r_product);

            // Allocate scratch buffers for double-buffer parallel bind
            var scratch: [8][]F = undefined;
            for (0..8) |i| {
                scratch[i] = try allocator.alloc(F, trace_len);
            }

            return Self{
                .left_is_rs1 = left_is_rs1,
                .rs1_value = rs1_value,
                .left_is_pc = left_is_pc,
                .unexpanded_pc = unexpanded_pc,
                .right_is_rs2 = right_is_rs2,
                .rs2_value = rs2_value,
                .right_is_imm = right_is_imm,
                .imm = imm,
                .scratch = scratch,
                .gruen_eq = gruen_eq,
                .gamma = gamma,
                .current_size = trace_len,
                .allocator = allocator,
                .thread_pool = thread_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.left_is_rs1);
            self.allocator.free(self.rs1_value);
            self.allocator.free(self.left_is_pc);
            self.allocator.free(self.unexpanded_pc);
            self.allocator.free(self.right_is_rs2);
            self.allocator.free(self.rs2_value);
            self.allocator.free(self.right_is_imm);
            self.allocator.free(self.imm);
            for (0..8) |i| self.allocator.free(self.scratch[i]);
            self.gruen_eq.deinit();
        }

        /// Compute round evaluations [p(0), p(1), p(2), p(3)] for degree-3 polynomial
        /// Uses Gruen split eq: factored E_out × E_in, evaluate at {0, ∞}, reconstruct cubic.
        pub fn computeRoundEvals(self: *Self, previous_claim: F) [4]F {
            const half = self.current_size / 2;
            const tables = self.gruen_eq.getWindowEqTables(self.gruen_eq.current_index, 1);
            const E_out = tables.E_out;
            const E_in = tables.E_in;
            const head_in_bits = tables.head_in_bits;
            const in_mask = (@as(usize, 1) << @intCast(head_in_bits)) -| 1;

            const Ctx = struct {
                left_is_rs1: []const F,
                rs1_value: []const F,
                left_is_pc: []const F,
                unexpanded_pc: []const F,
                right_is_rs2: []const F,
                rs2_value: []const F,
                right_is_imm: []const F,
                imm: []const F,
                E_out: []const F,
                E_in: []const F,
                head_in_bits: usize,
                in_mask: usize,
                gamma: F,
            };

            const ctx = Ctx{
                .left_is_rs1 = self.left_is_rs1,
                .rs1_value = self.rs1_value,
                .left_is_pc = self.left_is_pc,
                .unexpanded_pc = self.unexpanded_pc,
                .right_is_rs2 = self.right_is_rs2,
                .rs2_value = self.rs2_value,
                .right_is_imm = self.right_is_imm,
                .imm = self.imm,
                .E_out = E_out,
                .E_in = E_in,
                .head_in_bits = head_in_bits,
                .in_mask = in_mask,
                .gamma = self.gamma,
            };

            const mapFn = struct {
                fn map(c: Ctx, start: usize, end: usize) [2]F {
                    var q_constant = F.zero();
                    var q_quadratic = F.zero();

                    for (start..end) |j| {
                        // Factored eq: E_prefix = E_out[j >> head_in_bits] * E_in[j & mask]
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;
                        const E_prefix = (if (x_out < c.E_out.len) c.E_out[x_out] else F.one())
                            .mul(if (x_in < c.E_in.len) c.E_in[x_in] else F.one());

                        // Inner polynomial at X=0: val(0) = right(0) + γ*left(0)
                        const left_0 = c.left_is_rs1[2 * j].mul(c.rs1_value[2 * j])
                            .add(c.left_is_pc[2 * j].mul(c.unexpanded_pc[2 * j]));
                        const right_0 = c.right_is_rs2[2 * j].mul(c.rs2_value[2 * j])
                            .add(c.right_is_imm[2 * j].mul(c.imm[2 * j]));
                        const val_0 = right_0.add(c.gamma.mul(left_0));

                        // Inner polynomial "eval at ∞" = coefficient of X² in val(X).
                        // val(X) = right(X) + γ*left(X) where each is sum of products of linears.
                        // For f(X)*g(X) with f linear, g linear: coeff of X² = f_slope * g_slope.
                        const lis1_s = c.left_is_rs1[2 * j + 1].sub(c.left_is_rs1[2 * j]);
                        const rs1_s = c.rs1_value[2 * j + 1].sub(c.rs1_value[2 * j]);
                        const lipc_s = c.left_is_pc[2 * j + 1].sub(c.left_is_pc[2 * j]);
                        const pc_s = c.unexpanded_pc[2 * j + 1].sub(c.unexpanded_pc[2 * j]);
                        const ris2_s = c.right_is_rs2[2 * j + 1].sub(c.right_is_rs2[2 * j]);
                        const rs2_s = c.rs2_value[2 * j + 1].sub(c.rs2_value[2 * j]);
                        const riim_s = c.right_is_imm[2 * j + 1].sub(c.right_is_imm[2 * j]);
                        const imm_s = c.imm[2 * j + 1].sub(c.imm[2 * j]);

                        const left_inf = lis1_s.mul(rs1_s).add(lipc_s.mul(pc_s));
                        const right_inf = ris2_s.mul(rs2_s).add(riim_s.mul(imm_s));
                        const val_inf = right_inf.add(c.gamma.mul(left_inf));

                        // Accumulate: q_constant = Σ E_prefix * val(0)
                        //             q_quadratic = Σ E_prefix * val(∞)  [X² coefficient]
                        q_constant = q_constant.add(E_prefix.mul(val_0));
                        q_quadratic = q_quadratic.add(E_prefix.mul(val_inf));
                    }
                    return .{ q_constant, q_quadratic };
                }
            }.map;

            const reduceFn = struct {
                fn reduce(a: [2]F, b: [2]F) [2]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]) };
                }
            }.reduce;

            const sums = if (self.thread_pool) |tp|
                tp.parallelReduce([2]F, half, .{ F.zero(), F.zero() }, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            // Reconstruct degree-3 polynomial from {0, ∞} evaluations
            return self.gruen_eq.computeCubicRoundPoly(sums[0], sums[1], previous_claim);
        }

        /// Get main arrays as a fixed-size array of pointers (for parallel bind)
        fn getMainSlices(self: *Self) [8][]F {
            return .{
                self.left_is_rs1,  self.rs1_value,
                self.left_is_pc,   self.unexpanded_pc,
                self.right_is_rs2, self.rs2_value,
                self.right_is_imm, self.imm,
            };
        }

        /// Swap main ↔ scratch pointers after double-buffer bind
        fn swapBuffers(self: *Self) void {
            const pairs = .{
                .{ &self.left_is_rs1, &self.scratch[0] },
                .{ &self.rs1_value, &self.scratch[1] },
                .{ &self.left_is_pc, &self.scratch[2] },
                .{ &self.unexpanded_pc, &self.scratch[3] },
                .{ &self.right_is_rs2, &self.scratch[4] },
                .{ &self.rs2_value, &self.scratch[5] },
                .{ &self.right_is_imm, &self.scratch[6] },
                .{ &self.imm, &self.scratch[7] },
            };
            inline for (pairs) |pair| {
                const tmp = pair[0].*;
                pair[0].* = pair[1].*;
                pair[1].* = tmp;
            }
        }

        pub fn bind(self: *Self, r_j: F) void {
            const new_size = self.current_size / 2;

            // GPU bind path removed — InstructionInput uses GruenSplitEq for eq (no flat array)

            if (self.thread_pool) |tp| {
                if (new_size >= 256) {
                    // Double-buffer parallel bind: read from main, write to scratch,
                    // then swap pointers. No data races since src != dst.
                    const BindCtx = struct {
                        src: [8][]const F,
                        dst: [8][]F,
                        r: F,
                    };
                    const main = self.getMainSlices();
                    var src: [8][]const F = undefined;
                    for (0..8) |i| src[i] = main[i];
                    const ctx = BindCtx{
                        .src = src,
                        .dst = self.scratch,
                        .r = r_j,
                    };
                    tp.parallelForForce(new_size, ctx, struct {
                        fn run(c: BindCtx, i: usize) void {
                            inline for (0..8) |a| {
                                const low = c.src[a][2 * i];
                                const high = c.src[a][2 * i + 1];
                                c.dst[a][i] = low.add(c.r.mul(high.sub(low)));
                            }
                        }
                    }.run);
                    self.swapBuffers();
                    self.gruen_eq.bind(r_j);
                    self.current_size = new_size;
                    return;
                }
            }

            // Sequential fallback (small arrays — in-place is safe when sequential)
            for (0..new_size) |i| {
                self.left_is_rs1[i] = self.left_is_rs1[2 * i].add(r_j.mul(self.left_is_rs1[2 * i + 1].sub(self.left_is_rs1[2 * i])));
                self.rs1_value[i] = self.rs1_value[2 * i].add(r_j.mul(self.rs1_value[2 * i + 1].sub(self.rs1_value[2 * i])));
                self.left_is_pc[i] = self.left_is_pc[2 * i].add(r_j.mul(self.left_is_pc[2 * i + 1].sub(self.left_is_pc[2 * i])));
                self.unexpanded_pc[i] = self.unexpanded_pc[2 * i].add(r_j.mul(self.unexpanded_pc[2 * i + 1].sub(self.unexpanded_pc[2 * i])));
                self.right_is_rs2[i] = self.right_is_rs2[2 * i].add(r_j.mul(self.right_is_rs2[2 * i + 1].sub(self.right_is_rs2[2 * i])));
                self.rs2_value[i] = self.rs2_value[2 * i].add(r_j.mul(self.rs2_value[2 * i + 1].sub(self.rs2_value[2 * i])));
                self.right_is_imm[i] = self.right_is_imm[2 * i].add(r_j.mul(self.right_is_imm[2 * i + 1].sub(self.right_is_imm[2 * i])));
                self.imm[i] = self.imm[2 * i].add(r_j.mul(self.imm[2 * i + 1].sub(self.imm[2 * i])));
            }
            self.gruen_eq.bind(r_j);
            self.current_size = new_size;
        }

        pub fn finalClaims(self: *const Self) struct {
            left_is_rs1: F,
            rs1_value: F,
            left_is_pc: F,
            unexpanded_pc: F,
            right_is_rs2: F,
            rs2_value: F,
            right_is_imm: F,
            imm: F,
        } {
            return .{
                .left_is_rs1 = if (self.left_is_rs1.len > 0) self.left_is_rs1[0] else F.zero(),
                .rs1_value = if (self.rs1_value.len > 0) self.rs1_value[0] else F.zero(),
                .left_is_pc = if (self.left_is_pc.len > 0) self.left_is_pc[0] else F.zero(),
                .unexpanded_pc = if (self.unexpanded_pc.len > 0) self.unexpanded_pc[0] else F.zero(),
                .right_is_rs2 = if (self.right_is_rs2.len > 0) self.right_is_rs2[0] else F.zero(),
                .rs2_value = if (self.rs2_value.len > 0) self.rs2_value[0] else F.zero(),
                .right_is_imm = if (self.right_is_imm.len > 0) self.right_is_imm[0] else F.zero(),
                .imm = if (self.imm.len > 0) self.imm[0] else F.zero(),
            };
        }
    };
}
