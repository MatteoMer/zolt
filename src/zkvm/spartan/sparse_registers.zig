//! Sparse register matrix for Stage 4 sumcheck.
//!
//! Faithful port of Jolt's read_write_matrix/{registers,cycle_major,address_major}.rs.
//! Stores only non-zero entries (~3 per cycle) instead of dense K×T arrays.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ThreadPool = @import("../../utils/thread_pool.zig").ThreadPool;
const one_hot = @import("one_hot_coeffs.zig");
const OneHotCoeffLookupTable = one_hot.OneHotCoeffLookupTable;
const LookupTableIndex = one_hot.LookupTableIndex;
const gruen_mod = @import("gruen_eq.zig");
const GruenSplitEqPolynomial = gruen_mod.GruenSplitEqPolynomial;
const TraceStep = @import("../../tracer/mod.zig").TraceStep;
const ExecutionTrace = @import("../../tracer/mod.zig").ExecutionTrace;

pub const LOG_K: usize = 7;
pub const K: usize = 1 << LOG_K;

// ============================================================================
// Entry types
// ============================================================================

/// Cycle-major entry with comptime-selectable coefficient type.
/// use_lookups=true: ra/wa are LookupTableIndex (compact, first ~4 rounds)
/// use_lookups=false: ra/wa are F (after deref or for direct field ops)
pub fn CycleMajorEntry(comptime F: type, comptime use_lookups: bool) type {
    const C = if (use_lookups) LookupTableIndex else F;
    return struct {
        const Self = @This();
        val_coeff: F,
        prev_val: u64,
        next_val: u64,
        row: u32,
        col: u8,
        ra_coeff: C,
        wa_coeff: C,

        pub inline fn rowUsize(self: Self) usize {
            return @as(usize, self.row);
        }
        pub inline fn column(self: Self) usize {
            return @as(usize, self.col);
        }

        /// Compute [q(0), q_X2] contribution for this entry pair.
        /// Matches Jolt RegistersCycleMajorEntry::compute_evals.
        pub fn computeEvals(
            even: ?*const Self,
            odd: ?*const Self,
            inc_evals: [2]F,
            ra_table: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
            wa_table: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
        ) [2]F {
            if (even != null and odd != null) {
                const e = even.?;
                const o = odd.?;
                const ra_ev = coeffEvals(e.ra_coeff, o.ra_coeff, ra_table);
                const wa_ev = coeffEvals(e.wa_coeff, o.wa_coeff, wa_table);
                const val_ev = [2]F{ e.val_coeff, o.val_coeff.sub(e.val_coeff) };
                return .{
                    ra_ev[0].mul(val_ev[0]).add(wa_ev[0].mul(val_ev[0].add(inc_evals[0]))),
                    ra_ev[1].mul(val_ev[1]).add(wa_ev[1].mul(val_ev[1].add(inc_evals[1]))),
                };
            } else if (even != null) {
                const e = even.?;
                const odd_val = F.fromU64(e.next_val);
                const ra_ev = coeffEvalsEvenOnly(e.ra_coeff, ra_table);
                const wa_ev = coeffEvalsEvenOnly(e.wa_coeff, wa_table);
                const val_ev = [2]F{ e.val_coeff, odd_val.sub(e.val_coeff) };
                return .{
                    ra_ev[0].mul(val_ev[0]).add(wa_ev[0].mul(val_ev[0].add(inc_evals[0]))),
                    ra_ev[1].mul(val_ev[1]).add(wa_ev[1].mul(val_ev[1].add(inc_evals[1]))),
                };
            } else if (odd != null) {
                const o = odd.?;
                const even_val = F.fromU64(o.prev_val);
                const ra_ev = coeffEvalsOddOnly(o.ra_coeff, ra_table);
                const wa_ev = coeffEvalsOddOnly(o.wa_coeff, wa_table);
                const val_ev = [2]F{ even_val, o.val_coeff.sub(even_val) };
                // f(0) = 0 since even ra=wa=0
                return .{
                    F.zero(),
                    ra_ev[1].mul(val_ev[1]).add(wa_ev[1].mul(val_ev[1].add(inc_evals[1]))),
                };
            } else unreachable;
        }

        /// Bind a pair of entries. Matches Jolt RegistersCycleMajorEntry::bind_entries.
        pub fn bindEntries(
            even: ?*const Self,
            odd: ?*const Self,
            r: F,
            ra_table: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
            wa_table: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
        ) Self {
            if (even != null and odd != null) {
                const e = even.?;
                const o = odd.?;
                return .{
                    .row = e.row / 2,
                    .col = e.col,
                    .ra_coeff = bindCoeff(e.ra_coeff, o.ra_coeff, r, ra_table, true, true),
                    .wa_coeff = bindCoeff(e.wa_coeff, o.wa_coeff, r, wa_table, true, true),
                    .val_coeff = e.val_coeff.add(r.mul(o.val_coeff.sub(e.val_coeff))),
                    .prev_val = e.prev_val,
                    .next_val = o.next_val,
                };
            } else if (even != null) {
                const e = even.?;
                const odd_val = F.fromU64(e.next_val);
                return .{
                    .row = e.row / 2,
                    .col = e.col,
                    .ra_coeff = bindCoeff(e.ra_coeff, undefined, r, ra_table, true, false),
                    .wa_coeff = bindCoeff(e.wa_coeff, undefined, r, wa_table, true, false),
                    .val_coeff = e.val_coeff.add(r.mul(odd_val.sub(e.val_coeff))),
                    .prev_val = e.prev_val,
                    .next_val = e.next_val,
                };
            } else if (odd != null) {
                const o = odd.?;
                const even_val = F.fromU64(o.prev_val);
                return .{
                    .row = o.row / 2,
                    .col = o.col,
                    .ra_coeff = bindCoeff(undefined, o.ra_coeff, r, ra_table, false, true),
                    .wa_coeff = bindCoeff(undefined, o.wa_coeff, r, wa_table, false, true),
                    .val_coeff = even_val.add(r.mul(o.val_coeff.sub(even_val))),
                    .prev_val = o.prev_val,
                    .next_val = o.next_val,
                };
            } else unreachable;
        }

        /// Convert to AddressMajorEntry (resolve lookups to field elements).
        pub fn toAddressMajor(
            self: Self,
            ra_table: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
            wa_table: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
        ) AddressMajorEntry(F) {
            return .{
                .row = self.row,
                .col = self.col,
                .prev_val = F.fromU64(self.prev_val),
                .next_val = F.fromU64(self.next_val),
                .val_coeff = self.val_coeff,
                .ra_coeff = if (use_lookups) ra_table.deref(self.ra_coeff) else self.ra_coeff,
                .wa_coeff = if (use_lookups) wa_table.deref(self.wa_coeff) else self.wa_coeff,
            };
        }

        // -- Coefficient helpers (comptime-dispatched) --

        fn coeffEvals(even_c: C, odd_c: C, table: if (use_lookups) *const OneHotCoeffLookupTable(F) else void) [2]F {
            if (use_lookups) {
                return table.evals(even_c, odd_c);
            } else {
                return .{ even_c, odd_c.sub(even_c) };
            }
        }

        fn coeffEvalsEvenOnly(even_c: C, table: if (use_lookups) *const OneHotCoeffLookupTable(F) else void) [2]F {
            if (use_lookups) {
                return table.evals(even_c, null);
            } else {
                return .{ even_c, even_c.neg() };
            }
        }

        fn coeffEvalsOddOnly(odd_c: C, table: if (use_lookups) *const OneHotCoeffLookupTable(F) else void) [2]F {
            if (use_lookups) {
                return table.evals(null, odd_c);
            } else {
                return .{ F.zero(), odd_c };
            }
        }

        const undefined_c: C = if (use_lookups) LookupTableIndex.zero() else F.zero();
        const _unused_undefined = undefined_c;

        fn bindCoeff(
            even_c: C,
            odd_c: C,
            r: F,
            table: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
            has_even: bool,
            has_odd: bool,
        ) C {
            if (use_lookups) {
                return table.bindIndicesPreBind(
                    if (has_even) even_c else null,
                    if (has_odd) odd_c else null,
                );
            } else {
                const ev = if (has_even) even_c else F.zero();
                const od = if (has_odd) odd_c else F.zero();
                if (has_even and has_odd) {
                    return ev.add(r.mul(od.sub(ev)));
                } else if (has_even) {
                    return ev.sub(r.mul(ev));
                } else {
                    return r.mul(od);
                }
            }
        }
    };
}

/// Address-major entry (always uses F for coefficients).
pub fn AddressMajorEntry(comptime F: type) type {
    return struct {
        const Self = @This();
        prev_val: F,
        next_val: F,
        val_coeff: F,
        ra_coeff: F,
        wa_coeff: F,
        row: u32,
        col: u8,

        pub inline fn rowUsize(self: Self) usize {
            return @as(usize, self.row);
        }
        pub inline fn column(self: Self) usize {
            return @as(usize, self.col);
        }
        pub inline fn prevVal(self: Self) F {
            return self.prev_val;
        }
        pub inline fn nextVal(self: Self) F {
            return self.next_val;
        }

        /// Compute [f(0), f(2)] contribution. Matches Jolt RegistersAddressMajorEntry::compute_evals.
        pub fn computeEvals(
            even: ?*const Self,
            odd: ?*const Self,
            even_checkpoint: F,
            odd_checkpoint: F,
            inc_eval: F,
            eq_eval: F,
        ) [2]F {
            if (even != null and odd != null) {
                const e = even.?;
                const o = odd.?;
                // [f(0), f(2)] where f(2) = 2*odd - even
                const ra_0 = e.ra_coeff;
                const ra_2 = o.ra_coeff.add(o.ra_coeff).sub(e.ra_coeff);
                const wa_0 = e.wa_coeff;
                const wa_2 = o.wa_coeff.add(o.wa_coeff).sub(e.wa_coeff);
                const val_0 = e.val_coeff;
                const val_2 = o.val_coeff.add(o.val_coeff).sub(e.val_coeff);
                return .{
                    eq_eval.mul(ra_0.mul(val_0).add(wa_0.mul(val_0.add(inc_eval)))),
                    eq_eval.mul(ra_2.mul(val_2).add(wa_2.mul(val_2.add(inc_eval)))),
                };
            } else if (even != null) {
                const e = even.?;
                const ra_0 = e.ra_coeff;
                const ra_2 = e.ra_coeff.neg();
                const wa_0 = e.wa_coeff;
                const wa_2 = e.wa_coeff.neg();
                const val_0 = e.val_coeff;
                const val_2 = odd_checkpoint.add(odd_checkpoint).sub(e.val_coeff);
                return .{
                    eq_eval.mul(ra_0.mul(val_0).add(wa_0.mul(val_0.add(inc_eval)))),
                    eq_eval.mul(ra_2.mul(val_2).add(wa_2.mul(val_2.add(inc_eval)))),
                };
            } else if (odd != null) {
                const o = odd.?;
                const ra_2 = o.ra_coeff.add(o.ra_coeff);
                const wa_2 = o.wa_coeff.add(o.wa_coeff);
                const val_2 = o.val_coeff.add(o.val_coeff).sub(even_checkpoint);
                return .{
                    F.zero(),
                    eq_eval.mul(ra_2.mul(val_2).add(wa_2.mul(val_2.add(inc_eval)))),
                };
            } else unreachable;
        }

        /// Bind a pair of address-major entries. Matches Jolt RegistersAddressMajorEntry::bind_entries.
        pub fn bindEntries(
            even: ?*const Self,
            odd: ?*const Self,
            even_checkpoint: F,
            odd_checkpoint: F,
            r: F,
        ) Self {
            if (even != null and odd != null) {
                const e = even.?;
                const o = odd.?;
                return .{
                    .row = e.row,
                    .col = e.col / 2,
                    .ra_coeff = e.ra_coeff.add(r.mul(o.ra_coeff.sub(e.ra_coeff))),
                    .wa_coeff = e.wa_coeff.add(r.mul(o.wa_coeff.sub(e.wa_coeff))),
                    .val_coeff = e.val_coeff.add(r.mul(o.val_coeff.sub(e.val_coeff))),
                    .prev_val = e.prev_val.add(r.mul(o.prev_val.sub(e.prev_val))),
                    .next_val = e.next_val.add(r.mul(o.next_val.sub(e.next_val))),
                };
            } else if (even != null) {
                const e = even.?;
                const one_minus_r = F.one().sub(r);
                return .{
                    .row = e.row,
                    .col = e.col / 2,
                    .ra_coeff = one_minus_r.mul(e.ra_coeff),
                    .wa_coeff = one_minus_r.mul(e.wa_coeff),
                    .val_coeff = e.val_coeff.add(r.mul(odd_checkpoint.sub(e.val_coeff))),
                    .prev_val = e.prev_val.add(r.mul(odd_checkpoint.sub(e.prev_val))),
                    .next_val = e.next_val.add(r.mul(odd_checkpoint.sub(e.next_val))),
                };
            } else if (odd != null) {
                const o = odd.?;
                return .{
                    .row = o.row,
                    .col = o.col / 2,
                    .ra_coeff = r.mul(o.ra_coeff),
                    .wa_coeff = r.mul(o.wa_coeff),
                    .val_coeff = even_checkpoint.add(r.mul(o.val_coeff.sub(even_checkpoint))),
                    .prev_val = even_checkpoint.add(r.mul(o.prev_val.sub(even_checkpoint))),
                    .next_val = even_checkpoint.add(r.mul(o.next_val.sub(even_checkpoint))),
                };
            } else unreachable;
        }
    };
}

// ============================================================================
// Cycle-major sparse matrix
// ============================================================================

pub fn SparseRegistersCycleMajor(comptime F: type, comptime use_lookups: bool) type {
    const Entry = CycleMajorEntry(F, use_lookups);
    const OHTable = OneHotCoeffLookupTable(F);

    return struct {
        const Self = @This();

        entries: []Entry,
        ra_lookup_table: if (use_lookups) OHTable else void,
        wa_lookup_table: if (use_lookups) OHTable else void,
        val_init: []F,
        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.entries);
            if (use_lookups) {
                self.ra_lookup_table.deinit();
                self.wa_lookup_table.deinit();
            }
            self.allocator.free(self.val_init);
        }

        // ---- Construction from trace (only for use_lookups=true) ----

        pub fn fromTrace(allocator: Allocator, trace: *const ExecutionTrace, gamma: F, pool: ?*ThreadPool) !Self {
            if (!use_lookups) @compileError("fromTrace only for use_lookups=true");

            const trace_len = trace.steps.items.len;
            const T = std.math.ceilPowerOfTwo(usize, trace_len) catch trace_len;

            // Pass 1: count entries per cycle (parallelizable — each step independent)
            const counts = try allocator.alloc(u8, T);
            defer allocator.free(counts);
            @memset(counts, 0);

            if (pool != null and trace_len >= 256) {
                const CountCtx = struct {
                    steps: []const TraceStep,
                    cnts: []u8,
                };
                pool.?.parallelFor(trace_len, CountCtx{ .steps = trace.steps.items, .cnts = counts }, struct {
                    fn f(ctx: CountCtx, j: usize) void {
                        const step = ctx.steps[j];
                        if (!step.is_noop) {
                            ctx.cnts[j] = entryCountForStep(step);
                        }
                    }
                }.f);
            } else {
                for (trace.steps.items, 0..) |step, j| {
                    if (step.is_noop) continue;
                    counts[j] = entryCountForStep(step);
                }
            }

            // Prefix sum for offsets (sequential — O(T) additions on u8, negligible)
            var total: usize = 0;
            const offsets = try allocator.alloc(usize, T + 1);
            defer allocator.free(offsets);
            offsets[0] = 0;
            for (0..T) |j| {
                total += @as(usize, counts[j]);
                offsets[j + 1] = total;
            }

            // Pass 2: fill entries (parallelizable — each cycle writes to disjoint slices)
            // Uses TraceStep pre-recorded values (rs1_value, rs2_value, rd_pre_value)
            // instead of tracking register_values across cycles.
            const entries = try allocator.alloc(Entry, total);

            if (pool != null and trace_len >= 256) {
                const FillCtx = struct {
                    steps: []const TraceStep,
                    ents: []Entry,
                    offs: []const usize,
                    tlen: usize,
                };
                pool.?.parallelFor(T, FillCtx{
                    .steps = trace.steps.items,
                    .ents = entries,
                    .offs = offsets,
                    .tlen = trace_len,
                }, struct {
                    fn f(ctx: FillCtx, j: usize) void {
                        if (j >= ctx.tlen) return;
                        const step = ctx.steps[j];
                        const out = ctx.ents[ctx.offs[j]..ctx.offs[j + 1]];
                        if (!step.is_noop and out.len > 0) {
                            fillEntriesForStepIndependent(@intCast(j), step, out);
                        }
                    }
                }.f);
            } else {
                for (0..T) |j| {
                    if (j >= trace_len) continue;
                    const step = trace.steps.items[j];
                    const out = entries[offsets[j]..offsets[j + 1]];
                    if (!step.is_noop and out.len > 0) {
                        fillEntriesForStepIndependent(@intCast(j), step, out);
                    }
                }
            }

            // Sort entries by (row, col) - nearly sorted (filled in cycle order),
            // pdq detects this and runs near-O(N)
            std.sort.pdq(Entry, entries, {}, struct {
                fn cmp(_: void, a: Entry, b: Entry) bool {
                    if (a.row != b.row) return a.row < b.row;
                    return a.col < b.col;
                }
            }.cmp);

            // Initialize lookup tables
            const gamma_sq = gamma.mul(gamma);
            var ra_table = try OHTable.init(allocator, &.{
                F.zero(), // index 0: rs1=0, rs2=0
                gamma, // index 1: rs1=1, rs2=0
                gamma_sq, // index 2: rs1=0, rs2=1
                gamma.add(gamma_sq), // index 3: rs1=1, rs2=1
            });
            errdefer ra_table.deinit();

            var wa_table = try OHTable.initWa(allocator);
            errdefer wa_table.deinit();

            // val_init: all zeros for registers
            const val_init = try allocator.alloc(F, K);
            @memset(val_init, F.zero());

            return .{
                .entries = entries,
                .ra_lookup_table = ra_table,
                .wa_lookup_table = wa_table,
                .val_init = val_init,
                .allocator = allocator,
            };
        }

        fn entryCountForStep(step: TraceStep) u8 {
            var regs: [3]u8 = .{ 0, 0, 0 };
            var len: u8 = 0;

            if (step.rs1_read) {
                regs[len] = step.rs1_index;
                len += 1;
            }
            if (step.rs2_read) {
                var found = false;
                for (regs[0..len]) |r| {
                    if (r == step.rs2_index) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    regs[len] = step.rs2_index;
                    len += 1;
                }
            }
            if (step.rd_written) {
                var found = false;
                for (regs[0..len]) |r| {
                    if (r == step.rd_index) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    len += 1;
                }
            }
            return len;
        }

        fn fillEntriesForStep(row: u32, step: TraceStep, reg_vals: *const [K]u64, out: []Entry) void {
            var len: usize = 0;

            if (step.rs1_read) {
                const rs1 = step.rs1_index;
                const val = reg_vals[rs1];
                out[len] = .{
                    .row = row,
                    .col = rs1,
                    .prev_val = val,
                    .next_val = val,
                    .val_coeff = F.fromU64(val),
                    .ra_coeff = LookupTableIndex.fromU16(1), // gamma
                    .wa_coeff = LookupTableIndex.zero(),
                };
                len += 1;
            }

            if (step.rs2_read) {
                // Check if rs2 == rs1 (same register)
                var merged = false;
                for (out[0..len]) |*e| {
                    if (e.col == step.rs2_index) {
                        e.ra_coeff = LookupTableIndex.fromU16(3); // gamma + gamma^2
                        merged = true;
                        break;
                    }
                }
                if (!merged) {
                    const rs2 = step.rs2_index;
                    const val = reg_vals[rs2];
                    out[len] = .{
                        .row = row,
                        .col = rs2,
                        .prev_val = val,
                        .next_val = val,
                        .val_coeff = F.fromU64(val),
                        .ra_coeff = LookupTableIndex.fromU16(2), // gamma^2
                        .wa_coeff = LookupTableIndex.zero(),
                    };
                    len += 1;
                }
            }

            if (step.rd_written) {
                const rd = step.rd_index;
                var merged_rd = false;
                for (out[0..len]) |*e| {
                    if (e.col == rd) {
                        e.wa_coeff = LookupTableIndex.fromU16(1);
                        e.next_val = step.rd_value;
                        merged_rd = true;
                        break;
                    }
                }
                if (!merged_rd) {
                    const pre_val = reg_vals[rd];
                    out[len] = .{
                        .row = row,
                        .col = rd,
                        .prev_val = pre_val,
                        .next_val = step.rd_value,
                        .val_coeff = F.fromU64(pre_val),
                        .ra_coeff = if (use_lookups) LookupTableIndex.zero() else F.zero(),
                        .wa_coeff = if (use_lookups) LookupTableIndex.fromU16(1) else F.one(),
                    };
                    len += 1;
                }
            }

            std.debug.assert(len == out.len);

            // Sort by col (len <= 3, manual bubble sort)
            if (len == 2 and out[0].col > out[1].col) {
                std.mem.swap(Entry, &out[0], &out[1]);
            } else if (len == 3) {
                if (out[0].col > out[1].col) std.mem.swap(Entry, &out[0], &out[1]);
                if (out[1].col > out[2].col) std.mem.swap(Entry, &out[1], &out[2]);
                if (out[0].col > out[1].col) std.mem.swap(Entry, &out[0], &out[1]);
            }
        }

        /// Fill entries for a single step using TraceStep pre-recorded values.
        /// No external register_values array needed — reads rs1_value, rs2_value,
        /// rd_pre_value directly from the step. Safe for parallel use.
        fn fillEntriesForStepIndependent(row: u32, step: TraceStep, out: []Entry) void {
            var len: usize = 0;

            if (step.rs1_read) {
                const rs1 = step.rs1_index;
                const val = step.rs1_value;
                out[len] = .{
                    .row = row,
                    .col = rs1,
                    .prev_val = val,
                    .next_val = val,
                    .val_coeff = F.fromU64(val),
                    .ra_coeff = LookupTableIndex.fromU16(1), // gamma
                    .wa_coeff = LookupTableIndex.zero(),
                };
                len += 1;
            }

            if (step.rs2_read) {
                var merged = false;
                for (out[0..len]) |*e| {
                    if (e.col == step.rs2_index) {
                        e.ra_coeff = LookupTableIndex.fromU16(3); // gamma + gamma^2
                        merged = true;
                        break;
                    }
                }
                if (!merged) {
                    const rs2 = step.rs2_index;
                    const val = step.rs2_value;
                    out[len] = .{
                        .row = row,
                        .col = rs2,
                        .prev_val = val,
                        .next_val = val,
                        .val_coeff = F.fromU64(val),
                        .ra_coeff = LookupTableIndex.fromU16(2), // gamma^2
                        .wa_coeff = LookupTableIndex.zero(),
                    };
                    len += 1;
                }
            }

            if (step.rd_written) {
                const rd = step.rd_index;
                var merged_rd = false;
                for (out[0..len]) |*e| {
                    if (e.col == rd) {
                        e.wa_coeff = LookupTableIndex.fromU16(1);
                        e.next_val = step.rd_value;
                        merged_rd = true;
                        break;
                    }
                }
                if (!merged_rd) {
                    const pre_val = step.rd_pre_value;
                    out[len] = .{
                        .row = row,
                        .col = rd,
                        .prev_val = pre_val,
                        .next_val = step.rd_value,
                        .val_coeff = F.fromU64(pre_val),
                        .ra_coeff = if (use_lookups) LookupTableIndex.zero() else F.zero(),
                        .wa_coeff = if (use_lookups) LookupTableIndex.fromU16(1) else F.one(),
                    };
                    len += 1;
                }
            }

            std.debug.assert(len == out.len);

            // Sort by col (len <= 3, manual bubble sort)
            if (len == 2 and out[0].col > out[1].col) {
                std.mem.swap(Entry, &out[0], &out[1]);
            } else if (len == 3) {
                if (out[0].col > out[1].col) std.mem.swap(Entry, &out[0], &out[1]);
                if (out[1].col > out[2].col) std.mem.swap(Entry, &out[1], &out[2]);
                if (out[0].col > out[1].col) std.mem.swap(Entry, &out[0], &out[1]);
            }
        }

        // ---- Compute message (cycle-major round poly) ----

        /// Compute the sumcheck round polynomial for cycle variable binding.
        /// Returns [c0, c1, c2, c3] coefficients via gruenPolyDeg3.
        /// Matches Jolt ReadWriteMatrixCycleMajor::compute_message (registers.rs).
        pub fn computeMessage(
            self: *const Self,
            inc: []const F,
            gruen_eq: *const GruenSplitEqPolynomial(F),
            previous_claim: F,
        ) [4]F {
            const E_in = gruen_eq.E_in_current();
            const E_out = gruen_eq.E_out_current();
            const E_in_len = E_in.len;
            const num_x_in_bits: u6 = if (E_in_len > 1) @intCast(@ctz(E_in_len)) else 0;
            const x_bitmask: usize = if (num_x_in_bits > 0) ((@as(usize, 1) << num_x_in_bits) - 1) else 0;

            const ra_tbl = if (use_lookups) &self.ra_lookup_table else {};
            const wa_tbl = if (use_lookups) &self.wa_lookup_table else {};

            var q_0 = F.zero();
            var q_X2 = F.zero();

            // Iterate over entries grouped by row-pair (row/2)
            var idx: usize = 0;
            while (idx < self.entries.len) {
                const pair_id = self.entries[idx].row / 2;
                const group_start = idx;
                while (idx < self.entries.len and self.entries[idx].row / 2 == pair_id) {
                    idx += 1;
                }
                const group = self.entries[group_start..idx];

                // Split into even and odd rows
                var split: usize = 0;
                for (group) |e| {
                    if (e.row & 1 == 0) {
                        split += 1;
                    } else break;
                }
                const even_row = group[0..split];
                const odd_row = group[split..];

                const j_prime = @as(usize, pair_id);
                const x_in = if (num_x_in_bits > 0) (j_prime & x_bitmask) else 0;
                const x_out = j_prime >> num_x_in_bits;
                const E_in_eval = if (E_in_len <= 1) F.one() else if (x_in < E_in_len) E_in[x_in] else F.one();
                const E_out_eval = if (x_out < E_out.len) E_out[x_out] else F.one();

                // Inc evals
                const j_even = j_prime * 2;
                const j_odd = j_even + 1;
                const inc_0 = if (j_even < inc.len) inc[j_even] else F.zero();
                const inc_1 = if (j_odd < inc.len) inc[j_odd] else F.zero();
                const inc_evals = [2]F{ inc_0, inc_1.sub(inc_0) };

                // Two-pointer merge over columns
                const pair_evals = seqMergeComputeEvals(even_row, odd_row, inc_evals, ra_tbl, wa_tbl);

                // Weight by E_out * E_in
                const weight = E_out_eval.mul(E_in_eval);
                q_0 = q_0.add(weight.mul(pair_evals[0]));
                q_X2 = q_X2.add(weight.mul(pair_evals[1]));
            }

            return gruen_eq.gruenPolyDeg3(q_0, q_X2, previous_claim);
        }

        fn seqMergeComputeEvals(
            even_row: []const Entry,
            odd_row: []const Entry,
            inc_evals: [2]F,
            ra_tbl: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
            wa_tbl: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
        ) [2]F {
            var acc = [2]F{ F.zero(), F.zero() };
            var ei: usize = 0;
            var oi: usize = 0;

            while (ei < even_row.len and oi < odd_row.len) {
                if (even_row[ei].col == odd_row[oi].col) {
                    const evals = Entry.computeEvals(&even_row[ei], &odd_row[oi], inc_evals, ra_tbl, wa_tbl);
                    acc[0] = acc[0].add(evals[0]);
                    acc[1] = acc[1].add(evals[1]);
                    ei += 1;
                    oi += 1;
                } else if (even_row[ei].col < odd_row[oi].col) {
                    const evals = Entry.computeEvals(&even_row[ei], null, inc_evals, ra_tbl, wa_tbl);
                    acc[0] = acc[0].add(evals[0]);
                    acc[1] = acc[1].add(evals[1]);
                    ei += 1;
                } else {
                    const evals = Entry.computeEvals(null, &odd_row[oi], inc_evals, ra_tbl, wa_tbl);
                    acc[0] = acc[0].add(evals[0]);
                    acc[1] = acc[1].add(evals[1]);
                    oi += 1;
                }
            }
            while (ei < even_row.len) : (ei += 1) {
                const evals = Entry.computeEvals(&even_row[ei], null, inc_evals, ra_tbl, wa_tbl);
                acc[0] = acc[0].add(evals[0]);
                acc[1] = acc[1].add(evals[1]);
            }
            while (oi < odd_row.len) : (oi += 1) {
                const evals = Entry.computeEvals(null, &odd_row[oi], inc_evals, ra_tbl, wa_tbl);
                acc[0] = acc[0].add(evals[0]);
                acc[1] = acc[1].add(evals[1]);
            }
            return acc;
        }

        // ---- Bind (cycle variable) ----

        /// Bind a cycle variable. Matches Jolt ReadWriteMatrixCycleMajor::bind.
        pub fn bind(self: *Self, r: F) !void {
            const ra_tbl = if (use_lookups) &self.ra_lookup_table else {};
            const wa_tbl = if (use_lookups) &self.wa_lookup_table else {};

            // Pass 1: count output entries per row-pair group
            var total_output: usize = 0;
            {
                var idx: usize = 0;
                while (idx < self.entries.len) {
                    const pair_id = self.entries[idx].row / 2;
                    const group_start = idx;
                    while (idx < self.entries.len and self.entries[idx].row / 2 == pair_id) idx += 1;
                    const group = self.entries[group_start..idx];
                    var split: usize = 0;
                    for (group) |e| {
                        if (e.row & 1 == 0) split += 1 else break;
                    }
                    total_output += countMergedEntries(group[0..split], group[split..]);
                }
            }

            // Allocate output
            const new_entries = try self.allocator.alloc(Entry, total_output);

            // Pass 2: fill output
            {
                var idx: usize = 0;
                var out_idx: usize = 0;
                while (idx < self.entries.len) {
                    const pair_id = self.entries[idx].row / 2;
                    const group_start = idx;
                    while (idx < self.entries.len and self.entries[idx].row / 2 == pair_id) idx += 1;
                    const group = self.entries[group_start..idx];
                    var split: usize = 0;
                    for (group) |e| {
                        if (e.row & 1 == 0) split += 1 else break;
                    }
                    out_idx += seqBindRows(group[0..split], group[split..], r, new_entries[out_idx..], ra_tbl, wa_tbl);
                }
                std.debug.assert(out_idx == total_output);
            }

            // Replace entries
            self.allocator.free(self.entries);
            self.entries = new_entries;

            // Bind lookup tables
            if (use_lookups) {
                if (!self.ra_lookup_table.isSaturated()) {
                    try self.ra_lookup_table.bind(r);
                }
                if (!self.wa_lookup_table.isSaturated()) {
                    try self.wa_lookup_table.bind(r);
                }
            }
        }

        fn countMergedEntries(even: []const Entry, odd: []const Entry) usize {
            var ei: usize = 0;
            var oi: usize = 0;
            var count: usize = 0;
            while (ei < even.len and oi < odd.len) {
                if (even[ei].col == odd[oi].col) {
                    ei += 1;
                    oi += 1;
                } else if (even[ei].col < odd[oi].col) {
                    ei += 1;
                } else {
                    oi += 1;
                }
                count += 1;
            }
            count += even.len - ei;
            count += odd.len - oi;
            return count;
        }

        fn seqBindRows(
            even: []const Entry,
            odd: []const Entry,
            r: F,
            out: []Entry,
            ra_tbl: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
            wa_tbl: if (use_lookups) *const OneHotCoeffLookupTable(F) else void,
        ) usize {
            var ei: usize = 0;
            var oi: usize = 0;
            var k: usize = 0;

            while (ei < even.len and oi < odd.len) {
                if (even[ei].col == odd[oi].col) {
                    out[k] = Entry.bindEntries(&even[ei], &odd[oi], r, ra_tbl, wa_tbl);
                    ei += 1;
                    oi += 1;
                } else if (even[ei].col < odd[oi].col) {
                    out[k] = Entry.bindEntries(&even[ei], null, r, ra_tbl, wa_tbl);
                    ei += 1;
                } else {
                    out[k] = Entry.bindEntries(null, &odd[oi], r, ra_tbl, wa_tbl);
                    oi += 1;
                }
                k += 1;
            }
            while (ei < even.len) : (ei += 1) {
                out[k] = Entry.bindEntries(&even[ei], null, r, ra_tbl, wa_tbl);
                k += 1;
            }
            while (oi < odd.len) : (oi += 1) {
                out[k] = Entry.bindEntries(null, &odd[oi], r, ra_tbl, wa_tbl);
                k += 1;
            }
            return k;
        }

        // ---- Deref coefficients (lookup → field) ----

        /// Convert LookupTableIndex entries to F entries.
        /// Only available when use_lookups=true.
        pub fn derefCoeffs(self: *Self) !SparseRegistersCycleMajor(F, false) {
            if (!use_lookups) @compileError("derefCoeffs only for use_lookups=true");

            const FieldEntry = CycleMajorEntry(F, false);
            const new_entries = try self.allocator.alloc(FieldEntry, self.entries.len);

            for (self.entries, 0..) |entry, i| {
                new_entries[i] = .{
                    .val_coeff = entry.val_coeff,
                    .prev_val = entry.prev_val,
                    .next_val = entry.next_val,
                    .row = entry.row,
                    .col = entry.col,
                    .ra_coeff = self.ra_lookup_table.deref(entry.ra_coeff),
                    .wa_coeff = self.wa_lookup_table.deref(entry.wa_coeff),
                };
            }

            // Transfer val_init ownership, free old data
            const val_init = self.val_init;
            self.allocator.free(self.entries);
            self.ra_lookup_table.deinit();
            self.wa_lookup_table.deinit();
            // Prevent double-free: null out the fields
            self.entries = &.{};
            self.val_init = &.{};

            return .{
                .entries = new_entries,
                .ra_lookup_table = {},
                .wa_lookup_table = {},
                .val_init = val_init,
                .allocator = self.allocator,
            };
        }

        // ---- Transition to address-major ----

        /// Convert to address-major representation.
        /// Sorts entries by (col, row) and resolves lookup indices to F.
        pub fn toAddressMajor(self: *Self) !SparseRegistersAddressMajor(F) {
            const AMEntry = AddressMajorEntry(F);
            const ra_tbl = if (use_lookups) &self.ra_lookup_table else {};
            const wa_tbl = if (use_lookups) &self.wa_lookup_table else {};

            const am_entries = try self.allocator.alloc(AMEntry, self.entries.len);
            for (self.entries, 0..) |entry, i| {
                am_entries[i] = entry.toAddressMajor(ra_tbl, wa_tbl);
            }

            // Sort by (col, row) for address-major order
            // pdq has smaller constants than block sort and no auxiliary allocation
            std.sort.pdq(AMEntry, am_entries, {}, struct {
                fn cmp(_: void, a: AMEntry, b: AMEntry) bool {
                    if (a.col != b.col) return a.col < b.col;
                    return a.row < b.row;
                }
            }.cmp);

            // Transfer val_init
            const val_init = self.val_init;

            // Clean up cycle-major data
            self.allocator.free(self.entries);
            if (use_lookups) {
                self.ra_lookup_table.deinit();
                self.wa_lookup_table.deinit();
            }
            self.entries = &.{};
            self.val_init = &.{};

            return .{
                .entries = am_entries,
                .val_init = val_init,
                .allocator = self.allocator,
            };
        }

        /// Check if lookup tables are saturated (only for use_lookups=true).
        pub fn isLookupSaturated(self: *const Self) bool {
            if (!use_lookups) return false;
            return self.ra_lookup_table.isSaturated() or self.wa_lookup_table.isSaturated();
        }
    };
}

// ============================================================================
// Address-major sparse matrix
// ============================================================================

pub fn SparseRegistersAddressMajor(comptime F: type) type {
    const Entry = AddressMajorEntry(F);

    return struct {
        const Self = @This();

        entries: []Entry,
        val_init: []F,
        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.entries);
            self.allocator.free(self.val_init);
        }

        // ---- Compute round evaluations (address variable binding) ----

        /// Compute [eval_0, eval_2] for Phase 2 (address variable binding).
        /// Matches Jolt phase2_compute_message.
        pub fn computeRoundEvals(
            self: *const Self,
            inc: []const F,
            merged_eq: []const F,
        ) [2]F {
            var eval_0 = F.zero();
            var eval_2 = F.zero();

            // Iterate over entries grouped by column-pair (col/2)
            var idx: usize = 0;
            while (idx < self.entries.len) {
                const pair_id = self.entries[idx].col / 2;
                const group_start = idx;
                while (idx < self.entries.len and self.entries[idx].col / 2 == pair_id) idx += 1;
                const group = self.entries[group_start..idx];

                // Split into even and odd columns
                var split: usize = 0;
                for (group) |e| {
                    if (e.col & 1 == 0) split += 1 else break;
                }
                const even_col = group[0..split];
                const odd_col = group[split..];

                // Checkpoints from val_init
                const even_col_idx: usize = @as(usize, pair_id) * 2;
                const odd_col_idx = even_col_idx + 1;
                const even_cp = if (even_col_idx < self.val_init.len) self.val_init[even_col_idx] else F.zero();
                const odd_cp = if (odd_col_idx < self.val_init.len) self.val_init[odd_col_idx] else F.zero();

                const pair_evals = seqMergeComputeColEvals(even_col, odd_col, even_cp, odd_cp, inc, merged_eq);
                eval_0 = eval_0.add(pair_evals[0]);
                eval_2 = eval_2.add(pair_evals[1]);
            }

            return .{ eval_0, eval_2 };
        }

        fn seqMergeComputeColEvals(
            even: []const Entry,
            odd: []const Entry,
            initial_even_cp: F,
            initial_odd_cp: F,
            inc: []const F,
            eq: []const F,
        ) [2]F {
            var acc = [2]F{ F.zero(), F.zero() };
            var ei: usize = 0;
            var oi: usize = 0;
            var even_cp = initial_even_cp;
            var odd_cp = initial_odd_cp;

            while (ei < even.len and oi < odd.len) {
                if (even[ei].row == odd[oi].row) {
                    const row = even[ei].rowUsize();
                    const inc_eval = if (row < inc.len) inc[row] else F.zero();
                    const eq_eval = if (row < eq.len) eq[row] else F.zero();
                    const evals = Entry.computeEvals(&even[ei], &odd[oi], even_cp, odd_cp, inc_eval, eq_eval);
                    acc[0] = acc[0].add(evals[0]);
                    acc[1] = acc[1].add(evals[1]);
                    even_cp = even[ei].nextVal();
                    odd_cp = odd[oi].nextVal();
                    ei += 1;
                    oi += 1;
                } else if (even[ei].row < odd[oi].row) {
                    const row = even[ei].rowUsize();
                    const inc_eval = if (row < inc.len) inc[row] else F.zero();
                    const eq_eval = if (row < eq.len) eq[row] else F.zero();
                    const evals = Entry.computeEvals(&even[ei], null, even_cp, odd_cp, inc_eval, eq_eval);
                    acc[0] = acc[0].add(evals[0]);
                    acc[1] = acc[1].add(evals[1]);
                    even_cp = even[ei].nextVal();
                    ei += 1;
                } else {
                    const row = odd[oi].rowUsize();
                    const inc_eval = if (row < inc.len) inc[row] else F.zero();
                    const eq_eval = if (row < eq.len) eq[row] else F.zero();
                    const evals = Entry.computeEvals(null, &odd[oi], even_cp, odd_cp, inc_eval, eq_eval);
                    acc[0] = acc[0].add(evals[0]);
                    acc[1] = acc[1].add(evals[1]);
                    odd_cp = odd[oi].nextVal();
                    oi += 1;
                }
            }
            // Trailing entries: checkpoint NOT advanced (matches Jolt)
            while (ei < even.len) : (ei += 1) {
                const row = even[ei].rowUsize();
                const inc_eval = if (row < inc.len) inc[row] else F.zero();
                const eq_eval = if (row < eq.len) eq[row] else F.zero();
                const evals = Entry.computeEvals(&even[ei], null, even_cp, odd_cp, inc_eval, eq_eval);
                acc[0] = acc[0].add(evals[0]);
                acc[1] = acc[1].add(evals[1]);
            }
            while (oi < odd.len) : (oi += 1) {
                const row = odd[oi].rowUsize();
                const inc_eval = if (row < inc.len) inc[row] else F.zero();
                const eq_eval = if (row < eq.len) eq[row] else F.zero();
                const evals = Entry.computeEvals(null, &odd[oi], even_cp, odd_cp, inc_eval, eq_eval);
                acc[0] = acc[0].add(evals[0]);
                acc[1] = acc[1].add(evals[1]);
            }
            return acc;
        }

        // ---- Bind (address variable) ----

        /// Bind an address variable. Matches Jolt ReadWriteMatrixAddressMajor::bind.
        pub fn bind(self: *Self, r: F) !void {
            // Pass 1: count
            var total_output: usize = 0;
            {
                var idx: usize = 0;
                while (idx < self.entries.len) {
                    const pair_id = self.entries[idx].col / 2;
                    const group_start = idx;
                    while (idx < self.entries.len and self.entries[idx].col / 2 == pair_id) idx += 1;
                    const group = self.entries[group_start..idx];
                    var split: usize = 0;
                    for (group) |e| {
                        if (e.col & 1 == 0) split += 1 else break;
                    }
                    total_output += countMergedColEntries(group[0..split], group[split..]);
                }
            }

            // Allocate
            const new_entries = try self.allocator.alloc(Entry, total_output);

            // Pass 2: fill
            {
                var idx: usize = 0;
                var out_idx: usize = 0;
                while (idx < self.entries.len) {
                    const pair_id = self.entries[idx].col / 2;
                    const group_start = idx;
                    while (idx < self.entries.len and self.entries[idx].col / 2 == pair_id) idx += 1;
                    const group = self.entries[group_start..idx];
                    var split: usize = 0;
                    for (group) |e| {
                        if (e.col & 1 == 0) split += 1 else break;
                    }
                    const even_col_idx: usize = @as(usize, pair_id) * 2;
                    const odd_col_idx = even_col_idx + 1;
                    const even_cp = if (even_col_idx < self.val_init.len) self.val_init[even_col_idx] else F.zero();
                    const odd_cp = if (odd_col_idx < self.val_init.len) self.val_init[odd_col_idx] else F.zero();
                    out_idx += seqBindCols(group[0..split], group[split..], even_cp, odd_cp, r, new_entries[out_idx..]);
                }
                std.debug.assert(out_idx == total_output);
            }

            // Replace entries
            self.allocator.free(self.entries);
            self.entries = new_entries;

            // Bind val_init (low-to-high)
            const half = self.val_init.len / 2;
            for (0..half) |i| {
                self.val_init[i] = self.val_init[2 * i].add(r.mul(self.val_init[2 * i + 1].sub(self.val_init[2 * i])));
            }
            // Shrink val_init conceptually (just track the length)
            // Since we can't shrink the allocation, we'll just use half of it
            // The caller should use val_init[0..current_K] where current_K halves each round
        }

        fn countMergedColEntries(even: []const Entry, odd: []const Entry) usize {
            var ei: usize = 0;
            var oi: usize = 0;
            var count: usize = 0;
            while (ei < even.len and oi < odd.len) {
                if (even[ei].row == odd[oi].row) {
                    ei += 1;
                    oi += 1;
                } else if (even[ei].row < odd[oi].row) {
                    ei += 1;
                } else {
                    oi += 1;
                }
                count += 1;
            }
            count += even.len - ei;
            count += odd.len - oi;
            return count;
        }

        fn seqBindCols(
            even: []const Entry,
            odd: []const Entry,
            initial_even_cp: F,
            initial_odd_cp: F,
            r: F,
            out: []Entry,
        ) usize {
            var ei: usize = 0;
            var oi: usize = 0;
            var k: usize = 0;
            var even_cp = initial_even_cp;
            var odd_cp = initial_odd_cp;

            while (ei < even.len and oi < odd.len) {
                if (even[ei].row == odd[oi].row) {
                    out[k] = Entry.bindEntries(&even[ei], &odd[oi], even_cp, odd_cp, r);
                    even_cp = even[ei].nextVal();
                    odd_cp = odd[oi].nextVal();
                    ei += 1;
                    oi += 1;
                } else if (even[ei].row < odd[oi].row) {
                    out[k] = Entry.bindEntries(&even[ei], null, even_cp, odd_cp, r);
                    even_cp = even[ei].nextVal();
                    ei += 1;
                } else {
                    out[k] = Entry.bindEntries(null, &odd[oi], even_cp, odd_cp, r);
                    odd_cp = odd[oi].nextVal();
                    oi += 1;
                }
                k += 1;
            }
            // Trailing entries: checkpoint NOT advanced (matches Jolt)
            while (ei < even.len) : (ei += 1) {
                out[k] = Entry.bindEntries(&even[ei], null, even_cp, odd_cp, r);
                k += 1;
            }
            while (oi < odd.len) : (oi += 1) {
                out[k] = Entry.bindEntries(null, &odd[oi], even_cp, odd_cp, r);
                k += 1;
            }
            return k;
        }

        // ---- Materialize to dense ----

        /// Expand sparse entries to dense [ra, wa, val] arrays.
        /// Layout: address-major, index(k, t) = k * T_prime + t.
        pub fn materialize(self: *const Self, K_prime: usize, T_prime: usize) struct {
            ra: []F,
            wa: []F,
            val: []F,
        } {
            const len = K_prime * T_prime;
            const ra = self.allocator.alloc(F, len) catch @panic("materialize alloc");
            const wa = self.allocator.alloc(F, len) catch @panic("materialize alloc");
            const val = self.allocator.alloc(F, len) catch @panic("materialize alloc");
            @memset(ra, F.zero());
            @memset(wa, F.zero());
            @memset(val, F.zero());

            for (self.entries) |entry| {
                const idx = entry.column() * T_prime + entry.rowUsize();
                if (idx < len) {
                    ra[idx] = entry.ra_coeff;
                    wa[idx] = entry.wa_coeff;
                    val[idx] = entry.val_coeff;
                }
            }

            return .{ .ra = ra, .wa = wa, .val = val };
        }

        /// Get final claims after all variables are bound (single entry remaining).
        pub fn getFinalClaims(self: *const Self) struct {
            combined_ra: F,
            rd_wa: F,
            val: F,
        } {
            if (self.entries.len == 0) {
                return .{ .combined_ra = F.zero(), .rd_wa = F.zero(), .val = F.zero() };
            }
            // After all binding, there should be at most a few entries with col=0, row=0
            var ra_sum = F.zero();
            var wa_sum = F.zero();
            var val_sum = F.zero();
            for (self.entries) |entry| {
                ra_sum = ra_sum.add(entry.ra_coeff);
                wa_sum = wa_sum.add(entry.wa_coeff);
                val_sum = val_sum.add(entry.val_coeff);
            }
            return .{ .combined_ra = ra_sum, .rd_wa = wa_sum, .val = val_sum };
        }
    };
}

// ============================================================================
// Compute rs2_ra claim from trace (matches Jolt compute_rs2_ra_claim)
// ============================================================================

/// Compute rs2_ra(r_address, r_cycle) = Σ_j [has_rs2[j]] * eq(r_address, rs2[j]) * eq(r_cycle, j)
/// Uses 2-way split-eq optimization over the joint (cycle, address) space.
/// r_address and r_cycle are in big-endian order (r[0] = MSB).
pub fn computeRs2RaClaim(comptime F: type, trace: *const ExecutionTrace, r_address: []const F, r_cycle: []const F, allocator: Allocator) !F {
    const log_T = r_cycle.len;
    const addr_bits = r_address.len; // = 7
    const n = log_T + addr_bits;

    // Split: hi_bits from cycle, lo_bits from remaining cycle + address
    const hi_bits = @min(log_T, (n + 1) / 2);
    const lo_bits = n - hi_bits;

    // r_joint = [r_cycle..., r_address...]
    const r_joint = try allocator.alloc(F, n);
    defer allocator.free(r_joint);
    @memcpy(r_joint[0..log_T], r_cycle);
    @memcpy(r_joint[log_T..n], r_address);

    // Compute eq tables
    const E_hi = try computeEqTable(F, r_joint[0..hi_bits], allocator);
    defer allocator.free(E_hi);
    const E_lo = try computeEqTable(F, r_joint[hi_bits..n], allocator);
    defer allocator.free(E_lo);

    const cycle_bits_in_lo = lo_bits - addr_bits;
    const cycles_per_block: usize = @as(usize, 1) << @intCast(cycle_bits_in_lo);
    const cycle_lo_mask = cycles_per_block - 1;

    // Sum over hi blocks
    var result = F.zero();
    for (0..E_hi.len) |idx_hi| {
        const e_hi_val = E_hi[idx_hi];
        const block_start = idx_hi << @intCast(cycle_bits_in_lo);
        const block_end = @min(block_start + cycles_per_block, trace.steps.items.len);

        if (block_start >= trace.steps.items.len) continue;

        var inner_sum = F.zero();
        for (block_start..block_end) |j| {
            const step = trace.steps.items[j];
            if (step.is_noop or !step.rs2_read) continue;
            const rs2 = @as(usize, step.rs2_index);
            const j_in_block = j & cycle_lo_mask;
            const idx_lo = (j_in_block << @intCast(addr_bits)) | rs2;
            if (idx_lo < E_lo.len) {
                inner_sum = inner_sum.add(E_lo[idx_lo]);
            }
        }

        result = result.add(e_hi_val.mul(inner_sum));
    }

    return result;
}

/// Compute eq polynomial evaluation table (big-endian: r[0] = MSB of index).
fn computeEqTable(comptime F: type, r: []const F, allocator: Allocator) ![]F {
    const size: usize = @as(usize, 1) << @intCast(r.len);
    const table = try allocator.alloc(F, size);
    table[0] = F.one();
    var len: usize = 1;
    // Process from last to first (r[0] is MSB, assigned to highest bit)
    var i: usize = r.len;
    while (i > 0) {
        i -= 1;
        const ri = r[i];
        const one_m_r = F.one().sub(ri);
        var j: usize = len;
        while (j > 0) {
            j -= 1;
            table[2 * j + 1] = table[j].mul(ri);
            table[2 * j] = table[j].mul(one_m_r);
        }
        len *= 2;
    }
    return table;
}
