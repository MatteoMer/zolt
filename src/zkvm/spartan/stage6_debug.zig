//! Stage 6 Debug Verification Functions
//!
//! Contains debug-only verification logic extracted from stage6_prover.zig.
//! All functions are gated behind comptime debug_verbose — they compile to nothing
//! when verbose = false.

const std = @import("std");
const Allocator = std.mem.Allocator;

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;

/// Debug: verify opening claims after the batched sumcheck completes.
/// Checks booleanity verifier simulation, bytecode consistency, and IncClaimReduction consistency.
///
/// All verification is gated behind `comptime debug_verbose` and compiles to nothing in production.
pub fn debugVerifyOpeningClaims(
    comptime F: type,
    allocator: Allocator,
    // Claims
    instance_claims: [6]F,
    ram_inc_claim: F,
    rd_inc_claim: F,
    hamming_weight_claim: F,
    bytecode_ra_claims: []const F,
    ram_ra_virtual_claims: []const F,
    instruction_ra_virtual_claims: []const F,
    booleanity_ra_claims: []const F,
    // Provers (anytype to avoid circular imports)
    booleanity_prover: anytype,
    bytecode_prover: anytype,
    inc_prover: anytype,
    // Config
    instruction_d: usize,
    bytecode_d: usize,
    ram_d: usize,
    log_k_chunk: usize,
    n_cycle_vars: usize,
    max_num_rounds: usize,
    bytecode_log_k: usize,
    // Gamma values
    inc_gamma: F,
    inc_gamma2: F,
    // Challenges and opening points
    challenges: []const F,
    num_rounds_arr: [6]usize,
    lookups_ra_r_cycle: []const F,
    r_cycle_inc_ram_rwc: []const F,
    r_cycle_inc_ram_val: []const F,
    r_cycle_bc4_regs_rwc: []const F,
    r_cycle_bc5_regs_val: []const F,
) !void {
    if (comptime !debug_verbose) return;

    // ======================================================================
    // Debug: IncClaimReduction opening check
    // ======================================================================
    {
        const eq_r = inc_prover.eq_ram[0];
        const eq_d = inc_prover.eq_rd[0];
        const recomp = ram_inc_claim.mul(eq_r).add(inc_gamma2.mul(rd_inc_claim.mul(eq_d)));
        const er_be = eq_r.toBytesBE();
        const ed_be = eq_d.toBytesBE();
        const rc_be = recomp.toBytesBE();
        dbg("[INC_DEBUG] eq_ram[0]_LE=[", .{});
        for (0..32) |bi| dbg("{x:0>2}", .{er_be[31 - bi]});
        dbg("]\n  eq_rd[0]_LE=[", .{});
        for (0..32) |bi| dbg("{x:0>2}", .{ed_be[31 - bi]});
        dbg("]\n  recomp_LE=[", .{});
        for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
        dbg("]\n  instance[5]_LE=[", .{});
        const i5_be = instance_claims[5].toBytesBE();
        for (0..32) |bi| dbg("{x:0>2}", .{i5_be[31 - bi]});
        dbg("]\n", .{});
    }

    // ======================================================================
    // Debug: Bytecode RA claims and decomposition
    // ======================================================================
    {
        dbg("[S6P] Bytecode RA claims (d={d}):\n", .{bytecode_d});
        for (0..bytecode_d) |i| {
            const be = bytecode_ra_claims[i].toBytesBE();
            dbg("  ra[{d}]_LE=[", .{i});
            for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
            dbg("]\n", .{});
        }
        // Compute combined[0] from GruenSplitEq final scalars + entry correction
        {
            var comb0 = bytecode_prover.entry_correction_scalar;
            for (0..5) |s| {
                comb0 = comb0.add(bytecode_prover.bound_vals_phase2[s].mul(bytecode_prover.stage_gruen_eqs[s].?.current_scalar));
            }
            const comb0_be = comb0.toBytesBE();
            dbg("  combined[0]_LE=[", .{});
            for (0..32) |bi| dbg("{x:0>2}", .{comb0_be[31 - bi]});
            dbg("]\n", .{});
            var val_ra_prod = comb0;
            for (0..bytecode_d) |i| {
                val_ra_prod = val_ra_prod.mul(bytecode_ra_claims[i]);
            }
            const vrp_be = val_ra_prod.toBytesBE();
            dbg("  combined[0]*Π_ra_LE=[", .{});
            for (0..32) |bi| dbg("{x:0>2}", .{vrp_be[31 - bi]});
            dbg("]\n", .{});
        }
        const ic0_be = instance_claims[0].toBytesBE();
        dbg("  instance_claims[0]_LE=[", .{});
        for (0..32) |bi| dbg("{x:0>2}", .{ic0_be[31 - bi]});
        dbg("]\n", .{});

        // Per-stage decomposition
        const cycle_start = bytecode_log_k;
        var r_cycle_prime = try allocator.alloc(F, n_cycle_vars);
        defer allocator.free(r_cycle_prime);
        for (0..n_cycle_vars) |ci| {
            r_cycle_prime[ci] = challenges[cycle_start + n_cycle_vars - 1 - ci];
        }
        dbg("[DECOMP] r_cycle_prime (reversed cycle challenges, BE):\n", .{});
        for (0..@min(4, n_cycle_vars)) |ci| {
            const rcp_be = r_cycle_prime[ci].toBytesBE();
            dbg("  r_cycle_prime[{}]_LE=[", .{ci});
            for (0..8) |bi| dbg("{x:0>2}", .{rcp_be[31 - bi]});
            dbg("]\n", .{});
        }

        var decomp_sum = F.zero();
        for (0..5) |s| {
            var eq_mle = F.one();
            const r_s = bytecode_prover.stage_r_cycles[s];
            for (0..n_cycle_vars) |ci| {
                const a = r_s[ci];
                const b = r_cycle_prime[ci];
                const ab = a.mul(b);
                const term = F.one().sub(a).sub(b).add(ab).add(ab);
                eq_mle = eq_mle.mul(term);
            }

            const bv = bytecode_prover.bound_vals_stored[s];
            const stage_contrib = bv.mul(eq_mle);
            decomp_sum = decomp_sum.add(stage_contrib);

            const bv_be = bv.toBytesBE();
            const eq_be = eq_mle.toBytesBE();
            const sc_be = stage_contrib.toBytesBE();
            dbg("[DECOMP] stage[{}]: bound_val_LE=[", .{s});
            for (0..8) |bi| dbg("{x:0>2}", .{bv_be[31 - bi]});
            dbg("] eq_mle_LE=[", .{});
            for (0..8) |bi| dbg("{x:0>2}", .{eq_be[31 - bi]});
            dbg("] contrib_LE=[", .{});
            for (0..8) |bi| dbg("{x:0>2}", .{sc_be[31 - bi]});
            dbg("]\n", .{});
        }
        const ds_be = decomp_sum.toBytesBE();
        dbg("[DECOMP] val_sum_LE=[", .{});
        for (0..8) |bi| dbg("{x:0>2}", .{ds_be[31 - bi]});
        dbg("]\n", .{});

        for (0..5) |s| {
            const vwr = bytecode_prover.bound_vals_stored[s];
            const gp = bytecode_prover.gamma_powers[s];
            const vwr_be = vwr.toBytesBE();
            const gp_be = gp.toBytesBE();
            dbg("[DECOMP] stage[{}]: gamma_LE=[", .{s});
            for (0..8) |bi| dbg("{x:0>2}", .{gp_be[31 - bi]});
            dbg("] gamma*val_LE=[", .{});
            for (0..8) |bi| dbg("{x:0>2}", .{vwr_be[31 - bi]});
            dbg("]\n", .{});
        }
    }

    // ======================================================================
    // Debug: Booleanity claims from H final state
    // ======================================================================
    {
        const total_booleanity_polys = instruction_d + bytecode_d + ram_d;
        dbg("[STAGE6] Booleanity claims from H final state:\n", .{});
        for (0..@min(5, total_booleanity_polys)) |i| {
            const brc_be = booleanity_ra_claims[i].toBytesBE();
            dbg("  bool_claim[{}]_LE=[", .{i});
            for (0..8) |bi| dbg("{x:0>2}", .{brc_be[31 - bi]});
            dbg("]\n", .{});
        }
    }

    // ======================================================================
    // Debug: Booleanity verifier simulation
    // ======================================================================
    {
        const total_booleanity_polys = instruction_d + bytecode_d + ram_d;
        var sum_gamma_ra = F.zero();
        for (0..total_booleanity_polys) |i| {
            const ra = booleanity_ra_claims[i];
            sum_gamma_ra = sum_gamma_ra.add(booleanity_prover.gamma_powers_sq[i].mul(ra.mul(ra).sub(ra)));
        }
        const bp_eq_r_r = booleanity_prover.eq_r_r;
        const bp_eq_cycle_final = booleanity_prover.gruen_eq_cycle.current_scalar;
        const actual_output = bp_eq_r_r.mul(bp_eq_cycle_final).mul(sum_gamma_ra);

        const sg_be = sum_gamma_ra.toBytesBE();
        const err_be = bp_eq_r_r.toBytesBE();
        const ecf_be = bp_eq_cycle_final.toBytesBE();
        const ao_be = actual_output.toBytesBE();
        dbg("[BOOL_VERIFY] sum_gamma_ra_LE=[", .{});
        for (0..8) |bi| dbg("{x:0>2}", .{sg_be[31 - bi]});
        dbg("]\n", .{});
        dbg("[BOOL_VERIFY] eq_r_r_LE=[", .{});
        for (0..8) |bi| dbg("{x:0>2}", .{err_be[31 - bi]});
        dbg("]\n", .{});
        dbg("[BOOL_VERIFY] eq_cycle_final_LE=[", .{});
        for (0..8) |bi| dbg("{x:0>2}", .{ecf_be[31 - bi]});
        dbg("]\n", .{});
        dbg("[BOOL_VERIFY] actual_output_LE=[", .{});
        for (0..8) |bi| dbg("{x:0>2}", .{ao_be[31 - bi]});
        dbg("]\n", .{});

        const ic1_be = instance_claims[1].toBytesBE();
        dbg("[BOOL_VERIFY] instance_claims[1]_LE=[", .{});
        for (0..8) |bi| dbg("{x:0>2}", .{ic1_be[31 - bi]});
        dbg("]\n", .{});
        dbg("[BOOL_VERIFY] match={}\n", .{@intFromBool(actual_output.eql(instance_claims[1]))});

        // Compute eq(challenges, combined_r) directly
        {
            const bool_start_round = max_num_rounds - num_rounds_arr[1];
            dbg("[BOOL_VERIFY] bool_start_round={}, log_k={}, n_cycle={}\n", .{
                bool_start_round, log_k_chunk, n_cycle_vars,
            });

            var eq_direct = F.one();
            for (0..log_k_chunk) |m| {
                const ch_val = challenges[bool_start_round + m];
                const w_val = booleanity_prover.r_address_le[m];
                const prod = ch_val.mul(w_val);
                const eq_factor = F.one().sub(ch_val).sub(w_val).add(prod.add(prod));
                eq_direct = eq_direct.mul(eq_factor);

                const ch_be = ch_val.toBytesBE();
                const w_be = w_val.toBytesBE();
                const ef_be = eq_factor.toBytesBE();
                dbg("[BOOL_EQ_ZOLT] idx={} sc=[", .{m});
                for (0..8) |bi| dbg("{x:0>2}", .{ch_be[31 - bi]});
                dbg("] cr=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{w_be[31 - bi]});
                dbg("] eq_i=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ef_be[31 - bi]});
                dbg("]\n", .{});
            }
            for (0..n_cycle_vars) |m| {
                const ch_val = challenges[bool_start_round + log_k_chunk + m];
                const w_val = lookups_ra_r_cycle[m];
                const prod = ch_val.mul(w_val);
                const eq_factor = F.one().sub(ch_val).sub(w_val).add(prod.add(prod));
                eq_direct = eq_direct.mul(eq_factor);

                const ch_be = ch_val.toBytesBE();
                const w_be = w_val.toBytesBE();
                const ef_be = eq_factor.toBytesBE();
                dbg("[BOOL_EQ_ZOLT] idx={} sc=[", .{log_k_chunk + m});
                for (0..8) |bi| dbg("{x:0>2}", .{ch_be[31 - bi]});
                dbg("] cr=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{w_be[31 - bi]});
                dbg("] eq_i=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ef_be[31 - bi]});
                dbg("]\n", .{});
            }

            const eq_from_prover = bp_eq_r_r.mul(bp_eq_cycle_final);
            const ed_be = eq_direct.toBytesBE();
            const ep_be = eq_from_prover.toBytesBE();
            dbg("[BOOL_VERIFY] eq_direct_LE=[", .{});
            for (0..8) |bi| dbg("{x:0>2}", .{ed_be[31 - bi]});
            dbg("]\n", .{});
            dbg("[BOOL_VERIFY] eq_from_prover_LE=[", .{});
            for (0..8) |bi| dbg("{x:0>2}", .{ep_be[31 - bi]});
            dbg("]\n", .{});
            dbg("[BOOL_VERIFY] eq_match={}\n", .{@intFromBool(eq_direct.eql(eq_from_prover))});
        }
    }

    // ======================================================================
    // Debug: Opening claims hex dump
    // ======================================================================
    {
        dbg("[STAGE6] Opening claims (full LE hex):\n", .{});
        {
            const be = ram_inc_claim.toBytesBE();
            dbg("  ram_inc_LE=[", .{});
            for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
            dbg("]\n", .{});
        }
        {
            const be = rd_inc_claim.toBytesBE();
            dbg("  rd_inc_LE=[", .{});
            for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
            dbg("]\n", .{});
        }
        {
            const be = hamming_weight_claim.toBytesBE();
            dbg("  hamming_weight_LE=[", .{});
            for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
            dbg("]\n", .{});
        }
        for (0..bytecode_d) |i| {
            const be = bytecode_ra_claims[i].toBytesBE();
            dbg("  bytecode_ra[{d}]_LE=[", .{i});
            for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
            dbg("]\n", .{});
        }
        {
            const be = ram_ra_virtual_claims[0].toBytesBE();
            dbg("  ram_ra_virtual[0]_LE=[", .{});
            for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
            dbg("]\n", .{});
        }
        {
            const be = instruction_ra_virtual_claims[0].toBytesBE();
            dbg("  instruction_ra_virtual[0]_LE=[", .{});
            for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
            dbg("]\n", .{});
        }
        for (0..3) |i| {
            const be = booleanity_ra_claims[i].toBytesBE();
            dbg("  booleanity_ra[{d}]_LE=[", .{i});
            for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
            dbg("]\n", .{});
        }

        // Consistency check: instance_claims[0] should equal val * Π ra[i]
        {
            var bc_combined_val = bytecode_prover.entry_correction_scalar;
            for (0..5) |s| {
                bc_combined_val = bc_combined_val.add(bytecode_prover.bound_vals_phase2[s].mul(bytecode_prover.stage_gruen_eqs[s].?.current_scalar));
            }
            var bc_ra_prod = F.one();
            for (bytecode_ra_claims) |c| bc_ra_prod = bc_ra_prod.mul(c);
            const bc_recomputed = bc_combined_val.mul(bc_ra_prod);
            dbg("[STAGE6] Consistency check Instance 0:\n", .{});
            const cval_be = bc_combined_val.toBytesBE();
            dbg("  combined[0]_LE=[", .{});
            for (0..32) |bi| dbg("{x:0>2}", .{cval_be[31 - bi]});
            dbg("]\n", .{});
            for (0..bytecode_d) |i| {
                const ra_be = bytecode_ra_claims[i].toBytesBE();
                dbg("  ra[{}]_LE=[", .{i});
                for (0..32) |bi| dbg("{x:0>2}", .{ra_be[31 - bi]});
                dbg("]\n", .{});
            }
            dbg("  recomputed_LE=[", .{});
            const rc_be = bc_recomputed.toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
            dbg("]\n", .{});
            dbg("  instance[0]_LE=[", .{});
            const ic_be = instance_claims[0].toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{ic_be[31 - bi]});
            dbg("]\n", .{});
            dbg("  match = {}\n", .{@as(u8, if (std.mem.eql(u8, &bc_recomputed.toBytesBE(), &instance_claims[0].toBytesBE())) 1 else 0)});
        }

        // Consistency check Instance 5 (IncClaimReduction)
        {
            var opening_point = try allocator.alloc(F, n_cycle_vars);
            defer allocator.free(opening_point);
            const inc_offset = max_num_rounds - n_cycle_vars;
            for (0..n_cycle_vars) |i| {
                opening_point[n_cycle_vars - 1 - i] = challenges[inc_offset + i];
            }

            const computeEqEval = struct {
                fn eval(a: []const F, b: []const F) F {
                    var result = F.one();
                    for (0..a.len) |i| {
                        const prod = a[i].mul(b[i]);
                        const sum = a[i].add(b[i]);
                        result = result.mul(prod.add(prod).add(F.one()).sub(sum));
                    }
                    return result;
                }
            }.eval;

            const eq_r2 = computeEqEval(opening_point, r_cycle_inc_ram_rwc);
            const eq_r4 = computeEqEval(opening_point, r_cycle_inc_ram_val);
            const eq_s4 = computeEqEval(opening_point, r_cycle_bc4_regs_rwc);
            const eq_s5 = computeEqEval(opening_point, r_cycle_bc5_regs_val);

            const eq_ram_combined = eq_r2.add(inc_gamma.mul(eq_r4));
            const eq_rd_combined = eq_s4.add(inc_gamma.mul(eq_s5));

            const expected_inc = ram_inc_claim.mul(eq_ram_combined).add(inc_gamma2.mul(rd_inc_claim.mul(eq_rd_combined)));

            dbg("[STAGE6] Inc consistency check:\n", .{});
            dbg("  ram_inc_claim_LE=[", .{});
            const ric_be = ram_inc_claim.toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{ric_be[31 - bi]});
            dbg("]\n", .{});
            dbg("  rd_inc_claim_LE=[", .{});
            const rdc_be = rd_inc_claim.toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{rdc_be[31 - bi]});
            dbg("]\n", .{});
            dbg("  eq_r2_LE=[", .{});
            const er2 = eq_r2.toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{er2[31 - bi]});
            dbg("]\n", .{});
            dbg("  eq_r4_LE=[", .{});
            const er4 = eq_r4.toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{er4[31 - bi]});
            dbg("]\n", .{});
            dbg("  eq_s4_LE=[", .{});
            const es4 = eq_s4.toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{es4[31 - bi]});
            dbg("]\n", .{});
            dbg("  eq_s5_LE=[", .{});
            const es5 = eq_s5.toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{es5[31 - bi]});
            dbg("]\n", .{});
            dbg("  expected_inc_LE=[", .{});
            const eibc = expected_inc.toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{eibc[31 - bi]});
            dbg("]\n", .{});
            dbg("  instance[5]_LE=[", .{});
            const i5_be = instance_claims[5].toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{i5_be[31 - bi]});
            dbg("]\n", .{});
            dbg("  match = {}\n", .{@as(u8, if (std.mem.eql(u8, &expected_inc.toBytesBE(), &instance_claims[5].toBytesBE())) 1 else 0)});

            dbg("  r_cycle_inc_ram_rwc[0]_LE=[", .{});
            const rr0 = r_cycle_inc_ram_rwc[0].toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{rr0[31 - bi]});
            dbg("]\n", .{});
            dbg("  r_cycle_inc_ram_val[0]_LE=[", .{});
            const rv0 = r_cycle_inc_ram_val[0].toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{rv0[31 - bi]});
            dbg("]\n", .{});
            dbg("  r_cycle_bc4_regs_rwc[0]_LE=[", .{});
            const rc0 = r_cycle_bc4_regs_rwc[0].toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{rc0[31 - bi]});
            dbg("]\n", .{});
            dbg("  r_cycle_bc5_regs_val[0]_LE=[", .{});
            const rv5 = r_cycle_bc5_regs_val[0].toBytesBE();
            for (0..32) |bi| dbg("{x:0>2}", .{rv5[31 - bi]});
            dbg("]\n", .{});
        }
    }
}
