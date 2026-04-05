//! Stage 7 HammingWeightClaimReduction Prover
//!
//! Stage 7 runs a degree-2 sumcheck over log_k_chunk rounds to reduce
//! the Hamming-weight claims produced by Stage 6 into per-polynomial
//! opening claims G_i(rho).
//!
//! The prover computes:
//!   p(x) = Σ_i G_i(x) · [γ^{3i} + γ^{3i+1}·eq_bool(x) + γ^{3i+2}·eq_virt_i(x)]
//!
//! where G_i are the accumulated eq_cycle-weighted address-chunk indicator polynomials.

const std = @import("std");

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const pool_helpers = @import("zolt_pool").helpers;

const poly_mod = @import("zolt_arith").poly;
const transcripts = @import("zolt_arith").transcripts;
const jolt_types = @import("../jolt_types.zig");

const stage6_helpers = @import("stage6_helpers.zig");

pub fn Stage7Result(comptime F: type) type {
    return struct {
        const Self = @This();

        /// G_i claims for each ra polynomial (N elements)
        g_claims: []F,

        /// Unified [r_address_BE || r_cycle_BE] opening point
        /// Ownership transfers to caller — deinit does NOT free this.
        opening_point: []F,

        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.g_claims);
        }
    };
}

pub fn Stage7Prover(comptime F: type) type {
    return struct {
        const Self = @This();

        const Blake2bTranscript = transcripts.Blake2bTranscript(F);
        const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof(F);
        const CompressedUniPoly = jolt_types.CompressedUniPoly(F);
        const UniPoly = poly_mod.UniPoly(F);

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,

        pub fn init(allocator: Allocator) Self {
            return Self{ .allocator = allocator };
        }

        /// Generate Stage 7 HammingWeightClaimReduction proof.
        ///
        /// Runs a degree-2 sumcheck over log_k_chunk rounds, producing
        /// G_i opening claims and a unified opening point.
        ///
        /// Appends N cache_opening claims (G_i) to transcript before returning.
        pub fn generateStage7Proof(
            self: *Self,
            proof: *SumcheckInstanceProof,
            transcript: *Blake2bTranscript,
            stage6_result: anytype,
            stage5_challenges: []const F,
            r_address_raf: []const F,
            trace: anytype,
            memory_layout: anytype,
            pc_map: anytype,
        ) !Stage7Result(F) {
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
                        chunk[ci] = stage5_challenges[chunk_start + ci];
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
                const raf_len = r_address_raf.len;
                const padded_len = ((raf_len + s6_log_k_chunk - 1) / s6_log_k_chunk) * s6_log_k_chunk;
                const pad_count = padded_len - raf_len;

                for (0..s6_ram_d) |i| {
                    var chunk = try self.allocator.alloc(F, s6_log_k_chunk);
                    const chunk_start = i * s6_log_k_chunk;
                    for (0..s6_log_k_chunk) |ci| {
                        const src_idx = chunk_start + ci;
                        chunk[ci] = if (src_idx < pad_count) F.zero() else r_address_raf[src_idx - pad_count];
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
            const eq_cycle = try stage6_helpers.computeEqTableParallel(F, self.allocator, r_cycle_le, s6_n_cycle_vars, self.thread_pool);
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

            // Iterate over all cycles to populate G_i (parallelized)
            // G_i(k) = Σ_j eq(r_cycle, j) · [addr_chunk_i(j) == k]
            {
                // Populate G tables (parallelized via map-reduce when pool available)
                const mask128: u128 = (@as(u128, 1) << @intCast(s6_log_k_chunk)) - 1;
                const steps_slice = trace.steps.items[0..T_val];
                const eq_cycle_slice = eq_cycle[0..T_val];

                const LocalG = [][]F;
                const MapCtx = struct {
                    allocator_inner: Allocator,
                    steps_inner: @TypeOf(steps_slice),
                    eq_cycle_inner: []const F,
                    N_inner: usize,
                    k_chunk_inner: usize,
                    instruction_d_inner: usize,
                    bytecode_d_inner: usize,
                    ram_d_inner: usize,
                    log_k_chunk_inner: usize,
                    mask128_inner: u128,
                    pc_map_inner: @TypeOf(pc_map),
                    mem_layout_inner: @TypeOf(memory_layout),
                };
                const map_ctx = MapCtx{
                    .allocator_inner = self.allocator,
                    .steps_inner = steps_slice,
                    .eq_cycle_inner = eq_cycle_slice,
                    .N_inner = N,
                    .k_chunk_inner = k_chunk,
                    .instruction_d_inner = s6_instruction_d,
                    .bytecode_d_inner = s6_bytecode_d,
                    .ram_d_inner = s6_ram_d,
                    .log_k_chunk_inner = s6_log_k_chunk,
                    .mask128_inner = mask128,
                    .pc_map_inner = pc_map,
                    .mem_layout_inner = memory_layout,
                };
                const mapFn = struct {
                    fn f(c: MapCtx, start: usize, end: usize) LocalG {
                        // Allocate thread-local G table
                        const local_G = c.allocator_inner.alloc([]F, c.N_inner) catch @panic("OOM allocating G-table");
                        for (0..c.N_inner) |i| {
                            local_G[i] = c.allocator_inner.alloc(F, c.k_chunk_inner) catch @panic("OOM allocating G-table row");
                            @memset(local_G[i], F.zero());
                        }
                        for (start..end) |j| {
                            const step = c.steps_inner[j];
                            const eq_j = c.eq_cycle_inner[j];
                            // InstructionRa
                            {
                                const lookup_idx = stage6_helpers.computeLookupIndex(step);
                                for (0..c.instruction_d_inner) |i| {
                                    const shift = c.log_k_chunk_inner * (c.instruction_d_inner - 1 - i);
                                    const chunk_val: usize = @intCast((lookup_idx >> @intCast(shift)) & c.mask128_inner);
                                    if (chunk_val < c.k_chunk_inner) {
                                        local_G[i][chunk_val] = local_G[i][chunk_val].add(eq_j);
                                    }
                                }
                            }
                            // BytecodeRa
                            {
                                const pc_idx = c.pc_map_inner.getPCForStep(step);
                                for (0..c.bytecode_d_inner) |i| {
                                    const chunk_val = stage6_helpers.extractChunkMSB(@intCast(pc_idx), i, c.bytecode_d_inner, c.log_k_chunk_inner);
                                    const ra_idx = c.instruction_d_inner + i;
                                    if (chunk_val < c.k_chunk_inner) {
                                        local_G[ra_idx][chunk_val] = local_G[ra_idx][chunk_val].add(eq_j);
                                    }
                                }
                            }
                            // RamRa
                            {
                                if (step.memory_addr) |addr| {
                                    if (addr != 0) {
                                        if (c.mem_layout_inner.remapAddress(addr)) |raddr| {
                                            for (0..c.ram_d_inner) |i| {
                                                const chunk_val = stage6_helpers.extractChunkMSB(raddr, i, c.ram_d_inner, c.log_k_chunk_inner);
                                                const ra_idx = c.instruction_d_inner + c.bytecode_d_inner + i;
                                                if (chunk_val < c.k_chunk_inner) {
                                                    local_G[ra_idx][chunk_val] = local_G[ra_idx][chunk_val].add(eq_j);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        return local_G;
                    }
                }.f;
                const reduceFn = struct {
                    fn f(a: LocalG, b: LocalG) LocalG {
                        if (a.len == 0) return b;
                        if (b.len == 0) return a;
                        // Merge b into a
                        for (0..a.len) |i| {
                            for (0..a[i].len) |k| {
                                a[i][k] = a[i][k].add(b[i][k]);
                            }
                        }
                        return a;
                    }
                }.f;
                const empty_g: LocalG = &[_][]F{};
                const result_g = pool_helpers.parallelReduceOptional(LocalG, self.thread_pool, T_val, empty_g, map_ctx, mapFn, reduceFn);
                // Copy result into G and free the reduce result
                if (result_g.len > 0) {
                    for (0..N) |i| {
                        @memcpy(G[i], result_g[i]);
                        self.allocator.free(result_g[i]);
                    }
                    self.allocator.free(result_g);
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
            var eq_bool = try stage6_helpers.computeEqTableParallel(F, self.allocator, r_addr_bool_le, s6_log_k_chunk, self.thread_pool);
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
                eq_virt[i] = try stage6_helpers.computeEqTableParallel(F, self.allocator, r_virt_le, s6_log_k_chunk, self.thread_pool);
                self.allocator.free(r_virt_le);
            }

            // Sample gamma from transcript (matches Jolt's HammingWeightClaimReductionParams::new)
            // IMPORTANT: Jolt's HW code calls transcript.challenge_scalar() which uses
            // challenge_scalar_128_bits() -> F::from_bytes() = from_le_bytes_mod_order().
            // This is the FULL field element path, NOT the 125-bit optimized path.
            // So we must use challengeScalarFull() here.
            const gamma = transcript.challengeScalarFull();
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
                const a2 = a2_num.mul(UniPoly.INV2);

                const coeffs = try self.allocator.alloc(F, degree_bound);
                coeffs[0] = p0; // a0 = p(0) = constant term
                coeffs[1] = a2; // a2 = quadratic coefficient
                try proof.compressed_polys.append(self.allocator, .{
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
            }

            // Cache opening claims: G_i(ρ) for each ra_i
            // G_i[0] is the final value after all bindings
            // Order: InstructionRa(0..inst_d), BytecodeRa(0..bc_d), RamRa(0..ram_d)
            var g_claims = try self.allocator.alloc(F, N);
            for (0..N) |i| {
                const g_claim = G[i][0];
                g_claims[i] = g_claim;
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
                    const eq_bool_check = try stage6_helpers.computeEqTable(F, self.allocator, r_addr_bool_be, s6_log_k_chunk);
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

            return Stage7Result(F){
                .g_claims = g_claims,
                .opening_point = opening_point_storage,
                .allocator = self.allocator,
            };
        }
    };
}
