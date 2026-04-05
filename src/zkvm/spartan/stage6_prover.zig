//! Stage 6 Batched Sumcheck Prover
//!
//! Stage 6 is a batched sumcheck with 6 instances:
//! 0. BytecodeReadRaf: bytecode_log_k + n_cycle_vars rounds, degree bytecode_d + 1
//! 1. Booleanity: log_k_chunk + n_cycle_vars rounds, degree 3 (input_claim = 0)
//! 2. HammingBooleanity: n_cycle_vars rounds, degree 3 (input_claim = 0)
//! 3. RamRaVirtual: n_cycle_vars rounds, degree ram_d + 1
//! 4. LookupsRaVirtual: n_cycle_vars rounds, degree n_committed_per_virtual + 1
//! 5. IncClaimReduction: n_cycle_vars rounds, degree 2
//!
//! ALL instances use real sumcheck provers with actual polynomial materialization
//! from execution trace data. No shortcuts, no placeholders.

const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;
// Stage 6 fine-grained bench timing — enabled at runtime via ZOLT_BENCH=1
const s6_bench_timing = true;

// Maximum evaluation points for parallelReduce accumulator.
// Covers all sub-provers: LookupsRa (M+2 ≤ 10), RamRa (d+2 ≤ 6), BytecodeReadRaf (d+2 ≤ 4).
const MAX_RA_EVALS = 16;

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const pool_helpers = @import("zolt_pool").helpers;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;

const poly_mod = @import("zolt_arith").poly;
const UniPoly = poly_mod.UniPoly;
const transcripts = @import("zolt_arith").transcripts;
const Blake2bTranscript = transcripts.Blake2bTranscript;
const jolt_types = @import("../jolt_types.zig");
const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof;
const OpeningClaims = jolt_types.OpeningClaims;
const OpeningId = jolt_types.OpeningId;
const tracer = @import("../../tracer/mod.zig");
const ExecutionTrace = tracer.ExecutionTrace;
const ram = @import("../ram/mod.zig");
const jolt_device = @import("../jolt_device.zig");
const instruction_mod = @import("../instruction/mod.zig");
const CircuitFlags = instruction_mod.CircuitFlags;
const InstructionFlags = instruction_mod.InstructionFlags;
const preprocessing = @import("../preprocessing.zig");
const BytecodePCMapper = preprocessing.BytecodePCMapper;
const ra_poly_mod = @import("ra_poly.zig");
const UnreducedProductAccum = @import("zolt_arith").field.UnreducedProductAccum;
const sumcheck_helpers = @import("sumcheck_helpers.zig");

// Helper functions — extracted to stage6_helpers.zig
const stage6_helpers = @import("stage6_helpers.zig");
pub const dropInBackground = stage6_helpers.dropInBackground;
pub const computeEqTable = stage6_helpers.computeEqTable;
pub const computeEqTableParallel = stage6_helpers.computeEqTableParallel;
pub const fieldFromI128 = stage6_helpers.fieldFromI128;
pub const extractChunkMSB = stage6_helpers.extractChunkMSB;
pub const interleaveBits = stage6_helpers.interleaveBits;
pub const computeLookupIndex = stage6_helpers.computeLookupIndex;
const addEvalsAsMonomialToCoeffs = stage6_helpers.addEvalsAsMonomialToCoeffs;
const addInstanceEvalsToCombibed = stage6_helpers.addInstanceEvalsToCombibed;
const addFixedEvalsToCombibed = stage6_helpers.addFixedEvalsToCombibed;
const getLookupChunkInterleaved = stage6_helpers.getLookupChunkInterleaved;
const decodeImmediateU64 = stage6_helpers.decodeImmediateU64;
const debugIncClaimReductionInit = stage6_helpers.debugIncClaimReductionInit;
const debugBytecodeReadRafFieldComparisons = stage6_helpers.debugBytecodeReadRafFieldComparisons;

// Bytecode entry construction — extracted to bytecode_entries.zig
pub const bytecode_entry_mod = @import("bytecode_entries.zig");
pub const BytecodeEntry = bytecode_entry_mod.BytecodeEntry;
pub const buildBytecodeEntries = bytecode_entry_mod.buildBytecodeEntries;
const hasLookupTable = bytecode_entry_mod.hasLookupTable;
const getLookupTableIndex = bytecode_entry_mod.getLookupTableIndex;

/// Result of Stage 6 sumcheck
pub fn Stage6Result(comptime F: type) type {
    return struct {
        const Self = @This();

        /// All sumcheck challenges (stage6_max_rounds elements)
        challenges: []F,

        /// BytecodeReadRaf opening claims: BytecodeRa(i) for i in 0..bytecode_d
        bytecode_ra_claims: []F,

        /// HammingBooleanity opening claim: RamHammingWeight
        hamming_weight_claim: F,

        /// Booleanity opening claims: all RA polys [InstructionRa(0..inst_d), BytecodeRa(0..bc_d), RamRa(0..ram_d)]
        booleanity_ra_claims: []F,

        /// RamRaVirtualization opening claims: RamRa(i) for i in 0..ram_d
        ram_ra_virtual_claims: []F,

        /// InstructionRaVirtualization opening claims: InstructionRa(i) for i in 0..instruction_d
        instruction_ra_virtual_claims: []F,

        /// IncClaimReduction opening claims: [RamInc, RdInc]
        ram_inc_claim: F,
        rd_inc_claim: F,

        /// Stage 6 configuration for Stage 7 opening point extraction
        bytecode_log_k: usize,
        log_k_chunk: usize,
        n_cycle_vars: usize,
        bytecode_d: usize,
        ram_d: usize,
        instruction_d: usize,

        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.challenges);
            self.allocator.free(self.bytecode_ra_claims);
            self.allocator.free(self.booleanity_ra_claims);
            self.allocator.free(self.ram_ra_virtual_claims);
            self.allocator.free(self.instruction_ra_virtual_claims);
        }
    };
}

// Instance provers extracted to stage6_instances.zig
const stage6_instances = @import("stage6_instances.zig");
pub const IncClaimReductionProver = stage6_instances.IncClaimReductionProver;
pub const HammingBooleanityProver = stage6_instances.HammingBooleanityProver;
pub const RamRaVirtualProver = stage6_instances.RamRaVirtualProver;
pub const BooleanityProver = stage6_instances.BooleanityProver;

// LookupsRaVirtual instance prover — extracted to stage6_instances.zig
pub const LookupsRaVirtualProver = stage6_instances.LookupsRaVirtualProver;

// BytecodeReadRaf instance prover — extracted to stage6_bytecode_raf.zig
const stage6_bytecode_raf = @import("stage6_bytecode_raf.zig");
const BytecodeReadRafProver = stage6_bytecode_raf.BytecodeReadRafProver;
pub const computeBytecodeReadRafInputClaim = stage6_bytecode_raf.computeBytecodeReadRafInputClaim;

// =============================================================================
// Stage 6 Batched Sumcheck Prover (Main)
// =============================================================================
pub fn Stage6BatchedProver(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// GPU-accelerated bindLow: arr[j] = arr[2j] + r*(arr[2j+1] - arr[2j])
        /// Falls back to CPU when GPU unavailable or array too small.
        fn gpuBindLow(arr: []F, half: usize, r: F, gpu_ops: ?*GpuPolyOps) void {
            if (gpu_ops) |gpu| {
                if (half >= 16384) {
                    gpu.polyBindLow(arr[0 .. half * 2], r, arr[0..half]) catch {
                        cpuBindLow(arr, half, r);
                        return;
                    };
                    return;
                }
            }
            cpuBindLow(arr, half, r);
        }

        fn cpuBindLow(arr: []F, half: usize, r: F) void {
            for (0..half) |j| {
                arr[j] = arr[2 * j].add(r.montgomeryMul(arr[2 * j + 1].sub(arr[2 * j])));
            }
        }

        /// Generate Stage 6 batched sumcheck proof with real polynomial evaluation
        pub fn generateStage6Proof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            transcript: *Blake2bTranscript(F),
            opening_claims: *OpeningClaims(F),
            // Parameters
            n_cycle_vars: usize,
            bytecode_log_k: usize,
            log_k_chunk: usize,
            bytecode_d: usize,
            ram_d: usize,
            instruction_d: usize,
            lookups_ra_virtual_log_k_chunk: usize,
            // Execution trace
            trace: *const ExecutionTrace,
            // Opening points for BytecodeReadRaf (all BIG_ENDIAN)
            r_cycle_bc1_spartan_outer: []const F,
            r_cycle_bc2_product_virt: []const F,
            r_cycle_bc3_spartan_shift: []const F,
            r_cycle_bc4_regs_rwc: []const F,
            r_cycle_bc5_regs_val: []const F,
            // Opening points for IncClaimReduction (all BIG_ENDIAN)
            r_cycle_inc_ram_rwc: []const F, // RamReadWriteChecking
            r_cycle_inc_ram_val: []const F, // RamValEvaluation
            // Stage 5 challenges for deriving LookupsRaVirtual and RamRaVirtual points
            stage5_challenges: []const F,
            // RAM r_address from Stage 2 (BIG_ENDIAN) — the aligned address used by RamRaClaimReduction
            ram_r_address_stage2_be: []const F,
            // Memory layout for address remapping
            memory_layout: *const jolt_device.MemoryLayout,
            // Bytecode entry table for Val polynomial computation
            bytecode_entries: []const BytecodeEntry,
            // Register address opening points for Stages 4 and 5 (BIG_ENDIAN)
            r_register_4: []const F, // From RegistersReadWriteChecking (address portion)
            r_register_5: []const F, // From RegistersValEvaluation (address portion)
            // BytecodePCMapper for converting ELF addresses to bytecode array indices
            pc_map: *const BytecodePCMapper,
            entry_address: u64,
            // Stage 4 inc_poly copy for diagnostic comparison (pass null slice to skip)
            stage4_inc_poly_copy: []const F,
        ) !Stage6Result(F) {
            // Instance round counts
            const bytecodeReadRaf_rounds = bytecode_log_k + n_cycle_vars;
            const hammingBooleanity_rounds = n_cycle_vars;
            const booleanity_rounds = log_k_chunk + n_cycle_vars;
            const ramRaVirtual_rounds = n_cycle_vars;
            const lookupsRaVirtual_rounds = n_cycle_vars;
            const incClaimReduction_rounds = n_cycle_vars;

            const max_num_rounds = bytecodeReadRaf_rounds;

            // Instance degrees
            const bytecodeReadRaf_degree = bytecode_d + 1;
            const hammingBooleanity_degree: usize = 3;
            const booleanity_degree: usize = 3;
            const ramRaVirtual_degree = ram_d + 1;
            const n_committed_per_virtual = lookups_ra_virtual_log_k_chunk / log_k_chunk;
            const n_virtual_ra_polys = 128 / lookups_ra_virtual_log_k_chunk;
            const lookupsRaVirtual_degree = n_committed_per_virtual + 1;
            const incClaimReduction_degree: usize = 2;

            const max_degree = @max(
                @max(@max(bytecodeReadRaf_degree, hammingBooleanity_degree), @max(booleanity_degree, ramRaVirtual_degree)),
                @max(lookupsRaVirtual_degree, incClaimReduction_degree),
            );

            dbg("[STAGE6] Configuration:\n", .{});
            dbg("  bytecodeReadRaf: {} rounds (addr={}, cycle={}), degree {}\n", .{ bytecodeReadRaf_rounds, bytecode_log_k, n_cycle_vars, bytecodeReadRaf_degree });
            dbg("  hammingBooleanity: {} rounds, degree {}\n", .{ hammingBooleanity_rounds, hammingBooleanity_degree });
            dbg("  booleanity: {} rounds, degree {}\n", .{ booleanity_rounds, booleanity_degree });
            dbg("  ramRaVirtual: {} rounds, degree {}\n", .{ ramRaVirtual_rounds, ramRaVirtual_degree });
            dbg("  lookupsRaVirtual: {} rounds, degree {}\n", .{ lookupsRaVirtual_rounds, lookupsRaVirtual_degree });
            dbg("  incClaimReduction: {} rounds, degree {}\n", .{ incClaimReduction_rounds, incClaimReduction_degree });
            dbg("  max_num_rounds: {}, max_degree: {}\n", .{ max_num_rounds, max_degree });

            // ====================================================================
            // Sample gammas (must match Jolt verifier)
            // ====================================================================

            // Debug: dump transcript state at Stage 6 entry
            if (comptime debug_verbose) {
                dbg("[STAGE6] Transcript state at entry: {{ ", .{});
                for (transcript.state) |b| dbg("{x:0>2} ", .{b});
                dbg("}}, round={}\n", .{transcript.n_rounds});
            }

            dbg("[STAGE6] Transcript at entry: round={}\n", .{transcript.n_rounds});
            const bytecode_raf_gamma_powers = try transcript.challengeScalarPowers(self.allocator, 8);
            defer self.allocator.free(bytecode_raf_gamma_powers);

            // Debug: print first gamma to verify transcript sync
            {
                const g0_be = bytecode_raf_gamma_powers[1].toBytesBE(); // [1] is gamma itself
                dbg("[STAGE6] bytecodeRaf_gamma = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    g0_be[31], g0_be[30], g0_be[29], g0_be[28], g0_be[27], g0_be[26], g0_be[25], g0_be[24],
                });
            }

            const NUM_CIRCUIT_FLAGS: usize = 14;
            const stage1_gammas = try transcript.challengeScalarPowers(self.allocator, 2 + NUM_CIRCUIT_FLAGS);
            defer self.allocator.free(stage1_gammas);

            const stage2_gammas = try transcript.challengeScalarPowers(self.allocator, 4);
            defer self.allocator.free(stage2_gammas);

            const stage3_gammas = try transcript.challengeScalarPowers(self.allocator, 9);
            defer self.allocator.free(stage3_gammas);

            const stage4_gammas = try transcript.challengeScalarPowers(self.allocator, 3);
            defer self.allocator.free(stage4_gammas);

            const NUM_LOOKUP_TABLES: usize = 40;
            const stage5_gammas = try transcript.challengeScalarPowers(self.allocator, 2 + NUM_LOOKUP_TABLES);
            defer self.allocator.free(stage5_gammas);

            dbg("[STAGE6] Sampled BytecodeReadRaf gammas\n", .{});

            // BooleanitySumcheckParams::new() - conditional extra challenges
            // When Stage 5 address variables < log_k_chunk, Jolt samples extra challenges
            // to pad r_address to log_k_chunk length. This happens when LOOKUPS_LOG_K is
            // smaller than log_k_chunk, which doesn't happen in practice (128 > 4).
            if (lookups_ra_virtual_log_k_chunk < log_k_chunk) {
                const extra_count = log_k_chunk - lookups_ra_virtual_log_k_chunk;
                for (0..extra_count) |_| {
                    _ = transcript.challengeScalar();
                }
            }
            // Jolt samples 1 gamma via challenge_scalar_optimized() and derives powers:
            //   gamma_powers_square[i] = γ^(2i) for i = 0..total_d
            // The prover uses gamma_powers[i] = γ^i internally for polynomial scaling,
            // and the verifier uses gamma_powers_square[i] = γ^(2i) for expected_output_claim.
            const total_d = instruction_d + bytecode_d + ram_d;
            const booleanity_gamma = transcript.challengeScalar();
            // Handle degenerate gamma=0 case (same as Jolt: replace with 1)
            const booleanity_gamma_f: F = if (booleanity_gamma.isZero()) F.one() else booleanity_gamma;
            const booleanity_gamma_sq = booleanity_gamma_f.mul(booleanity_gamma_f);
            const booleanity_gammas = try self.allocator.alloc(F, total_d);
            booleanity_gammas[0] = F.one(); // γ^0 = 1
            for (1..total_d) |i| {
                booleanity_gammas[i] = booleanity_gammas[i - 1].mul(booleanity_gamma_sq); // γ^(2i)
            }
            // Also compute γ^i powers for Phase 2 pre-scaling optimization
            const booleanity_gamma_unsq = try self.allocator.alloc(F, total_d);
            booleanity_gamma_unsq[0] = F.one(); // γ^0 = 1
            for (1..total_d) |i| {
                booleanity_gamma_unsq[i] = booleanity_gamma_unsq[i - 1].mul(booleanity_gamma_f); // γ^i
            }

            // LookupsRa::new() - gamma powers for virtual RA batching
            const lookups_ra_gamma_powers = try transcript.challengeScalarPowers(self.allocator, n_virtual_ra_polys);
            defer self.allocator.free(lookups_ra_gamma_powers);
            {
                dbg("[STAGE6] lookups_ra_gamma_powers:\n", .{});
                for (0..@min(n_virtual_ra_polys, 4)) |i| {
                    const gp_le = lookups_ra_gamma_powers[i].toBytes();
                    dbg("  gamma_powers[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        i, gp_le[0], gp_le[1], gp_le[2], gp_le[3], gp_le[4], gp_le[5], gp_le[6], gp_le[7],
                    });
                }
            }

            // IncClaimReduction::new() - gamma
            // Jolt uses challenge_scalar() (FULL 128-bit) for inc gamma, not optimized
            const inc_gamma = transcript.challengeScalarFull();

            // ====================================================================
            // Compute input claims
            // ====================================================================

            const bcraf_result = computeBytecodeReadRafInputClaim(
                F,
                opening_claims,
                bytecode_raf_gamma_powers,
                stage1_gammas,
                stage2_gammas,
                stage3_gammas,
                stage4_gammas,
                stage5_gammas,
            );
            var bytecodeReadRaf_input = bcraf_result.total.add(bytecode_raf_gamma_powers[7]);
            const bcraf_per_stage_claims = bcraf_result.per_stage;

            const hammingBooleanity_input = F.zero();
            const booleanity_input = F.zero();

            const ramRaVirtual_input = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRaClaimReduction } },
            ) orelse F.zero();

            var lookupsRaVirtual_input = F.zero();
            for (0..n_virtual_ra_polys) |i| {
                const ra_claim = opening_claims.get(
                    .{ .Virtual = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionReadRaf } },
                ) orelse F.zero();
                lookupsRaVirtual_input = lookupsRaVirtual_input.add(lookups_ra_gamma_powers[i].mul(ra_claim));
            }

            const inc_gamma2 = inc_gamma.mul(inc_gamma);
            const inc_gamma3 = inc_gamma2.mul(inc_gamma);

            const v1_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamReadWriteChecking } },
            ) orelse F.zero();
            const v2_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } },
            ) orelse F.zero();
            const w1_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } },
            ) orelse F.zero();
            const w2_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersValEvaluation } },
            ) orelse F.zero();

            // Debug: dump inc_gamma and individual claims
            {
                const ig_be = inc_gamma.toBytesBE();
                const v1_be = v1_claim.toBytesBE();
                const v2_be = v2_claim.toBytesBE();
                const w1_be = w1_claim.toBytesBE();
                const w2_be = w2_claim.toBytesBE();
                dbg("[STAGE6] inc_gamma = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    ig_be[31], ig_be[30], ig_be[29], ig_be[28], ig_be[27], ig_be[26], ig_be[25], ig_be[24],
                });
                dbg("[STAGE6] IncClaim v1(RamInc@RamRWC) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    v1_be[31], v1_be[30], v1_be[29], v1_be[28], v1_be[27], v1_be[26], v1_be[25], v1_be[24],
                });
                dbg("[STAGE6] IncClaim v2(RamInc@RamVal) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    v2_be[31], v2_be[30], v2_be[29], v2_be[28], v2_be[27], v2_be[26], v2_be[25], v2_be[24],
                });
                dbg("[STAGE6] IncClaim w1(RdInc@RegsRWC) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    w1_be[31], w1_be[30], w1_be[29], w1_be[28], w1_be[27], w1_be[26], w1_be[25], w1_be[24],
                });
                dbg("[STAGE6] IncClaim w2(RdInc@RegsVal) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    w2_be[31], w2_be[30], w2_be[29], w2_be[28], w2_be[27], w2_be[26], w2_be[25], w2_be[24],
                });
            }

            const incClaimReduction_input = v1_claim
                .add(inc_gamma.mul(v2_claim))
                .add(inc_gamma2.mul(w1_claim))
                .add(inc_gamma3.mul(w2_claim));

            dbg("[STAGE6] Input claims (LE first 8):\n", .{});
            // Print components for IncClaimReduction
            {
                const v1_be = v1_claim.toBytesBE();
                const v2_be = v2_claim.toBytesBE();
                const w1_be = w1_claim.toBytesBE();
                const w2_be = w2_claim.toBytesBE();
                dbg("  IncClaim components: v1=[{x:0>2},{x:0>2},...] v2=[{x:0>2},{x:0>2},...] w1=[{x:0>2},{x:0>2},...] w2=[{x:0>2},{x:0>2},...]\n", .{
                    v1_be[31], v1_be[30], v2_be[31], v2_be[30], w1_be[31], w1_be[30], w2_be[31], w2_be[30],
                });
            }
            // Print LookupsRa claims
            for (0..@min(n_virtual_ra_polys, 4)) |i| {
                const ra_c = opening_claims.get(
                    .{ .Virtual = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionReadRaf } },
                ) orelse F.zero();
                const ra_be = ra_c.toBytesBE();
                dbg("  InstructionRa[{}] = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    i, ra_be[31], ra_be[30], ra_be[29], ra_be[28], ra_be[27], ra_be[26], ra_be[25], ra_be[24],
                });
            }
            // Print BytecodeReadRaf components
            {
                const bc_be = bytecodeReadRaf_input.toBytesBE();
                dbg("  bytecodeReadRaf_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    bc_be[31], bc_be[30], bc_be[29], bc_be[28], bc_be[27], bc_be[26], bc_be[25], bc_be[24],
                });
            }
            {
                const ram_be = ramRaVirtual_input.toBytesBE();
                dbg("  ramRaVirtual_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    ram_be[31], ram_be[30], ram_be[29], ram_be[28], ram_be[27], ram_be[26], ram_be[25], ram_be[24],
                });
            }
            {
                const look_be = lookupsRaVirtual_input.toBytesBE();
                dbg("  lookupsRaVirtual_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    look_be[31], look_be[30], look_be[29], look_be[28], look_be[27], look_be[26], look_be[25], look_be[24],
                });
            }
            {
                const inc_be = incClaimReduction_input.toBytesBE();
                dbg("  incClaimReduction_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    inc_be[31], inc_be[30], inc_be[29], inc_be[28], inc_be[27], inc_be[26], inc_be[25], inc_be[24],
                });
            }

            // ====================================================================
            // Derive opening points for RamRaVirtual and LookupsRaVirtual from Stage 5
            // ====================================================================

            const LOOKUPS_LOG_K: usize = 128;
            const ram_log_k: usize = ram_r_address_stage2_be.len;

            // RamRaVirtual: r_cycle from Stage 5 RamRaClaimReduction, r_address from Stage 2
            // RamRaClaimReduction is cycle-only (log_T rounds), NOT address+cycle.
            // The r_address comes from Stage 2's aligned RAM address, stored in ram_r_address_stage2_be.
            const stage5_max_rounds = LOOKUPS_LOG_K + n_cycle_vars;
            // RamRaClaimReduction has n_cycle_vars rounds (cycle-only), offset = stage5_max - n_cycle_vars
            const ram_ra_offset = stage5_max_rounds - n_cycle_vars;
            dbg("[STAGE6] RamRa challenge offset: stage5_max={}, ram_ra_rounds={}, offset={}\n", .{
                stage5_max_rounds, n_cycle_vars, ram_ra_offset,
            });
            var ram_ra_r_cycle = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(ram_ra_r_cycle);
            for (0..n_cycle_vars) |i| {
                // Reverse cycle part: challenges[offset..offset+n_cycle_vars] reversed (BE)
                ram_ra_r_cycle[i] = stage5_challenges[ram_ra_offset + n_cycle_vars - 1 - i];
            }

            // r_address for RamRa: from Stage 2 aligned RAM address (already BIG_ENDIAN)
            // Pad with leading zeros to make length a multiple of log_k_chunk (matching Jolt's compute_r_address_chunks)
            const padded_ram_len = ((ram_log_k + log_k_chunk - 1) / log_k_chunk) * log_k_chunk;
            var ram_ra_r_address_be: []F = undefined;
            var ram_ra_r_address_allocated = false;
            if (padded_ram_len != ram_log_k) {
                ram_ra_r_address_be = try self.allocator.alloc(F, padded_ram_len);
                ram_ra_r_address_allocated = true;
                const pad_count = padded_ram_len - ram_log_k;
                @memset(ram_ra_r_address_be[0..pad_count], F.zero());
                @memcpy(ram_ra_r_address_be[pad_count..], ram_r_address_stage2_be);
            } else {
                ram_ra_r_address_be = @constCast(ram_r_address_stage2_be);
            }
            defer if (ram_ra_r_address_allocated) self.allocator.free(ram_ra_r_address_be);

            // Split r_address into chunks (BIG_ENDIAN, chunk[0] = MSB)
            var ram_ra_addr_chunks = try self.allocator.alloc([]const F, ram_d);
            defer self.allocator.free(ram_ra_addr_chunks);
            for (0..ram_d) |i| {
                const chunk_start = i * log_k_chunk;
                const chunk_end = chunk_start + log_k_chunk;
                ram_ra_addr_chunks[i] = ram_ra_r_address_be[chunk_start..chunk_end];
            }

            // LookupsRaVirtual: r_cycle and r_addr_chunks from InstructionReadRaf (Stage 5 Instance 1)
            // InstructionReadRaf has LOOKUPS_LOG_K + n_cycle_vars = 136 rounds
            // normalize_opening_point: address NOT reversed, cycle IS reversed
            var lookups_ra_r_cycle = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(lookups_ra_r_cycle);
            for (0..n_cycle_vars) |i| {
                // Reverse cycle part: challenges[128..136] reversed
                lookups_ra_r_cycle[i] = stage5_challenges[LOOKUPS_LOG_K + n_cycle_vars - 1 - i];
            }
            // Debug: print lookups_ra_r_cycle to compare with Jolt's r_cycle
            for (0..n_cycle_vars) |dbg_i| {
                const dbg_b = lookups_ra_r_cycle[dbg_i].toBytesBE();
                dbg("[S6_RCYCLE] lookups_ra_r_cycle[{}] LE=[", .{dbg_i});
                for (0..8) |bi| dbg("{x:0>2}", .{dbg_b[31 - bi]});
                dbg("]\n", .{});
            }

            // r_address for Lookups: challenges[0..128] NOT reversed (stays LITTLE_ENDIAN)
            // Then compute_r_address_chunks splits into log_k_chunk-sized pieces
            var lookups_ra_addr_chunks = try self.allocator.alloc([]const F, instruction_d);
            defer self.allocator.free(lookups_ra_addr_chunks);
            for (0..instruction_d) |i| {
                const chunk_start = i * log_k_chunk;
                const chunk_end = @min(chunk_start + log_k_chunk, LOOKUPS_LOG_K);
                lookups_ra_addr_chunks[i] = stage5_challenges[chunk_start..chunk_end];
            }

            // ====================================================================
            // Initialize ALL sumcheck instances
            // ====================================================================
            const bench_s6 = (std.posix.getenv("ZOLT_BENCH") != null);
            const t_s6_overall_start = if (bench_s6) std.time.nanoTimestamp() else 0;
            var s6_init_timer: if (s6_bench_timing) std.time.Timer else void = if (comptime s6_bench_timing) std.time.Timer.start() catch unreachable else {};

            // Instance 5: IncClaimReduction (degree 2)
            // IncClaimReduction uses RAM r_cycles (not BytecodeReadRaf r_cycles)
            const t_init_inc = if (bench_s6) std.time.nanoTimestamp() else 0;
            var inc_prover = try IncClaimReductionProver(F).init(
                self.allocator,
                trace,
                inc_gamma,
                r_cycle_inc_ram_rwc,
                r_cycle_inc_ram_val,
                r_cycle_bc4_regs_rwc,
                r_cycle_bc5_regs_val,
                self.thread_pool,
            );
            inc_prover.gpu = self.gpu_ops;
            defer inc_prover.deinit();
            const t_after_inc = if (bench_s6) std.time.nanoTimestamp() else 0;

            // Diagnostic: compare IncClaimReduction inc_poly and verify component sums
            try debugIncClaimReductionInit(
                F,
                self.allocator,
                n_cycle_vars,
                inc_prover.rd_inc,
                inc_prover.ram_inc,
                @as(usize, 1) << @intCast(n_cycle_vars),
                stage4_inc_poly_copy,
                r_cycle_inc_ram_rwc,
                r_cycle_inc_ram_val,
                r_cycle_bc4_regs_rwc,
                r_cycle_bc5_regs_val,
                v1_claim,
                v2_claim,
                w1_claim,
                w2_claim,
                trace,
            );

            // Instance 1: HammingBooleanity (degree 3)
            const t_init_hamming = if (bench_s6) std.time.nanoTimestamp() else 0;
            var hamming_prover = try HammingBooleanityProver(F).init(
                self.allocator,
                trace,
                r_cycle_bc1_spartan_outer,
                self.thread_pool,
            );
            hamming_prover.gpu = self.gpu_ops;
            defer hamming_prover.deinit();
            const t_after_hamming = if (bench_s6) std.time.nanoTimestamp() else 0;

            // Instance 3: RamRaVirtual (degree ram_d+1)
            const t_init_ram = if (bench_s6) std.time.nanoTimestamp() else 0;
            var ram_ra_prover = try RamRaVirtualProver(F).init(
                self.allocator,
                trace,
                ram_ra_r_cycle,
                ram_ra_addr_chunks,
                ram_d,
                memory_layout,
                log_k_chunk,
                self.thread_pool,
            );
            ram_ra_prover.gpu = self.gpu_ops;
            defer ram_ra_prover.deinit();
            const t_after_ram = if (bench_s6) std.time.nanoTimestamp() else 0;

            // Instance 4: LookupsRaVirtual (degree n_committed_per_virtual+1)
            const t_init_lookups = if (bench_s6) std.time.nanoTimestamp() else 0;
            var lookups_ra_prover = try LookupsRaVirtualProver(F).init(
                self.allocator,
                trace,
                lookups_ra_r_cycle,
                lookups_ra_addr_chunks,
                lookups_ra_gamma_powers,
                n_committed_per_virtual,
                n_virtual_ra_polys,
                log_k_chunk,
                instruction_d,
                self.thread_pool,
            );
            lookups_ra_prover.gpu = self.gpu_ops;
            defer lookups_ra_prover.deinit();
            const t_after_lookups = if (bench_s6) std.time.nanoTimestamp() else 0;

            // Verify: eq table partition of unity (Σ eq[j] = 1)
            if (comptime debug_verbose) {
                var eq_sum = F.zero();
                for (0..lookups_ra_prover.current_len) |j| {
                    eq_sum = eq_sum.add(lookups_ra_prover.e_out[j]);
                }
                dbg("[LR_EQ] Σeq==1? {} T={}\n", .{ eq_sum.eql(F.one()), lookups_ra_prover.current_len });
            }

            // Instance 2: Booleanity (degree 3, two-phase)
            const t_init_booleanity = if (bench_s6) std.time.nanoTimestamp() else 0;
            var booleanity_prover = try stage6_instances.initBooleanityProver(
                F,
                self.allocator,
                self.thread_pool,
                self.gpu_ops,
                trace,
                stage5_challenges,
                lookups_ra_r_cycle,
                booleanity_gammas,
                booleanity_gamma_unsq,
                instruction_d,
                bytecode_d,
                ram_d,
                log_k_chunk,
                n_cycle_vars,
                memory_layout,
                pc_map,
            );
            defer booleanity_prover.deinit();
            const t_after_booleanity = if (bench_s6) std.time.nanoTimestamp() else 0;

            // Instance 0: BytecodeReadRaf (degree bytecode_d+1)
            // Compute Val polynomials from bytecode entries and stage gammas
            const bytecode_K: usize = @as(usize, 1) << @intCast(bytecode_log_k);
            var bytecode_val_polys: [5][]F = undefined;

            // Precompute eq tables for Stages 4 and 5 register addresses
            // r_register_4 and r_register_5 are the address portions from
            // RegistersReadWriteChecking and RegistersValEvaluation opening points
            const REGISTER_COUNT_LOG2: usize = 7; // log2(128 registers: 32 RISC-V + 96 virtual)
            dbg("[STAGE6] r_register_4 (len={}):\n", .{r_register_4.len});
            for (r_register_4, 0..) |rv, i| {
                dbg("  r_register_4[{}] mont_limbs=[0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{ i, rv.limbs[0], rv.limbs[1], rv.limbs[2], rv.limbs[3] });
            }
            dbg("[STAGE6] r_register_5 (len={}):\n", .{r_register_5.len});
            for (r_register_5, 0..) |rv, i| {
                dbg("  r_register_5[{}] mont_limbs=[0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{ i, rv.limbs[0], rv.limbs[1], rv.limbs[2], rv.limbs[3] });
            }
            // Jolt's EqPolynomial::evals uses BIG-ENDIAN bit indexing:
            // r[0] maps to MSB of index, r[n-1] maps to LSB.
            // Our computeEqTable uses LITTLE-ENDIAN: r[0] maps to LSB.
            // Fix: reverse the input array so our LE computation produces BE-indexed results.
            var r_register_4_rev = try self.allocator.alloc(F, r_register_4.len);
            defer self.allocator.free(r_register_4_rev);
            for (0..r_register_4.len) |i| {
                r_register_4_rev[i] = r_register_4[r_register_4.len - 1 - i];
            }
            var r_register_5_rev = try self.allocator.alloc(F, r_register_5.len);
            defer self.allocator.free(r_register_5_rev);
            for (0..r_register_5.len) |i| {
                r_register_5_rev[i] = r_register_5[r_register_5.len - 1 - i];
            }
            const eq_table_4 = try computeEqTable(F, self.allocator, r_register_4_rev, REGISTER_COUNT_LOG2);
            defer self.allocator.free(eq_table_4);
            const eq_table_5 = try computeEqTable(F, self.allocator, r_register_5_rev, REGISTER_COUNT_LOG2);
            defer self.allocator.free(eq_table_5);
            // Print eq_table_4 entries in LE hex for comparison with Jolt
            dbg("[STAGE6] eq_table_4 (len={}):\n", .{eq_table_4.len});
            for ([_]usize{ 0, 1, 2, 8, 10, 15, 31, 127 }) |idx| {
                if (idx < eq_table_4.len) {
                    const vbe = eq_table_4[idx].toBytesBE();
                    dbg("  eq4[{}]_LE=[", .{idx});
                    for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
                    dbg("]\n", .{});
                }
            }
            // Print stage4_gammas in LE hex
            dbg("[STAGE6] stage4_gammas:\n", .{});
            for (0..3) |i| {
                const gbe = stage4_gammas[i].toBytesBE();
                dbg("  gamma4[{}]_LE=[", .{i});
                for (0..32) |bi| dbg("{x:0>2}", .{gbe[31 - bi]});
                dbg("]\n", .{});
            }

            for (0..5) |s| {
                bytecode_val_polys[s] = try self.allocator.alloc(F, bytecode_K);
                @memset(bytecode_val_polys[s], F.zero());
            }

            for (0..bytecode_K) |k| {
                if (k >= bytecode_entries.len) break;
                const entry = bytecode_entries[k];

                // Stage 1: unexpanded_pc + γ₁¹·imm + Σ γ₁^(2+i)·circuit_flag_i
                // CRITICAL: The Imm encoding must match Jolt's vanilla verifier exactly.
                // Jolt's NormalizedOperands.imm is i128, but how it gets there depends
                // on the instruction FORMAT type:
                //   FormatI (I-type): u64 as i128 → zero-extended (always positive)
                //   FormatU (U-type): u64 as i128 → zero-extended (always positive)
                //   FormatJ (J-type): u64 as i128 → zero-extended (always positive)
                //   FormatB (B-type): i128 directly → signed
                //   FormatS (S-type): i64 as i128 → sign-extended (signed)
                //   Virtual (0x0B, 0x2B): u64 as i128 (from emit_i helper)
                // Then Jolt calls from_i128(operands.imm) to get the field element.
                const imm_field: F = blk: {
                    const opcode_for_imm = entry.opcode;
                    // Jolt stores imm as i128 in NormalizedOperands, then uses from_i128().
                    // The i128 value depends on the instruction format's source type:
                    //   FormatI (u64): u64 as i128 → zero-extended (always positive)
                    //   FormatU (u64): u64 as i128 → zero-extended (always positive)
                    //   FormatJ (u64): u64 as i128 → zero-extended (always positive)
                    //   FormatB (i128): direct → can be negative
                    //   FormatS (i64): i64 as i128 → sign-extended (can be negative)
                    //   FormatLoad (i64): i64 as i128 → sign-extended (can be negative)
                    // We must match: signed formats use fieldFromI128, unsigned use fromU64.
                    // Signed encoding: must match R1CS witness and Jolt verifier.
                    const is_signed_format = (opcode_for_imm == 0x63) or // B-type (branches: FormatB i128)
                        (opcode_for_imm == 0x23) or // S-type (stores: FormatS i64)
                        (opcode_for_imm == 0x03) or // Load (FormatLoad: i64 sign-extended to i128)
                        (opcode_for_imm == 0x22); // VirtualAssert (FormatAssert: signed i64)
                    if (is_signed_format) {
                        break :blk fieldFromI128(F, @as(i128, entry.imm));
                    } else {
                        // I-type, U-type, J-type, Virtual: u64 zero-extended to i128.
                        // from_i128(u64 as i128) = from_u64(u64), so fromU64(@bitCast) matches.
                        break :blk F.fromU64(@as(u64, @bitCast(entry.imm)));
                    }
                };
                var val1 = F.fromU64(entry.address); // No gamma[0] - Jolt formula: unexpanded_pc + γ¹·imm + Σγ^(2+i)·cf[i]
                val1 = val1.add(stage1_gammas[1].mul(imm_field));
                for (0..14) |i| {
                    if (entry.circuit_flags[i]) {
                        val1 = val1.add(stage1_gammas[2 + i]);
                    }
                }
                bytecode_val_polys[0][k] = val1;

                // Debug: print details for mismatching entries
                if (k == 3 or k == 4 or k == 10 or k == 16 or k == 18 or k == 27 or k == 29 or k == 35) {
                    const addr_be = F.fromU64(entry.address).toBytesBE();
                    const imm_be = imm_field.toBytesBE();
                    dbg("[ZOLT_BC_ENTRY] k={}: addr=0x{x:0>8} imm_LE=[", .{ k, entry.address });
                    for (0..8) |bi| dbg("{x:0>2}", .{imm_be[31 - bi]});
                    dbg("] opcode=0x{x:0>2} raw_imm={} cf=[", .{ entry.opcode, entry.imm });
                    for (0..14) |ci| {
                        if (entry.circuit_flags[ci]) dbg("1", .{}) else dbg("0", .{});
                    }
                    dbg("]\n", .{});
                    _ = addr_be;
                }

                // Stage 2: γ₂⁰·jump + γ₂¹·branch + γ₂²·write_lookup_to_rd + γ₂³·virtual_instruction
                // Matches upstream a16z/jolt (no IsRdNotZero — that was fork-only)
                var val2 = F.zero();
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.Jump)]) {
                    val2 = val2.add(stage2_gammas[0]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.Branch)]) {
                    val2 = val2.add(stage2_gammas[1]);
                }
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)]) {
                    val2 = val2.add(stage2_gammas[2]);
                }
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)]) {
                    val2 = val2.add(stage2_gammas[3]);
                }
                bytecode_val_polys[1][k] = val2;

                // Stage 3: γ₃⁰·imm + γ₃¹·unexpanded_pc + γ₃²·L_is_rs1 + γ₃³·L_is_pc
                //         + γ₃⁴·R_is_rs2 + γ₃⁵·R_is_imm + γ₃⁶·is_noop
                //         + γ₃⁷·virtual_instruction + γ₃⁸·is_first_in_sequence
                // Uses same signed Imm encoding as Stage 1 (see comment above)
                var val3 = imm_field; // No gamma[0] - Jolt formula: imm + γ¹·unexpanded_pc + Σγ^(2+i)·flags[i]
                val3 = val3.add(stage3_gammas[1].mul(F.fromU64(entry.address)));
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)]) {
                    val3 = val3.add(stage3_gammas[2]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.LeftOperandIsPC)]) {
                    val3 = val3.add(stage3_gammas[3]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)]) {
                    val3 = val3.add(stage3_gammas[4]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.RightOperandIsImm)]) {
                    val3 = val3.add(stage3_gammas[5]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.IsNoop)]) {
                    val3 = val3.add(stage3_gammas[6]);
                }
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)]) {
                    val3 = val3.add(stage3_gammas[7]);
                }
                if (entry.is_first_in_sequence) {
                    val3 = val3.add(stage3_gammas[8]);
                }
                bytecode_val_polys[2][k] = val3;

                // Stage 4: γ₄⁰·eq(rd, r_reg4) + γ₄¹·eq(rs1, r_reg4) + γ₄²·eq(rs2, r_reg4)
                const REGISTER_COUNT: usize = 128; // 32 RISC-V + 96 virtual
                var val4 = F.zero();
                if (entry.rd < REGISTER_COUNT) {
                    val4 = val4.add(stage4_gammas[0].mul(eq_table_4[entry.rd]));
                }
                if (entry.rs1 < REGISTER_COUNT) {
                    val4 = val4.add(stage4_gammas[1].mul(eq_table_4[entry.rs1]));
                }
                if (entry.rs2 < REGISTER_COUNT) {
                    val4 = val4.add(stage4_gammas[2].mul(eq_table_4[entry.rs2]));
                }
                bytecode_val_polys[3][k] = val4;

                // Stage 5: eq(rd, r_reg5) + γ₅¹·!is_interleaved + Σ γ₅^(2+i)·table_flag_i
                var val5 = F.zero();
                if (entry.rd < REGISTER_COUNT) {
                    val5 = val5.add(eq_table_5[entry.rd]);
                }
                if (!entry.is_interleaved) {
                    val5 = val5.add(stage5_gammas[1]);
                }
                if (entry.lookup_table_index < 40) {
                    val5 = val5.add(stage5_gammas[2 + @as(usize, entry.lookup_table_index)]);
                }
                bytecode_val_polys[4][k] = val5;
            }

            // Debug: Print Stage 3 Val poly for comparison with Jolt verifier
            if (comptime debug_verbose) {
                dbg("[STAGE6] Val[3] (Stage 4/RegistersRWC) entries:\n", .{});
                for (0..bytecode_K) |k| {
                    const vbe = bytecode_val_polys[3][k].toBytesBE();
                    dbg("  Val[3][{}]_LE=[", .{k});
                    for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
                    dbg("]\n", .{});
                }
            }
            if (debug_verbose) {
                for ([_]usize{ 0, 1, 2, 4 }) |s| {
                    for (0..bytecode_K) |k| {
                        const vbe = bytecode_val_polys[s][k].toBytesBE();
                        dbg("  Val[{}][{}]_LE=[", .{ s, k });
                        for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            // Debug: Dump bytecode entries
            if (comptime debug_verbose) {
                dbg("[STAGE6] Bytecode entries (ALL k=0..{}):\n", .{bytecode_K});
                for (0..@min(bytecode_K, 64)) |k| {
                    if (k >= bytecode_entries.len) break;
                    const entry = bytecode_entries[k];
                    dbg("[STAGE6] entry[{}]: addr=0x{x:0>8} rd={} rs1={} rs2={} imm={} cf=[", .{ k, entry.address, entry.rd, entry.rs1, entry.rs2, entry.imm });
                    for (0..14) |i| {
                        if (i > 0) dbg(",", .{});
                        if (entry.circuit_flags[i]) dbg("1", .{}) else dbg("0", .{});
                    }
                    dbg("] if=[", .{});
                    for (0..7) |i| {
                        if (i > 0) dbg(",", .{});
                        if (entry.instruction_flags[i]) dbg("1", .{}) else dbg("0", .{});
                    }
                    dbg("] lt={} interleaved={}\n", .{ entry.lookup_table_index, @intFromBool(entry.is_interleaved) });
                }
            }

            // Build identity polynomial
            var bytecode_int_poly = try self.allocator.alloc(F, bytecode_K);
            for (0..bytecode_K) |k| {
                bytecode_int_poly[k] = F.fromU64(@intCast(k));
            }

            // DEBUG: Per-field comparison for BytecodeReadRaf Stages 1-4
            try debugBytecodeReadRafFieldComparisons(
                F,
                self.allocator,
                self.thread_pool,
                trace,
                pc_map,
                n_cycle_vars,
                bytecode_K,
                bytecode_entries,
                r_cycle_bc1_spartan_outer,
                r_cycle_bc2_product_virt,
                r_cycle_bc4_regs_rwc,
                r_cycle_bc5_regs_val,
                eq_table_4,
                eq_table_5,
                opening_claims,
                stage1_gammas,
                stage2_gammas,
                stage4_gammas,
                bytecode_raf_gamma_powers,
                bcraf_per_stage_claims,
            );

            var bytecode_gamma_arr: [8]F = undefined;
            for (0..8) |i| {
                bytecode_gamma_arr[i] = bytecode_raf_gamma_powers[i];
            }
            const entry_bytecode_index = pc_map.getPC(entry_address, 0);
            const t_init_bcraf = if (bench_s6) std.time.nanoTimestamp() else 0;
            var bytecode_prover = try BytecodeReadRafProver(F).init(
                self.allocator,
                trace,
                pc_map,
                bytecode_val_polys,
                bytecode_log_k,
                n_cycle_vars,
                bytecode_d,
                log_k_chunk,
                bytecode_gamma_arr,
                [5][]const F{
                    r_cycle_bc1_spartan_outer,
                    r_cycle_bc2_product_virt,
                    r_cycle_bc3_spartan_shift,
                    r_cycle_bc4_regs_rwc,
                    r_cycle_bc5_regs_val,
                },
                bytecode_int_poly,
                bcraf_per_stage_claims,
                entry_bytecode_index,
                self.thread_pool,
            );
            bytecode_prover.gpu = self.gpu_ops;
            defer bytecode_prover.deinit();
            const t_after_bcraf = if (bench_s6) std.time.nanoTimestamp() else 0;

            // pc_maps now consistent — no override needed

            // Debug: Compare prover's initial BytecodeReadRaf claim with opening-claims-derived claim
            if (comptime debug_verbose) {
                var prover_initial = F.zero();
                for (0..5) |s| {
                    prover_initial = prover_initial.add(bytecode_prover.gamma_powers[s].mul(bytecode_prover.stage_claims[s]));
                }
                const pi_be = prover_initial.toBytesBE();
                const oc_be = bytecodeReadRaf_input.toBytesBE();
                dbg("\n[S6P_BCRAF_COMPARE] prover_initial_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{pi_be[31 - bi]});
                dbg("]\n[S6P_BCRAF_COMPARE] opening_claims_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{oc_be[31 - bi]});
                dbg("]\n[S6P_BCRAF_COMPARE] match={}\n", .{@as(u8, if (prover_initial.eql(bytecodeReadRaf_input)) 1 else 0)});

                for (0..5) |s| {
                    const ps_be = bytecode_prover.stage_claims[s].toBytesBE();
                    const os_be = bcraf_per_stage_claims[s].toBytesBE();
                    const sm = @as(u8, if (bytecode_prover.stage_claims[s].eql(bcraf_per_stage_claims[s])) 1 else 0);
                    if (sm == 0) {
                        dbg("[S6P_BCRAF_COMPARE] stage[{}] MISMATCH! prover_LE=[", .{s});
                        for (0..32) |bi| dbg("{x:0>2}", .{ps_be[31 - bi]});
                        dbg("] opening_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{os_be[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            // Debug: print r_cycle values for comparison with Jolt
            {
                const r_cycles = [5][]const F{
                    r_cycle_bc1_spartan_outer,
                    r_cycle_bc2_product_virt,
                    r_cycle_bc3_spartan_shift,
                    r_cycle_bc4_regs_rwc,
                    r_cycle_bc5_regs_val,
                };
                for (0..5) |s| {
                    dbg("[ZOLT_BCRAF] r_cycle[{}] (len={}):", .{ s, r_cycles[s].len });
                    for (0..@min(r_cycles[s].len, 4)) |i| {
                        const v_le = r_cycles[s][i].toBytes();
                        dbg(" [{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                            v_le[0], v_le[1], v_le[2], v_le[3], v_le[4], v_le[5], v_le[6], v_le[7],
                        });
                    }
                    if (r_cycles[s].len > 4) dbg("...", .{});
                    dbg("\n", .{});
                }
            }

            // ====================================================================
            // Append input claims and get batching coefficients
            // ====================================================================

            dbg("[STAGE6] Transcript before input_claims: round={}\n", .{transcript.n_rounds});

            transcript.appendScalar("sumcheck_claim", bytecodeReadRaf_input);
            transcript.appendScalar("sumcheck_claim", booleanity_input);
            transcript.appendScalar("sumcheck_claim", hammingBooleanity_input);
            transcript.appendScalar("sumcheck_claim", ramRaVirtual_input);
            transcript.appendScalar("sumcheck_claim", lookupsRaVirtual_input);
            transcript.appendScalar("sumcheck_claim", incClaimReduction_input);

            const batch = try self.allocator.alloc(F, 6);
            defer self.allocator.free(batch);
            for (0..6) |i| {
                batch[i] = transcript.challengeScalarFull();
            }

            const input_claims = [6]F{
                bytecodeReadRaf_input,
                booleanity_input,
                hammingBooleanity_input,
                ramRaVirtual_input,
                lookupsRaVirtual_input,
                incClaimReduction_input,
            };
            const num_rounds_arr = [6]usize{
                bytecodeReadRaf_rounds,
                booleanity_rounds,
                hammingBooleanity_rounds,
                ramRaVirtual_rounds,
                lookupsRaVirtual_rounds,
                incClaimReduction_rounds,
            };

            var batched_claim = F.zero();
            for (0..6) |i| {
                const scale = max_num_rounds - num_rounds_arr[i];
                var scaled = input_claims[i];
                for (0..scale) |_| scaled = scaled.add(scaled);
                batched_claim = batched_claim.add(batch[i].mul(scaled));
            }

            // Debug: print the initial batched claim and all batch coefficients
            {
                const bc_be = batched_claim.toBytesBE();
                dbg("[S6P_BATCHED] initial_batched_claim_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{bc_be[31 - bi]});
                dbg("]\n", .{});
                for (0..6) |i| {
                    const b_be = batch[i].toBytesBE();
                    const ic_be = input_claims[i].toBytesBE();
                    dbg("[S6P_BATCHED] batch[{}]_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{b_be[31 - bi]});
                    dbg("] input_claim_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{ic_be[31 - bi]});
                    dbg("] rounds={}\n", .{num_rounds_arr[i]});
                }
            }

            // ====================================================================
            // Run batched sumcheck
            // ====================================================================

            var challenges = try self.allocator.alloc(F, max_num_rounds);
            errdefer self.allocator.free(challenges);

            var instance_claims: [6]F = input_claims;
            var current_batched_claim = batched_claim;

            const num_compressed = max_degree;

            // Track Phase 1 address challenges for BytecodeReadRaf
            var bytecode_addr_challenges = try self.allocator.alloc(F, bytecode_log_k);
            defer self.allocator.free(bytecode_addr_challenges);

            // Stage 6 fine-grained timing (gated by ZOLT_BENCH env var)
            if (bench_s6) {
                const toMs = struct {
                    fn f(ns: i128) f64 {
                        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
                    }
                }.f;
                std.debug.print("    [STAGE6-BENCH] Init total: {d:7.1}ms\n", .{
                    @as(f64, @floatFromInt(s6_init_timer.read())) / 1_000_000.0,
                });
                std.debug.print("    [STAGE6-BENCH]   IncClaim init:       {d:7.1}ms\n", .{toMs(t_after_inc - t_init_inc)});
                std.debug.print("    [STAGE6-BENCH]   Hamming init:        {d:7.1}ms\n", .{toMs(t_after_hamming - t_init_hamming)});
                std.debug.print("    [STAGE6-BENCH]   RamRaVirtual init:   {d:7.1}ms\n", .{toMs(t_after_ram - t_init_ram)});
                std.debug.print("    [STAGE6-BENCH]   LookupsRa init:      {d:7.1}ms\n", .{toMs(t_after_lookups - t_init_lookups)});
                std.debug.print("    [STAGE6-BENCH]   Booleanity init:     {d:7.1}ms\n", .{toMs(t_after_booleanity - t_init_booleanity)});
                std.debug.print("    [STAGE6-BENCH]   BytecodeRaf init:    {d:7.1}ms\n", .{toMs(t_after_bcraf - t_init_bcraf)});
                std.debug.print("    [STAGE6-BENCH]   Val polys+eq+other:  {d:7.1}ms\n", .{
                    toMs((t_init_bcraf - t_after_booleanity) + (t_init_booleanity - t_after_lookups) + (t_init_lookups - t_after_ram) + (t_init_ram - t_after_hamming) + (t_init_hamming - t_after_inc)),
                });
            }
            var s6_t_compute: if (s6_bench_timing) [6]u64 else void = if (comptime s6_bench_timing) [6]u64{ 0, 0, 0, 0, 0, 0 } else {};
            var s6_t_bind: if (s6_bench_timing) [6]u64 else void = if (comptime s6_bench_timing) [6]u64{ 0, 0, 0, 0, 0, 0 } else {};
            var s6_t_transcript: if (s6_bench_timing) u64 else void = if (comptime s6_bench_timing) @as(u64, 0) else {};
            var s6_timer: if (s6_bench_timing) std.time.Timer else void = if (comptime s6_bench_timing) std.time.Timer.start() catch unreachable else {};

            for (0..max_num_rounds) |round| {
                const remaining_rounds = max_num_rounds - round;

                // Monomial-form batched polynomial: combined_coeffs[i] = coefficient of x^i
                // This matches Jolt's approach: each instance returns a UniPoly in monomial form,
                // and the batched poly is Σ batch[i] * poly_i in coefficient space.
                var combined_coeffs = try self.allocator.alloc(F, max_degree + 1);
                defer self.allocator.free(combined_coeffs);
                @memset(combined_coeffs, F.zero());

                // Per-instance cached round poly evals for claim tracking
                // We cache each instance's round poly so we don't recompute after challenge
                // Phase 1: degree-2 coefficients [a0, a1, a2] for p(x) = a0 + a1*x + a2*x^2
                var cached_bc_phase1_coeffs: [3]F = undefined;
                var cached_bc_phase1_per_stage: [5][2]F = undefined;
                var cached_bc_phase2: ?[]F = null;
                var cached_hamming: [4]F = undefined;
                var cached_ram_ra: ?[]F = null;
                var cached_lookups_ra: ?[]F = null;
                var cached_inc: [3]F = undefined; // Vandermonde: [p(0), p(1), p(2)]
                var cached_inc_p1: F = F.zero(); // recovered p(1)

                // Track which instances are active this round
                var inst_active: [6]bool = .{ false, false, false, false, false, false };
                const debug_r5 = (round == 5 or round == 6);
                // Debug: per-instance contribution to combined_coeffs[0] and [1]
                var dbg_inst_p0: [6]F = .{F.zero()} ** 6;
                var dbg_inst_p1: [6]F = .{F.zero()} ** 6;

                // Instance 0: BytecodeReadRaf - REAL prover
                if (bench_s6) s6_timer.reset();
                {
                    const inst = 0;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        // Not started yet - constant polynomial (degree 0)
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        if (bytecode_prover.phase == 0) {
                            // Phase 1: address binding (degree-2 poly)
                            // computeRoundPolyPhase1 returns aggregated [p(0), p(2)] and per-stage evals
                            const phase1_result = bytecode_prover.computeRoundPolyPhase1();
                            cached_bc_phase1_per_stage = phase1_result.per_stage;
                            const p0 = phase1_result.agg[0];
                            const p2 = phase1_result.agg[1];
                            // Recover p(1) from sumcheck constraint: p(0) + p(1) = claim
                            const p1 = instance_claims[inst].sub(p0);

                            if (round < 2) {
                                const bc_sum = p0.add(p1);
                                dbg("  [S6P] R{} BC_Phase1 p(0)={any} p(1)={any} p(2)={any} sum={any} claim={any}\n", .{
                                    round,
                                    p0.toBytesBE()[0..8],
                                    p1.toBytesBE()[0..8],
                                    p2.toBytesBE()[0..8],
                                    bc_sum.toBytesBE()[0..8],
                                    instance_claims[0].toBytesBE()[0..8],
                                });
                            }

                            // Interpolate degree-2 coefficients from evals at {0, 1, 2}
                            // p(x) = a0 + a1*x + a2*x^2
                            // a0 = p(0)
                            // a2 = (p(2) - 2*p(1) + p(0)) / 2
                            // a1 = p(1) - p(0) - a2
                            const two = F.fromU64(2);
                            const two_inv = two.inverse().?;
                            const a0 = p0;
                            const a2 = p2.sub(p1.add(p1)).add(p0).mul(two_inv);
                            const a1 = p1.sub(p0).sub(a2);
                            cached_bc_phase1_coeffs = [3]F{ a0, a1, a2 };

                            // Add degree-2 monomial coefficients [a0, a1, a2] to combined_coeffs
                            combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(a0));
                            combined_coeffs[1] = combined_coeffs[1].add(batch[inst].mul(a1));
                            combined_coeffs[2] = combined_coeffs[2].add(batch[inst].mul(a2));
                        } else {
                            // Phase 2: cycle binding (degree bytecode_d+1)
                            // Returns Toom-Cook evals: [p(0), p(1), ..., p(d), p(∞)]
                            const polys = try bytecode_prover.computeRoundPolyPhase2(self.allocator);
                            cached_bc_phase2 = polys;
                            if (debug_r5) {
                                const p01 = polys[0].add(polys[1]);
                                const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                                dbg("  [R5_DBG] inst0_phase2 polys_len={} p(0)+p(1)=claim? {}\n", .{ polys.len, p01_ok });
                            }
                            // Convert Toom-Cook evaluations to monomial coefficients
                            const mono = try UniPoly(F).fromEvalsToom(self.allocator, polys);
                            defer self.allocator.free(mono);
                            for (0..mono.len) |ci| {
                                combined_coeffs[ci] = combined_coeffs[ci].add(batch[inst].mul(mono[ci]));
                            }
                        }
                    }
                }

                dbg_inst_p0[0] = combined_coeffs[0];
                dbg_inst_p1[0] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst0: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (bench_s6) s6_t_compute[0] += s6_timer.read();
                // Instance 1: Booleanity - REAL prover (degree 3)
                if (bench_s6) s6_timer.reset();
                var cached_booleanity: ?[]F = null;
                {
                    const inst = 1;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        const polys = try booleanity_prover.computeRoundPoly(self.allocator, instance_claims[inst]);
                        cached_booleanity = polys;
                        {
                            const p01 = polys[0].add(polys[1]);
                            const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            const p0b = polys[0].toBytesBE();
                            const p1b = polys[1].toBytesBE();
                            dbg("  [S6P] R{} Bool p(0)+p(1)=claim? {} phase={} p0=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}] p1=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                                round,   p01_ok,  if (booleanity_prover.round < booleanity_prover.log_k_chunk) @as(u8, 1) else 2,
                                p0b[31], p0b[30], p0b[29],
                                p0b[28], p1b[31], p1b[30],
                                p1b[29], p1b[28],
                            });
                        }
                        // Convert degree-3 evals [p(0), p(1), p(2), p(3)] to monomial coefficients
                        // using finite differences, then add batch[inst] * coeffs to combined_coeffs
                        addEvalsAsMonomialToCoeffs(F, combined_coeffs, polys, 4, batch[inst]);
                    }
                }
                dbg_inst_p0[1] = combined_coeffs[0];
                dbg_inst_p1[1] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst1: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (bench_s6) s6_t_compute[1] += s6_timer.read();
                // Instance 2: HammingBooleanity - REAL prover
                if (bench_s6) s6_timer.reset();
                {
                    const inst = 2;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        const polys = hamming_prover.computeRoundPoly(instance_claims[inst]);
                        cached_hamming = polys;
                        addEvalsAsMonomialToCoeffs(F, combined_coeffs, &polys, 4, batch[inst]);
                    }
                }
                dbg_inst_p0[2] = combined_coeffs[0];
                dbg_inst_p1[2] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst2: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (bench_s6) s6_t_compute[2] += s6_timer.read();
                // Instance 3: RamRaVirtual - REAL prover
                if (bench_s6) s6_timer.reset();
                {
                    const inst = 3;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        // computeRoundPoly now returns monomial coefficients directly (Toom-Cook quotient approach)
                        const mono = try ram_ra_prover.computeRoundPoly(self.allocator, instance_claims[inst]);
                        cached_ram_ra = mono;
                        if (debug_r5) {
                            // Check p(0)+p(1)=claim for RamRaVirtual (mono format: eval via Horner)
                            var p0 = mono[mono.len - 1];
                            var ci_dbg: usize = mono.len - 1;
                            while (ci_dbg > 0) {
                                ci_dbg -= 1;
                                p0 = p0.mul(F.zero()).add(mono[ci_dbg]);
                            }
                            var p1 = mono[mono.len - 1];
                            ci_dbg = mono.len - 1;
                            while (ci_dbg > 0) {
                                ci_dbg -= 1;
                                p1 = p1.mul(F.one()).add(mono[ci_dbg]);
                            }
                            const p01 = p0.add(p1);
                            const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            dbg("  [R5_DBG] inst3 polys_len={} p(0)+p(1)=claim? {}\n", .{ mono.len, p01_ok });
                        }
                        for (0..mono.len) |ci| {
                            combined_coeffs[ci] = combined_coeffs[ci].add(batch[inst].mul(mono[ci]));
                        }
                    }
                }
                dbg_inst_p0[3] = combined_coeffs[0];
                dbg_inst_p1[3] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst3: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (bench_s6) s6_t_compute[3] += s6_timer.read();
                // Instance 4: LookupsRaVirtual - REAL prover
                // Overlap with previous instances via join when both are active
                if (bench_s6) s6_timer.reset();
                {
                    const inst = 4;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        // computeRoundPoly now returns monomial coefficients directly (Toom-Cook quotient approach)
                        const mono = try lookups_ra_prover.computeRoundPoly(self.allocator, instance_claims[inst]);
                        cached_lookups_ra = mono;
                        for (0..mono.len) |ci| {
                            combined_coeffs[ci] = combined_coeffs[ci].add(batch[inst].mul(mono[ci]));
                        }
                    }
                }
                dbg_inst_p0[4] = combined_coeffs[0];
                dbg_inst_p1[4] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst4: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (bench_s6) s6_t_compute[4] += s6_timer.read();
                // Instance 5: IncClaimReduction - REAL prover
                if (bench_s6) s6_timer.reset();
                {
                    const inst = 5;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        const polys = inc_prover.computeRoundPoly();
                        cached_inc = polys;
                        // polys = [p(0), p(1), p(2)] in Vandermonde format for degree 2
                        const p0 = polys[0];
                        const p1 = polys[1];
                        cached_inc_p1 = p1;
                        if (debug_r5) {
                            const p01_ok: u8 = if (std.mem.eql(u8, &p0.add(p1).toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            dbg("  [R5_DBG] inst5 p(0)+p(1)=claim? {} p(0)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] p(1)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                p01_ok,
                                p0.toBytes()[0],
                                p0.toBytes()[1],
                                p0.toBytes()[2],
                                p0.toBytes()[3],
                                p0.toBytes()[4],
                                p0.toBytes()[5],
                                p0.toBytes()[6],
                                p0.toBytes()[7],
                                p1.toBytes()[0],
                                p1.toBytes()[1],
                                p1.toBytes()[2],
                                p1.toBytes()[3],
                                p1.toBytes()[4],
                                p1.toBytes()[5],
                                p1.toBytes()[6],
                                p1.toBytes()[7],
                            });
                        }

                        // IncClaimReduction is degree 2 in Vandermonde format [p(0), p(1), p(2)].
                        // Interpolate monomial coefficients: a0 + a1*x + a2*x^2
                        const a0 = p0;
                        const two = F.fromU64(2);
                        const two_inv = two.inverse().?;
                        const a2_coeff = polys[2].sub(p1.add(p1)).add(p0).mul(two_inv);
                        const a1 = p1.sub(a0).sub(a2_coeff);

                        // Add monomial coefficients to combined_coeffs
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(a0));
                        combined_coeffs[1] = combined_coeffs[1].add(batch[inst].mul(a1));
                        combined_coeffs[2] = combined_coeffs[2].add(batch[inst].mul(a2_coeff));
                    }
                }
                dbg_inst_p0[5] = combined_coeffs[0];
                dbg_inst_p1[5] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst5: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                    // In monomial form, p(0)+p(1) = 2*c0 + c1 + c2 + ... + cd
                    var sum = combined_coeffs[0].add(combined_coeffs[0]); // 2*c0
                    for (1..max_degree + 1) |ci| sum = sum.add(combined_coeffs[ci]); // + c1 + c2 + ... + cd
                    const sum_le = sum.toBytes();
                    const claim_le = current_batched_claim.toBytes();
                    dbg("  [R5_DBG] sum=e[0]+e[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        sum_le[0], sum_le[1], sum_le[2], sum_le[3], sum_le[4], sum_le[5], sum_le[6], sum_le[7],
                    });
                    dbg("  [R5_DBG] claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        claim_le[0], claim_le[1], claim_le[2], claim_le[3], claim_le[4], claim_le[5], claim_le[6], claim_le[7],
                    });
                    // Also check each instance's expected contribution to sum
                    for (0..6) |ii| {
                        const ic_le = instance_claims[ii].toBytes();
                        const ba_le = batch[ii].toBytes();
                        dbg("  [R5_DBG] inst[{}] claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] batch_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] rounds={}\n", .{
                            ii,
                            ic_le[0],
                            ic_le[1],
                            ic_le[2],
                            ic_le[3],
                            ic_le[4],
                            ic_le[5],
                            ic_le[6],
                            ic_le[7],
                            ba_le[0],
                            ba_le[1],
                            ba_le[2],
                            ba_le[3],
                            ba_le[4],
                            ba_le[5],
                            ba_le[6],
                            ba_le[7],
                            num_rounds_arr[ii],
                        });
                    }
                    // Recompute expected batched claim for round 5
                    // At round 5, remaining_rounds = 13-5 = 8
                    // inst 0 (13 rounds): active, scale = 0
                    // inst 1 (8 rounds): remaining 8 > 8? no, so active, scale = 0
                    // inst 2 (8 rounds): active, scale = 0
                    // inst 3 (8 rounds): active, scale = 0
                    // inst 4 (8 rounds): active, scale = 0
                    // inst 5 (8 rounds): active, scale = 0
                    // All active! Batched claim = Σ batch[i] * instance_claims[i]
                    var expected_sum = F.zero();
                    for (0..6) |ii| {
                        expected_sum = expected_sum.add(batch[ii].mul(instance_claims[ii]));
                    }
                    const exp_le = expected_sum.toBytes();
                    dbg("  [R5_DBG] expected_batched_Σ(b*c)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        exp_le[0], exp_le[1], exp_le[2], exp_le[3], exp_le[4], exp_le[5], exp_le[6], exp_le[7],
                    });
                }

                // Debug: check sumcheck invariant p(0)+p(1)=claim for ALL rounds
                // In monomial form: p(0)+p(1) = 2*c0 + c1 + c2 + ... + cd
                if (comptime debug_verbose) {
                    var p01_sum = combined_coeffs[0].add(combined_coeffs[0]); // 2*c0
                    for (1..max_degree + 1) |cii| p01_sum = p01_sum.add(combined_coeffs[cii]);
                    const p01_match = p01_sum.eql(current_batched_claim);
                    if (!p01_match) {
                        dbg("  [S6P] R{} *** SUMCHECK INVARIANT VIOLATED *** p(0)+p(1) != claim\n", .{round});
                        const ps = p01_sum.toBytes();
                        const cb = current_batched_claim.toBytes();
                        dbg("    p(0)+p(1)_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{ps[bi]});
                        dbg("]\n    claim_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{cb[bi]});
                        dbg("]\n", .{});
                        // Print each instance's contribution and per-instance p(0)+p(1) check
                        for (0..6) |di| {
                            const di_claim = instance_claims[di].toBytes();
                            dbg("    inst[{}] claim_LE=[", .{di});
                            for (0..32) |bi| dbg("{x:0>2}", .{di_claim[bi]});
                            dbg("] active={} rounds={}\n", .{ @as(u8, if (inst_active[di]) 1 else 0), num_rounds_arr[di] });
                        }
                        // Recompute expected batched claim from per-instance claims
                        var recomp = F.zero();
                        for (0..6) |di| {
                            if (inst_active[di]) {
                                recomp = recomp.add(batch[di].mul(instance_claims[di]));
                            } else {
                                const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[di], remaining_rounds, num_rounds_arr[di]);
                                recomp = recomp.add(batch[di].mul(scaled).add(batch[di].mul(scaled)));
                            }
                        }
                        const rc_le = recomp.toBytes();
                        dbg("    recomputed_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{rc_le[bi]});
                        dbg("] match_claim={}\n", .{@as(u8, if (recomp.eql(current_batched_claim)) 1 else 0)});
                        // Per-instance p(0)+p(1) vs batch*claim check using cumulative deltas
                        var prev_p0 = F.zero();
                        var prev_p1 = F.zero();
                        for (0..6) |di| {
                            const inst_p0 = dbg_inst_p0[di].sub(prev_p0);
                            const inst_p1 = dbg_inst_p1[di].sub(prev_p1);
                            const inst_sum = inst_p0.add(inst_p1);
                            const expected_contrib = batch[di].mul(instance_claims[di]);
                            if (!inst_sum.eql(expected_contrib)) {
                                const is_le = inst_sum.toBytes();
                                const ex_le = expected_contrib.toBytes();
                                dbg("    *** MISMATCH inst[{}]: batch*(p0+p1)_LE=[", .{di});
                                for (0..32) |bi| dbg("{x:0>2}", .{is_le[bi]});
                                dbg("] batch*claim_LE=[", .{});
                                for (0..32) |bi| dbg("{x:0>2}", .{ex_le[bi]});
                                dbg("]\n", .{});
                            } else {
                                dbg("    inst[{}] p(0)+p(1)=claim OK\n", .{di});
                            }
                            prev_p0 = dbg_inst_p0[di];
                            prev_p1 = dbg_inst_p1[di];
                        }
                    }
                }

                // Debug: print monomial coefficients for round 7
                if (comptime debug_verbose) {
                    if (round == 7) {
                        dbg("  [S6P] R7 monomial coeffs:\n", .{});
                        for (0..max_degree + 1) |ci_idx| {
                            const ci_le = combined_coeffs[ci_idx].toBytes();
                            dbg("    c[{}]=[", .{ci_idx});
                            for (0..32) |bi| dbg("{x:0>2}", .{ci_le[bi]});
                            dbg("]\n", .{});
                        }
                        // p(0)+p(1) = 2*c0 + c1 + c2 + ... + cd
                        var sum01 = combined_coeffs[0].add(combined_coeffs[0]);
                        for (1..max_degree + 1) |ci_idx| sum01 = sum01.add(combined_coeffs[ci_idx]);
                        const sum_le = sum01.toBytes();
                        const hint_le = current_batched_claim.toBytes();
                        dbg("    p(0)+p(1)=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{sum_le[bi]});
                        dbg("]\n    hint    =[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{hint_le[bi]});
                        dbg("]\n    match={}\n", .{sum01.eql(current_batched_claim)});
                    }
                }

                if (bench_s6) s6_t_compute[5] += s6_timer.read();
                if (bench_s6) s6_timer.reset();
                // Compress: strip c1 (linear term) from monomial coefficients
                // compressed = [c0, c2, c3, ..., c_d] (same as Jolt's UniPoly::compress)
                const compressed = try self.allocator.alloc(F, max_degree);
                defer self.allocator.free(compressed);
                compressed[0] = combined_coeffs[0]; // c0
                for (1..max_degree) |ci_idx| {
                    compressed[ci_idx] = combined_coeffs[ci_idx + 1]; // c2, c3, ..., c_d
                }

                // Debug: print compressed coefficients LE for ALL rounds
                if (comptime debug_verbose) {
                    var c_idx: usize = 0;
                    while (c_idx < compressed.len) : (c_idx += 1) {
                        const le = compressed[c_idx].toBytes();
                        dbg("  [S6P] R{} coeff[{}]=[", .{ round, c_idx });
                        for (0..32) |bi| dbg("{x:0>2}", .{le[bi]});
                        dbg("]\n", .{});
                    }
                }

                const coeffs = try self.allocator.alloc(F, num_compressed);
                for (0..num_compressed) |j| {
                    coeffs[j] = if (j < compressed.len) compressed[j] else F.zero();
                }

                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                // Write diagnostic data to file for R0 - BEFORE appending to transcript
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const diag_file = std.fs.cwd().createFile("/tmp/s6p_diag.bin", .{}) catch null;
                        if (diag_file) |f| {
                            defer f.close();
                            f.writeAll(&transcript.state) catch {};
                            for (0..num_compressed) |j| {
                                const le = coeffs[j].toBytes();
                                f.writeAll(&le) catch {};
                            }
                        }
                    }
                }

                transcript.appendScalars("sumcheck_poly", coeffs[0..num_compressed]);

                // Dump transcript state AFTER appending R0 polynomial
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const diag_after = std.fs.cwd().createFile("/tmp/s6p_state_after_r0.bin", .{}) catch null;
                        if (diag_after) |fa| {
                            defer fa.close();
                            fa.writeAll(&transcript.state) catch {};
                            var nr_buf: [4]u8 = undefined;
                            std.mem.writeInt(u32, &nr_buf, transcript.n_rounds, .little);
                            fa.writeAll(&nr_buf) catch {};
                        }
                    }
                }

                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                // Write R0 challenge to diagnostic file
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const diag2 = std.fs.cwd().createFile("/tmp/s6p_r0_challenge.bin", .{}) catch null;
                        if (diag2) |f2| {
                            defer f2.close();
                            const ch_le = challenge.toBytes();
                            f2.writeAll(&ch_le) catch {};
                        }
                    }
                }

                // Evaluate combined polynomial at challenge using evalFromHintGeneral
                current_batched_claim = UniPoly(F).evalFromHintGeneral(coeffs[0..num_compressed], current_batched_claim, challenge);

                if (comptime debug_verbose) {
                    // Verify: directly evaluate combined_coeffs at challenge via Horner
                    var direct_eval = combined_coeffs[max_degree];
                    {
                        var ci_rev = max_degree;
                        while (ci_rev > 0) {
                            ci_rev -= 1;
                            direct_eval = direct_eval.mul(challenge).add(combined_coeffs[ci_rev]);
                        }
                    }
                    const efh_match = direct_eval.eql(current_batched_claim);
                    if (!efh_match) {
                        const efh_le = direct_eval.toBytes();
                        const vdm_le = current_batched_claim.toBytes();
                        dbg("  [S6P] R{} EVAL_MISMATCH! direct_eval=[", .{round});
                        for (0..32) |bi| dbg("{x:0>2}", .{efh_le[bi]});
                        dbg("]\n  [S6P] R{} EVAL_MISMATCH! evalFromHint=[", .{round});
                        for (0..32) |bi| dbg("{x:0>2}", .{vdm_le[bi]});
                        dbg("]\n", .{});
                        dbg("  [S6P] R{} num_compressed={}, compressed.len={}\n", .{ round, num_compressed, compressed.len });
                    }
                    dbg("  [S6P] R{} efh_match={}\n", .{ round, @intFromBool(efh_match) });
                }

                if (comptime debug_verbose) {
                    const ch_le = challenge.toBytes();
                    const cl_le = current_batched_claim.toBytes();
                    dbg("  [S6P] R{} challenge_LE=[", .{round});
                    for (0..32) |bi| dbg("{x:0>2}", .{ch_le[bi]});
                    dbg("]\n", .{});
                    dbg("  [S6P] R{} new_claim_LE=[", .{round});
                    for (0..32) |bi| dbg("{x:0>2}", .{cl_le[bi]});
                    dbg("]\n", .{});
                }

                if (bench_s6) s6_t_transcript += s6_timer.read();
                // Update per-instance claims from CACHED round polys and bind challenge
                // Instance 0: BytecodeReadRaf
                if (bench_s6) s6_timer.reset();
                if (inst_active[0]) {
                    if (bytecode_prover.phase == 0) {
                        // Phase 1: degree-2 poly, p(r) = a0 + a1*r + a2*r^2
                        const bc_a0 = cached_bc_phase1_coeffs[0];
                        const bc_a1 = cached_bc_phase1_coeffs[1];
                        const bc_a2 = cached_bc_phase1_coeffs[2];
                        instance_claims[0] = bc_a0.add(challenge.mul(bc_a1.add(challenge.mul(bc_a2))));
                        if (comptime debug_verbose) {
                            const ic_le = instance_claims[0].toBytes();
                            dbg("  [S6P] R{} inst0_from_poly_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                round, ic_le[0], ic_le[1], ic_le[2], ic_le[3], ic_le[4], ic_le[5], ic_le[6], ic_le[7],
                            });
                        }
                        bytecode_addr_challenges[bytecode_prover.addr_rounds_done] = challenge;
                        bytecode_prover.bindChallengePhase1(challenge, cached_bc_phase1_per_stage);
                        if (comptime debug_verbose) {
                            // Check invariant: instance_claims[0] == Σ gamma^s * stage_claims[s]
                            var agg_check = F.zero();
                            for (0..5) |si| {
                                agg_check = agg_check.add(bytecode_prover.gamma_powers[si].mul(bytecode_prover.stage_claims[si]));
                            }
                            const ac_le = agg_check.toBytes();
                            const ic_le2 = instance_claims[0].toBytes();
                            for (0..5) |si| {
                                const scl = bytecode_prover.stage_claims[si].toBytes();
                                dbg("[INVARIANT_CHECK] R{} stage[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                    round,  si,
                                    scl[0], scl[1],
                                    scl[2], scl[3],
                                    scl[4], scl[5],
                                    scl[6], scl[7],
                                });
                            }
                            dbg("[INVARIANT_CHECK] R{} agg_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] inst0_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match={}\n", .{
                                round,
                                ac_le[0],
                                ac_le[1],
                                ac_le[2],
                                ac_le[3],
                                ac_le[4],
                                ac_le[5],
                                ac_le[6],
                                ac_le[7],
                                ic_le2[0],
                                ic_le2[1],
                                ic_le2[2],
                                ic_le2[3],
                                ic_le2[4],
                                ic_le2[5],
                                ic_le2[6],
                                ic_le2[7],
                                @as(u8, if (agg_check.eql(instance_claims[0])) 1 else 0),
                            });
                            const bc_a0_ = cached_bc_phase1_coeffs[0];
                            const bc_a1_ = cached_bc_phase1_coeffs[1];
                            const bc_a2_ = cached_bc_phase1_coeffs[2];
                            const manual_eval = bc_a0_.add(challenge.mul(bc_a1_.add(challenge.mul(bc_a2_))));
                            const me_le = manual_eval.toBytes();
                            dbg("[INVARIANT_CHECK] R{} manual_eval_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match_inst={}\n", .{
                                round,
                                me_le[0],
                                me_le[1],
                                me_le[2],
                                me_le[3],
                                me_le[4],
                                me_le[5],
                                me_le[6],
                                me_le[7],
                                @as(u8, if (manual_eval.eql(instance_claims[0])) 1 else 0),
                            });
                        }
                        if (bytecode_prover.addr_rounds_done == bytecode_log_k) {
                            if (comptime debug_verbose) {
                                // BEFORE transition: check Σ_s gamma^s * stage_claims[s] vs instance_claims[0]
                                var agg_from_stages = F.zero();
                                for (0..5) |si| {
                                    agg_from_stages = agg_from_stages.add(bytecode_prover.gamma_powers[si].mul(bytecode_prover.stage_claims[si]));
                                }
                                const afs_le = agg_from_stages.toBytes();
                                const ic0_le = instance_claims[0].toBytes();
                                dbg("[PHASE_TRANSITION_PRE] agg_stages_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] inst0_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match={}\n", .{
                                    afs_le[0],                                                      afs_le[1], afs_le[2], afs_le[3], afs_le[4], afs_le[5], afs_le[6], afs_le[7],
                                    ic0_le[0],                                                      ic0_le[1], ic0_le[2], ic0_le[3], ic0_le[4], ic0_le[5], ic0_le[6], ic0_le[7],
                                    @as(u8, if (agg_from_stages.eql(instance_claims[0])) 1 else 0),
                                });
                                for (0..5) |si| {
                                    const sc_le2 = bytecode_prover.stage_claims[si].toBytes();
                                    dbg("[PHASE_TRANSITION_PRE] stage[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                        si, sc_le2[0], sc_le2[1], sc_le2[2], sc_le2[3], sc_le2[4], sc_le2[5], sc_le2[6], sc_le2[7],
                                    });
                                }
                            }
                            try bytecode_prover.transitionToPhase2(bytecode_addr_challenges);
                            if (comptime debug_verbose) {
                                // After transition, check Phase 2 polynomial sum
                                // (combined[] replaced by GruenSplitEq — full verification
                                // requires materializing eq tables, skipped in debug mode)
                                dbg("[PHASE_TRANSITION] inst0 transition complete\n", .{});
                            }
                        }
                    } else {
                        // Phase 2: evaluate from Toom-Cook cached evals
                        // Convert to monomials, evaluate at challenge, free
                        const bc_p2_mono = try UniPoly(F).fromEvalsToom(self.allocator, cached_bc_phase2.?);
                        defer self.allocator.free(bc_p2_mono);
                        var bc_p2_val = F.zero();
                        var x_pow = F.one();
                        for (bc_p2_mono) |coeff| {
                            bc_p2_val = bc_p2_val.add(coeff.mul(x_pow));
                            x_pow = x_pow.mul(challenge);
                        }
                        instance_claims[0] = bc_p2_val;
                        self.allocator.free(cached_bc_phase2.?);
                        cached_bc_phase2 = null;
                        bytecode_prover.bindChallengePhase2(challenge);
                    }
                }

                if (bench_s6) s6_t_bind[0] += s6_timer.read();
                // Instance 1: Booleanity (real prover)
                if (bench_s6) s6_timer.reset();
                if (inst_active[1]) {
                    if (cached_booleanity) |polys| {
                        // Evaluate degree-3 poly at challenge from Vandermonde [p(0), p(1), p(2), p(3)]
                        const evals_arr = [4]F{ polys[0], polys[1], polys[2], polys[3] };
                        instance_claims[1] = UniPoly(F).evalFromEvalsDeg3(evals_arr, challenge);
                        self.allocator.free(polys);
                        cached_booleanity = null;
                    }
                    try booleanity_prover.bindChallenge(challenge);
                    if (comptime debug_verbose) {
                        if (booleanity_prover.round == booleanity_prover.log_k_chunk) {
                            const ic1_be = instance_claims[1].toBytesBE();
                            dbg("[BOOL_TRANSITION] inst_claim[1] after Ph1 LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                                ic1_be[31], ic1_be[30], ic1_be[29], ic1_be[28], ic1_be[27], ic1_be[26], ic1_be[25], ic1_be[24],
                            });
                        }
                    }
                }

                if (bench_s6) s6_t_bind[1] += s6_timer.read();
                // Instance 2: HammingBooleanity
                if (bench_s6) s6_timer.reset();
                if (inst_active[2]) {
                    instance_claims[2] = UniPoly(F).evalFromEvalsDeg3(cached_hamming, challenge);
                    try hamming_prover.bindChallenge(challenge);
                }

                if (bench_s6) s6_t_bind[2] += s6_timer.read();
                // Instance 3: RamRaVirtual
                if (bench_s6) s6_timer.reset();
                if (inst_active[3]) {
                    // Monomial coefficients — evaluate via Horner's method
                    const ram_mono = cached_ram_ra.?;
                    var ram_val = ram_mono[ram_mono.len - 1];
                    var ram_ci: usize = ram_mono.len - 1;
                    while (ram_ci > 0) {
                        ram_ci -= 1;
                        ram_val = ram_val.mul(challenge).add(ram_mono[ram_ci]);
                    }
                    instance_claims[3] = ram_val;
                    self.allocator.free(ram_mono);
                    cached_ram_ra = null;
                    try ram_ra_prover.bindChallenge(challenge);
                }

                if (bench_s6) s6_t_bind[3] += s6_timer.read();
                // Instance 4: LookupsRaVirtual
                if (bench_s6) s6_timer.reset();
                if (inst_active[4]) {
                    // Monomial coefficients — evaluate via Horner's method
                    const mono = cached_lookups_ra.?;
                    var val = mono[mono.len - 1];
                    var ci: usize = mono.len - 1;
                    while (ci > 0) {
                        ci -= 1;
                        val = val.mul(challenge).add(mono[ci]);
                    }
                    instance_claims[4] = val;
                    self.allocator.free(mono);
                    cached_lookups_ra = null;
                    try lookups_ra_prover.bindChallenge(challenge);
                }

                if (bench_s6) s6_t_bind[4] += s6_timer.read();
                // Instance 5: IncClaimReduction
                if (bench_s6) s6_timer.reset();
                if (inst_active[5]) {
                    instance_claims[5] = UniPoly(F).evalFromEvalsDeg2(cached_inc, challenge);

                    try inc_prover.bindChallenge(challenge);
                }
                if (bench_s6) s6_t_bind[5] += s6_timer.read();

                // NOTE: Instance claims for inactive instances are NOT halved here.
                // In Zolt, instance_claims starts at the UNSCALED input_claims (not 2^offset-scaled),
                // and the inactive round contributions are computed directly from input_claims with
                // the correct power-of-2 scaling. When an instance first becomes active,
                // instance_claims[i] = input_claims[i] = the correct unscaled claim.
            }

            if (bench_s6) {
                const names = [6][]const u8{ "BcRaf", "Bool ", "Hamm ", "RamRa", "LkRa ", "Inc  " };
                var total_compute: u64 = 0;
                var total_bind: u64 = 0;
                for (0..6) |i| {
                    total_compute += s6_t_compute[i];
                    total_bind += s6_t_bind[i];
                }
                const s6_sumcheck_wall_ns: i128 = std.time.nanoTimestamp() - t_s6_overall_start;
                const toMsU = struct {
                    fn f(ns: u64) f64 {
                        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
                    }
                }.f;
                const toMsI = struct {
                    fn f(ns: i128) f64 {
                        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
                    }
                }.f;
                std.debug.print("\n    [STAGE6-BENCH] Sumcheck loop ({} rounds):\n", .{max_num_rounds});
                for (0..6) |i| {
                    std.debug.print("    [STAGE6-BENCH]   {s}: compute={d:7.1}ms  bind={d:7.1}ms  total={d:7.1}ms\n", .{
                        names[i],
                        toMsU(s6_t_compute[i]),
                        toMsU(s6_t_bind[i]),
                        toMsU(s6_t_compute[i] + s6_t_bind[i]),
                    });
                }
                std.debug.print("    [STAGE6-BENCH]   transcript+compress: {d:7.1}ms\n", .{toMsU(s6_t_transcript)});
                std.debug.print("    [STAGE6-BENCH]   Sumcheck TOTAL: compute={d:7.1}ms  bind={d:7.1}ms  transcript={d:7.1}ms\n", .{
                    toMsU(total_compute),
                    toMsU(total_bind),
                    toMsU(s6_t_transcript),
                });
                std.debug.print("    [STAGE6-BENCH]   Stage 6 overall wall: {d:7.1}ms\n", .{toMsI(s6_sumcheck_wall_ns)});
            }

            // ====================================================================
            // Extract opening claims from all real provers
            // ====================================================================

            const inc_opening = inc_prover.openingClaims();
            const ram_inc_claim = inc_opening.ram_inc;
            const rd_inc_claim = inc_opening.rd_inc;
            if (comptime debug_verbose) {
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

            const hamming_weight_claim = hamming_prover.openingClaim();

            const bytecode_ra_claims = try bytecode_prover.getOpeningClaims(self.allocator);
            if (comptime debug_verbose) {
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
                    // Compute val_from_prover = combined[0] * Π ra[i]
                    var val_ra_prod = comb0;
                    for (0..bytecode_d) |i| {
                        val_ra_prod = val_ra_prod.mul(bytecode_ra_claims[i]);
                    }
                    const vrp_be = val_ra_prod.toBytesBE();
                    dbg("  combined[0]*Π_ra_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{vrp_be[31 - bi]});
                    dbg("]\n", .{});
                }
                // Compare with instance_claims[0]
                const ic0_be = instance_claims[0].toBytesBE();
                dbg("  instance_claims[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{ic0_be[31 - bi]});
                dbg("]\n", .{});

                // === PER-STAGE DECOMPOSITION ===
                // Recompute combined[0] = Σ_s bound_vals[s] * eq_mle(r_cycle_s, r_cycle_prime)
                // r_cycle_prime = reversed Phase 2 challenges (matching Jolt's normalize_opening_point)
                const cycle_start = bytecode_log_k;
                var r_cycle_prime = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(r_cycle_prime);
                for (0..n_cycle_vars) |ci| {
                    r_cycle_prime[ci] = challenges[cycle_start + n_cycle_vars - 1 - ci];
                }
                // Print r_cycle_prime
                dbg("[DECOMP] r_cycle_prime (reversed cycle challenges, BE):\n", .{});
                for (0..@min(4, n_cycle_vars)) |ci| {
                    const rcp_be = r_cycle_prime[ci].toBytesBE();
                    dbg("  r_cycle_prime[{}]_LE=[", .{ci});
                    for (0..8) |bi| dbg("{x:0>2}", .{rcp_be[31 - bi]});
                    dbg("]\n", .{});
                }

                var decomp_sum = F.zero();
                for (0..5) |s| {
                    // Compute eq_mle(r_cycle_s, r_cycle_prime) = Π_i (r_s[i]*r_p[i] + (1-r_s[i])(1-r_p[i]))
                    // Both r_cycle_s and r_cycle_prime are in BE order
                    var eq_mle = F.one();
                    const r_s = bytecode_prover.stage_r_cycles[s];
                    for (0..n_cycle_vars) |ci| {
                        const a = r_s[ci];
                        const b = r_cycle_prime[ci];
                        // eq term: a*b + (1-a)*(1-b) = 1 - a - b + 2*a*b
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

                // Also print val_with_raf bound values (without gamma)
                for (0..5) |s| {
                    const vwr = bytecode_prover.bound_vals_stored[s];
                    const gp = bytecode_prover.gamma_powers[s];
                    // val_with_raf[s][0] = bound_vals[s] / gamma[s]
                    // Print bound_val directly (it already includes gamma)
                    const vwr_be = vwr.toBytesBE();
                    const gp_be = gp.toBytesBE();
                    dbg("[DECOMP] stage[{}]: gamma_LE=[", .{s});
                    for (0..8) |bi| dbg("{x:0>2}", .{gp_be[31 - bi]});
                    dbg("] gamma*val_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{vwr_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            const ram_ra_virtual_claims = try ram_ra_prover.getOpeningClaims(self.allocator);

            const instruction_ra_virtual_claims = try lookups_ra_prover.getOpeningClaims(self.allocator, lookups_ra_gamma_powers);

            // Get booleanity claims directly from the prover's final H state.
            // After all Phase 2 rounds, H[i][0] = ra_i(ρ_addr, ρ_cycle).
            const booleanity_ra_claims = try booleanity_prover.getBooleanityClaims(self.allocator);
            if (comptime debug_verbose) {
                const total_booleanity_polys = instruction_d + bytecode_d + ram_d;
                dbg("[STAGE6] Booleanity claims from H final state:\n", .{});
                for (0..@min(5, total_booleanity_polys)) |i| {
                    const brc_be = booleanity_ra_claims[i].toBytesBE();
                    dbg("  bool_claim[{}]_LE=[", .{i});
                    for (0..8) |bi| dbg("{x:0>2}", .{brc_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            if (comptime debug_verbose) {
                // Debug: compute what the verifier would compute for Instance 1 (Booleanity)
                // expected = eq(challenges, combined_r) * Σ gamma^{2i} * (ra_i^2 - ra_i)
                // combined_r = r_address.reversed ++ r_cycle.reversed
                // In Jolt: r_address reversed means the original r_address (from params) reversed.
                // The booleanity params store r_address in LE format. "reversed" in Jolt means
                // going from LE to reversed-LE. But actually Jolt stores r_address and r_cycle in a
                // specific order from BooleanitySumcheckParams::new, and then reverses them.
                {
                    const total_booleanity_polys = instruction_d + bytecode_d + ram_d;
                    // Jolt's BooleanitySumcheckParams stores r_address and r_cycle from Stage 5.
                    // r_address = last log_k_chunk challenges from the InstructionReadRaf address.
                    // r_cycle = cycle challenges from InstructionReadRaf.
                    // The verifier reverses both: combined_r = rev(r_address) ++ rev(r_cycle).
                    //
                    // In our code:
                    // r_address_bool_le = [ch[log_k-1], ch[log_k-2], ..., ch[0]] (from stage5 MSB-first)
                    // But the Jolt params store them in a specific order based on Stage 5's binding.
                    // Jolt's BooleanitySumcheckParams::new extracts r_address from the accumulator
                    // which stores them in the binding order from Stage 5 InstructionReadRaf.
                    //
                    // For now, let me compute the expected claim using the data I have:
                    // The sumcheck challenges for booleanity rounds are challenges[0..log_k+n_cycle].
                    // Booleanity uses rounds 0..log_k for address, log_k..log_k+n_cycle for cycle.
                    //
                    // The actual output_claim from the sumcheck should be:
                    //   eq_r_r * eq_cycle_final * Σ gamma^{2i} * (H[i][0]^2 - H[i][0])
                    // where eq_cycle_final is what eq_cycle[0] becomes after all Phase 2 bindings.
                    //
                    // Let me just compute Σ gamma^{2i} * (ra_i^2 - ra_i) and the eq parts.
                    var sum_gamma_ra = F.zero();
                    for (0..total_booleanity_polys) |i| {
                        const ra = booleanity_ra_claims[i];
                        sum_gamma_ra = sum_gamma_ra.add(booleanity_prover.gamma_powers_sq[i].mul(ra.mul(ra).sub(ra)));
                    }
                    // Also, get the actual eq values from the prover
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

                    // Compare with instance_claims[1] (the sumcheck output claim for booleanity)
                    const ic1_be = instance_claims[1].toBytesBE();
                    dbg("[BOOL_VERIFY] instance_claims[1]_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{ic1_be[31 - bi]});
                    dbg("]\n", .{});
                    dbg("[BOOL_VERIFY] match={}\n", .{@intFromBool(actual_output.eql(instance_claims[1]))});

                    // Now compute eq(challenges, combined_r) directly, the way the verifier does.
                    // combined_r = rev(r_address_LE) ++ rev(r_cycle_LE)
                    // r_address_LE (in Jolt) = last log_k_chunk elements of Stage5 addr reversed to LE
                    // In our code: the ORIGINAL r_address_bool_le (before reversal in init) is the LE version.
                    // After init() reversed it, booleanity_prover.r_address_le[m] = MSB at m=0.
                    // To get Jolt's LE r_address, we need to reverse it back.
                    // Then rev(r_address_LE) = booleanity_prover.r_address_le (as-is, since it was reversed to BE)
                    //
                    // combined_r_addr[m] = r_address_LE[log_k-1-m] = booleanity_prover.r_address_le[m]
                    // combined_r_cycle[m] = r_cycle_LE[n_cycle-1-m]
                    //
                    // r_cycle_LE = lookups_ra_r_cycle (the original, before computeEqTable)
                    // combined_r_cycle[m] = lookups_ra_r_cycle[n_cycle-1-m]
                    //
                    // eq(ch[m], combined_r[m]) for m < log_k:
                    //   = eq(ch[m], booleanity_prover.r_address_le[m])
                    // eq(ch[log_k+m], combined_r[log_k+m]) for m < n_cycle:
                    //   = eq(ch[log_k+m], lookups_ra_r_cycle[n_cycle-1-m])
                    {
                        const bool_start_round = max_num_rounds - num_rounds_arr[1];
                        dbg("[BOOL_VERIFY] bool_start_round={}, log_k={}, n_cycle={}\n", .{
                            bool_start_round, log_k_chunk, n_cycle_vars,
                        });

                        // Print ALL eq factors matching Jolt's format
                        // Jolt: combined_r = rev(r_address_LE) ++ rev(r_cycle_LE)
                        // Zolt: r_address_le[m] = MSB at 0 (reversed in init) = rev(r_address_LE)[m]
                        // Zolt: combined_r_cycle[m] = r_cycle_LE[n_cycle-1-m] = lookups_ra_r_cycle[n_cycle-1-m]
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
                            // Jolt: combined_r_cycle[m] = rev(r_cycle_LE)[m] = r_cycle_LE[n-1-m]
                            // Since lookups_ra_r_cycle is BE (MSB at 0), and Jolt r_cycle_LE[n-1-m] = lookups[m]
                            const w_val = lookups_ra_r_cycle[m]; // direct index, no reversal
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
            } // end if (comptime debug_verbose) for BOOL_VERIFY

            if (comptime debug_verbose) {
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
                // where val = GruenSplitEq final scalar sum + entry correction
                {
                    var bc_combined_val = bytecode_prover.entry_correction_scalar;
                    for (0..5) |s| {
                        bc_combined_val = bc_combined_val.add(bytecode_prover.bound_vals_phase2[s].mul(bytecode_prover.stage_gruen_eqs[s].?.current_scalar));
                    }
                    var bc_ra_prod = F.one();
                    for (bytecode_ra_claims) |c| bc_ra_prod = bc_ra_prod.mul(c);
                    const bc_recomputed = bc_combined_val.mul(bc_ra_prod);
                    dbg("[STAGE6] Consistency check Instance 0:\n", .{});
                    // Print combined[0] as LE hex for comparison with Jolt's "val (sum)"
                    const cval_be = bc_combined_val.toBytesBE();
                    dbg("  combined[0]_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{cval_be[31 - bi]});
                    dbg("]\n", .{});
                    // Print ra claims
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

                // Consistency check Instance 5 (IncClaimReduction):
                // expected = ram_inc * eq_ram_combined(rho) + gamma^2 * rd_inc * eq_rd_combined(rho)
                // where rho = reversed sumcheck challenges (opening point in BE)
                {
                    // Build opening point: reverse challenges for LE->BE
                    var opening_point = try self.allocator.alloc(F, n_cycle_vars);
                    defer self.allocator.free(opening_point);
                    // Instance 5 has n_cycle_vars rounds; offset = max_num_rounds - n_cycle_vars
                    const inc_offset = max_num_rounds - n_cycle_vars;
                    for (0..n_cycle_vars) |i| {
                        opening_point[n_cycle_vars - 1 - i] = challenges[inc_offset + i];
                    }

                    // Compute eq evaluations at opening_point vs each r_cycle
                    // eq(a, b) = prod_i (a[i]*b[i] + (1-a[i])*(1-b[i]))
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

                    // Also print the r_cycle values themselves
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
            } // end if (comptime debug_verbose)

            // ====================================================================
            // Cache openings to transcript
            // ====================================================================

            dbg("[STAGE6] Transcript before cache_openings: round={}\n", .{transcript.n_rounds});

            // Instance 0: BytecodeReadRaf
            for (bytecode_ra_claims) |claim| {
                transcript.appendScalar("opening_claim", claim);
            }
            dbg("[STAGE6] After BytecodeReadRaf openings ({}): round={}\n", .{ bytecode_ra_claims.len, transcript.n_rounds });

            // Instance 1: Booleanity
            // Upstream aliasing: when bytecode_log_k is a multiple of log_k_chunk,
            // BytecodeRa(0)/Booleanity has the same opening point as BytecodeRa(0)/BytecodeReadRaf
            // (no zero-padding in compute_r_address_chunks), so the verifier aliases it
            // and does NOT flush it to transcript.
            const bytecode_ra0_aliases = (bytecode_log_k % log_k_chunk == 0);
            const bool_skip_index = instruction_ra_virtual_claims.len; // BytecodeRa(0) is at index instruction_d in Booleanity's polynomial_types
            for (booleanity_ra_claims, 0..) |claim, i| {
                if (bytecode_ra0_aliases and i == bool_skip_index) continue;
                transcript.appendScalar("opening_claim", claim);
            }

            // Instance 2: HammingBooleanity
            transcript.appendScalar("opening_claim", hamming_weight_claim);

            // Instance 3: RamRaVirtual
            for (ram_ra_virtual_claims) |claim| {
                transcript.appendScalar("opening_claim", claim);
            }

            // Instance 4: LookupsRaVirtual
            for (instruction_ra_virtual_claims) |claim| {
                transcript.appendScalar("opening_claim", claim);
            }

            dbg("[STAGE6] After LookupsRaVirtual openings ({}): round={}\n", .{ instruction_ra_virtual_claims.len, transcript.n_rounds });

            // Instance 5: IncClaimReduction
            transcript.appendScalar("opening_claim", ram_inc_claim);
            transcript.appendScalar("opening_claim", rd_inc_claim);
            dbg("[STAGE6] After ALL cache_openings: round={}\n", .{transcript.n_rounds});

            return Stage6Result(F){
                .challenges = challenges,
                .bytecode_ra_claims = bytecode_ra_claims,
                .hamming_weight_claim = hamming_weight_claim,
                .booleanity_ra_claims = booleanity_ra_claims,
                .ram_ra_virtual_claims = ram_ra_virtual_claims,
                .instruction_ra_virtual_claims = instruction_ra_virtual_claims,
                .ram_inc_claim = ram_inc_claim,
                .rd_inc_claim = rd_inc_claim,
                .bytecode_log_k = bytecode_log_k,
                .log_k_chunk = log_k_chunk,
                .n_cycle_vars = n_cycle_vars,
                .bytecode_d = bytecode_d,
                .ram_d = ram_d,
                .instruction_d = instruction_d,
                .allocator = self.allocator,
            };
        }

    };
}

// Helper functions, eq table construction, polynomial interpolation, and tests
// have been extracted to stage6_helpers.zig
