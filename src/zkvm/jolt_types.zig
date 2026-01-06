//! Jolt-Compatible Proof Types
//!
//! This module defines proof structures that are byte-compatible with
//! Jolt's Rust implementation, enabling cross-verification of proofs.
//!
//! Reference: jolt-core/src/zkvm/proof_serialization.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

// =============================================================================
// SumcheckId - Identifies which sumcheck a claim belongs to
// =============================================================================

/// Matches Jolt's SumcheckId enum (22 variants)
/// Reference: jolt-core/src/poly/opening_proof.rs
pub const SumcheckId = enum(u8) {
    SpartanOuter = 0,
    SpartanProductVirtualization = 1,
    SpartanShift = 2,
    InstructionClaimReduction = 3,
    InstructionInputVirtualization = 4,
    InstructionReadRaf = 5,
    InstructionRaVirtualization = 6,
    RamReadWriteChecking = 7,
    RamRafEvaluation = 8,
    RamOutputCheck = 9,
    RamValEvaluation = 10,
    RamValFinalEvaluation = 11,
    RamRaClaimReduction = 12,
    RamHammingBooleanity = 13,
    RamRaVirtualization = 14,
    RegistersClaimReduction = 15,
    RegistersReadWriteChecking = 16,
    RegistersValEvaluation = 17,
    BytecodeReadRaf = 18,
    Booleanity = 19,
    IncClaimReduction = 20,
    HammingWeightClaimReduction = 21,

    pub const COUNT: usize = 22;
};

// =============================================================================
// CommittedPolynomial - Polynomials that have PCS commitments
// =============================================================================

/// Matches Jolt's CommittedPolynomial enum
/// Reference: jolt-core/src/zkvm/witness.rs
pub const CommittedPolynomial = union(enum) {
    RdInc,
    RamInc,
    InstructionRa: usize,
    BytecodeRa: usize,
    RamRa: usize,

    /// Serialize in Jolt's compact format
    pub fn serialize(self: CommittedPolynomial, writer: anytype) !void {
        switch (self) {
            .RdInc => try writer.writeByte(0),
            .RamInc => try writer.writeByte(1),
            .InstructionRa => |i| {
                try writer.writeByte(2);
                try writer.writeByte(@truncate(i));
            },
            .BytecodeRa => |i| {
                try writer.writeByte(3);
                try writer.writeByte(@truncate(i));
            },
            .RamRa => |i| {
                try writer.writeByte(4);
                try writer.writeByte(@truncate(i));
            },
        }
    }

    /// Deserialize from Jolt's compact format
    pub fn deserialize(reader: anytype) !CommittedPolynomial {
        const discriminant = try reader.readByte();
        return switch (discriminant) {
            0 => .RdInc,
            1 => .RamInc,
            2 => CommittedPolynomial{ .InstructionRa = try reader.readByte() },
            3 => CommittedPolynomial{ .BytecodeRa = try reader.readByte() },
            4 => CommittedPolynomial{ .RamRa = try reader.readByte() },
            else => error.InvalidData,
        };
    }
};

// =============================================================================
// VirtualPolynomial - Polynomials derived from other polynomials
// =============================================================================

/// Matches Jolt's VirtualPolynomial enum
/// Reference: jolt-core/src/zkvm/witness.rs
pub const VirtualPolynomial = union(enum) {
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    NextIsVirtual,
    NextIsFirstInSequence,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,
    Product,
    ShouldJump,
    ShouldBranch,
    WritePCtoRD,
    WriteLookupOutputToRD,
    Rd,
    Imm,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    InstructionRaf,
    InstructionRafFlag,
    InstructionRa: usize,
    RegistersVal,
    RamAddress,
    RamRa,
    RamReadValue,
    RamWriteValue,
    RamVal,
    RamValInit,
    RamValFinal,
    RamHammingWeight,
    UnivariateSkip,
    OpFlags: u8,
    InstructionFlags: u8,
    LookupTableFlag: usize,

    /// Compare payloads of two VirtualPolynomials with the same tag
    /// Returns .eq if they're completely equal, otherwise .lt or .gt
    pub fn orderByPayload(a: VirtualPolynomial, b: VirtualPolynomial) std.math.Order {
        switch (a) {
            .OpFlags => |val_a| {
                const val_b = b.OpFlags;
                return std.math.order(val_a, val_b);
            },
            .InstructionFlags => |val_a| {
                const val_b = b.InstructionFlags;
                return std.math.order(val_a, val_b);
            },
            .InstructionRa => |val_a| {
                const val_b = b.InstructionRa;
                return std.math.order(val_a, val_b);
            },
            .LookupTableFlag => |val_a| {
                const val_b = b.LookupTableFlag;
                return std.math.order(val_a, val_b);
            },
            // All other variants have no payload, so they're equal if tags match
            else => return .eq,
        }
    }

    /// Serialize in Jolt's compact format
    pub fn serialize(self: VirtualPolynomial, writer: anytype) !void {
        switch (self) {
            .PC => try writer.writeByte(0),
            .UnexpandedPC => try writer.writeByte(1),
            .NextPC => try writer.writeByte(2),
            .NextUnexpandedPC => try writer.writeByte(3),
            .NextIsNoop => try writer.writeByte(4),
            .NextIsVirtual => try writer.writeByte(5),
            .NextIsFirstInSequence => try writer.writeByte(6),
            .LeftLookupOperand => try writer.writeByte(7),
            .RightLookupOperand => try writer.writeByte(8),
            .LeftInstructionInput => try writer.writeByte(9),
            .RightInstructionInput => try writer.writeByte(10),
            .Product => try writer.writeByte(11),
            .ShouldJump => try writer.writeByte(12),
            .ShouldBranch => try writer.writeByte(13),
            .WritePCtoRD => try writer.writeByte(14),
            .WriteLookupOutputToRD => try writer.writeByte(15),
            .Rd => try writer.writeByte(16),
            .Imm => try writer.writeByte(17),
            .Rs1Value => try writer.writeByte(18),
            .Rs2Value => try writer.writeByte(19),
            .RdWriteValue => try writer.writeByte(20),
            .Rs1Ra => try writer.writeByte(21),
            .Rs2Ra => try writer.writeByte(22),
            .RdWa => try writer.writeByte(23),
            .LookupOutput => try writer.writeByte(24),
            .InstructionRaf => try writer.writeByte(25),
            .InstructionRafFlag => try writer.writeByte(26),
            .InstructionRa => |i| {
                try writer.writeByte(27);
                try writer.writeByte(@truncate(i));
            },
            .RegistersVal => try writer.writeByte(28),
            .RamAddress => try writer.writeByte(29),
            .RamRa => try writer.writeByte(30),
            .RamReadValue => try writer.writeByte(31),
            .RamWriteValue => try writer.writeByte(32),
            .RamVal => try writer.writeByte(33),
            .RamValInit => try writer.writeByte(34),
            .RamValFinal => try writer.writeByte(35),
            .RamHammingWeight => try writer.writeByte(36),
            .UnivariateSkip => try writer.writeByte(37),
            .OpFlags => |f| {
                try writer.writeByte(38);
                try writer.writeByte(f);
            },
            .InstructionFlags => |f| {
                try writer.writeByte(39);
                try writer.writeByte(f);
            },
            .LookupTableFlag => |f| {
                try writer.writeByte(40);
                try writer.writeByte(@truncate(f));
            },
        }
    }
};

// =============================================================================
// OpeningId - Identifies a polynomial opening claim
// =============================================================================

/// Matches Jolt's OpeningId enum
/// Reference: jolt-core/src/poly/opening_proof.rs
pub const OpeningId = union(enum) {
    Committed: struct { poly: CommittedPolynomial, sumcheck_id: SumcheckId },
    Virtual: struct { poly: VirtualPolynomial, sumcheck_id: SumcheckId },
    UntrustedAdvice: SumcheckId,
    TrustedAdvice: SumcheckId,

    // Encoding bases for compact serialization
    pub const UNTRUSTED_ADVICE_BASE: u8 = 0;
    pub const TRUSTED_ADVICE_BASE: u8 = UNTRUSTED_ADVICE_BASE + SumcheckId.COUNT;
    pub const COMMITTED_BASE: u8 = TRUSTED_ADVICE_BASE + SumcheckId.COUNT;
    pub const VIRTUAL_BASE: u8 = COMMITTED_BASE + SumcheckId.COUNT;

    /// Serialize in Jolt's compact format
    pub fn serialize(self: OpeningId, writer: anytype) !void {
        switch (self) {
            .UntrustedAdvice => |id| {
                try writer.writeByte(UNTRUSTED_ADVICE_BASE + @intFromEnum(id));
            },
            .TrustedAdvice => |id| {
                try writer.writeByte(TRUSTED_ADVICE_BASE + @intFromEnum(id));
            },
            .Committed => |c| {
                try writer.writeByte(COMMITTED_BASE + @intFromEnum(c.sumcheck_id));
                try c.poly.serialize(writer);
            },
            .Virtual => |v| {
                try writer.writeByte(VIRTUAL_BASE + @intFromEnum(v.sumcheck_id));
                try v.poly.serialize(writer);
            },
        }
    }

    /// Compare for ordering (needed for BTreeMap-like structure)
    pub fn order(a: OpeningId, b: OpeningId) std.math.Order {
        // Compare by variant first
        const a_tag = @intFromEnum(std.meta.activeTag(a));
        const b_tag = @intFromEnum(std.meta.activeTag(b));
        if (a_tag != b_tag) {
            return std.math.order(a_tag, b_tag);
        }

        // Same variant - compare by content
        switch (a) {
            .UntrustedAdvice => |id_a| {
                const id_b = b.UntrustedAdvice;
                return std.math.order(@intFromEnum(id_a), @intFromEnum(id_b));
            },
            .TrustedAdvice => |id_a| {
                const id_b = b.TrustedAdvice;
                return std.math.order(@intFromEnum(id_a), @intFromEnum(id_b));
            },
            .Committed => |c_a| {
                const c_b = b.Committed;
                // Compare sumcheck_id first, then poly
                const cmp = std.math.order(@intFromEnum(c_a.sumcheck_id), @intFromEnum(c_b.sumcheck_id));
                if (cmp != .eq) return cmp;
                // For simplicity, use memory comparison (could be more detailed)
                return std.math.order(@intFromEnum(std.meta.activeTag(c_a.poly)), @intFromEnum(std.meta.activeTag(c_b.poly)));
            },
            .Virtual => |v_a| {
                const v_b = b.Virtual;
                const cmp = std.math.order(@intFromEnum(v_a.sumcheck_id), @intFromEnum(v_b.sumcheck_id));
                if (cmp != .eq) return cmp;
                // Compare polynomial by tag first
                const tag_a = @intFromEnum(std.meta.activeTag(v_a.poly));
                const tag_b = @intFromEnum(std.meta.activeTag(v_b.poly));
                if (tag_a != tag_b) {
                    return std.math.order(tag_a, tag_b);
                }
                // Same tag - compare payload if applicable
                return VirtualPolynomial.orderByPayload(v_a.poly, v_b.poly);
            },
        }
    }
};

// =============================================================================
// CompressedUniPoly - Compressed univariate polynomial for sumcheck
// =============================================================================

/// Compressed univariate polynomial
/// Matches Jolt's CompressedUniPoly
///
/// Compression: For polynomial p(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ
/// we store [a₀, a₂, a₃, ..., aₙ] (excluding a₁, the linear term)
///
/// The verifier can recover a₁ from the sumcheck claim:
///   claim = p(0) + p(1) = a₀ + (a₀ + a₁ + a₂ + ... + aₙ) = 2a₀ + a₁ + a₂ + ... + aₙ
///   => a₁ = claim - 2a₀ - a₂ - a₃ - ... - aₙ
pub fn CompressedUniPoly(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Coefficients excluding the linear term (index 1)
        /// For [a₀, a₁, a₂, ..., aₙ], stores [a₀, a₂, a₃, ..., aₙ]
        coeffs_except_linear_term: []F,
        allocator: Allocator,

        /// Create from full coefficient array, compressing by removing the linear term
        pub fn init(allocator: Allocator, coeffs: []const F) !Self {
            // Jolt format: [coeffs[0]] ++ coeffs[2..]
            // i.e., remove index 1 (linear term)
            if (coeffs.len <= 1) {
                // Degree 0 or less: just copy all
                const copy = try allocator.alloc(F, coeffs.len);
                @memcpy(copy, coeffs);
                return Self{
                    .coeffs_except_linear_term = copy,
                    .allocator = allocator,
                };
            }

            // Remove linear term at index 1
            // Result length: coeffs.len - 1
            const compressed_len = coeffs.len - 1;
            const copy = try allocator.alloc(F, compressed_len);

            // Copy coeffs[0] (constant term)
            copy[0] = coeffs[0];

            // Copy coeffs[2..] (quadratic and higher)
            if (coeffs.len > 2) {
                @memcpy(copy[1..], coeffs[2..]);
            }

            return Self{
                .coeffs_except_linear_term = copy,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.coeffs_except_linear_term);
        }

        /// Serialize to match Jolt's format
        pub fn serialize(self: *const Self, writer: anytype) !void {
            // Write length as u64
            try writer.writeInt(u64, self.coeffs_except_linear_term.len, .little);
            // Write each coefficient
            for (self.coeffs_except_linear_term) |coeff| {
                var buf: [32]u8 = undefined;
                for (0..4) |i| {
                    std.mem.writeInt(u64, buf[i * 8 ..][0..8], coeff.limbs[i], .little);
                }
                try writer.writeAll(&buf);
            }
        }
    };
}

// =============================================================================
// SumcheckInstanceProof - Proof for a batched sumcheck instance
// =============================================================================

/// Matches Jolt's SumcheckInstanceProof
/// Reference: jolt-core/src/subprotocols/sumcheck.rs
pub fn SumcheckInstanceProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Compressed univariate polynomials for each round
        compressed_polys: std.ArrayListUnmanaged(CompressedUniPoly(F)),
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .compressed_polys = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.compressed_polys.items) |*poly| {
                poly.deinit();
            }
            self.compressed_polys.deinit(self.allocator);
        }

        pub fn addRoundPoly(self: *Self, coeffs: []const F) !void {
            const poly = try CompressedUniPoly(F).init(self.allocator, coeffs);
            try self.compressed_polys.append(self.allocator, poly);
        }

        /// Serialize to match Jolt's format
        pub fn serialize(self: *const Self, writer: anytype) !void {
            // Write number of round polynomials
            try writer.writeInt(u64, self.compressed_polys.items.len, .little);
            // Write each polynomial
            for (self.compressed_polys.items) |*poly| {
                try poly.serialize(writer);
            }
        }
    };
}

// =============================================================================
// UniSkipFirstRoundProof - First round proof with univariate skipping
// =============================================================================

/// Matches Jolt's UniSkipFirstRoundProof
/// Reference: jolt-core/src/subprotocols/univariate_skip.rs
pub fn UniSkipFirstRoundProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The univariate polynomial for the first round
        uni_poly: []F,
        allocator: Allocator,

        pub fn init(allocator: Allocator, coeffs: []const F) !Self {
            const copy = try allocator.alloc(F, coeffs.len);
            @memcpy(copy, coeffs);
            return Self{
                .uni_poly = copy,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.uni_poly);
        }

        /// Serialize to match Jolt's format
        pub fn serialize(self: *const Self, writer: anytype) !void {
            // Write number of coefficients
            try writer.writeInt(u64, self.uni_poly.len, .little);
            // Write each coefficient
            for (self.uni_poly) |coeff| {
                var buf: [32]u8 = undefined;
                for (0..4) |i| {
                    std.mem.writeInt(u64, buf[i * 8 ..][0..8], coeff.limbs[i], .little);
                }
                try writer.writeAll(&buf);
            }
        }
    };
}

// =============================================================================
// OpeningClaims - Map of opening IDs to claims
// =============================================================================

/// Map of OpeningId to claim values
/// Matches Jolt's Claims<F> wrapper around Openings<F>
pub fn OpeningClaims(comptime F: type) type {
    return struct {
        const Self = @This();
        const Entry = struct {
            id: OpeningId,
            claim: F,
        };

        /// Entries stored sorted by OpeningId
        entries: std.ArrayListUnmanaged(Entry),
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .entries = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.entries.deinit(self.allocator);
        }

        /// Insert a claim, maintaining sorted order
        pub fn insert(self: *Self, id: OpeningId, claim: F) !void {
            // Find insertion point
            var insert_idx: usize = 0;
            for (self.entries.items, 0..) |entry, i| {
                if (id.order(entry.id) != .gt) {
                    insert_idx = i;
                    break;
                }
                insert_idx = i + 1;
            }

            // Check for duplicate
            if (insert_idx < self.entries.items.len) {
                if (id.order(self.entries.items[insert_idx].id) == .eq) {
                    // Update existing
                    self.entries.items[insert_idx].claim = claim;
                    return;
                }
            }

            // Insert new entry
            try self.entries.insert(self.allocator, insert_idx, Entry{ .id = id, .claim = claim });
        }

        /// Get number of entries
        pub fn len(self: *const Self) usize {
            return self.entries.items.len;
        }

        /// Get claim by OpeningId, returns null if not found
        pub fn get(self: *const Self, id: OpeningId) ?F {
            for (self.entries.items) |entry| {
                if (id.order(entry.id) == .eq) {
                    return entry.claim;
                }
            }
            return null;
        }

        /// Serialize to match Jolt's format
        pub fn serialize(self: *const Self, writer: anytype) !void {
            // Write number of entries
            try writer.writeInt(u64, self.entries.items.len, .little);
            // Write each (key, claim) pair
            for (self.entries.items) |entry| {
                try entry.id.serialize(writer);
                // Write claim as 32-byte LE
                var buf: [32]u8 = undefined;
                for (0..4) |i| {
                    std.mem.writeInt(u64, buf[i * 8 ..][0..8], entry.claim.limbs[i], .little);
                }
                try writer.writeAll(&buf);
            }
        }
    };
}

// =============================================================================
// JoltProof - Complete Jolt-compatible proof structure
// =============================================================================

/// Complete proof structure matching Jolt's JoltProof
/// Reference: jolt-core/src/zkvm/proof_serialization.rs
pub fn JoltProof(comptime F: type, comptime Commitment: type, comptime Proof: type) type {
    return struct {
        const Self = @This();

        /// Opening claims map
        opening_claims: OpeningClaims(F),

        /// Polynomial commitments
        commitments: std.ArrayListUnmanaged(Commitment),

        /// Stage 1: Outer Spartan
        stage1_uni_skip_first_round_proof: ?UniSkipFirstRoundProof(F),
        stage1_sumcheck_proof: SumcheckInstanceProof(F),

        /// Stage 2: Product virtualization
        stage2_uni_skip_first_round_proof: ?UniSkipFirstRoundProof(F),
        stage2_sumcheck_proof: SumcheckInstanceProof(F),

        /// Stages 3-7: Various reductions
        stage3_sumcheck_proof: SumcheckInstanceProof(F),
        stage4_sumcheck_proof: SumcheckInstanceProof(F),
        stage5_sumcheck_proof: SumcheckInstanceProof(F),
        stage6_sumcheck_proof: SumcheckInstanceProof(F),
        stage7_sumcheck_proof: SumcheckInstanceProof(F),

        /// Joint opening proof (PCS batched opening)
        joint_opening_proof: ?Proof,

        /// Advice opening proofs
        trusted_advice_val_evaluation_proof: ?Proof,
        trusted_advice_val_final_proof: ?Proof,
        untrusted_advice_val_evaluation_proof: ?Proof,
        untrusted_advice_val_final_proof: ?Proof,
        untrusted_advice_commitment: ?Commitment,

        /// Configuration parameters
        trace_length: usize,
        ram_K: usize,
        bytecode_K: usize,
        log_k_chunk: usize,
        lookups_ra_virtual_log_k_chunk: usize,

        /// Allocator
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .opening_claims = OpeningClaims(F).init(allocator),
                .commitments = .{},
                .stage1_uni_skip_first_round_proof = null,
                .stage1_sumcheck_proof = SumcheckInstanceProof(F).init(allocator),
                .stage2_uni_skip_first_round_proof = null,
                .stage2_sumcheck_proof = SumcheckInstanceProof(F).init(allocator),
                .stage3_sumcheck_proof = SumcheckInstanceProof(F).init(allocator),
                .stage4_sumcheck_proof = SumcheckInstanceProof(F).init(allocator),
                .stage5_sumcheck_proof = SumcheckInstanceProof(F).init(allocator),
                .stage6_sumcheck_proof = SumcheckInstanceProof(F).init(allocator),
                .stage7_sumcheck_proof = SumcheckInstanceProof(F).init(allocator),
                .joint_opening_proof = null,
                .trusted_advice_val_evaluation_proof = null,
                .trusted_advice_val_final_proof = null,
                .untrusted_advice_val_evaluation_proof = null,
                .untrusted_advice_val_final_proof = null,
                .untrusted_advice_commitment = null,
                .trace_length = 0,
                .ram_K = 0,
                .bytecode_K = 0,
                .log_k_chunk = 0,
                .lookups_ra_virtual_log_k_chunk = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.opening_claims.deinit();
            self.commitments.deinit(self.allocator);

            if (self.stage1_uni_skip_first_round_proof) |*p| p.deinit();
            self.stage1_sumcheck_proof.deinit();

            if (self.stage2_uni_skip_first_round_proof) |*p| p.deinit();
            self.stage2_sumcheck_proof.deinit();

            self.stage3_sumcheck_proof.deinit();
            self.stage4_sumcheck_proof.deinit();
            self.stage5_sumcheck_proof.deinit();
            self.stage6_sumcheck_proof.deinit();
            self.stage7_sumcheck_proof.deinit();
        }
    };
}

// =============================================================================
// JoltProofWithDory - JoltProof bundled with Dory commitments for serialization
// =============================================================================

/// Bundle of JoltProof with Dory commitments (GT elements)
///
/// This structure carries the proof along with the GT element commitments
/// that were computed during proving. This ensures the same commitments
/// are used for both the transcript and the serialized proof.
pub fn JoltProofWithDory(comptime F: type, comptime Commitment: type, comptime Proof: type) type {
    const Dory = @import("../poly/commitment/dory.zig");
    const GT = Dory.GT;

    return struct {
        const Self = @This();

        proof: JoltProof(F, Commitment, Proof),

        /// Dory commitments (GT elements) computed during proving
        /// Order: bytecode, memory, memory_final, registers, registers_final
        dory_commitments: [5]GT,

        /// Polynomial evaluations used to compute the Dory commitments
        /// These are stored so they can be used for generating the opening proof
        bytecode_evals: []F,
        memory_evals: []F,
        memory_final_evals: []F,
        register_evals: []F,
        register_final_evals: []F,

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .proof = JoltProof(F, Commitment, Proof).init(allocator),
                .dory_commitments = [_]GT{GT.one()} ** 5, // one() is the multiplicative identity for GT
                .bytecode_evals = &[_]F{},
                .memory_evals = &[_]F{},
                .memory_final_evals = &[_]F{},
                .register_evals = &[_]F{},
                .register_final_evals = &[_]F{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.proof.deinit();
            if (self.bytecode_evals.len > 0) self.allocator.free(self.bytecode_evals);
            if (self.memory_evals.len > 0) self.allocator.free(self.memory_evals);
            if (self.memory_final_evals.len > 0) self.allocator.free(self.memory_final_evals);
            if (self.register_evals.len > 0) self.allocator.free(self.register_evals);
            if (self.register_final_evals.len > 0) self.allocator.free(self.register_final_evals);
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = @import("../field/mod.zig").BN254Scalar;

test "SumcheckId count" {
    try testing.expectEqual(@as(usize, 22), SumcheckId.COUNT);
}

test "OpeningId encoding bases" {
    try testing.expectEqual(@as(u8, 0), OpeningId.UNTRUSTED_ADVICE_BASE);
    try testing.expectEqual(@as(u8, 22), OpeningId.TRUSTED_ADVICE_BASE);
    try testing.expectEqual(@as(u8, 44), OpeningId.COMMITTED_BASE);
    try testing.expectEqual(@as(u8, 66), OpeningId.VIRTUAL_BASE);
}

test "OpeningClaims ordering" {
    const Claims = OpeningClaims(BN254Scalar);
    var claims = Claims.init(testing.allocator);
    defer claims.deinit();

    // Insert in non-sorted order
    try claims.insert(.{ .UntrustedAdvice = .RamValEvaluation }, BN254Scalar.fromU64(1));
    try claims.insert(.{ .TrustedAdvice = .SpartanOuter }, BN254Scalar.fromU64(2));
    try claims.insert(.{ .UntrustedAdvice = .SpartanOuter }, BN254Scalar.fromU64(3));

    try testing.expectEqual(@as(usize, 3), claims.len());

    // Verify sorted order: UntrustedAdvice comes before TrustedAdvice
    try testing.expectEqual(SumcheckId.SpartanOuter, claims.entries.items[0].id.UntrustedAdvice);
    try testing.expectEqual(SumcheckId.RamValEvaluation, claims.entries.items[1].id.UntrustedAdvice);
    try testing.expectEqual(SumcheckId.SpartanOuter, claims.entries.items[2].id.TrustedAdvice);
}

test "SumcheckInstanceProof basic" {
    const Proof = SumcheckInstanceProof(BN254Scalar);
    var proof = Proof.init(testing.allocator);
    defer proof.deinit();

    const coeffs = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
    };

    try proof.addRoundPoly(&coeffs);
    try proof.addRoundPoly(&coeffs);

    try testing.expectEqual(@as(usize, 2), proof.compressed_polys.items.len);
}

test "UniSkipFirstRoundProof basic" {
    const UniProof = UniSkipFirstRoundProof(BN254Scalar);

    const coeffs = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(4),
    };

    var proof = try UniProof.init(testing.allocator, &coeffs);
    defer proof.deinit();

    try testing.expectEqual(@as(usize, 4), proof.uni_poly.len);
}

test "JoltProof initialization" {
    // Use a dummy commitment type for testing
    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    const JProof = JoltProof(BN254Scalar, DummyCommitment, DummyProof);
    var proof = JProof.init(testing.allocator);
    defer proof.deinit();

    try testing.expectEqual(@as(usize, 0), proof.trace_length);
    try testing.expectEqual(@as(usize, 0), proof.opening_claims.len());
}
