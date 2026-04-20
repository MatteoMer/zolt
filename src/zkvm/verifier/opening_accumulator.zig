//! Verifier Opening Accumulator
//!
//! Collects polynomial opening claims across verification stages.
//! Each stage's sumcheck instances call `cacheOpenings()` to register
//! claims, which are then:
//!   - Read by subsequent stages via `get()` (e.g., Stage 1 uni-skip → Stage 1 remaining)
//!   - Flushed to the Fiat-Shamir transcript at stage boundaries
//!   - Collected for batch Dory PCS verification in Stage 8
//!
//! Mirrors Rust's `VerifierOpeningAccumulator` from opening_proof.rs.

const std = @import("std");
const Allocator = std.mem.Allocator;

const jolt_types = @import("../jolt_types.zig");
const OpeningId = jolt_types.OpeningId;
const VirtualPolynomial = jolt_types.VirtualPolynomial;
const CommittedPolynomial = jolt_types.CommittedPolynomial;
const SumcheckId = jolt_types.SumcheckId;

/// An accumulated opening claim: the polynomial evaluation point and claimed value.
pub fn OpeningEntry(comptime F: type) type {
    return struct {
        /// The sumcheck challenge point where the polynomial is opened
        point: []const F,
        /// The claimed evaluation value
        claim: F,
    };
}

/// Accumulates polynomial opening claims across all verification stages.
///
/// Claims are stored by OpeningId and can be retrieved by subsequent stages.
/// Pending claims (not yet flushed) are tracked separately for transcript binding.
pub fn VerifierOpeningAccumulator(comptime F: type) type {
    return struct {
        const Self = @This();
        const Entry = OpeningEntry(F);

        /// All accumulated openings, keyed by OpeningId.
        openings: std.ArrayListUnmanaged(IdEntry),
        /// Indices of claims not yet flushed to transcript.
        pending_indices: std.ArrayListUnmanaged(usize),
        allocator: Allocator,

        const IdEntry = struct {
            id: OpeningId,
            entry: Entry,
        };

        pub fn init(allocator: Allocator) Self {
            return Self{
                .openings = .{},
                .pending_indices = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.openings.deinit(self.allocator);
            self.pending_indices.deinit(self.allocator);
        }

        /// Register a virtual polynomial opening claim.
        pub fn appendVirtual(
            self: *Self,
            poly: VirtualPolynomial,
            sumcheck_id: SumcheckId,
            point: []const F,
            claim: F,
        ) !void {
            const id = OpeningId{ .Virtual = .{ .poly = poly, .sumcheck_id = sumcheck_id } };
            try self.append(id, point, claim);
        }

        /// Register a committed (dense/sparse) polynomial opening claim.
        pub fn appendCommitted(
            self: *Self,
            poly: CommittedPolynomial,
            sumcheck_id: SumcheckId,
            point: []const F,
            claim: F,
        ) !void {
            const id = OpeningId{ .Committed = .{ .poly = poly, .sumcheck_id = sumcheck_id } };
            try self.append(id, point, claim);
        }

        /// Register a trusted advice opening claim.
        pub fn appendTrustedAdvice(
            self: *Self,
            sumcheck_id: SumcheckId,
            point: []const F,
            claim: F,
        ) !void {
            const id = OpeningId{ .TrustedAdvice = sumcheck_id };
            try self.append(id, point, claim);
        }

        /// Register an untrusted advice opening claim.
        pub fn appendUntrustedAdvice(
            self: *Self,
            sumcheck_id: SumcheckId,
            point: []const F,
            claim: F,
        ) !void {
            const id = OpeningId{ .UntrustedAdvice = sumcheck_id };
            try self.append(id, point, claim);
        }

        /// Get a virtual polynomial opening by (poly, sumcheck_id).
        /// Returns null if not found.
        pub fn getVirtual(
            self: *const Self,
            poly: VirtualPolynomial,
            sumcheck_id: SumcheckId,
        ) ?Entry {
            const id = OpeningId{ .Virtual = .{ .poly = poly, .sumcheck_id = sumcheck_id } };
            return self.get(id);
        }

        /// Get a committed polynomial opening by (poly, sumcheck_id).
        /// Returns null if not found.
        pub fn getCommitted(
            self: *const Self,
            poly: CommittedPolynomial,
            sumcheck_id: SumcheckId,
        ) ?Entry {
            const id = OpeningId{ .Committed = .{ .poly = poly, .sumcheck_id = sumcheck_id } };
            return self.get(id);
        }

        /// Look up an opening by exact OpeningId.
        pub fn get(self: *const Self, id: OpeningId) ?Entry {
            for (self.openings.items) |entry| {
                if (id.order(entry.id) == .eq) {
                    return entry.entry;
                }
            }
            return null;
        }

        /// Flush all pending claims to the Fiat-Shamir transcript.
        /// After flushing, the pending buffer is cleared.
        pub fn flushToTranscript(self: *Self, transcript: anytype) void {
            if (self.pending_indices.items.len == 0) return;

            // Append each pending claim value to transcript
            for (self.pending_indices.items) |idx| {
                transcript.appendScalar("opening_claim", self.openings.items[idx].entry.claim);
            }
            self.pending_indices.clearRetainingCapacity();
        }

        /// Number of total accumulated claims.
        pub fn count(self: *const Self) usize {
            return self.openings.items.len;
        }

        /// Number of claims pending transcript flush.
        pub fn pendingCount(self: *const Self) usize {
            return self.pending_indices.items.len;
        }

        // Internal: append a claim and mark it pending.
        fn append(self: *Self, id: OpeningId, point: []const F, claim: F) !void {
            // Check for existing entry with same id — update if found
            for (self.openings.items, 0..) |*entry, i| {
                if (id.order(entry.id) == .eq) {
                    entry.entry = Entry{ .point = point, .claim = claim };
                    // Re-add to pending
                    try self.pending_indices.append(self.allocator, i);
                    return;
                }
            }

            const idx = self.openings.items.len;
            try self.openings.append(self.allocator, IdEntry{
                .id = id,
                .entry = Entry{ .point = point, .claim = claim },
            });
            try self.pending_indices.append(self.allocator, idx);
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const zolt_arith = @import("zolt_arith");
const BN254Scalar = zolt_arith.field.BN254Scalar;

test "opening accumulator: append and retrieve" {
    const F_ = BN254Scalar;
    var acc = VerifierOpeningAccumulator(F_).init(testing.allocator);
    defer acc.deinit();

    const point = [_]F_{ F_.fromU64(42) };
    const claim = F_.fromU64(100);

    try acc.appendVirtual(.PC, .SpartanOuter, &point, claim);

    const entry = acc.getVirtual(.PC, .SpartanOuter);
    try testing.expect(entry != null);
    try testing.expect(entry.?.claim.eql(claim));
    try testing.expectEqual(@as(usize, 1), acc.count());
    try testing.expectEqual(@as(usize, 1), acc.pendingCount());
}

test "opening accumulator: committed polynomial" {
    const F_ = BN254Scalar;
    var acc = VerifierOpeningAccumulator(F_).init(testing.allocator);
    defer acc.deinit();

    const point = [_]F_{ F_.fromU64(7), F_.fromU64(8) };
    try acc.appendCommitted(.RdInc, .SpartanOuter, &point, F_.fromU64(99));

    const entry = acc.getCommitted(.RdInc, .SpartanOuter);
    try testing.expect(entry != null);
    try testing.expect(entry.?.claim.eql(F_.fromU64(99)));
}

test "opening accumulator: flush to transcript clears pending" {
    const F_ = BN254Scalar;
    const Blake2bTranscript = zolt_arith.transcripts.Blake2bTranscript;

    var acc = VerifierOpeningAccumulator(F_).init(testing.allocator);
    defer acc.deinit();

    const point = [_]F_{ F_.fromU64(1) };
    try acc.appendVirtual(.PC, .SpartanOuter, &point, F_.fromU64(10));
    try acc.appendVirtual(.NextPC, .SpartanOuter, &point, F_.fromU64(20));

    try testing.expectEqual(@as(usize, 2), acc.pendingCount());

    var transcript = Blake2bTranscript(F_).init("test");
    acc.flushToTranscript(&transcript);

    try testing.expectEqual(@as(usize, 0), acc.pendingCount());
    try testing.expectEqual(@as(usize, 2), acc.count()); // still stored
}

test "opening accumulator: update existing entry" {
    const F_ = BN254Scalar;
    var acc = VerifierOpeningAccumulator(F_).init(testing.allocator);
    defer acc.deinit();

    const point = [_]F_{ F_.fromU64(1) };
    try acc.appendVirtual(.PC, .SpartanOuter, &point, F_.fromU64(10));
    try acc.appendVirtual(.PC, .SpartanOuter, &point, F_.fromU64(20));

    // Should have updated, not duplicated
    try testing.expectEqual(@as(usize, 1), acc.count());
    const entry = acc.getVirtual(.PC, .SpartanOuter);
    try testing.expect(entry.?.claim.eql(F_.fromU64(20)));
}

test "opening accumulator: miss returns null" {
    const F_ = BN254Scalar;
    var acc = VerifierOpeningAccumulator(F_).init(testing.allocator);
    defer acc.deinit();

    try testing.expect(acc.getVirtual(.PC, .SpartanOuter) == null);
}
