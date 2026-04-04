//! Dory Verifier Setup Serialization
//!
//! Precomputed pairing values for Dory-based verification.
//! Matches Jolt's VerifierSetup<BN254> structure.

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const dory = zolt_arith.poly.commitment.dory;
const pairing = zolt_arith.field.pairing;
const field_mod = zolt_arith.field;
const Fp = field_mod.BN254BaseField;
const ThreadPool = @import("zolt_pool").ThreadPool;

pub const GT = dory.GT;
pub const G1Point = dory.G1Point;
pub const G2Point = dory.G2Point;
pub const DorySRS = dory.DorySRS;

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

/// Convert G1Point x-coordinate to Fp
/// G1Point stores x,y in BN254Scalar (Fr) Montgomery form
/// We need to convert to BN254BaseField (Fp) for pairing
fn g1PointXToFp(p: G1Point) Fp {
    // The x coordinate is stored in Montgomery form for Fr
    // Both Fr and Fp have the same limb structure, just different moduli
    // Since the G1 generator coords are valid in both fields (they're small),
    // we can interpret the limbs directly
    return Fp{ .limbs = p.x.limbs };
}

fn g1PointYToFp(p: G1Point) Fp {
    return Fp{ .limbs = p.y.limbs };
}

/// Multi-pairing of G1 and G2 vectors using batch Miller loop + shared final exponentiation.
/// Optionally parallelizes Miller loop computation across threads.
fn multiPair(g1_vec: []const G1Point, g2_vec: []const G2Point, tp: ?*ThreadPool) GT {
    const n = @min(g1_vec.len, g2_vec.len);
    if (n == 0) return GT.one();

    const Ctx = struct { g1: []const G1Point, g2: []const G2Point };
    const ctx = Ctx{ .g1 = g1_vec, .g2 = g2_vec };

    const mapFn = struct {
        fn map(c: Ctx, start: usize, end: usize) pairing.Fp12 {
            var acc = pairing.Fp12.one();
            for (start..end) |i| {
                if (c.g1[i].infinity or c.g2[i].infinity) continue;
                const g1_fp = pairing.G1PointFp{
                    .x = g1PointXToFp(c.g1[i]),
                    .y = g1PointYToFp(c.g1[i]),
                    .infinity = false,
                };
                const ml = pairing.millerLoopArkworks(g1_fp, c.g2[i]);
                acc = acc.mul(ml);
            }
            return acc;
        }
    }.map;

    const reduceFn = struct {
        fn reduce(a: pairing.Fp12, b: pairing.Fp12) pairing.Fp12 {
            return a.mul(b);
        }
    }.reduce;

    const miller_acc = if (tp) |pool|
        pool.parallelReduceForce(pairing.Fp12, n, pairing.Fp12.one(), ctx, mapFn, reduceFn)
    else
        mapFn(ctx, 0, n);

    if (miller_acc.eql(pairing.Fp12.one())) return GT.one();
    return pairing.finalExponentiation(miller_acc);
}

/// DoryVerifierSetup - precomputed pairing values for verification
/// Matches Jolt's VerifierSetup<BN254> structure
pub const DoryVerifierSetup = struct {
    /// Δ₁L[k] = e(Γ₁[..2^(k-1)], Γ₂[..2^(k-1)])
    delta_1l: std.ArrayListUnmanaged(GT),
    /// Δ₁R[k] = e(Γ₁[2^(k-1)..2^k], Γ₂[..2^(k-1)])
    delta_1r: std.ArrayListUnmanaged(GT),
    /// Δ₂L[k] = same as Δ₁L[k]
    delta_2l: std.ArrayListUnmanaged(GT),
    /// Δ₂R[k] = e(Γ₁[..2^(k-1)], Γ₂[2^(k-1)..2^k])
    delta_2r: std.ArrayListUnmanaged(GT),
    /// χ[k] = e(Γ₁[..2^k], Γ₂[..2^k])
    chi: std.ArrayListUnmanaged(GT),
    /// First G1 generator
    g1_0: G1Point,
    /// First G2 generator
    g2_0: G2Point,
    /// Blinding generator in G1
    h1: G1Point,
    /// Blinding generator in G2
    h2: G2Point,
    /// h_t = e(h₁, h₂)
    ht: GT,
    /// Maximum log₂ of polynomial size supported
    max_log_n: usize,
    /// Allocator used
    allocator: Allocator,

    pub fn deinit(self: *DoryVerifierSetup) void {
        self.delta_1l.deinit(self.allocator);
        self.delta_1r.deinit(self.allocator);
        self.delta_2l.deinit(self.allocator);
        self.delta_2r.deinit(self.allocator);
        self.chi.deinit(self.allocator);
    }

    /// Create verifier setup from prover setup (SRS)
    pub fn fromSRS(allocator: Allocator, srs: *const DorySRS, tp: ?*ThreadPool) !DoryVerifierSetup {
        const max_num_rounds = std.math.log2_int(usize, srs.g1_vec.len);

        var delta_1l = std.ArrayListUnmanaged(GT){};
        var delta_1r = std.ArrayListUnmanaged(GT){};
        var delta_2r = std.ArrayListUnmanaged(GT){};
        var chi = std.ArrayListUnmanaged(GT){};

        try delta_1l.ensureTotalCapacity(allocator, max_num_rounds + 1);
        try delta_1r.ensureTotalCapacity(allocator, max_num_rounds + 1);
        try delta_2r.ensureTotalCapacity(allocator, max_num_rounds + 1);
        try chi.ensureTotalCapacity(allocator, max_num_rounds + 1);

        for (0..(max_num_rounds + 1)) |k| {
            if (k == 0) {
                // Base case: identities for deltas, single pairing for chi
                try delta_1l.append(allocator, GT.one());
                try delta_1r.append(allocator, GT.one());
                try delta_2r.append(allocator, GT.one());
                // chi[0] = e(g1_vec[0], g2_vec[0])
                const g1_0_fp = pairing.G1PointFp{
                    .x = g1PointXToFp(srs.g1_vec[0]),
                    .y = g1PointYToFp(srs.g1_vec[0]),
                    .infinity = srs.g1_vec[0].infinity,
                };
                const chi_0 = pairing.pairingFp(g1_0_fp, srs.g2_vec[0]);
                try chi.append(allocator, chi_0);
            } else {
                const half_len = @as(usize, 1) << @intCast(k - 1);
                const full_len = @as(usize, 1) << @intCast(k);

                const g1_first_half = srs.g1_vec[0..half_len];
                const g1_second_half = srs.g1_vec[half_len..full_len];
                const g2_first_half = srs.g2_vec[0..half_len];
                const g2_second_half = srs.g2_vec[half_len..full_len];

                // Δ₁L[k] = χ[k-1] (reuse previous chi)
                try delta_1l.append(allocator, chi.items[k - 1]);

                // Compute 3 independent multi-pairings:
                //   Δ₁R[k] = e(Γ₁[2^(k-1)..2^k], Γ₂[..2^(k-1)])
                //   Δ₂R[k] = e(Γ₁[..2^(k-1)], Γ₂[2^(k-1)..2^k])
                //   cross  = e(Γ₁[2^(k-1)..2^k], Γ₂[2^(k-1)..2^k])
                // Batch all 3 independent multi-pairings into one parallelReduceForce call.
                const batch = dory.multiPairBatched(3, .{
                    dory.PairGroup{ .g1 = g1_second_half, .g2 = g2_first_half }, // delta_1r
                    dory.PairGroup{ .g1 = g1_first_half, .g2 = g2_second_half }, // delta_2r
                    dory.PairGroup{ .g1 = g1_second_half, .g2 = g2_second_half }, // cross
                }, tp);
                try delta_1r.append(allocator, batch[0]);
                try delta_2r.append(allocator, batch[1]);

                // χ[k] = χ[k-1] * cross
                const chi_k = chi.items[k - 1].mul(batch[2]);
                try chi.append(allocator, chi_k);
            }
        }

        // delta_2l == delta_1l (clone)
        var delta_2l = std.ArrayListUnmanaged(GT){};
        try delta_2l.ensureTotalCapacity(allocator, delta_1l.items.len);
        for (delta_1l.items) |item| {
            try delta_2l.append(allocator, item);
        }

        // Use h1, h2 from the SRS (these are separate blinding generators)
        const h1 = srs.h1;
        const h2 = srs.h2;
        const h1_fp = pairing.G1PointFp{
            .x = g1PointXToFp(h1),
            .y = g1PointYToFp(h1),
            .infinity = h1.infinity,
        };
        const ht = pairing.pairingFp(h1_fp, h2);

        return DoryVerifierSetup{
            .delta_1l = delta_1l,
            .delta_1r = delta_1r,
            .delta_2l = delta_2l,
            .delta_2r = delta_2r,
            .chi = chi,
            .g1_0 = srs.g1_vec[0],
            .g2_0 = srs.g2_vec[0],
            .h1 = h1,
            .h2 = h2,
            .ht = ht,
            .max_log_n = max_num_rounds * 2, // Since square matrices
            .allocator = allocator,
        };
    }

    /// Serialize to arkworks-compatible format
    /// Matches the CanonicalSerialize impl for VerifierSetup<BN254>
    pub fn serialize(self: *const DoryVerifierSetup, writer: anytype) !void {
        // Serialize delta_1l: Vec<GT>
        try writer.writeInt(u64, @intCast(self.delta_1l.items.len), .little);
        for (self.delta_1l.items) |gt| {
            try serializeGT(gt, writer);
        }

        // Serialize delta_1r: Vec<GT>
        try writer.writeInt(u64, @intCast(self.delta_1r.items.len), .little);
        for (self.delta_1r.items) |gt| {
            try serializeGT(gt, writer);
        }

        // Serialize delta_2l: Vec<GT>
        try writer.writeInt(u64, @intCast(self.delta_2l.items.len), .little);
        for (self.delta_2l.items) |gt| {
            try serializeGT(gt, writer);
        }

        // Serialize delta_2r: Vec<GT>
        try writer.writeInt(u64, @intCast(self.delta_2r.items.len), .little);
        for (self.delta_2r.items) |gt| {
            try serializeGT(gt, writer);
        }

        // Serialize chi: Vec<GT>
        try writer.writeInt(u64, @intCast(self.chi.items.len), .little);
        for (self.chi.items) |gt| {
            try serializeGT(gt, writer);
        }

        // Serialize g1_0: G1
        try serializeG1(self.g1_0, writer);

        // Serialize g2_0: G2
        try serializeG2(self.g2_0, writer);

        // Serialize h1: G1
        try serializeG1(self.h1, writer);

        // Serialize h2: G2
        try serializeG2(self.h2, writer);

        // Serialize ht: GT
        try serializeGT(self.ht, writer);

        // Serialize max_log_n: usize (as u64)
        try writer.writeInt(u64, @intCast(self.max_log_n), .little);
    }
};

/// Serialize GT element in arkworks format (uncompressed Fq12)
pub fn serializeGT(gt: GT, writer: anytype) !void {
    // GT is Fp12 = (Fp6, Fp6) where Fp6 = (Fp2, Fp2, Fp2)
    // arkworks serializes as c0 first, then c1
    // Each Fp2 is (c0, c1) where each Fp is 32 bytes LE

    // Serialize gt.c0 (Fp6)
    try serializeFp6(&gt.c0, writer);
    // Serialize gt.c1 (Fp6)
    try serializeFp6(&gt.c1, writer);
}

pub fn serializeFp6(fp6: *const pairing.Fp6, writer: anytype) !void {
    // Fp6 = (Fp2, Fp2, Fp2)
    try serializeFp2(&fp6.c0, writer);
    try serializeFp2(&fp6.c1, writer);
    try serializeFp2(&fp6.c2, writer);
}

pub fn serializeFp2(fp2: *const pairing.Fp2, writer: anytype) !void {
    // Fp2 = (c0, c1) where each is Fp
    try serializeFp(&fp2.c0, writer);
    try serializeFp(&fp2.c1, writer);
}

pub fn serializeFp(fp: *const Fp, writer: anytype) !void {
    // Fp is 32 bytes in little-endian (standard form)
    const std_form = fp.fromMontgomery();
    for (0..4) |i| {
        try writer.writeInt(u64, std_form.limbs[i], .little);
    }
}

pub fn serializeG1(point: G1Point, writer: anytype) !void {
    // G1 compressed serialization: 32 bytes (x coordinate + flags)
    // arkworks compressed format: x with flags in MSB of last limb
    if (point.infinity) {
        // Point at infinity: write 0s with infinity flag (bit 62)
        for (0..3) |_| {
            try writer.writeInt(u64, 0, .little);
        }
        // Set infinity flag (bit 62 = 0x4000_0000_0000_0000)
        try writer.writeInt(u64, 0x4000000000000000, .little);
        return;
    }

    // Write x coordinate (32 bytes LE, standard form)
    const x_std = point.x.fromMontgomery();
    for (0..3) |i| {
        try writer.writeInt(u64, x_std.limbs[i], .little);
    }

    // Set flag bits in top byte of last limb of x:
    // - bit 63: y sign (positive_y_over_neg_y flag)
    // - bit 62: infinity (already handled above)
    const neg_y = point.y.neg();
    const y_is_positive = lexicographicallyLess(point.y, neg_y);
    var last_limb = x_std.limbs[3];
    if (!y_is_positive) {
        last_limb |= 0x8000000000000000; // Set bit 63 if y is "negative"
    }
    try writer.writeInt(u64, last_limb, .little);
}

pub fn serializeG2(point: G2Point, writer: anytype) !void {
    // G2 compressed serialization: 64 bytes (x as Fp2 + flags)
    // arkworks compressed format: x.c0, x.c1 with flags in MSB of x.c1's last limb
    if (point.infinity) {
        // Point at infinity: write 0s with infinity flag (bit 62 of x.c1)
        for (0..7) |_| {
            try writer.writeInt(u64, 0, .little);
        }
        // Set infinity flag (bit 62 = 0x4000_0000_0000_0000)
        try writer.writeInt(u64, 0x4000000000000000, .little);
        return;
    }

    // Write x.c0 (32 bytes)
    const x_c0_std = point.x.c0.fromMontgomery();
    for (0..4) |i| {
        try writer.writeInt(u64, x_c0_std.limbs[i], .little);
    }

    // Write x.c1 (32 bytes) with flags in MSB
    const x_c1_std = point.x.c1.fromMontgomery();
    for (0..3) |i| {
        try writer.writeInt(u64, x_c1_std.limbs[i], .little);
    }

    // Set flag bits in top byte of last limb of x.c1:
    // - bit 63: y sign (for Fp2, compare lexicographically)
    // - bit 62: infinity (already handled above)
    const neg_y = point.y.neg();
    const y_is_positive = lexicographicallyLessFp2(point.y, neg_y);
    var last_limb = x_c1_std.limbs[3];
    if (!y_is_positive) {
        last_limb |= 0x8000000000000000; // Set bit 63 if y is "negative"
    }
    try writer.writeInt(u64, last_limb, .little);
}

pub fn lexicographicallyLess(a: Fp, b: Fp) bool {
    const a_std = a.fromMontgomery();
    const b_std = b.fromMontgomery();
    var i: usize = 4;
    while (i > 0) {
        i -= 1;
        if (a_std.limbs[i] < b_std.limbs[i]) return true;
        if (a_std.limbs[i] > b_std.limbs[i]) return false;
    }
    return false; // Equal
}

pub fn lexicographicallyLessFp2(a: pairing.Fp2, b: pairing.Fp2) bool {
    // For Fp2 = (c0, c1), compare c1 first (more significant), then c0
    // This matches arkworks' lexicographic ordering for Fp2
    const a_c1_std = a.c1.fromMontgomery();
    const b_c1_std = b.c1.fromMontgomery();

    // Compare c1 (more significant)
    var i: usize = 4;
    while (i > 0) {
        i -= 1;
        if (a_c1_std.limbs[i] < b_c1_std.limbs[i]) return true;
        if (a_c1_std.limbs[i] > b_c1_std.limbs[i]) return false;
    }

    // c1 is equal, compare c0
    const a_c0_std = a.c0.fromMontgomery();
    const b_c0_std = b.c0.fromMontgomery();

    i = 4;
    while (i > 0) {
        i -= 1;
        if (a_c0_std.limbs[i] < b_c0_std.limbs[i]) return true;
        if (a_c0_std.limbs[i] > b_c0_std.limbs[i]) return false;
    }
    return false; // Equal
}

test "DoryVerifierSetup serialization" {
    const allocator = std.testing.allocator;
    const DoryCommitmentScheme = dory.DoryCommitmentScheme(@import("zolt_arith").field.BN254Scalar);

    // Create a small SRS for testing
    var srs = try DoryCommitmentScheme.setup(allocator, 4);
    defer srs.deinit();

    // Create verifier setup from SRS
    var verifier_setup = try DoryVerifierSetup.fromSRS(allocator, &srs, null);
    defer verifier_setup.deinit();

    // Check that delta/chi arrays have correct sizes
    // For 4 variables (16 coeffs), we get sigma=2, nu=2
    // g1_vec.len = 4, g2_vec.len = 4
    // max_num_rounds = log2(4) = 2
    // So we have 3 entries (k=0,1,2)
    try std.testing.expect(verifier_setup.delta_1l.items.len == 3);
    try std.testing.expect(verifier_setup.chi.items.len == 3);

    // Test serialization
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try verifier_setup.serialize(buf.writer(allocator));

    // Should produce non-empty output
    try std.testing.expect(buf.items.len > 0);
    dbg("Verifier setup serialized to {} bytes\n", .{buf.items.len});
}
