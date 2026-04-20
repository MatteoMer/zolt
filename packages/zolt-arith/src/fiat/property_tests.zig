//! Algebraic property tests for fiat-crypto field types.
//!
//! Validates field axioms (commutativity, associativity, distributivity,
//! identity, inverse, square consistency, serialization round-trip) and
//! cross-validates every operation against MontgomeryField.

const std = @import("std");
const testing = std.testing;
const fiat_bn254 = @import("bn254.zig");
const FiatFp = fiat_bn254.Fp;
const FiatFr = fiat_bn254.Fr;
const MontFp = @import("../curves/bn254/mod.zig").Fp;
const MontFr = @import("../curves/bn254/mod.zig").Fr;

const ITERATIONS = 1000;
const SEED: u64 = 0xdeadbeef_cafebabe;

fn randomLimbs(rng: *std.Random.Xoshiro256) [4]u64 {
    return .{ rng.random().int(u64), rng.random().int(u64), rng.random().int(u64), rng.random().int(u64) };
}

fn randomField(comptime F: type, rng: *std.Random.Xoshiro256) F {
    // Create from raw limbs — fromRaw handles reduction via to_montgomery.
    return F.fromRaw(randomLimbs(rng));
}

// ── Commutativity ──────────────────────────────────────────────────────

fn testCommutativity(comptime F: type) !void {
    var rng = std.Random.Xoshiro256.init(SEED);
    for (0..ITERATIONS) |_| {
        const a = randomField(F, &rng);
        const b = randomField(F, &rng);
        try testing.expect(F.eql(F.add(a, b), F.add(b, a)));
        try testing.expect(F.eql(F.mul(a, b), F.mul(b, a)));
    }
}

test "FiatFp: commutativity" { try testCommutativity(FiatFp); }
test "FiatFr: commutativity" { try testCommutativity(FiatFr); }

// ── Associativity ──────────────────────────────────────────────────────

fn testAssociativity(comptime F: type) !void {
    var rng = std.Random.Xoshiro256.init(SEED +% 1);
    for (0..ITERATIONS) |_| {
        const a = randomField(F, &rng);
        const b = randomField(F, &rng);
        const c = randomField(F, &rng);
        try testing.expect(F.eql(F.add(F.add(a, b), c), F.add(a, F.add(b, c))));
        try testing.expect(F.eql(F.mul(F.mul(a, b), c), F.mul(a, F.mul(b, c))));
    }
}

test "FiatFp: associativity" { try testAssociativity(FiatFp); }
test "FiatFr: associativity" { try testAssociativity(FiatFr); }

// ── Distributivity ─────────────────────────────────────────────────────

fn testDistributivity(comptime F: type) !void {
    var rng = std.Random.Xoshiro256.init(SEED +% 2);
    for (0..ITERATIONS) |_| {
        const a = randomField(F, &rng);
        const b = randomField(F, &rng);
        const c = randomField(F, &rng);
        try testing.expect(F.eql(F.mul(a, F.add(b, c)), F.add(F.mul(a, b), F.mul(a, c))));
    }
}

test "FiatFp: distributivity" { try testDistributivity(FiatFp); }
test "FiatFr: distributivity" { try testDistributivity(FiatFr); }

// ── Identity elements ──────────────────────────────────────────────────

fn testIdentity(comptime F: type) !void {
    var rng = std.Random.Xoshiro256.init(SEED +% 3);
    for (0..ITERATIONS) |_| {
        const a = randomField(F, &rng);
        try testing.expect(F.eql(F.add(a, F.zero()), a));
        try testing.expect(F.eql(F.mul(a, F.one()), a));
    }
}

test "FiatFp: identity" { try testIdentity(FiatFp); }
test "FiatFr: identity" { try testIdentity(FiatFr); }

// ── Inverse ────────────────────────────────────────────────────────────

fn testInverse(comptime F: type) !void {
    var rng = std.Random.Xoshiro256.init(SEED +% 4);
    for (0..ITERATIONS) |_| {
        const a = randomField(F, &rng);
        try testing.expect(F.eql(F.add(a, F.neg(a)), F.zero()));
        if (!a.isZero()) {
            const inv_a = F.inverse(a) orelse return error.InverseOfNonZero;
            try testing.expect(F.eql(F.mul(a, inv_a), F.one()));
        }
    }
}

test "FiatFp: inverse" { try testInverse(FiatFp); }
test "FiatFr: inverse" { try testInverse(FiatFr); }

// ── Square consistency ─────────────────────────────────────────────────

fn testSquare(comptime F: type) !void {
    var rng = std.Random.Xoshiro256.init(SEED +% 5);
    for (0..ITERATIONS) |_| {
        const a = randomField(F, &rng);
        try testing.expect(F.eql(F.square(a), F.mul(a, a)));
    }
}

test "FiatFp: square == mul(a,a)" { try testSquare(FiatFp); }
test "FiatFr: square == mul(a,a)" { try testSquare(FiatFr); }

// ── Serialization round-trip ───────────────────────────────────────────

fn testSerialization(comptime F: type) !void {
    var rng = std.Random.Xoshiro256.init(SEED +% 6);
    for (0..ITERATIONS) |_| {
        const a = randomField(F, &rng);
        const bytes = F.toBytesBE(a);
        const recovered = F.fromBytesBE(&bytes);
        try testing.expect(F.eql(a, recovered));
    }
}

test "FiatFp: fromBytesBE(toBytesBE(a)) == a" { try testSerialization(FiatFp); }
test "FiatFr: fromBytesBE(toBytesBE(a)) == a" { try testSerialization(FiatFr); }

// ── Cross-validation vs MontgomeryField ────────────────────────────────

fn testCrossValidation(comptime Fiat: type, comptime Mont: type) !void {
    var rng = std.Random.Xoshiro256.init(SEED +% 7);
    for (0..ITERATIONS) |_| {
        const limbs_a = randomLimbs(&rng);
        const limbs_b = randomLimbs(&rng);

        const fa = Fiat.fromRaw(limbs_a);
        const fb = Fiat.fromRaw(limbs_b);
        const ma = Mont.fromRaw(limbs_a);
        const mb = Mont.fromRaw(limbs_b);

        // add
        try testing.expectEqual(Fiat.add(fa, fb).limbs, Mont.add(ma, mb).limbs);
        // sub
        try testing.expectEqual(Fiat.sub(fa, fb).limbs, Mont.sub(ma, mb).limbs);
        // mul
        try testing.expectEqual(Fiat.mul(fa, fb).limbs, Mont.mul(ma, mb).limbs);
        // square
        try testing.expectEqual(Fiat.square(fa).limbs, Mont.square(ma).limbs);
        // neg
        try testing.expectEqual(Fiat.neg(fa).limbs, Mont.neg(ma).limbs);
        // toBytesBE
        try testing.expectEqual(Fiat.toBytesBE(fa), Mont.toBytesBE(ma));
    }
}

test "Cross: FiatFp vs MontFp (1000 random)" { try testCrossValidation(FiatFp, MontFp); }
test "Cross: FiatFr vs MontFr (1000 random)" { try testCrossValidation(FiatFr, MontFr); }
