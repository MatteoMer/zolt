const std = @import("std");
const parser = @import("vector_parser.zig");
const zolt = @import("zolt");
const diff_config = @import("diff_config");

const field = zolt.field;
const pairing = field.pairing;
const msm = zolt.msm;
const glv = msm.glv;
const accumulators = field.accumulators;
const transcripts = zolt.transcripts;
const dory = zolt.poly.commitment.dory;

const Fr = field.BN254Scalar;
const Fp = field.BN254BaseField;
const Fp2 = pairing.Fp2;
const Fp6 = pairing.Fp6;
const Fp12 = pairing.Fp12;
const G1Affine = msm.AffinePoint(Fp);
const G2Point = pairing.G2Point;
const G1MSM = msm.MSM(Fr, Fp);
const Transcript = transcripts.Blake2bTranscript(Fr);

fn readFixtureAlloc(allocator: std.mem.Allocator, relative_path: []const u8) ![]u8 {
    const full_path = try std.fs.path.join(allocator, &.{ diff_config.fixtures_root, relative_path });
    defer allocator.free(full_path);
    return std.Io.Dir.cwd().readFileAlloc(std.Io.Threaded.global_single_threaded.io(), full_path, allocator, .limited(16 * 1024 * 1024));
}

fn fieldToBytesLE(comptime F: type, value: F) [32]u8 {
    const standard = value.fromMontgomery();
    var bytes: [32]u8 = undefined;
    inline for (0..4) |i| {
        std.mem.writeInt(u64, bytes[i * 8 ..][0..8], standard.limbs[i], .little);
    }
    return bytes;
}

fn parseFrHex(text: []const u8) !Fr {
    const bytes = try parser.parseHexBytesExact(32, text);
    return Fr.fromBytesBE(&bytes);
}

fn parseFpHex(text: []const u8) !Fp {
    const bytes = try parser.parseHexBytesExact(32, text);
    return Fp.fromBytesBE(&bytes);
}

fn generateG1Bases(allocator: std.mem.Allocator, n: usize) ![]G1Affine {
    var bases = try allocator.alloc(G1Affine, n);
    const gen = G1Affine.generator();
    var proj = msm.ProjectivePoint(Fp).fromAffine(gen);
    for (0..n) |i| {
        bases[i] = proj.toAffine();
        proj = proj.double();
    }
    return bases;
}

fn generateG2Bases(allocator: std.mem.Allocator, n: usize) ![]G2Point {
    var bases = try allocator.alloc(G2Point, n);
    const gen = G2Point.generator();
    var acc = pairing.G2Projective.fromAffine(gen);
    for (0..n) |i| {
        bases[i] = acc.toAffine();
        acc = acc.double();
    }
    return bases;
}

fn parseFrCsv(allocator: std.mem.Allocator, text: []const u8) ![]Fr {
    return parser.parseCsvExact(Fr, text, allocator, parseFrHex);
}

fn parseI128Csv(allocator: std.mem.Allocator, text: []const u8) ![]i128 {
    return parser.parseCsvExact(i128, text, allocator, struct {
        fn parseOne(part: []const u8) !i128 {
            return parser.parseDecimal(i128, part);
        }
    }.parseOne);
}

test "zolt-arith differential field fixtures" {
    const allocator = std.testing.allocator;
    const fixture_sets = [_]struct { path: []const u8, field_type: type }{
        .{ .path = "field/fr_ops.txt", .field_type = Fr },
        .{ .path = "field/fp_ops.txt", .field_type = Fp },
    };

    inline for (fixture_sets) |fixture_set| {
        const fixture_text = try readFixtureAlloc(allocator, fixture_set.path);
        defer allocator.free(fixture_text);
        var lines = std.mem.splitScalar(u8, fixture_text, '\n');
        while (lines.next()) |raw_line| {
            const line = parser.cleanLine(raw_line) orelse continue;
            const fields_split = try parser.splitFieldsExact(4, line, '|');

            const F = fixture_set.field_type;
            const a_bytes = try parser.parseHexBytesExact(32, fields_split[1]);
            const expected_bytes = try parser.parseHexBytesExact(32, fields_split[3]);
            const a = F.fromBytesBE(&a_bytes);
            const expected = F.fromBytesBE(&expected_bytes);

            if (std.mem.eql(u8, fields_split[0], "inv")) {
                try std.testing.expect(a.inverse().?.eql(expected));
                continue;
            }

            const b_bytes = try parser.parseHexBytesExact(32, fields_split[2]);
            const b = F.fromBytesBE(&b_bytes);
            if (std.mem.eql(u8, fields_split[0], "add")) {
                try std.testing.expect(a.add(b).eql(expected));
            } else if (std.mem.eql(u8, fields_split[0], "sub")) {
                try std.testing.expect(a.sub(b).eql(expected));
            } else if (std.mem.eql(u8, fields_split[0], "mul")) {
                try std.testing.expect(a.mul(b).eql(expected));
            } else {
                return error.UnknownFieldOperation;
            }
        }
    }
}

test "zolt-arith differential pairing fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "pairing/generator_cases.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(4, line, '|');

        const g1_scalar = Fr.fromU64(try parser.parseDecimal(u64, fields_split[1]));
        const g2_scalar = Fr.fromU64(try parser.parseDecimal(u64, fields_split[2]));

        const g1_affine = G1MSM.scalarMul(pairing.G1PointInFp.generator(), g1_scalar).toAffine();
        const g1 = pairing.G1PointFp{
            .x = g1_affine.x,
            .y = g1_affine.y,
            .infinity = g1_affine.infinity,
        };
        const g2 = G2Point.generator().scalarMul(g2_scalar);

        const expected = try parser.parseHexBytesExact(384, fields_split[3]);
        const actual = pairing.pairingFp(g1, g2).toBytes();
        try std.testing.expectEqualSlices(u8, &expected, &actual);
    }
}

test "zolt-arith differential msm g1 fr fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "msm/g1_fr_cases.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(5, line, '|');
        const scalars = try parseFrCsv(allocator, fields_split[1]);
        defer allocator.free(scalars);

        const bases = try generateG1Bases(allocator, scalars.len);
        defer allocator.free(bases);

        const actual = G1MSM.computeWithPool(bases, scalars, null);
        const expected_infinity = try parser.parseDecimal(u8, fields_split[2]);
        try std.testing.expectEqual(expected_infinity == 1, actual.infinity);
        if (!actual.infinity) {
            const expected_x = try parser.parseHexBytesExact(32, fields_split[3]);
            const expected_y = try parser.parseHexBytesExact(32, fields_split[4]);
            const actual_x = fieldToBytesLE(Fp, actual.x);
            const actual_y = fieldToBytesLE(Fp, actual.y);
            try std.testing.expectEqualSlices(u8, &expected_x, &actual_x);
            try std.testing.expectEqualSlices(u8, &expected_y, &actual_y);
        }
    }
}

test "zolt-arith differential msm g1 i128 fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "msm/g1_i128_cases.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(5, line, '|');
        const scalars = try parseI128Csv(allocator, fields_split[1]);
        defer allocator.free(scalars);

        const bases = try generateG1Bases(allocator, scalars.len);
        defer allocator.free(bases);

        const actual = G1MSM.computeI128(bases, scalars, null);
        const expected_infinity = try parser.parseDecimal(u8, fields_split[2]);
        try std.testing.expectEqual(expected_infinity == 1, actual.infinity);
        if (!actual.infinity) {
            const expected_x = try parser.parseHexBytesExact(32, fields_split[3]);
            const expected_y = try parser.parseHexBytesExact(32, fields_split[4]);
            const actual_x = fieldToBytesLE(Fp, actual.x);
            const actual_y = fieldToBytesLE(Fp, actual.y);
            try std.testing.expectEqualSlices(u8, &expected_x, &actual_x);
            try std.testing.expectEqualSlices(u8, &expected_y, &actual_y);
        }
    }
}

test "zolt-arith differential msm g2 fr fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "msm/g2_fr_cases.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(7, line, '|');
        const scalars = try parseFrCsv(allocator, fields_split[1]);
        defer allocator.free(scalars);

        const bases = try generateG2Bases(allocator, scalars.len);
        defer allocator.free(bases);

        const actual = zolt.poly.commitment.dory.msmG2Bench(Fr, bases, scalars, null);
        const expected_infinity = try parser.parseDecimal(u8, fields_split[2]);
        try std.testing.expectEqual(expected_infinity == 1, actual.infinity);
        if (!actual.infinity) {
            const expected_x_c0 = try parser.parseHexBytesExact(32, fields_split[3]);
            const expected_x_c1 = try parser.parseHexBytesExact(32, fields_split[4]);
            const expected_y_c0 = try parser.parseHexBytesExact(32, fields_split[5]);
            const expected_y_c1 = try parser.parseHexBytesExact(32, fields_split[6]);
            const actual_x_c0 = fieldToBytesLE(Fp, actual.x.c0);
            const actual_x_c1 = fieldToBytesLE(Fp, actual.x.c1);
            const actual_y_c0 = fieldToBytesLE(Fp, actual.y.c0);
            const actual_y_c1 = fieldToBytesLE(Fp, actual.y.c1);
            try std.testing.expectEqualSlices(u8, &expected_x_c0, &actual_x_c0);
            try std.testing.expectEqualSlices(u8, &expected_x_c1, &actual_x_c1);
            try std.testing.expectEqualSlices(u8, &expected_y_c0, &actual_y_c0);
            try std.testing.expectEqualSlices(u8, &expected_y_c1, &actual_y_c1);
        }
    }
}

// ============================================================================
// Accumulator differential tests
// ============================================================================

test "zolt-arith differential sum_of_products fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "accumulator/sum_of_products.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(6, line, '|');

        const a0 = try parseFrHex(fields_split[1]);
        const b0 = try parseFrHex(fields_split[2]);
        const a1 = try parseFrHex(fields_split[3]);
        const b1 = try parseFrHex(fields_split[4]);
        const expected = try parseFrHex(fields_split[5]);

        const actual = Fr.sumOfProducts(.{ a0, a1 }, .{ b0, b1 });
        try std.testing.expect(actual.eql(expected));
    }
}

test "zolt-arith differential batch_inverse fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "accumulator/batch_inverse.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(4, line, '|');

        const count = try parser.parseDecimal(usize, fields_split[1]);
        const inputs = try parseFrCsv(allocator, fields_split[2]);
        defer allocator.free(inputs);
        const expected_vals = try parseFrCsv(allocator, fields_split[3]);
        defer allocator.free(expected_vals);

        try std.testing.expectEqual(count, inputs.len);
        try std.testing.expectEqual(count, expected_vals.len);

        const results = try allocator.alloc(Fr, count);
        defer allocator.free(results);
        try accumulators.BatchOps.batchInverse(results, inputs, allocator);

        for (0..count) |i| {
            try std.testing.expect(results[i].eql(expected_vals[i]));
        }
    }
}

test "zolt-arith differential mul_u64 fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "accumulator/mul_u64.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(4, line, '|');

        const field_val = try parseFrHex(fields_split[1]);
        const scalar = try parser.parseDecimal(u64, fields_split[2]);
        const expected = try parseFrHex(fields_split[3]);

        const actual = accumulators.mulU64(field_val, scalar);
        try std.testing.expect(actual.eql(expected));
    }
}

test "zolt-arith differential mul_u128 fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "accumulator/mul_u128.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(4, line, '|');

        const field_val = try parseFrHex(fields_split[1]);
        const scalar = try parser.parseDecimal(u128, fields_split[2]);
        const expected = try parseFrHex(fields_split[3]);

        const unreduced = accumulators.mulU128Unreduced(field_val, scalar);
        const actual = accumulators.reduceMulU128(unreduced);
        try std.testing.expect(actual.eql(expected));
    }
}

// ============================================================================
// Transcript differential tests
// ============================================================================

/// Apply a semicolon-separated list of public transcript operations.
/// Format: "append_label:data;append_u64:count:999;append_scalar:val:7"
fn applyTranscriptOps(transcript: *Transcript, ops_desc: []const u8) !void {
    if (std.mem.eql(u8, ops_desc, "-")) return;
    var ops = std.mem.splitScalar(u8, ops_desc, ';');
    while (ops.next()) |op| {
        var parts = std.mem.splitScalar(u8, op, ':');
        const kind = parts.next() orelse continue;
        if (std.mem.eql(u8, kind, "append_label")) {
            transcript.appendLabel(parts.next() orelse return error.MissingArg);
        } else if (std.mem.eql(u8, kind, "append_u64")) {
            const label = parts.next() orelse return error.MissingArg;
            const val_str = parts.next() orelse return error.MissingArg;
            transcript.appendU64(label, try parser.parseDecimal(u64, val_str));
        } else if (std.mem.eql(u8, kind, "append_scalar")) {
            const label = parts.next() orelse return error.MissingArg;
            const val_str = parts.next() orelse return error.MissingArg;
            transcript.appendScalar(label, Fr.fromU64(try parser.parseDecimal(u64, val_str)));
        } else {
            return error.UnknownTranscriptOp;
        }
    }
}

test "zolt-arith differential transcript state fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "transcript/state_vectors.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        // Format: name|init_label|ops_desc|expected_state_hex|expected_rounds
        const fields_split = try parser.splitFieldsExact(5, line, '|');

        const init_label = fields_split[1];
        const ops_desc = fields_split[2];
        const expected_state = try parser.parseHexBytesExact(32, fields_split[3]);
        const expected_rounds = try parser.parseDecimal(u32, fields_split[4]);

        var transcript = Transcript.init(init_label);
        try applyTranscriptOps(&transcript, ops_desc);

        try std.testing.expectEqualSlices(u8, &expected_state, &transcript.debugState());
        try std.testing.expectEqual(expected_rounds, transcript.n_rounds);
    }
}

test "zolt-arith differential transcript challenge fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "transcript/challenge_vectors.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        // Format: name|init_label|ops_desc|expected_u128|expected_limb2_hex|expected_limb3_hex
        const fields_split = try parser.splitFieldsExact(6, line, '|');

        const init_label = fields_split[1];
        const ops_desc = fields_split[2];
        const expected_u128_str = fields_split[3];
        const expected_low_str = fields_split[4];
        const expected_high_str = fields_split[5];

        // Check challenge_u128
        if (!std.mem.eql(u8, expected_u128_str, "-")) {
            var transcript = Transcript.init(init_label);
            try applyTranscriptOps(&transcript, ops_desc);
            const expected_u128 = try parser.parseDecimal(u128, expected_u128_str);
            try std.testing.expectEqual(expected_u128, transcript.challengeU128());
        }

        // Check challenge_scalar_128bits
        if (!std.mem.eql(u8, expected_low_str, "-") and !std.mem.eql(u8, expected_high_str, "-")) {
            var transcript = Transcript.init(init_label);
            try applyTranscriptOps(&transcript, ops_desc);
            const expected_low = try std.fmt.parseInt(u64, expected_low_str, 16);
            const expected_high = try std.fmt.parseInt(u64, expected_high_str, 16);
            const challenge = transcript.challengeScalar128Bits();
            try std.testing.expectEqual(@as(u64, 0), challenge.limbs[0]);
            try std.testing.expectEqual(@as(u64, 0), challenge.limbs[1]);
            try std.testing.expectEqual(expected_low, challenge.limbs[2]);
            try std.testing.expectEqual(expected_high, challenge.limbs[3]);
        }
    }
}

// ============================================================================
// Extension field differential tests
// ============================================================================

fn fpFromBytesLE(bytes: *const [32]u8) Fp {
    var limbs: [4]u64 = undefined;
    inline for (0..4) |i| {
        limbs[i] = std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little);
    }
    const raw = Fp{ .limbs = limbs };
    return raw.toMontgomery();
}

fn parseFp2HexLE(text: []const u8) !Fp2 {
    const bytes = try parser.parseHexBytesExact(64, text);
    return Fp2.init(fpFromBytesLE(bytes[0..32]), fpFromBytesLE(bytes[32..64]));
}

fn parseFp6HexLE(text: []const u8) !Fp6 {
    const bytes = try parser.parseHexBytesExact(192, text);
    return Fp6{
        .c0 = Fp2.init(fpFromBytesLE(bytes[0..32]), fpFromBytesLE(bytes[32..64])),
        .c1 = Fp2.init(fpFromBytesLE(bytes[64..96]), fpFromBytesLE(bytes[96..128])),
        .c2 = Fp2.init(fpFromBytesLE(bytes[128..160]), fpFromBytesLE(bytes[160..192])),
    };
}

fn parseFp12HexLE(text: []const u8) !Fp12 {
    const bytes = try parser.parseHexBytesExact(384, text);
    return Fp12.fromBytes(&bytes);
}

test "zolt-arith differential fp2 fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "extensions/fp2_ops.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(4, line, '|');
        const op = fields_split[0];
        const a = try parseFp2HexLE(fields_split[1]);
        const expected = try parseFp2HexLE(fields_split[3]);

        if (std.mem.eql(u8, op, "add")) {
            const b = try parseFp2HexLE(fields_split[2]);
            try std.testing.expect(a.add(b).eql(expected));
        } else if (std.mem.eql(u8, op, "sub")) {
            const b = try parseFp2HexLE(fields_split[2]);
            try std.testing.expect(a.sub(b).eql(expected));
        } else if (std.mem.eql(u8, op, "mul")) {
            const b = try parseFp2HexLE(fields_split[2]);
            try std.testing.expect(a.mul(b).eql(expected));
        } else if (std.mem.eql(u8, op, "square")) {
            try std.testing.expect(a.square().eql(expected));
        } else if (std.mem.eql(u8, op, "inv")) {
            try std.testing.expect(a.inverse().?.eql(expected));
        } else if (std.mem.eql(u8, op, "conjugate")) {
            try std.testing.expect(a.conjugate().eql(expected));
        } else {
            return error.UnknownExtensionFieldOp;
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 40);
}

test "zolt-arith differential fp6 fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "extensions/fp6_ops.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(4, line, '|');
        const op = fields_split[0];
        const a = try parseFp6HexLE(fields_split[1]);
        const expected = try parseFp6HexLE(fields_split[3]);

        if (std.mem.eql(u8, op, "add")) {
            const b = try parseFp6HexLE(fields_split[2]);
            try std.testing.expect(a.add(b).eql(expected));
        } else if (std.mem.eql(u8, op, "sub")) {
            const b = try parseFp6HexLE(fields_split[2]);
            try std.testing.expect(a.sub(b).eql(expected));
        } else if (std.mem.eql(u8, op, "mul")) {
            const b = try parseFp6HexLE(fields_split[2]);
            try std.testing.expect(a.mul(b).eql(expected));
        } else if (std.mem.eql(u8, op, "square")) {
            try std.testing.expect(a.square().eql(expected));
        } else if (std.mem.eql(u8, op, "inv")) {
            try std.testing.expect(a.inverse().?.eql(expected));
        } else {
            return error.UnknownExtensionFieldOp;
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 30);
}

test "zolt-arith differential fp12 fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "extensions/fp12_ops.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(4, line, '|');
        const op = fields_split[0];
        const a = try parseFp12HexLE(fields_split[1]);
        const expected = try parseFp12HexLE(fields_split[3]);

        if (std.mem.eql(u8, op, "add")) {
            const b = try parseFp12HexLE(fields_split[2]);
            try std.testing.expect(a.add(b).eql(expected));
        } else if (std.mem.eql(u8, op, "sub")) {
            const b = try parseFp12HexLE(fields_split[2]);
            try std.testing.expect(a.sub(b).eql(expected));
        } else if (std.mem.eql(u8, op, "mul")) {
            const b = try parseFp12HexLE(fields_split[2]);
            try std.testing.expect(a.mul(b).eql(expected));
        } else if (std.mem.eql(u8, op, "square")) {
            try std.testing.expect(a.square().eql(expected));
        } else if (std.mem.eql(u8, op, "inv")) {
            try std.testing.expect(a.inverse().?.eql(expected));
        } else if (std.mem.eql(u8, op, "conjugate")) {
            try std.testing.expect(a.conjugate().eql(expected));
        } else if (std.mem.eql(u8, op, "frobenius")) {
            try std.testing.expect(a.frobenius().eql(expected));
        } else if (std.mem.eql(u8, op, "frobenius2")) {
            try std.testing.expect(a.frobenius2().eql(expected));
        } else if (std.mem.eql(u8, op, "frobenius3")) {
            try std.testing.expect(a.frobenius3().eql(expected));
        } else if (std.mem.eql(u8, op, "cyclotomic_square")) {
            try std.testing.expect(a.cyclotomicSquare().eql(expected));
        } else {
            return error.UnknownExtensionFieldOp;
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 60);
}

// ============================================================================
// G2 curve ops differential tests
// ============================================================================

fn compareG2Coords(actual: G2Point, fields: struct { xc0: []const u8, xc1: []const u8, yc0: []const u8, yc1: []const u8 }) !void {
    const expected_x_c0 = try parser.parseHexBytesExact(32, fields.xc0);
    const expected_x_c1 = try parser.parseHexBytesExact(32, fields.xc1);
    const expected_y_c0 = try parser.parseHexBytesExact(32, fields.yc0);
    const expected_y_c1 = try parser.parseHexBytesExact(32, fields.yc1);
    const actual_x_c0 = fieldToBytesLE(Fp, actual.x.c0);
    const actual_x_c1 = fieldToBytesLE(Fp, actual.x.c1);
    const actual_y_c0 = fieldToBytesLE(Fp, actual.y.c0);
    const actual_y_c1 = fieldToBytesLE(Fp, actual.y.c1);
    try std.testing.expectEqualSlices(u8, &expected_x_c0, &actual_x_c0);
    try std.testing.expectEqualSlices(u8, &expected_x_c1, &actual_x_c1);
    try std.testing.expectEqualSlices(u8, &expected_y_c0, &actual_y_c0);
    try std.testing.expectEqualSlices(u8, &expected_y_c1, &actual_y_c1);
}

test "zolt-arith differential g2 ops fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "g2/g2_ops.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    const gen = G2Point.generator();

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(8, line, '|');
        const op = fields_split[0];
        const expected_infinity = try parser.parseDecimal(u8, fields_split[3]);

        if (std.mem.eql(u8, op, "scalar_mul")) {
            const scalar = try parseFrHex(fields_split[1]);
            const actual = gen.scalarMul(scalar);
            try std.testing.expectEqual(expected_infinity == 1, actual.infinity);
            if (!actual.infinity) {
                try compareG2Coords(actual, .{ .xc0 = fields_split[4], .xc1 = fields_split[5], .yc0 = fields_split[6], .yc1 = fields_split[7] });
            }
        } else if (std.mem.eql(u8, op, "add")) {
            const s1 = try parseFrHex(fields_split[1]);
            const s2 = try parseFrHex(fields_split[2]);
            const actual = gen.scalarMul(s1).add(gen.scalarMul(s2));
            try std.testing.expectEqual(expected_infinity == 1, actual.infinity);
            if (!actual.infinity) {
                try compareG2Coords(actual, .{ .xc0 = fields_split[4], .xc1 = fields_split[5], .yc0 = fields_split[6], .yc1 = fields_split[7] });
            }
        } else if (std.mem.eql(u8, op, "double")) {
            const s = try parseFrHex(fields_split[1]);
            const p = gen.scalarMul(s);
            const actual = p.add(p);
            try std.testing.expectEqual(expected_infinity == 1, actual.infinity);
            if (!actual.infinity) {
                try compareG2Coords(actual, .{ .xc0 = fields_split[4], .xc1 = fields_split[5], .yc0 = fields_split[6], .yc1 = fields_split[7] });
            }
        } else if (std.mem.eql(u8, op, "neg")) {
            const s = try parseFrHex(fields_split[1]);
            const actual = gen.scalarMul(s).neg();
            try std.testing.expectEqual(expected_infinity == 1, actual.infinity);
            if (!actual.infinity) {
                try compareG2Coords(actual, .{ .xc0 = fields_split[4], .xc1 = fields_split[5], .yc0 = fields_split[6], .yc1 = fields_split[7] });
            }
        } else {
            return error.UnknownG2Op;
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 10);
}

// ============================================================================
// Point compression differential tests
// ============================================================================

test "zolt-arith differential g1 point compression fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "point_compression/g1_compress.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(4, line, '|');

        const expected_compressed = try parser.parseHexBytesExact(32, fields_split[3]);

        // Reconstruct point from uncompressed coords (LE hex)
        // Identity case: x and y are empty strings
        if (fields_split[1].len == 0) {
            // Identity point
            const identity = G1Affine.identity();
            const actual_compressed = dory.compressG1(identity);
            try std.testing.expectEqualSlices(u8, &expected_compressed, &actual_compressed);
            // Decompress and verify
            const decompressed = dory.decompressG1(&actual_compressed);
            try std.testing.expect(decompressed != null);
            try std.testing.expect(decompressed.?.infinity);
        } else {
            const x_bytes = try parser.parseHexBytesExact(32, fields_split[1]);
            const y_bytes = try parser.parseHexBytesExact(32, fields_split[2]);
            const x = fpFromBytesLE(&x_bytes);
            const y = fpFromBytesLE(&y_bytes);
            const point = G1Affine{ .x = x, .y = y, .infinity = false };

            // Test compression
            const actual_compressed = dory.compressG1(point);
            try std.testing.expectEqualSlices(u8, &expected_compressed, &actual_compressed);

            // Test decompression roundtrip
            const decompressed = dory.decompressG1(&actual_compressed);
            try std.testing.expect(decompressed != null);
            try std.testing.expect(!decompressed.?.infinity);
            try std.testing.expectEqualSlices(u8, &x_bytes, &fieldToBytesLE(Fp, decompressed.?.x));
            try std.testing.expectEqualSlices(u8, &y_bytes, &fieldToBytesLE(Fp, decompressed.?.y));
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 6);
}

test "zolt-arith differential g2 point compression fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "point_compression/g2_compress.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(6, line, '|');

        if (fields_split[1].len == 0) {
            // Identity: verify compress roundtrip
            const identity = G2Point.identity();
            const compressed = dory.compressG2(identity);
            const decompressed = dory.decompressG2(&compressed);
            try std.testing.expect(decompressed != null);
            try std.testing.expect(decompressed.?.infinity);
        } else {
            const x_c0_bytes = try parser.parseHexBytesExact(32, fields_split[1]);
            const x_c1_bytes = try parser.parseHexBytesExact(32, fields_split[2]);
            const y_c0_bytes = try parser.parseHexBytesExact(32, fields_split[3]);
            const y_c1_bytes = try parser.parseHexBytesExact(32, fields_split[4]);
            const point = G2Point{
                .x = Fp2.init(fpFromBytesLE(&x_c0_bytes), fpFromBytesLE(&x_c1_bytes)),
                .y = Fp2.init(fpFromBytesLE(&y_c0_bytes), fpFromBytesLE(&y_c1_bytes)),
                .infinity = false,
            };

            // Verify full compress → decompress roundtrip
            const compressed = dory.compressG2(point);
            const decompressed = dory.decompressG2(&compressed);
            try std.testing.expect(decompressed != null);
            try std.testing.expect(!decompressed.?.infinity);
            try std.testing.expectEqualSlices(u8, &x_c0_bytes, &fieldToBytesLE(Fp, decompressed.?.x.c0));
            try std.testing.expectEqualSlices(u8, &x_c1_bytes, &fieldToBytesLE(Fp, decompressed.?.x.c1));
            try std.testing.expectEqualSlices(u8, &y_c0_bytes, &fieldToBytesLE(Fp, decompressed.?.y.c0));
            try std.testing.expectEqualSlices(u8, &y_c1_bytes, &fieldToBytesLE(Fp, decompressed.?.y.c1));
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 6);
}

// ============================================================================
// GLV scalar multiplication differential tests
// ============================================================================

test "zolt-arith differential glv g1 scalar mul fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "glv/glv_g1_scalar_mul.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(5, line, '|');

        const scalar = try parseFrHex(fields_split[1]);
        const expected_infinity = try parser.parseDecimal(u8, fields_split[2]);

        const actual_proj = glv.glvScalarMulG1(G1Affine.generator(), scalar);
        const actual = actual_proj.toAffine();

        try std.testing.expectEqual(expected_infinity == 1, actual.infinity);
        if (!actual.infinity) {
            const expected_x = try parser.parseHexBytesExact(32, fields_split[3]);
            const expected_y = try parser.parseHexBytesExact(32, fields_split[4]);
            try std.testing.expectEqualSlices(u8, &expected_x, &fieldToBytesLE(Fp, actual.x));
            try std.testing.expectEqualSlices(u8, &expected_y, &fieldToBytesLE(Fp, actual.y));
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 8);
}

test "zolt-arith differential glv g2 scalar mul fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "glv/glv_g2_scalar_mul.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(7, line, '|');

        const scalar = try parseFrHex(fields_split[1]);
        const expected_infinity = try parser.parseDecimal(u8, fields_split[2]);

        const actual_proj = glv.glvScalarMulG2(G2Point.generator(), scalar);
        const actual = actual_proj.toAffine();

        try std.testing.expectEqual(expected_infinity == 1, actual.infinity);
        if (!actual.infinity) {
            try compareG2Coords(actual, .{ .xc0 = fields_split[3], .xc1 = fields_split[4], .yc0 = fields_split[5], .yc1 = fields_split[6] });
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 8);
}

// ============================================================================
// Dory commitment differential tests
// ============================================================================

test "zolt-arith differential dory commit fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "dory/commit_cases.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    const DoryScheme = dory.DoryCommitmentScheme(Fr);

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(4, line, '|');

        const max_num_vars = try parser.parseDecimal(usize, fields_split[1]);
        const expected_commitment = try parseFp12HexLE(fields_split[3]);

        // Parse evals
        const evals = try parseFrCsv(allocator, fields_split[2]);
        defer allocator.free(evals);

        // Generate SRS matching the Rust generator
        var srs = try DoryScheme.setup(allocator, max_num_vars);
        defer srs.deinit();

        // Compute commitment
        const actual = DoryScheme.commitWithPool(&srs, evals, null);

        // Compare Fp12 bytes
        const expected_bytes = expected_commitment.toBytes();
        const actual_bytes = actual.toBytes();
        try std.testing.expectEqualSlices(u8, &expected_bytes, &actual_bytes);
        case_count += 1;
    }
    try std.testing.expect(case_count >= 2);
}

// ============================================================================
// GPU field crossover differential tests
// ============================================================================

test "zolt-arith differential gpu field crossover fixtures" {
    const allocator = std.testing.allocator;
    const fixture_text = try readFixtureAlloc(allocator, "gpu/field_crossover.txt");
    defer allocator.free(fixture_text);
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = parser.cleanLine(raw_line) orelse continue;
        const fields_split = try parser.splitFieldsExact(5, line, '|');
        const op = fields_split[0];

        // Parse inputs
        const a_vals = try parseFrCsv(allocator, fields_split[2]);
        defer allocator.free(a_vals);
        const expected_vals = try parseFrCsv(allocator, fields_split[4]);
        defer allocator.free(expected_vals);

        // Verify CPU path against fixture
        if (std.mem.eql(u8, op, "mul")) {
            const b_vals = try parseFrCsv(allocator, fields_split[3]);
            defer allocator.free(b_vals);
            for (a_vals, 0..) |a, i| {
                try std.testing.expect(a.mul(b_vals[i]).eql(expected_vals[i]));
            }
        } else if (std.mem.eql(u8, op, "add")) {
            const b_vals = try parseFrCsv(allocator, fields_split[3]);
            defer allocator.free(b_vals);
            for (a_vals, 0..) |a, i| {
                try std.testing.expect(a.add(b_vals[i]).eql(expected_vals[i]));
            }
        } else if (std.mem.eql(u8, op, "sub")) {
            const b_vals = try parseFrCsv(allocator, fields_split[3]);
            defer allocator.free(b_vals);
            for (a_vals, 0..) |a, i| {
                try std.testing.expect(a.sub(b_vals[i]).eql(expected_vals[i]));
            }
        } else if (std.mem.eql(u8, op, "neg")) {
            for (a_vals, 0..) |a, i| {
                try std.testing.expect(a.neg().eql(expected_vals[i]));
            }
        } else {
            return error.UnknownGpuOp;
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 10);
}
