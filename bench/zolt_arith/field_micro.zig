const std = @import("std");
const zolt = @import("zolt");
const harness = @import("bench_harness.zig");

const Fr = zolt.field.BN254Scalar;
const Fp = zolt.field.BN254BaseField;

const ELEMENTS: usize = 128;

fn fillFieldInputs(comptime F: type, a: *[ELEMENTS]F, b: *[ELEMENTS]F) void {
    for (0..ELEMENTS) |i| {
        a[i] = F.fromU64(@as(u64, @intCast(i + 1)) *% 0x9E3779B185EBCA87 +% 17);
        b[i] = F.fromU64(@as(u64, @intCast(i + 1)) *% 0xD6E8FEB86659FD93 +% 29);
    }
}

fn benchField(comptime F: type, comptime field_name: []const u8) void {
    const Ctx = struct {
        var a: [ELEMENTS]F = undefined;
        var b: [ELEMENTS]F = undefined;

        fn add(i: usize) F { return a[i % ELEMENTS].add(b[i % ELEMENTS]); }
        fn sub(i: usize) F { return a[i % ELEMENTS].sub(b[i % ELEMENTS]); }
        fn mul(i: usize) F { return a[i % ELEMENTS].mul(b[i % ELEMENTS]); }
        fn square(i: usize) F { return a[i % ELEMENTS].square(); }
        fn inverse(i: usize) F { return a[i % ELEMENTS].inverse() orelse F.zero(); }
        fn toMontgomery(i: usize) F { return a[i % ELEMENTS].fromMontgomery().toMontgomery(); }
        fn fromMontgomery(i: usize) F { return a[i % ELEMENTS].fromMontgomery(); }
        fn sumOfProducts(i: usize) F {
            const idx = i % ELEMENTS;
            return F.sumOfProducts(.{ a[idx], b[idx] }, .{ b[idx], a[idx] });
        }
    };

    fillFieldInputs(F, &Ctx.a, &Ctx.b);

    const cfg = harness.Config.field;
    _ = harness.run(field_name, "add", cfg, F, Ctx.add);
    _ = harness.run(field_name, "sub", cfg, F, Ctx.sub);
    _ = harness.run(field_name, "mul", cfg, F, Ctx.mul);
    _ = harness.run(field_name, "square", cfg, F, Ctx.square);
    _ = harness.run(field_name, "inverse", cfg, F, Ctx.inverse);
    _ = harness.run(field_name, "toMontgomery", cfg, F, Ctx.toMontgomery);
    _ = harness.run(field_name, "fromMontgomery", cfg, F, Ctx.fromMontgomery);
    _ = harness.run(field_name, "sumOfProducts", cfg, F, Ctx.sumOfProducts);
}

pub fn main() !void {
    std.debug.print("=== Zolt-Arith Field Microbench ===\n", .{});
    benchField(Fp, "field_Fp");
    benchField(Fr, "field_Fr");
}
