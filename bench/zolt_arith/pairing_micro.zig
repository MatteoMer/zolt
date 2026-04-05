const std = @import("std");
const zolt = @import("zolt");
const harness = @import("bench_harness.zig");

const field = zolt.field;
const p = field.pairing;

const Fp = field.BN254BaseField;
const Fp12 = p.Fp12;
const G1PointFp = p.G1PointFp;
const G2Point = p.G2Point;
const G2Projective = p.G2Projective;
const G2Prepared = p.G2Prepared;

const NUM_PAIRS: usize = 4;

// File-scope inputs populated by setupInputs().
var g1_points: [NUM_PAIRS]G1PointFp = undefined;
var g2_points: [NUM_PAIRS]G2Point = undefined;
var g2_prepared: [NUM_PAIRS]G2Prepared = undefined;
var fp12_inputs: [NUM_PAIRS]Fp12 = undefined;

fn setupInputs() void {
    // G1: use generator (1, 2) for all pairs — pairing cost is
    // dominated by the fixed Miller loop, not the specific point.
    for (0..NUM_PAIRS) |i| {
        g1_points[i] = .{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    }

    // G2: varied points via repeated doubling of generator.
    var g2_proj = G2Projective.fromAffine(G2Point.generator());
    for (0..NUM_PAIRS) |i| {
        g2_points[i] = g2_proj.toAffine();
        g2_prepared[i] = G2Prepared.fromG2Point(g2_points[i]);
        g2_proj = g2_proj.double();
    }

    // Fp12: Miller loop outputs for finalExponentiation benchmark.
    for (0..NUM_PAIRS) |i| {
        fp12_inputs[i] = p.millerLoopArkworks(g1_points[i], g2_points[i]);
    }
}

// --- Benchmark bodies ---

fn benchPairingFp(i: usize) Fp12 {
    const idx = i % NUM_PAIRS;
    return p.pairingFp(g1_points[idx], g2_points[idx]);
}

fn benchMillerLoop(i: usize) Fp12 {
    const idx = i % NUM_PAIRS;
    return p.millerLoopArkworks(g1_points[idx], g2_points[idx]);
}

fn benchMillerLoopPrepared(i: usize) Fp12 {
    const idx = i % NUM_PAIRS;
    return p.millerLoopPrepared(g1_points[idx], &g2_prepared[idx]);
}

fn benchFinalExp(i: usize) Fp12 {
    const idx = i % NUM_PAIRS;
    return p.finalExponentiation(fp12_inputs[idx]);
}

pub fn main() !void {
    std.debug.print("=== Zolt-Arith Pairing Microbench ===\n", .{});
    setupInputs();

    const cfg = harness.Config.pairing;
    _ = harness.run("pairing", "pairingFp", cfg, Fp12, benchPairingFp);
    _ = harness.run("pairing", "millerLoop", cfg, Fp12, benchMillerLoop);
    _ = harness.run("pairing", "millerLoopPrepared", cfg, Fp12, benchMillerLoopPrepared);
    _ = harness.run("pairing", "finalExp", cfg, Fp12, benchFinalExp);
}
