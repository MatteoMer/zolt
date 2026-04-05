/// ARM64 field arithmetic micro-benchmarks.
/// Verifies claims about carry-chain pipelining, branchless vs branched reduction,
/// and the impact of cinc codegen on the Zig u128 fallback.
const std = @import("std");
const builtin = @import("builtin");
const zolt = @import("zolt");
const Fr = zolt.field.BN254Scalar;

const ITERS: u64 = 200_000_000;
const MOD = [4]u64{ 0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029 };

// ── Claim A: bare 4-limb add pipelines at ~1 cycle/iter via OoO ────────────
// Expected: ~0.3 ns   (NOT the 4-cycle chain latency)
noinline fn bareAddLoop(acc: *[4]u64, b: *const [4]u64, n: u64) void {
    comptime if (builtin.cpu.arch != .aarch64) @compileError("aarch64 only");
    var counter: u64 = undefined;
    asm volatile (
    // load b (loop-invariant)
        \\ ldp x4, x5, [x10]
        \\ ldp x6, x7, [x10, #16]
        // load acc
        \\ ldp x0, x1, [x9]
        \\ ldp x2, x3, [x9, #16]
        \\ 1:
        \\ adds x0, x0, x4
        \\ adcs x1, x1, x5
        \\ adcs x2, x2, x6
        \\ adc  x3, x3, x7
        \\ subs x11, x11, #1
        \\ b.ne 1b
        // store result
        \\ stp x0, x1, [x9]
        \\ stp x2, x3, [x9, #16]
        : [_cnt] "={x11}" (counter),
        : [_acc] "{x9}" (acc),
          [_b] "{x10}" (b),
          [_n] "{x11}" (n),
        : .{ .x0 = true, .x1 = true, .x2 = true, .x3 = true, .x4 = true, .x5 = true, .x6 = true, .x7 = true, .nzcv = true, .memory = true }); // note: Zig requires trailing comma after last clobber field
}

// ── Claim B: branchless add (adds/adcs + subs/sbcs + csel) ≈ 7 cycles ─────
// Expected: ~2.0 ns
noinline fn branchlessAddLoop(acc: *[4]u64, b: *const [4]u64, m: *const [4]u64, n: u64) void {
    comptime if (builtin.cpu.arch != .aarch64) @compileError("aarch64 only");
    var counter: u64 = undefined;
    asm volatile (
    // load b (x4-x7), mod (x12,x13,x14,x15), acc (x0-x3)
        \\ ldp x4, x5, [x10]
        \\ ldp x6, x7, [x10, #16]
        \\ ldp x12, x13, [x20]
        \\ ldp x14, x15, [x20, #16]
        \\ ldp x0, x1, [x9]
        \\ ldp x2, x3, [x9, #16]
        \\ 1:
        \\ adds x0, x0, x4
        \\ adcs x1, x1, x5
        \\ adcs x2, x2, x6
        \\ adcs x3, x3, x7
        \\ adc  x17, xzr, xzr
        // speculative subtract
        \\ subs x21, x0, x12
        \\ sbcs x22, x1, x13
        \\ sbcs x23, x2, x14
        \\ sbcs x24, x3, x15
        \\ sbcs x17, x17, xzr
        // select
        \\ csel x0, x21, x0, hs
        \\ csel x1, x22, x1, hs
        \\ csel x2, x23, x2, hs
        \\ csel x3, x24, x3, hs
        \\ subs x11, x11, #1
        \\ b.ne 1b
        \\ stp x0, x1, [x9]
        \\ stp x2, x3, [x9, #16]
        : [_cnt] "={x11}" (counter),
        : [_acc] "{x9}" (acc),
          [_b] "{x10}" (b),
          [_m] "{x20}" (m),
          [_n] "{x11}" (n),
        : .{ .x0 = true, .x1 = true, .x2 = true, .x3 = true, .x4 = true, .x5 = true, .x6 = true, .x7 = true, .x12 = true, .x13 = true, .x14 = true, .x15 = true, .x17 = true, .x20 = true, .x21 = true, .x22 = true, .x23 = true, .x24 = true, .nzcv = true, .memory = true });
}

// ── Claim C: b−p trick with branch ≈ 3 cycles average (0.8 ns) ────────────
// add (b−p) always; if carry clear, add p back.
// Expected: ~0.8 ns  (OoO pipelines the fast path at ~1 cyc/iter)
noinline fn bpBranchAddLoop(acc: *[4]u64, bmp: *const [4]u64, m: *const [4]u64, n: u64) void {
    comptime if (builtin.cpu.arch != .aarch64) @compileError("aarch64 only");
    var counter: u64 = undefined;
    asm volatile (
    // bmp = b - p  (x4-x7),  mod (x12-x15),  acc (x0-x3)
        \\ ldp x4, x5, [x10]
        \\ ldp x6, x7, [x10, #16]
        \\ ldp x12, x13, [x20]
        \\ ldp x14, x15, [x20, #16]
        \\ ldp x0, x1, [x9]
        \\ ldp x2, x3, [x9, #16]
        \\ 1:
        \\ adds x0, x0, x4
        \\ adcs x1, x1, x5
        \\ adcs x2, x2, x6
        \\ adcs x3, x3, x7
        \\ b.cs 2f
        // no carry → over-subtracted, add p back
        \\ adds x0, x0, x12
        \\ adcs x1, x1, x13
        \\ adcs x2, x2, x14
        \\ adc  x3, x3, x15
        \\ 2:
        \\ subs x11, x11, #1
        \\ b.ne 1b
        \\ stp x0, x1, [x9]
        \\ stp x2, x3, [x9, #16]
        : [_cnt] "={x11}" (counter),
        : [_acc] "{x9}" (acc),
          [_bmp] "{x10}" (bmp),
          [_m] "{x20}" (m),
          [_n] "{x11}" (n),
        : .{ .x0 = true, .x1 = true, .x2 = true, .x3 = true, .x4 = true, .x5 = true, .x6 = true, .x7 = true, .x12 = true, .x13 = true, .x14 = true, .x15 = true, .x20 = true, .nzcv = true, .memory = true });
}

// ── Claim D: sub with proper sbcs + mask correction ≈ 2 ns ─────────────────
noinline fn maskSubLoop(acc: *[4]u64, b: *const [4]u64, m: *const [4]u64, n: u64) void {
    comptime if (builtin.cpu.arch != .aarch64) @compileError("aarch64 only");
    var counter: u64 = undefined;
    asm volatile (
        \\ ldp x4, x5, [x10]
        \\ ldp x6, x7, [x10, #16]
        \\ ldp x12, x13, [x20]
        \\ ldp x14, x15, [x20, #16]
        \\ ldp x0, x1, [x9]
        \\ ldp x2, x3, [x9, #16]
        \\ 1:
        \\ subs x0, x0, x4
        \\ sbcs x1, x1, x5
        \\ sbcs x2, x2, x6
        \\ sbcs x3, x3, x7
        // mask = borrow ? -1 : 0
        \\ sbc  x17, xzr, xzr
        \\ and  x21, x17, x12
        \\ and  x22, x17, x13
        \\ and  x23, x17, x14
        \\ and  x24, x17, x15
        \\ adds x0, x0, x21
        \\ adcs x1, x1, x22
        \\ adcs x2, x2, x23
        \\ adc  x3, x3, x24
        \\ subs x11, x11, #1
        \\ b.ne 1b
        \\ stp x0, x1, [x9]
        \\ stp x2, x3, [x9, #16]
        : [_cnt] "={x11}" (counter),
        : [_acc] "{x9}" (acc),
          [_b] "{x10}" (b),
          [_m] "{x20}" (m),
          [_n] "{x11}" (n),
        : .{ .x0 = true, .x1 = true, .x2 = true, .x3 = true, .x4 = true, .x5 = true, .x6 = true, .x7 = true, .x12 = true, .x13 = true, .x14 = true, .x15 = true, .x17 = true, .x20 = true, .x21 = true, .x22 = true, .x23 = true, .x24 = true, .nzcv = true, .memory = true });
}

// ── Claim E: sub with branch (fast path: no borrow) ────────────────────────
noinline fn branchSubLoop(acc: *[4]u64, b: *const [4]u64, m: *const [4]u64, n: u64) void {
    comptime if (builtin.cpu.arch != .aarch64) @compileError("aarch64 only");
    var counter: u64 = undefined;
    asm volatile (
        \\ ldp x4, x5, [x10]
        \\ ldp x6, x7, [x10, #16]
        \\ ldp x12, x13, [x20]
        \\ ldp x14, x15, [x20, #16]
        \\ ldp x0, x1, [x9]
        \\ ldp x2, x3, [x9, #16]
        \\ 1:
        \\ subs x0, x0, x4
        \\ sbcs x1, x1, x5
        \\ sbcs x2, x2, x6
        \\ sbcs x3, x3, x7
        // C=1 → no borrow, result OK; C=0 → borrow, add mod
        \\ b.cs 2f
        \\ adds x0, x0, x12
        \\ adcs x1, x1, x13
        \\ adcs x2, x2, x14
        \\ adc  x3, x3, x15
        \\ 2:
        \\ subs x11, x11, #1
        \\ b.ne 1b
        \\ stp x0, x1, [x9]
        \\ stp x2, x3, [x9, #16]
        : [_cnt] "={x11}" (counter),
        : [_acc] "{x9}" (acc),
          [_b] "{x10}" (b),
          [_m] "{x20}" (m),
          [_n] "{x11}" (n),
        : .{ .x0 = true, .x1 = true, .x2 = true, .x3 = true, .x4 = true, .x5 = true, .x6 = true, .x7 = true, .x12 = true, .x13 = true, .x14 = true, .x15 = true, .x20 = true, .nzcv = true, .memory = true });
}

// ─────────────────────────────────────────────────────────────────────────────
fn timer_ns(t: *std.time.Timer) f64 {
    return @as(f64, @floatFromInt(t.read())) / @as(f64, @floatFromInt(ITERS));
}

pub fn main() !void {
    // Proper Montgomery-form values via the field library
    const fr_a = Fr.fromU64(1);
    const fr_b = Fr.fromU64(1000);
    const acc: [4]u64 = fr_a.limbs;
    const b_val: [4]u64 = fr_b.limbs;

    // b − p  (unsigned wraparound)
    var b_minus_p: [4]u64 = undefined;
    {
        var borrow: u1 = 0;
        inline for (0..4) |i| {
            const r1 = @subWithOverflow(b_val[i], MOD[i]);
            const r2 = @subWithOverflow(r1[0], @as(u64, borrow));
            b_minus_p[i] = r2[0];
            borrow = r1[1] | r2[1];
        }
    }

    std.debug.print("=== ARM64 field-op claim verification ===\n", .{});
    std.debug.print("Iterations: {}  (200 M)\n\n", .{ITERS});

    // ── reference: current Zig u128 add ──
    {
        var a = Fr{ .limbs = acc };
        const b_fr = Fr{ .limbs = b_val };
        var t = std.time.Timer.start() catch unreachable;
        for (0..ITERS) |_| a = a.add(b_fr);
        const ns = timer_ns(&t);
        std.mem.doNotOptimizeAway(&a);
        std.debug.print("Ref  Zig u128 add:          {d:.2} ns\n", .{ns});
    }

    // ── reference: current Zig u128 sub ──
    {
        var a = Fr{ .limbs = acc };
        const b_fr = Fr{ .limbs = b_val };
        var t = std.time.Timer.start() catch unreachable;
        for (0..ITERS) |_| a = a.sub(b_fr);
        const ns = timer_ns(&t);
        std.mem.doNotOptimizeAway(&a);
        std.debug.print("Ref  Zig u128 sub:          {d:.2} ns\n\n", .{ns});
    }

    // A – bare 4-limb add (no reduction)
    {
        var a = acc;
        var t = std.time.Timer.start() catch unreachable;
        bareAddLoop(&a, &b_val, ITERS);
        const ns = timer_ns(&t);
        std.mem.doNotOptimizeAway(&a);
        std.debug.print("A.   Bare add  (no mod):    {d:.2} ns   (expect ~0.3)\n", .{ns});
    }

    // B – branchless add (subs/sbcs/csel)
    {
        var a = acc;
        var m = MOD;
        var t = std.time.Timer.start() catch unreachable;
        branchlessAddLoop(&a, &b_val, &m, ITERS);
        const ns = timer_ns(&t);
        std.mem.doNotOptimizeAway(&a);
        std.debug.print("B.   Branchless add:        {d:.2} ns   (expect ~2.0)\n", .{ns});
    }

    // C – b−p trick + branch
    {
        var a = acc;
        var m = MOD;
        var t = std.time.Timer.start() catch unreachable;
        bpBranchAddLoop(&a, &b_minus_p, &m, ITERS);
        const ns = timer_ns(&t);
        std.mem.doNotOptimizeAway(&a);
        std.debug.print("C.   b-p trick + branch:    {d:.2} ns   (expect ~0.8)\n\n", .{ns});
    }

    // D – sub with mask correction
    {
        var a = acc;
        var m = MOD;
        var t = std.time.Timer.start() catch unreachable;
        maskSubLoop(&a, &b_val, &m, ITERS);
        const ns = timer_ns(&t);
        std.mem.doNotOptimizeAway(&a);
        std.debug.print("D.   Mask sub:              {d:.2} ns   (expect ~2.0)\n", .{ns});
    }

    // E – sub with branch
    {
        var a = acc;
        var m = MOD;
        var t = std.time.Timer.start() catch unreachable;
        branchSubLoop(&a, &b_val, &m, ITERS);
        const ns = timer_ns(&t);
        std.mem.doNotOptimizeAway(&a);
        std.debug.print("E.   Branch sub:            {d:.2} ns   (expect ~0.8)\n", .{ns});
    }
}
