//! SIMD-accelerated field operations.
//!
//! Uses Zig's built-in SIMD vectors for parallel limb operations where beneficial.

const std = @import("std");
const mod = @import("mod.zig");

const BN254Scalar = mod.BN254Scalar;
const BN254_MODULUS = mod.BN254_MODULUS;

/// SIMD-accelerated field operations
/// Uses Zig's built-in SIMD vectors for parallel limb operations where beneficial
pub const SimdOps = struct {
    /// Vector type for 4 limbs (one field element)
    pub const Limb4 = @Vector(4, u64);

    /// Convert scalar to SIMD vector
    pub inline fn toVec(scalar: BN254Scalar) Limb4 {
        return Limb4{ scalar.limbs[0], scalar.limbs[1], scalar.limbs[2], scalar.limbs[3] };
    }

    /// Convert SIMD vector to scalar
    pub inline fn fromVec(vec: Limb4) BN254Scalar {
        return BN254Scalar{ .limbs = .{ vec[0], vec[1], vec[2], vec[3] } };
    }

    /// SIMD modulus vector
    pub const modulus_vec: Limb4 = Limb4{
        BN254_MODULUS[0],
        BN254_MODULUS[1],
        BN254_MODULUS[2],
        BN254_MODULUS[3],
    };

    /// Parallel comparison: returns true if all limbs of a < b (lexicographically)
    pub inline fn lessThan(a: Limb4, b: Limb4) bool {
        // Compare from most significant limb
        if (a[3] != b[3]) return a[3] < b[3];
        if (a[2] != b[2]) return a[2] < b[2];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[0] < b[0];
    }

    /// SIMD-parallel addition with reduction (vectorized limb operations)
    /// For cases where we add many field elements, this allows better instruction pipelining
    pub fn simdAdd4(a: [4]BN254Scalar, b: [4]BN254Scalar) [4]BN254Scalar {
        var results: [4]BN254Scalar = undefined;
        // Process all 4 additions - compiler can vectorize limb operations
        comptime var i = 0;
        inline while (i < 4) : (i += 1) {
            results[i] = a[i].add(b[i]);
        }
        return results;
    }

    /// SIMD-parallel multiplication (for pipelining 4 muls together)
    pub fn simdMul4(a: [4]BN254Scalar, b: [4]BN254Scalar) [4]BN254Scalar {
        var results: [4]BN254Scalar = undefined;
        comptime var i = 0;
        inline while (i < 4) : (i += 1) {
            results[i] = a[i].mul(b[i]);
        }
        return results;
    }

    /// Process slices in chunks of 4 for better vectorization
    pub fn batchAddSimd(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);

        const len = results.len;
        const chunks = len / 4;
        const remainder = len % 4;

        // Process in SIMD-friendly chunks of 4
        var i: usize = 0;
        while (i < chunks * 4) : (i += 4) {
            const a4 = [4]BN254Scalar{ a[i], a[i + 1], a[i + 2], a[i + 3] };
            const b4 = [4]BN254Scalar{ b[i], b[i + 1], b[i + 2], b[i + 3] };
            const r4 = simdAdd4(a4, b4);
            results[i] = r4[0];
            results[i + 1] = r4[1];
            results[i + 2] = r4[2];
            results[i + 3] = r4[3];
        }

        // Handle remainder
        for (i..i + remainder) |j| {
            results[j] = a[j].add(b[j]);
        }
    }

    /// Process multiplications in chunks of 4 for better vectorization
    pub fn batchMulSimd(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);

        const len = results.len;
        const chunks = len / 4;
        const remainder = len % 4;

        // Process in SIMD-friendly chunks of 4
        var i: usize = 0;
        while (i < chunks * 4) : (i += 4) {
            const a4 = [4]BN254Scalar{ a[i], a[i + 1], a[i + 2], a[i + 3] };
            const b4 = [4]BN254Scalar{ b[i], b[i + 1], b[i + 2], b[i + 3] };
            const r4 = simdMul4(a4, b4);
            results[i] = r4[0];
            results[i + 1] = r4[1];
            results[i + 2] = r4[2];
            results[i + 3] = r4[3];
        }

        // Handle remainder
        for (i..i + remainder) |j| {
            results[j] = a[j].mul(b[j]);
        }
    }

    /// Inner product with unrolled accumulation for better pipelining
    pub fn innerProductSimd(a: []const BN254Scalar, b: []const BN254Scalar) BN254Scalar {
        std.debug.assert(a.len == b.len);

        const len = a.len;
        const chunks = len / 4;
        const remainder = len % 4;

        // Use 4 accumulators for better instruction-level parallelism
        var acc0 = BN254Scalar.zero();
        var acc1 = BN254Scalar.zero();
        var acc2 = BN254Scalar.zero();
        var acc3 = BN254Scalar.zero();

        var i: usize = 0;
        while (i < chunks * 4) : (i += 4) {
            acc0 = acc0.add(a[i].mul(b[i]));
            acc1 = acc1.add(a[i + 1].mul(b[i + 1]));
            acc2 = acc2.add(a[i + 2].mul(b[i + 2]));
            acc3 = acc3.add(a[i + 3].mul(b[i + 3]));
        }

        // Handle remainder
        for (i..i + remainder) |j| {
            acc0 = acc0.add(a[j].mul(b[j]));
        }

        // Combine accumulators
        return acc0.add(acc1).add(acc2.add(acc3));
    }
};

test "simd operations" {
    var a: [8]BN254Scalar = undefined;
    var b: [8]BN254Scalar = undefined;
    var results: [8]BN254Scalar = undefined;

    for (0..8) |i| {
        a[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 1)));
        b[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 10)));
    }

    // Test SIMD batch add
    SimdOps.batchAddSimd(&results, &a, &b);
    for (0..8) |i| {
        const expected = BN254Scalar.fromU64(@as(u64, @intCast(2 * i + 11)));
        try std.testing.expect(results[i].eql(expected));
    }

    // Test SIMD batch mul
    SimdOps.batchMulSimd(&results, &a, &b);
    for (0..8) |i| {
        const expected = BN254Scalar.fromU64(@as(u64, @intCast((i + 1) * (i + 10))));
        try std.testing.expect(results[i].eql(expected));
    }

    // Test SIMD inner product: sum((i+1) * (i+10)) for i=0..7
    // = 1*10 + 2*11 + 3*12 + 4*13 + 5*14 + 6*15 + 7*16 + 8*17
    // = 10 + 22 + 36 + 52 + 70 + 90 + 112 + 136 = 528
    const ip = SimdOps.innerProductSimd(&a, &b);
    try std.testing.expect(ip.eql(BN254Scalar.fromU64(528)));
}
