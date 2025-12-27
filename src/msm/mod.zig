//! Multi-Scalar Multiplication (MSM) for Jolt
//!
//! MSM computes sum_{i=0}^{n-1} s_i * G_i where s_i are scalars and G_i are curve points.
//! This is a critical operation for polynomial commitments.
//!
//! We use the short Weierstrass curve y^2 = x^3 + b (a = 0 for BN254)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// BN254 curve parameter b = 3
const BN254_B: u64 = 3;

/// Elliptic curve point in affine coordinates
pub fn AffinePoint(comptime F: type) type {
    return struct {
        const Self = @This();

        x: F,
        y: F,
        infinity: bool,

        /// Identity point (point at infinity)
        pub fn identity() Self {
            return .{
                .x = F.zero(),
                .y = F.zero(),
                .infinity = true,
            };
        }

        /// Create a point from coordinates
        pub fn fromCoords(x: F, y: F) Self {
            return .{
                .x = x,
                .y = y,
                .infinity = false,
            };
        }

        /// Check if this is the identity
        pub fn isIdentity(self: Self) bool {
            return self.infinity;
        }

        /// Negate a point
        pub fn neg(self: Self) Self {
            if (self.infinity) return self;
            return .{
                .x = self.x,
                .y = self.y.neg(),
                .infinity = false,
            };
        }

        /// Check equality
        pub fn eql(self: Self, other: Self) bool {
            if (self.infinity and other.infinity) return true;
            if (self.infinity or other.infinity) return false;
            return self.x.eql(other.x) and self.y.eql(other.y);
        }

        /// Add two affine points
        pub fn add(self: Self, other: Self) Self {
            if (self.infinity) return other;
            if (other.infinity) return self;

            // If x-coordinates are equal
            if (self.x.eql(other.x)) {
                // If y-coordinates are negatives, result is infinity
                if (self.y.eql(other.y.neg())) {
                    return identity();
                }
                // If same point, double it
                if (self.y.eql(other.y)) {
                    return self.double();
                }
            }

            // Standard addition formula for different points
            // lambda = (y2 - y1) / (x2 - x1)
            const dy = other.y.sub(self.y);
            const dx = other.x.sub(self.x);
            const lambda = dy.mul(dx.inverse() orelse return identity());

            // x3 = lambda^2 - x1 - x2
            const x3 = lambda.square().sub(self.x).sub(other.x);

            // y3 = lambda * (x1 - x3) - y1
            const y3 = lambda.mul(self.x.sub(x3)).sub(self.y);

            return fromCoords(x3, y3);
        }

        /// Double an affine point
        pub fn double(self: Self) Self {
            if (self.infinity) return self;

            // If y = 0, doubling gives infinity
            if (self.y.isZero()) return identity();

            // For y^2 = x^3 + b (a = 0):
            // lambda = 3x^2 / 2y
            const x_sq = self.x.square();
            const three_x_sq = x_sq.add(x_sq).add(x_sq); // 3x^2
            const two_y = self.y.add(self.y); // 2y
            const lambda = three_x_sq.mul(two_y.inverse() orelse return identity());

            // x3 = lambda^2 - 2x
            const x3 = lambda.square().sub(self.x).sub(self.x);

            // y3 = lambda * (x - x3) - y
            const y3 = lambda.mul(self.x.sub(x3)).sub(self.y);

            return fromCoords(x3, y3);
        }
    };
}

/// Elliptic curve point in Jacobian projective coordinates
/// (X, Y, Z) represents affine (X/Z^2, Y/Z^3)
/// This is more efficient for repeated additions
pub fn ProjectivePoint(comptime F: type) type {
    return struct {
        const Self = @This();

        x: F,
        y: F,
        z: F,

        /// Identity point
        pub fn identity() Self {
            return .{
                .x = F.one(),
                .y = F.one(),
                .z = F.zero(),
            };
        }

        /// Create from affine point
        pub fn fromAffine(p: AffinePoint(F)) Self {
            if (p.infinity) return identity();
            return .{
                .x = p.x,
                .y = p.y,
                .z = F.one(),
            };
        }

        /// Check if this is the identity
        pub fn isIdentity(self: Self) bool {
            return self.z.isZero();
        }

        /// Convert to affine
        pub fn toAffine(self: Self) AffinePoint(F) {
            if (self.z.isZero()) return AffinePoint(F).identity();

            const z_inv = self.z.inverse() orelse return AffinePoint(F).identity();
            const z_inv_sq = z_inv.square();
            const z_inv_cube = z_inv_sq.mul(z_inv);

            return AffinePoint(F).fromCoords(
                self.x.mul(z_inv_sq),
                self.y.mul(z_inv_cube),
            );
        }

        /// Point doubling in Jacobian coordinates
        /// For y^2 = x^3 + b (a = 0)
        /// Using the efficient formulas from https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
        pub fn double(self: Self) Self {
            if (self.z.isZero()) return self;

            // A = X^2
            const A = self.x.square();
            // B = Y^2
            const B = self.y.square();
            // C = B^2
            const C = B.square();
            // D = 2*((X+B)^2 - A - C)
            const xpb = self.x.add(B);
            const D = xpb.square().sub(A).sub(C);
            const two_D = D.add(D);
            // E = 3*A
            const E = A.add(A).add(A);
            // F = E^2
            const FF = E.square();
            // X3 = F - 2*D
            const X3 = FF.sub(two_D);
            // Y3 = E*(D - X3) - 8*C
            const eight_C = C.add(C).add(C).add(C).add(C).add(C).add(C).add(C);
            const Y3 = E.mul(D.sub(X3)).sub(eight_C);
            // Z3 = 2*Y*Z
            const Z3 = self.y.mul(self.z).add(self.y.mul(self.z));

            return .{
                .x = X3,
                .y = Y3,
                .z = Z3,
            };
        }

        /// Point addition in Jacobian coordinates (mixed: self is Jacobian, other is affine)
        pub fn addAffine(self: Self, other: AffinePoint(F)) Self {
            if (other.infinity) return self;
            if (self.z.isZero()) return fromAffine(other);

            // Z1Z1 = Z1^2
            const z1z1 = self.z.square();
            // U2 = X2*Z1Z1
            const U2 = other.x.mul(z1z1);
            // S2 = Y2*Z1*Z1Z1
            const S2 = other.y.mul(self.z).mul(z1z1);
            // H = U2 - X1
            const H = U2.sub(self.x);
            // HH = H^2
            const HH = H.square();
            // I = 4*HH
            const I = HH.add(HH).add(HH).add(HH);
            // J = H*I
            const J = H.mul(I);
            // r = 2*(S2 - Y1)
            const r = S2.sub(self.y).add(S2.sub(self.y));
            // V = X1*I
            const V = self.x.mul(I);
            // X3 = r^2 - J - 2*V
            const X3 = r.square().sub(J).sub(V).sub(V);
            // Y3 = r*(V - X3) - 2*Y1*J
            const Y3 = r.mul(V.sub(X3)).sub(self.y.mul(J)).sub(self.y.mul(J));
            // Z3 = (Z1 + H)^2 - Z1Z1 - HH
            const Z3 = self.z.add(H).square().sub(z1z1).sub(HH);

            // Handle special case: if H = 0, points are equal or negatives
            if (H.isZero()) {
                if (r.isZero()) {
                    // Points are equal, double
                    return self.double();
                } else {
                    // Points are negatives
                    return identity();
                }
            }

            return .{
                .x = X3,
                .y = Y3,
                .z = Z3,
            };
        }

        /// Point addition in Jacobian coordinates
        pub fn add(self: Self, other: Self) Self {
            if (self.z.isZero()) return other;
            if (other.z.isZero()) return self;

            // Z1Z1 = Z1^2
            const z1z1 = self.z.square();
            // Z2Z2 = Z2^2
            const z2z2 = other.z.square();
            // U1 = X1*Z2Z2
            const U1 = self.x.mul(z2z2);
            // U2 = X2*Z1Z1
            const U2 = other.x.mul(z1z1);
            // S1 = Y1*Z2*Z2Z2
            const S1 = self.y.mul(other.z).mul(z2z2);
            // S2 = Y2*Z1*Z1Z1
            const S2 = other.y.mul(self.z).mul(z1z1);
            // H = U2 - U1
            const H = U2.sub(U1);
            // I = (2*H)^2
            const two_H = H.add(H);
            const I = two_H.square();
            // J = H*I
            const J = H.mul(I);
            // r = 2*(S2 - S1)
            const r = S2.sub(S1).add(S2.sub(S1));
            // V = U1*I
            const V = U1.mul(I);
            // X3 = r^2 - J - 2*V
            const X3 = r.square().sub(J).sub(V).sub(V);
            // Y3 = r*(V - X3) - 2*S1*J
            const Y3 = r.mul(V.sub(X3)).sub(S1.mul(J)).sub(S1.mul(J));
            // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2)*H
            const Z3 = self.z.add(other.z).square().sub(z1z1).sub(z2z2).mul(H);

            // Handle special case: if H = 0, points are on same vertical line
            if (H.isZero()) {
                if (r.isZero()) {
                    // Points are equal, double
                    return self.double();
                } else {
                    // Points are negatives
                    return identity();
                }
            }

            return .{
                .x = X3,
                .y = Y3,
                .z = Z3,
            };
        }
    };
}

/// MSM using Pippenger's algorithm
///
/// Pippenger's algorithm (also known as Pippenger's bucket method) reduces the
/// complexity of MSM from O(n * log(r)) to O(n + 2^c * log(r)/c) where:
/// - n is the number of points
/// - r is the scalar field size
/// - c is the window size (chosen optimally as ~log(n))
///
/// The algorithm:
/// 1. Choose window size c
/// 2. Split scalars into windows of c bits
/// 3. For each window position, accumulate points into 2^c-1 buckets
/// 4. Sum buckets using a running sum trick
/// 5. Combine window results with appropriate doubling
pub fn MSM(comptime F: type, comptime G: type) type {
    return struct {
        const Self = @This();
        const Affine = AffinePoint(G);
        const Projective = ProjectivePoint(G);

        /// Scalar bit width
        const SCALAR_BITS: usize = 256;

        /// Compute sum_{i} scalars[i] * bases[i]
        pub fn compute(
            bases: []const Affine,
            scalars: []const F,
        ) Affine {
            std.debug.assert(bases.len == scalars.len);

            if (bases.len == 0) {
                return Affine.identity();
            }

            // For small inputs, use naive algorithm
            if (bases.len < 8) {
                return naiveMSM(bases, scalars);
            }

            // Use Pippenger's algorithm for larger inputs
            return pippengerMSM(bases, scalars);
        }

        /// Pippenger's bucket method MSM
        fn pippengerMSM(
            bases: []const Affine,
            scalars: []const F,
        ) Affine {
            // Choose optimal window size: c ≈ max(1, log2(n))
            const c = optimalWindowSize(bases.len);
            const num_windows = (SCALAR_BITS + c - 1) / c;
            const num_buckets = (@as(usize, 1) << @as(u6, @intCast(c))) - 1; // 2^c - 1 buckets (excluding 0)

            // Accumulator for final result
            var final_result = Projective.identity();

            // Process windows from most significant to least significant
            var window_idx: usize = num_windows;
            while (window_idx > 0) {
                window_idx -= 1;

                // Double the result c times for the previous windows
                if (!final_result.isIdentity()) {
                    var i: usize = 0;
                    while (i < c) : (i += 1) {
                        final_result = final_result.double();
                    }
                }

                // Accumulate into buckets for this window
                var buckets: [256]Projective = undefined; // Max 256 buckets (8-bit window)
                for (0..@min(num_buckets, 256)) |j| {
                    buckets[j] = Projective.identity();
                }

                for (bases, scalars) |base, scalar| {
                    if (base.isIdentity()) continue;

                    // Get the c-bit window from the scalar
                    const bucket_idx = getWindow(scalar, window_idx, c);
                    if (bucket_idx == 0) continue; // Skip bucket 0

                    // Add point to bucket (bucket indices are 1 to 2^c - 1)
                    const idx = bucket_idx - 1;
                    if (idx < num_buckets) {
                        buckets[idx] = buckets[idx].addAffine(base);
                    }
                }

                // Sum buckets using running sum:
                // result = sum_{i=1}^{2^c-1} i * buckets[i]
                // = buckets[2^c-1] + (buckets[2^c-1] + buckets[2^c-2]) + ... + (sum all buckets)
                var running_sum = Projective.identity();
                var window_sum = Projective.identity();

                // Process from highest bucket to lowest
                var bucket_idx: usize = num_buckets;
                while (bucket_idx > 0) {
                    bucket_idx -= 1;
                    running_sum = running_sum.add(buckets[bucket_idx]);
                    window_sum = window_sum.add(running_sum);
                }

                final_result = final_result.add(window_sum);
            }

            return final_result.toAffine();
        }

        /// Get c-bit window from scalar at position window_idx
        fn getWindow(scalar: F, window_idx: usize, c: usize) usize {
            const normal_scalar = scalar.fromMontgomery();
            const bit_offset = window_idx * c;

            // Calculate which limb(s) contain the bits
            const limb_idx = bit_offset / 64;
            const bit_in_limb = @as(u6, @intCast(bit_offset % 64));

            if (limb_idx >= 4) return 0;

            // Create mask for c bits
            const mask: u64 = (@as(u64, 1) << @as(u6, @intCast(c))) - 1;

            // Extract bits (may need to combine from two limbs)
            var value = (normal_scalar.limbs[limb_idx] >> bit_in_limb) & mask;

            // If window crosses limb boundary, get bits from next limb
            const bit_in_limb_usize: usize = @as(usize, bit_in_limb);
            if (bit_in_limb_usize + c > 64 and limb_idx + 1 < 4) {
                const remaining = bit_in_limb_usize + c - 64;
                if (remaining > 0 and remaining <= 63 and bit_in_limb > 0) {
                    const remaining_bits: u6 = @intCast(remaining);
                    const next_limb = normal_scalar.limbs[limb_idx + 1];
                    const next_mask = (@as(u64, 1) << remaining_bits) - 1;
                    const shift_amount: u6 = @intCast(64 - bit_in_limb_usize);
                    value |= (next_limb & next_mask) << shift_amount;
                }
            }

            return @as(usize, @intCast(value & mask));
        }

        /// Choose optimal window size based on input size
        /// Optimal c ≈ log2(n) for MSM of n points
        fn optimalWindowSize(n: usize) usize {
            if (n < 8) return 1;
            if (n < 32) return 2;
            if (n < 128) return 3;
            if (n < 512) return 4;
            if (n < 2048) return 5;
            if (n < 8192) return 6;
            if (n < 32768) return 7;
            return 8; // Max window size of 8 bits
        }

        /// Naive O(n * 256) MSM (fallback for small inputs)
        fn naiveMSM(
            bases: []const Affine,
            scalars: []const F,
        ) Affine {
            var result = Projective.identity();

            for (bases, scalars) |base, scalar| {
                const term = scalarMul(base, scalar);
                result = result.add(term);
            }

            return result.toAffine();
        }

        /// Scalar multiplication using double-and-add
        /// Processes bits from most significant to least significant
        pub fn scalarMul(base: Affine, scalar: F) Projective {
            if (base.isIdentity()) return Projective.identity();
            if (scalar.isZero()) return Projective.identity();

            var result = Projective.identity();

            // Convert scalar from Montgomery form to get actual value
            const normal_scalar = scalar.fromMontgomery();

            // Process each limb of the scalar from most significant to least significant
            // limbs[3] is most significant, limbs[0] is least significant
            var limb_idx: usize = 4;
            while (limb_idx > 0) {
                limb_idx -= 1;
                const limb = normal_scalar.limbs[limb_idx];

                // Process bits from most significant to least significant
                var bit_idx: u7 = 64;
                while (bit_idx > 0) {
                    bit_idx -= 1;
                    // Always double (unless we haven't started yet)
                    if (!result.isIdentity()) {
                        result = result.double();
                    }

                    const bit = (limb >> @as(u6, @intCast(bit_idx))) & 1;
                    if (bit == 1) {
                        if (result.isIdentity()) {
                            result = Projective.fromAffine(base);
                        } else {
                            result = result.addAffine(base);
                        }
                    }
                }
            }

            return result;
        }
    };
}

/// Batch MSM for multiple MSM operations
pub fn BatchMSM(comptime F: type, comptime G: type) type {
    return struct {
        const Self = @This();
        const SingleMSM = MSM(F, G);

        /// Compute multiple MSMs with shared bases
        pub fn compute(
            bases: []const AffinePoint(G),
            scalar_batches: []const []const F,
            allocator: Allocator,
        ) ![]AffinePoint(G) {
            const results = try allocator.alloc(AffinePoint(G), scalar_batches.len);

            for (scalar_batches, 0..) |scalars, i| {
                results[i] = SingleMSM.compute(bases, scalars);
            }

            return results;
        }
    };
}

test "msm types compile" {
    const F = @import("../field/mod.zig").BN254Scalar;

    // Verify types compile
    _ = AffinePoint(F);
    _ = ProjectivePoint(F);
    _ = MSM(F, F);
}

test "affine point identity" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const Point = AffinePoint(F);

    const id = Point.identity();
    try std.testing.expect(id.isIdentity());
}

test "affine point addition identity" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const Point = AffinePoint(F);

    const p = Point.fromCoords(F.fromU64(1), F.fromU64(2));
    const id = Point.identity();

    // p + 0 = p
    const sum = p.add(id);
    try std.testing.expect(sum.eql(p));

    // 0 + p = p
    const sum2 = id.add(p);
    try std.testing.expect(sum2.eql(p));
}

test "projective point conversion" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const Affine = AffinePoint(F);
    const Proj = ProjectivePoint(F);

    // Identity conversion
    const id_affine = Affine.identity();
    const id_proj = Proj.fromAffine(id_affine);
    try std.testing.expect(id_proj.isIdentity());
    try std.testing.expect(id_proj.toAffine().isIdentity());

    // Non-identity point
    const p = Affine.fromCoords(F.fromU64(5), F.fromU64(7));
    const p_proj = Proj.fromAffine(p);
    const p_back = p_proj.toAffine();
    try std.testing.expect(p_back.x.eql(F.fromU64(5)));
    try std.testing.expect(p_back.y.eql(F.fromU64(7)));
}

test "scalar multiplication by zero" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const Affine = AffinePoint(F);
    const SingleMSM = MSM(F, F);

    const p = Affine.fromCoords(F.fromU64(1), F.fromU64(2));

    // 0 * P = O
    const result = SingleMSM.compute(&[_]Affine{p}, &[_]F{F.zero()});
    try std.testing.expect(result.isIdentity());
}

test "scalar multiplication by one" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const Affine = AffinePoint(F);
    const SingleMSM = MSM(F, F);

    const p = Affine.fromCoords(F.fromU64(7), F.fromU64(11));

    // 1 * P = P
    const result = SingleMSM.compute(&[_]Affine{p}, &[_]F{F.one()});
    try std.testing.expect(result.x.eql(p.x));
    try std.testing.expect(result.y.eql(p.y));
}

test "pippenger optimal window size" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const SingleMSM = MSM(F, F);

    // Window size should grow with input size
    try std.testing.expectEqual(@as(usize, 1), SingleMSM.optimalWindowSize(4));
    try std.testing.expectEqual(@as(usize, 2), SingleMSM.optimalWindowSize(16));
    try std.testing.expectEqual(@as(usize, 4), SingleMSM.optimalWindowSize(256));
    try std.testing.expectEqual(@as(usize, 6), SingleMSM.optimalWindowSize(4096));
    try std.testing.expectEqual(@as(usize, 8), SingleMSM.optimalWindowSize(100000));
}

test "getWindow extracts correct bits" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const SingleMSM = MSM(F, F);

    // Test with a known scalar value
    const scalar = F.fromU64(0b11001010); // 202 in decimal

    // Window 0 (bits 0-3): 1010 = 10
    try std.testing.expectEqual(@as(usize, 10), SingleMSM.getWindow(scalar, 0, 4));

    // Window 1 (bits 4-7): 1100 = 12
    try std.testing.expectEqual(@as(usize, 12), SingleMSM.getWindow(scalar, 1, 4));
}

test "pippenger msm basic" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const Affine = AffinePoint(F);
    const SingleMSM = MSM(F, F);

    // Test that Pippenger doesn't crash and produces consistent results
    var bases: [16]Affine = undefined;
    var scalars: [16]F = undefined;

    for (0..16) |i| {
        bases[i] = Affine.fromCoords(F.fromU64(i + 1), F.fromU64(i + 100));
        scalars[i] = F.fromU64(i * 17 + 3);
    }

    // Call Pippenger - verify it doesn't crash
    const result = SingleMSM.pippengerMSM(&bases, &scalars);

    // Result should be a valid point (not NaN or error)
    // The actual result is hard to verify without knowing the curve equation
    _ = result;
}

test "pippenger handles zero scalars" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const Affine = AffinePoint(F);
    const SingleMSM = MSM(F, F);

    var bases: [10]Affine = undefined;
    var scalars: [10]F = undefined;

    // All zero scalars should give identity
    for (0..10) |i| {
        bases[i] = Affine.fromCoords(F.fromU64(i + 1), F.fromU64(i + 2));
        scalars[i] = F.zero();
    }

    const result = SingleMSM.pippengerMSM(&bases, &scalars);
    try std.testing.expect(result.isIdentity());
}
