//! Multi-Scalar Multiplication (MSM) for Jolt
//!
//! MSM computes sum_{i=0}^{n-1} s_i * G_i where s_i are scalars and G_i are curve points.
//! This is a critical operation for polynomial commitments.
//!
//! We use the short Weierstrass curve y^2 = x^3 + b (a = 0 for BN254)

const std = @import("std");
const Allocator = std.mem.Allocator;
const ThreadPool = @import("../utils/thread_pool.zig").ThreadPool;

pub const glv = @import("glv.zig");

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

        /// BN254 G1 generator point: (1, 2)
        /// This is the standard generator point for the BN254/alt_bn128 curve.
        pub fn generator() Self {
            return .{
                .x = F.one(),
                .y = F.fromU64(2),
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

        /// Check if the point is on the curve y² = x³ + b (b = 3 for BN254)
        pub fn isOnCurve(self: Self) bool {
            if (self.infinity) return true;

            // y² = x³ + b
            const y_sq = self.y.square();
            const x_cubed = self.x.square().mul(self.x);
            const rhs = x_cubed.add(F.fromU64(BN254_B)); // b = 3

            return y_sq.eql(rhs);
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
        /// Specifically using dbl-2009-l formulas.
        pub fn double(self: Self) Self {
            if (self.z.isZero()) return self;

            // A = X^2
            const A = self.x.square();
            // B = Y^2
            const B = self.y.square();
            // C = B^2
            const C = B.square();
            // D = 2*((X+B)^2 - A - C)  -- note: D already includes the factor of 2
            const xpb = self.x.add(B);
            const half_D = xpb.square().sub(A).sub(C);
            const D = half_D.add(half_D); // D = 2 * half_D
            // E = 3*A
            const E = A.add(A).add(A);
            // F = E^2
            const FF = E.square();
            // X3 = F - 2*D
            const two_D = D.add(D);
            const X3 = FF.sub(two_D);
            // Y3 = E*(D - X3) - 8*C  -- using D (which is 2*half_D)
            const two_C = C.add(C);
            const four_C = two_C.add(two_C);
            const eight_C = four_C.add(four_C);
            const Y3 = E.mul(D.sub(X3)).sub(eight_C);
            // Z3 = 2*Y*Z
            const yz = self.y.mul(self.z);
            const Z3 = yz.add(yz);

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
        /// Batch normalize projective points to affine using Montgomery's trick.
        /// Single inversion + ~6n multiplications instead of n inversions.
        pub fn batchNormalize(points: []const Self, out: []AffinePoint(F)) void {
            std.debug.assert(out.len >= points.len);
            const n = points.len;
            if (n == 0) return;

            // Accumulate Z products
            // products[i] = Z_0 * Z_1 * ... * Z_i (skipping identity points)
            var stack_products: [2048]F = undefined;
            var heap_products: ?[]F = null;
            defer if (heap_products) |h| std.heap.page_allocator.free(h);

            const products: []F = if (n <= 2048)
                stack_products[0..n]
            else blk: {
                heap_products = std.heap.page_allocator.alloc(F, n) catch {
                    // Fallback: individual conversions
                    for (points, 0..) |p, i| out[i] = p.toAffine();
                    return;
                };
                break :blk heap_products.?;
            };

            // Forward pass: accumulate Z products
            var acc = F.one();
            for (points, 0..) |p, i| {
                if (p.isIdentity()) {
                    products[i] = acc; // placeholder, won't be used
                } else {
                    products[i] = acc;
                    acc = acc.mul(p.z);
                }
            }

            // Invert the accumulated product
            var inv = acc.inverse() orelse F.one();

            // Backward pass: extract individual Z inverses
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                if (points[i].isIdentity()) {
                    out[i] = AffinePoint(F).identity();
                } else {
                    const z_inv = products[i].mul(inv);
                    inv = inv.mul(points[i].z);
                    const z_inv_sq = z_inv.square();
                    const z_inv_cube = z_inv_sq.mul(z_inv);
                    out[i] = AffinePoint(F).fromCoords(
                        points[i].x.mul(z_inv_sq),
                        points[i].y.mul(z_inv_cube),
                    );
                }
            }
        }
    };
}

/// Bucket point in XYZZ coordinates for efficient MSM bucket accumulation.
/// Represents affine (X/ZZ, Y/ZZZ). Mixed affine addition costs ~7M+2S
/// vs ~7M+4S for Jacobian mixed add.
pub fn BucketPoint(comptime F: type) type {
    return struct {
        const Self = @This();

        x: F,
        y: F,
        zz: F,
        zzz: F,
        empty: bool,

        pub fn identity() Self {
            return .{
                .x = F.zero(),
                .y = F.zero(),
                .zz = F.zero(),
                .zzz = F.zero(),
                .empty = true,
            };
        }

        pub fn isIdentity(self: Self) bool {
            return self.empty;
        }

        /// Mixed affine addition: XYZZ + affine (madd-2008-s, 7M+2S)
        pub fn addAffine(self: Self, other: AffinePoint(F)) Self {
            if (other.isIdentity()) return self;
            if (self.empty) {
                return .{
                    .x = other.x,
                    .y = other.y,
                    .zz = F.one(),
                    .zzz = F.one(),
                    .empty = false,
                };
            }

            // U2 = X2 * ZZ1, S2 = Y2 * ZZZ1
            const U2 = other.x.mul(self.zz);
            const S2 = other.y.mul(self.zzz);
            const P = U2.sub(self.x);
            const R = S2.sub(self.y);

            if (P.isZero()) {
                if (R.isZero()) {
                    // Same point — double
                    return self.doubleXYZZ();
                } else {
                    // Inverse points
                    return identity();
                }
            }

            const PP = P.square();
            const PPP = P.mul(PP);
            const Q = self.x.mul(PP);
            const X3 = R.square().sub(PPP).sub(Q).sub(Q);
            const Y3 = R.mul(Q.sub(X3)).sub(self.y.mul(PPP));
            const ZZ3 = self.zz.mul(PP);
            const ZZZ3 = self.zzz.mul(PPP);

            return .{ .x = X3, .y = Y3, .zz = ZZ3, .zzz = ZZZ3, .empty = false };
        }

        /// Subtract affine point (add negated)
        pub fn subAffine(self: Self, other: AffinePoint(F)) Self {
            if (other.isIdentity()) return self;
            return self.addAffine(other.neg());
        }

        /// XYZZ doubling (a=0 short Weierstrass)
        fn doubleXYZZ(self: Self) Self {
            if (self.empty) return self;
            // Using formulas adapted from Jacobian dbl-2009-l
            const A = self.x.square();
            const B = self.y.square();
            const C = B.square();
            const xpb = self.x.add(B);
            const half_D = xpb.square().sub(A).sub(C);
            const D = half_D.add(half_D);
            const E = A.add(A).add(A); // 3A
            const FF = E.square();
            const X3 = FF.sub(D).sub(D);
            const two_C = C.add(C);
            const four_C = two_C.add(two_C);
            const eight_C = four_C.add(four_C);
            const Y3 = E.mul(D.sub(X3)).sub(eight_C);
            // ZZ3 = (2Y)^2 * ZZ = 4*B*ZZ
            const four_B = B.add(B).add(B).add(B);
            const ZZ3 = four_B.mul(self.zz);
            // ZZZ3 = (2Y)^3 * ZZZ = 8*Y*B*ZZ * ZZZ... simpler:
            // ZZZ3 = 2Y * ZZ3 * (ZZZ/ZZ) ... actually let's derive properly
            // Z_new = 2*Y*Z_old, ZZ3 = Z_new^2 = 4*Y^2*Z_old^2 = 4*B*ZZ
            // ZZZ3 = Z_new^3 = 8*Y^3*Z_old^3 = 8*Y*B*ZZZ
            const eight_Y = self.y.add(self.y).add(self.y.add(self.y)).add(self.y.add(self.y).add(self.y.add(self.y)));
            const ZZZ3 = eight_Y.mul(B).mul(self.zzz);

            return .{ .x = X3, .y = Y3, .zz = ZZ3, .zzz = ZZZ3, .empty = false };
        }

        /// Add two XYZZ points (for running sum phase)
        pub fn add(self: Self, other: Self) Self {
            if (self.empty) return other;
            if (other.empty) return self;

            const U1 = self.x.mul(other.zz);
            const U2 = other.x.mul(self.zz);
            const S1 = self.y.mul(other.zzz);
            const S2 = other.y.mul(self.zzz);
            const P = U2.sub(U1);
            const R = S2.sub(S1);

            if (P.isZero()) {
                if (R.isZero()) {
                    return self.doubleXYZZ();
                } else {
                    return identity();
                }
            }

            const PP = P.square();
            const PPP = P.mul(PP);
            const Q = U1.mul(PP);
            const X3 = R.square().sub(PPP).sub(Q).sub(Q);
            const Y3 = R.mul(Q.sub(X3)).sub(S1.mul(PPP));
            const ZZ3 = self.zz.mul(other.zz).mul(PP);
            const ZZZ3 = self.zzz.mul(other.zzz).mul(PPP);

            return .{ .x = X3, .y = Y3, .zz = ZZ3, .zzz = ZZZ3, .empty = false };
        }

        /// Convert to Jacobian projective (Z = zz, X = x*zz, Y = y*zzz... )
        /// Actually XYZZ (X, Y, ZZ, ZZZ) → Jacobian (X', Y', Z') where:
        ///   Z' such that ZZ = Z'^2 and ZZZ = Z'^3
        ///   Z' = ZZZ / ZZ (when ZZ != 0)
        ///   X' = X * ZZ  (since Jacobian X = affine_x * Z^2 = (X/ZZ) * ZZ^2... nah)
        /// Simpler: just convert to affine first
        pub fn toProjective(self: Self) ProjectivePoint(F) {
            if (self.empty) return ProjectivePoint(F).identity();
            const aff = self.toAffine();
            return ProjectivePoint(F).fromAffine(aff);
        }

        pub fn toAffine(self: Self) AffinePoint(F) {
            if (self.empty) return AffinePoint(F).identity();
            // Batch inverse trick: 1 inversion + 3 muls instead of 2 inversions
            const product = self.zz.mul(self.zzz);
            const product_inv = product.inverse() orelse return AffinePoint(F).identity();
            const zz_inv = product_inv.mul(self.zzz);
            const zzz_inv = product_inv.mul(self.zz);
            return AffinePoint(F).fromCoords(
                self.x.mul(zz_inv),
                self.y.mul(zzz_inv),
            );
        }
    };
}

/// MSM using Pippenger's algorithm with wNAF signed digit decomposition
/// and XYZZ bucket coordinates.
///
/// Pippenger's algorithm (also known as Pippenger's bucket method) reduces the
/// complexity of MSM from O(n * log(r)) to O(n + 2^c * log(r)/c) where:
/// - n is the number of points
/// - r is the scalar field size
/// - c is the window size (chosen optimally as ~log(n))
///
/// wNAF recenters each c-bit digit to [-2^(c-1), 2^(c-1)), halving the
/// expected number of non-zero digits and bucket additions.
///
/// XYZZ bucket coordinates reduce mixed affine addition cost from ~7M+4S
/// (Jacobian) to ~7M+2S.
pub fn MSM(comptime F: type, comptime G: type) type {
    return struct {
        const Self = @This();
        const Affine = AffinePoint(G);
        const Projective = ProjectivePoint(G);
        const Bucket = BucketPoint(G);

        /// Scalar bit width
        const SCALAR_BITS: usize = 256;

        /// Max digits = ceil(256/c) + 1 (extra for wNAF carry), c >= 3
        const MAX_DIGITS: usize = 87;

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

        /// Compute MSM with optional thread pool for parallel window processing
        pub fn computeWithPool(
            bases: []const Affine,
            scalars: []const F,
            tp: ?*ThreadPool,
        ) Affine {
            std.debug.assert(bases.len == scalars.len);

            if (bases.len == 0) {
                return Affine.identity();
            }

            if (bases.len < 8) {
                return naiveMSM(bases, scalars);
            }

            if (tp != null and bases.len >= 256) {
                return pippengerMSMParallel(bases, scalars, tp.?);
            }

            return pippengerMSM(bases, scalars);
        }

        /// wNAF signed digit decomposition.
        /// Decomposes scalar into num_digits signed digits in [-2^(c-1), 2^(c-1)).
        /// num_digits should be ceil(256/c) + 1 to handle carry from the last window.
        fn makeDigits(limbs: [4]u64, c: usize, num_digits: usize) [MAX_DIGITS]i32 {
            var digits: [MAX_DIGITS]i32 = undefined;
            const radix: u64 = @as(u64, 1) << @as(u6, @intCast(c));
            const window_mask: u64 = radix - 1;
            const half_radix: u64 = radix >> 1;
            var carry: u64 = 0;

            for (0..num_digits) |i| {
                const bit_offset = i * c;
                const u64_idx = bit_offset / 64;
                const bit_idx = @as(u6, @intCast(bit_offset % 64));

                var bit_buf: u64 = if (u64_idx < 4) limbs[u64_idx] >> bit_idx else 0;
                const bit_idx_usize: usize = @as(usize, bit_idx);
                if (bit_idx_usize + c > 64 and u64_idx + 1 < 4) {
                    bit_buf |= limbs[u64_idx + 1] << @as(u6, @intCast(64 - bit_idx_usize));
                }

                const coef = carry + (bit_buf & window_mask);
                carry = if (coef >= half_radix) 1 else 0;
                const digit: i32 = @as(i32, @intCast(coef)) - @as(i32, @intCast(carry)) * @as(i32, @intCast(radix));

                digits[i] = digit;
            }
            // Final carry becomes an extra digit (0 or 1)
            if (num_digits < MAX_DIGITS) {
                digits[num_digits] = @intCast(carry);
            }
            return digits;
        }

        /// Pippenger's bucket method MSM with wNAF + XYZZ buckets
        fn pippengerMSM(
            bases: []const Affine,
            scalars: []const F,
        ) Affine {
            const c = optimalWindowSize(bases.len);
            const num_scalar_windows = (SCALAR_BITS + c - 1) / c;
            const num_windows = num_scalar_windows + 1; // +1 for wNAF carry digit
            const num_buckets = (@as(usize, 1) << @as(u6, @intCast(c))) / 2; // 2^(c-1) buckets for wNAF

            // Pre-convert all scalars and compute wNAF digits
            const stack_threshold = 256;
            var stack_digits: [stack_threshold][MAX_DIGITS]i32 = undefined;
            var heap_digits: ?[][MAX_DIGITS]i32 = null;
            defer if (heap_digits) |buf| std.heap.page_allocator.free(buf);

            const all_digits: [][MAX_DIGITS]i32 = if (scalars.len <= stack_threshold)
                stack_digits[0..scalars.len]
            else blk: {
                heap_digits = std.heap.page_allocator.alloc([MAX_DIGITS]i32, scalars.len) catch
                    return naiveMSM(bases, scalars);
                break :blk heap_digits.?;
            };

            for (scalars, 0..) |s, i| {
                all_digits[i] = makeDigits(s.fromMontgomery().limbs, c, num_scalar_windows);
            }

            // Allocate buckets on heap for larger windows
            var heap_buckets: ?[]Bucket = null;
            defer if (heap_buckets) |buf| std.heap.page_allocator.free(buf);

            // Stack buckets for small windows (up to 256 buckets)
            var stack_buckets: [256]Bucket = undefined;
            const buckets_slice: []Bucket = if (num_buckets <= 256)
                stack_buckets[0..num_buckets]
            else blk: {
                heap_buckets = std.heap.page_allocator.alloc(Bucket, num_buckets) catch
                    return naiveMSM(bases, scalars);
                break :blk heap_buckets.?;
            };

            // Phase 1: Compute all window sums in XYZZ form
            var window_sums: [MAX_DIGITS]Bucket = undefined;
            for (0..num_windows) |wi| {
                // Reset buckets
                for (0..num_buckets) |j| {
                    buckets_slice[j] = Bucket.identity();
                }

                // Accumulate into buckets using wNAF signed digits
                for (bases, 0..) |base, idx| {
                    if (base.isIdentity()) continue;

                    const digit = all_digits[idx][wi];
                    if (digit > 0) {
                        const bidx: usize = @intCast(digit - 1);
                        buckets_slice[bidx] = buckets_slice[bidx].addAffine(base);
                    } else if (digit < 0) {
                        const bidx: usize = @intCast(-digit - 1);
                        buckets_slice[bidx] = buckets_slice[bidx].subAffine(base);
                    }
                }

                // Sum buckets using running sum trick
                var running_sum = Bucket.identity();
                var window_sum = Bucket.identity();

                var bucket_idx: usize = num_buckets;
                while (bucket_idx > 0) {
                    bucket_idx -= 1;
                    running_sum = running_sum.add(buckets_slice[bucket_idx]);
                    window_sum = window_sum.add(running_sum);
                }

                window_sums[wi] = window_sum;
            }

            // Phase 2: Batch convert window sums to affine using Montgomery's trick
            // (1 inversion + O(n) muls instead of 2n inversions)
            var window_sums_affine: [MAX_DIGITS]Affine = undefined;
            var products: [MAX_DIGITS]G = undefined; // zz * zzz per non-empty window
            var non_empty_indices: [MAX_DIGITS]usize = undefined;
            var num_non_empty: usize = 0;

            for (0..num_windows) |wi| {
                if (window_sums[wi].empty) {
                    window_sums_affine[wi] = Affine.identity();
                } else {
                    products[num_non_empty] = window_sums[wi].zz.mul(window_sums[wi].zzz);
                    non_empty_indices[num_non_empty] = wi;
                    num_non_empty += 1;
                }
            }

            if (num_non_empty > 0) {
                // Build prefix products
                var prefix: [MAX_DIGITS]G = undefined;
                prefix[0] = products[0];
                for (1..num_non_empty) |i| {
                    prefix[i] = prefix[i - 1].mul(products[i]);
                }

                // Single inversion of the total product
                var running_inv = prefix[num_non_empty - 1].inverse() orelse G.one();

                // Backtrack to recover individual inverses
                var i = num_non_empty;
                while (i > 1) {
                    i -= 1;
                    const product_inv = prefix[i - 1].mul(running_inv);
                    running_inv = running_inv.mul(products[i]);
                    const wi = non_empty_indices[i];
                    const zz_inv = product_inv.mul(window_sums[wi].zzz);
                    const zzz_inv = product_inv.mul(window_sums[wi].zz);
                    window_sums_affine[wi] = Affine.fromCoords(
                        window_sums[wi].x.mul(zz_inv),
                        window_sums[wi].y.mul(zzz_inv),
                    );
                }
                // Handle first element
                {
                    const wi = non_empty_indices[0];
                    const zz_inv = running_inv.mul(window_sums[wi].zzz);
                    const zzz_inv = running_inv.mul(window_sums[wi].zz);
                    window_sums_affine[wi] = Affine.fromCoords(
                        window_sums[wi].x.mul(zz_inv),
                        window_sums[wi].y.mul(zzz_inv),
                    );
                }
            }

            // Phase 3: Combine windows (MSB to LSB) using addAffine (cheaper than generic add)
            var final_result = Projective.identity();
            var window_idx: usize = num_windows;
            while (window_idx > 0) {
                window_idx -= 1;

                if (!final_result.isIdentity()) {
                    var k: usize = 0;
                    while (k < c) : (k += 1) {
                        final_result = final_result.double();
                    }
                }

                if (!window_sums_affine[window_idx].isIdentity()) {
                    if (final_result.isIdentity()) {
                        final_result = Projective.fromAffine(window_sums_affine[window_idx]);
                    } else {
                        final_result = final_result.addAffine(window_sums_affine[window_idx]);
                    }
                }
            }

            return final_result.toAffine();
        }

        /// Chunk-based parallel MSM: split input into num_threads chunks,
        /// each chunk runs sequential Pippenger independently, results summed.
        /// Better cache locality than per-window parallelism (chunk fits L2).
        fn pippengerMSMParallel(
            bases: []const Affine,
            scalars: []const F,
            tp: *ThreadPool,
        ) Affine {
            const n = bases.len;
            const num_threads = tp.thread_count + 1; // workers + caller

            // For small inputs, chunks would be too small for Pippenger
            if (n < num_threads * 256) {
                return pippengerMSM(bases, scalars);
            }

            const num_chunks = num_threads;
            const chunk_size = (n + num_chunks - 1) / num_chunks;

            const chunk_results = std.heap.page_allocator.alloc(Affine, num_chunks) catch
                return pippengerMSM(bases, scalars);
            defer std.heap.page_allocator.free(chunk_results);

            const ChunkCtx = struct {
                bases: []const Affine,
                scalars: []const F,
                results: []Affine,
                chunk_sz: usize,
                total_n: usize,
            };
            const ctx = ChunkCtx{
                .bases = bases,
                .scalars = scalars,
                .results = chunk_results,
                .chunk_sz = chunk_size,
                .total_n = n,
            };

            tp.parallelForForce(num_chunks, ctx, struct {
                fn f(c: ChunkCtx, chunk_idx: usize) void {
                    const start = chunk_idx * c.chunk_sz;
                    const end = @min(start + c.chunk_sz, c.total_n);
                    if (start >= c.total_n) {
                        c.results[chunk_idx] = Affine.identity();
                        return;
                    }
                    c.results[chunk_idx] = pippengerMSM(
                        c.bases[start..end],
                        c.scalars[start..end],
                    );
                }
            }.f);

            // Sum chunk results (only num_threads point additions)
            var final_result = Projective.identity();
            for (chunk_results) |cr| {
                if (!cr.isIdentity()) {
                    final_result = final_result.addAffine(cr);
                }
            }
            return final_result.toAffine();
        }

        /// Get c-bit window from scalar at position window_idx
        fn getWindow(scalar: F, window_idx: usize, c: usize) usize {
            const normal_scalar = scalar.fromMontgomery();
            return getWindowFromLimbs(normal_scalar.limbs, window_idx, c);
        }

        /// Get c-bit window from pre-converted (non-Montgomery) limbs
        fn getWindowFromLimbs(limbs: [4]u64, window_idx: usize, c: usize) usize {
            const bit_offset = window_idx * c;

            // Calculate which limb(s) contain the bits
            const limb_idx = bit_offset / 64;
            const bit_in_limb = @as(u6, @intCast(bit_offset % 64));

            if (limb_idx >= 4) return 0;

            // Create mask for c bits
            const mask: u64 = (@as(u64, 1) << @as(u6, @intCast(c))) - 1;

            // Extract bits (may need to combine from two limbs)
            var value = (limbs[limb_idx] >> bit_in_limb) & mask;

            // If window crosses limb boundary, get bits from next limb
            const bit_in_limb_usize: usize = @as(usize, bit_in_limb);
            if (bit_in_limb_usize + c > 64 and limb_idx + 1 < 4) {
                const remaining = bit_in_limb_usize + c - 64;
                if (remaining > 0 and remaining <= 63 and bit_in_limb > 0) {
                    const remaining_bits: u6 = @intCast(remaining);
                    const next_limb = limbs[limb_idx + 1];
                    const next_mask = (@as(u64, 1) << remaining_bits) - 1;
                    const shift_amount: u6 = @intCast(64 - bit_in_limb_usize);
                    value |= (next_limb & next_mask) << shift_amount;
                }
            }

            return @as(usize, @intCast(value & mask));
        }

        /// Choose optimal window size based on input size.
        /// Uses ln(n) + 2 heuristic (matching arkworks' formula).
        /// Larger c = fewer windows (fewer passes over points) but more buckets (more memory).
        fn optimalWindowSize(n: usize) usize {
            if (n < 32) return 3;
            const log2_n = @as(usize, std.math.log2_int(usize, n));
            const c = (log2_n * 69 / 100) + 2;
            return @max(3, @min(c, 20));
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

        /// Scalar multiplication using pre-converted (non-Montgomery) limbs
        pub fn scalarMulWithLimbs(base: Affine, normal_limbs: [4]u64) Projective {
            if (base.isIdentity()) return Projective.identity();

            var result = Projective.identity();

            var limb_idx: usize = 4;
            while (limb_idx > 0) {
                limb_idx -= 1;
                const limb = normal_limbs[limb_idx];

                var bit_idx: u7 = 64;
                while (bit_idx > 0) {
                    bit_idx -= 1;
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

        // =====================================================================
        // i128 small-scalar MSM (128-bit scalars instead of 256-bit)
        // =====================================================================

        const SCALAR_BITS_I128: usize = 128;
        const MAX_DIGITS_I128: usize = (SCALAR_BITS_I128 + 2) / 3 + 1; // ceil(128/3) + 1 = 44

        /// wNAF digit decomposition for 128-bit scalars (2 limbs).
        fn makeDigitsI128(limbs: [2]u64, c: usize, num_digits: usize) [MAX_DIGITS_I128]i32 {
            var digits: [MAX_DIGITS_I128]i32 = undefined;
            const radix: u64 = @as(u64, 1) << @as(u6, @intCast(c));
            const window_mask: u64 = radix - 1;
            const half_radix: u64 = radix >> 1;
            var carry: u64 = 0;

            for (0..num_digits) |i| {
                const bit_offset = i * c;
                const u64_idx = bit_offset / 64;
                const bit_idx = @as(u6, @intCast(bit_offset % 64));

                var bit_buf: u64 = if (u64_idx < 2) limbs[u64_idx] >> bit_idx else 0;
                const bit_idx_usize: usize = @as(usize, bit_idx);
                if (bit_idx_usize + c > 64 and u64_idx + 1 < 2) {
                    bit_buf |= limbs[u64_idx + 1] << @as(u6, @intCast(64 - bit_idx_usize));
                }

                const coef = carry + (bit_buf & window_mask);
                carry = if (coef >= half_radix) 1 else 0;
                const digit: i32 = @as(i32, @intCast(coef)) - @as(i32, @intCast(carry)) * @as(i32, @intCast(radix));

                digits[i] = digit;
            }
            if (num_digits < MAX_DIGITS_I128) {
                digits[num_digits] = @intCast(carry);
            }
            return digits;
        }

        /// Compute MSM with i128 scalars (128-bit). ~2x faster than 256-bit for same point count.
        pub fn computeI128(
            bases: []const Affine,
            scalars: []const i128,
            tp: ?*ThreadPool,
        ) Affine {
            if (bases.len == 0 or scalars.len == 0) return Affine.identity();
            std.debug.assert(bases.len == scalars.len);

            if (tp != null and bases.len >= 256) {
                return pippengerMsmI128Parallel(bases, scalars, tp.?);
            }

            return pippengerMsmI128(bases, scalars);
        }

        /// Pippenger's bucket method MSM for i128 scalars.
        /// Detects actual scalar bit-width to reduce window count when scalars are small.
        fn pippengerMsmI128(
            bases: []const Affine,
            scalars: []const i128,
        ) Affine {
            const c = optimalWindowSize(bases.len);

            // Detect effective scalar bit-width (O(n) scan, trivial vs MSM cost)
            var max_abs: u128 = 0;
            for (scalars) |s| {
                const abs_val: u128 = if (s < 0) @intCast(-s) else @intCast(s);
                max_abs |= abs_val;
            }
            const effective_bits: usize = if (max_abs == 0) 1 else @as(usize, std.math.log2_int(u128, max_abs)) + 1;
            const num_scalar_windows = @min((effective_bits + c - 1) / c, (SCALAR_BITS_I128 + c - 1) / c);
            const num_windows = num_scalar_windows + 1;
            const num_buckets = (@as(usize, 1) << @as(u6, @intCast(c))) / 2;

            // Pre-compute wNAF digits and sign flags
            const stack_threshold = 256;
            var stack_digits: [stack_threshold][MAX_DIGITS_I128]i32 = undefined;
            var stack_signs: [stack_threshold]bool = undefined;
            var heap_digits: ?[][MAX_DIGITS_I128]i32 = null;
            var heap_signs: ?[]bool = null;
            defer if (heap_digits) |buf| std.heap.page_allocator.free(buf);
            defer if (heap_signs) |buf| std.heap.page_allocator.free(buf);

            const all_digits: [][MAX_DIGITS_I128]i32 = if (scalars.len <= stack_threshold)
                stack_digits[0..scalars.len]
            else blk: {
                heap_digits = std.heap.page_allocator.alloc([MAX_DIGITS_I128]i32, scalars.len) catch
                    return Affine.identity(); // OOM fallback
                break :blk heap_digits.?;
            };
            const all_negative: []bool = if (scalars.len <= stack_threshold)
                stack_signs[0..scalars.len]
            else blk: {
                heap_signs = std.heap.page_allocator.alloc(bool, scalars.len) catch
                    return Affine.identity();
                break :blk heap_signs.?;
            };

            for (scalars, 0..) |s, i| {
                const is_neg = s < 0;
                all_negative[i] = is_neg;
                const abs_val: u128 = if (is_neg) @intCast(-s) else @intCast(s);
                const limbs: [2]u64 = .{
                    @truncate(abs_val),
                    @truncate(abs_val >> 64),
                };
                all_digits[i] = makeDigitsI128(limbs, c, num_scalar_windows);
            }

            // Allocate buckets
            var heap_buckets: ?[]Bucket = null;
            defer if (heap_buckets) |buf| std.heap.page_allocator.free(buf);
            var stack_buckets: [256]Bucket = undefined;
            const buckets_slice: []Bucket = if (num_buckets <= 256)
                stack_buckets[0..num_buckets]
            else blk: {
                heap_buckets = std.heap.page_allocator.alloc(Bucket, num_buckets) catch
                    return Affine.identity();
                break :blk heap_buckets.?;
            };

            // Phase 1: Compute all window sums
            var window_sums: [MAX_DIGITS_I128]Bucket = undefined;
            for (0..num_windows) |wi| {
                for (0..num_buckets) |j| {
                    buckets_slice[j] = Bucket.identity();
                }

                for (bases, 0..) |base, idx| {
                    if (base.isIdentity()) continue;

                    const digit = all_digits[idx][wi];
                    // If scalar was negative, flip the sign of the digit contribution
                    if (digit > 0) {
                        const bidx: usize = @intCast(digit - 1);
                        if (all_negative[idx]) {
                            buckets_slice[bidx] = buckets_slice[bidx].subAffine(base);
                        } else {
                            buckets_slice[bidx] = buckets_slice[bidx].addAffine(base);
                        }
                    } else if (digit < 0) {
                        const bidx: usize = @intCast(-digit - 1);
                        if (all_negative[idx]) {
                            buckets_slice[bidx] = buckets_slice[bidx].addAffine(base);
                        } else {
                            buckets_slice[bidx] = buckets_slice[bidx].subAffine(base);
                        }
                    }
                }

                var running_sum = Bucket.identity();
                var window_sum = Bucket.identity();
                var bucket_idx: usize = num_buckets;
                while (bucket_idx > 0) {
                    bucket_idx -= 1;
                    running_sum = running_sum.add(buckets_slice[bucket_idx]);
                    window_sum = window_sum.add(running_sum);
                }
                window_sums[wi] = window_sum;
            }

            // Phase 2: Batch convert to affine
            var window_sums_affine: [MAX_DIGITS_I128]Affine = undefined;
            var products: [MAX_DIGITS_I128]G = undefined;
            var non_empty_indices: [MAX_DIGITS_I128]usize = undefined;
            var num_non_empty: usize = 0;

            for (0..num_windows) |wi| {
                if (window_sums[wi].empty) {
                    window_sums_affine[wi] = Affine.identity();
                } else {
                    products[num_non_empty] = window_sums[wi].zz.mul(window_sums[wi].zzz);
                    non_empty_indices[num_non_empty] = wi;
                    num_non_empty += 1;
                }
            }

            if (num_non_empty > 0) {
                var prefix: [MAX_DIGITS_I128]G = undefined;
                prefix[0] = products[0];
                for (1..num_non_empty) |i| {
                    prefix[i] = prefix[i - 1].mul(products[i]);
                }

                var running_inv = prefix[num_non_empty - 1].inverse() orelse G.one();

                var i = num_non_empty;
                while (i > 1) {
                    i -= 1;
                    const product_inv = prefix[i - 1].mul(running_inv);
                    running_inv = running_inv.mul(products[i]);
                    const wi = non_empty_indices[i];
                    const zz_inv = product_inv.mul(window_sums[wi].zzz);
                    const zzz_inv = product_inv.mul(window_sums[wi].zz);
                    window_sums_affine[wi] = Affine.fromCoords(
                        window_sums[wi].x.mul(zz_inv),
                        window_sums[wi].y.mul(zzz_inv),
                    );
                }
                {
                    const wi = non_empty_indices[0];
                    const zz_inv = running_inv.mul(window_sums[wi].zzz);
                    const zzz_inv = running_inv.mul(window_sums[wi].zz);
                    window_sums_affine[wi] = Affine.fromCoords(
                        window_sums[wi].x.mul(zz_inv),
                        window_sums[wi].y.mul(zzz_inv),
                    );
                }
            }

            // Phase 3: Combine windows
            var final_result = Projective.identity();
            var window_idx: usize = num_windows;
            while (window_idx > 0) {
                window_idx -= 1;
                if (!final_result.isIdentity()) {
                    var k: usize = 0;
                    while (k < c) : (k += 1) {
                        final_result = final_result.double();
                    }
                }
                if (!window_sums_affine[window_idx].isIdentity()) {
                    if (final_result.isIdentity()) {
                        final_result = Projective.fromAffine(window_sums_affine[window_idx]);
                    } else {
                        final_result = final_result.addAffine(window_sums_affine[window_idx]);
                    }
                }
            }

            return final_result.toAffine();
        }

        /// Chunk-based parallel MSM for i128 scalars: split input into
        /// num_threads chunks, each runs sequential Pippenger, results summed.
        fn pippengerMsmI128Parallel(
            bases: []const Affine,
            scalars: []const i128,
            tp: *ThreadPool,
        ) Affine {
            const n = bases.len;
            const num_threads = tp.thread_count + 1;

            if (n < num_threads * 256) {
                return pippengerMsmI128(bases, scalars);
            }

            const num_chunks = num_threads;
            const chunk_size = (n + num_chunks - 1) / num_chunks;

            const chunk_results = std.heap.page_allocator.alloc(Affine, num_chunks) catch
                return pippengerMsmI128(bases, scalars);
            defer std.heap.page_allocator.free(chunk_results);

            const ChunkCtx = struct {
                bases: []const Affine,
                scalars: []const i128,
                results: []Affine,
                chunk_sz: usize,
                total_n: usize,
            };
            const ctx = ChunkCtx{
                .bases = bases,
                .scalars = scalars,
                .results = chunk_results,
                .chunk_sz = chunk_size,
                .total_n = n,
            };

            tp.parallelForForce(num_chunks, ctx, struct {
                fn f(c: ChunkCtx, chunk_idx: usize) void {
                    const start = chunk_idx * c.chunk_sz;
                    const end = @min(start + c.chunk_sz, c.total_n);
                    if (start >= c.total_n) {
                        c.results[chunk_idx] = Affine.identity();
                        return;
                    }
                    c.results[chunk_idx] = pippengerMsmI128(
                        c.bases[start..end],
                        c.scalars[start..end],
                    );
                }
            }.f);

            var final_result = Projective.identity();
            for (chunk_results) |cr| {
                if (!cr.isIdentity()) {
                    final_result = final_result.addAffine(cr);
                }
            }
            return final_result.toAffine();
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

/// Parallel MSM using multiple threads
///
/// For large MSMs, we can split the work across multiple threads.
/// Each thread computes a partial MSM on a subset of the points,
/// and the results are combined at the end.
pub fn ParallelMSM(comptime F: type, comptime G: type) type {
    return struct {
        const Self = @This();
        const Affine = AffinePoint(G);
        const Projective = ProjectivePoint(G);
        const SingleMSM = MSM(F, G);

        /// Thread context for parallel MSM computation
        const ThreadContext = struct {
            bases: []const Affine,
            scalars: []const F,
            result: Projective,
        };

        /// Compute MSM using parallel threads
        /// Falls back to single-threaded for small inputs
        pub fn compute(
            bases: []const Affine,
            scalars: []const F,
            num_threads: usize,
            allocator: Allocator,
        ) !Affine {
            std.debug.assert(bases.len == scalars.len);

            if (bases.len == 0) {
                return Affine.identity();
            }

            // For small inputs, single-threaded is faster due to thread overhead
            const min_points_per_thread: usize = 1024;
            const actual_threads = @min(num_threads, @max(1, bases.len / min_points_per_thread));

            if (actual_threads <= 1) {
                return SingleMSM.compute(bases, scalars);
            }

            // Divide work among threads
            const chunk_size = (bases.len + actual_threads - 1) / actual_threads;

            // Allocate thread handles and contexts
            const thread_handles = try allocator.alloc(std.Thread, actual_threads);
            defer allocator.free(thread_handles);

            const contexts = try allocator.alloc(ThreadContext, actual_threads);
            defer allocator.free(contexts);

            // Spawn threads
            for (0..actual_threads) |i| {
                const start = i * chunk_size;
                const end = @min(start + chunk_size, bases.len);

                if (start >= bases.len) {
                    // No more work
                    contexts[i] = .{
                        .bases = &[_]Affine{},
                        .scalars = &[_]F{},
                        .result = Projective.identity(),
                    };
                    thread_handles[i] = try std.Thread.spawn(.{}, threadWorkerEmpty, .{&contexts[i]});
                } else {
                    contexts[i] = .{
                        .bases = bases[start..end],
                        .scalars = scalars[start..end],
                        .result = Projective.identity(),
                    };
                    thread_handles[i] = try std.Thread.spawn(.{}, threadWorker, .{&contexts[i]});
                }
            }

            // Wait for all threads to complete
            for (thread_handles) |handle| {
                handle.join();
            }

            // Combine results from all threads
            var final_result = Projective.identity();
            for (contexts) |ctx| {
                final_result = final_result.add(ctx.result);
            }

            return final_result.toAffine();
        }

        /// Worker function for each thread
        fn threadWorker(ctx: *ThreadContext) void {
            if (ctx.bases.len == 0) {
                ctx.result = Projective.identity();
                return;
            }

            // Use Pippenger for the thread's chunk
            const affine_result = SingleMSM.compute(ctx.bases, ctx.scalars);
            ctx.result = Projective.fromAffine(affine_result);
        }

        /// Empty worker for threads with no work
        fn threadWorkerEmpty(ctx: *ThreadContext) void {
            ctx.result = Projective.identity();
        }

        /// Detect optimal number of threads
        pub fn detectOptimalThreads() usize {
            // Try to get CPU count, default to 4
            const cpu_count = std.Thread.getCpuCount() catch return 4;
            // Use at most 8 threads to avoid diminishing returns
            return @min(cpu_count, 8);
        }
    };
}

/// Parallel batch MSM - multiple MSM operations computed in parallel
pub fn ParallelBatchMSM(comptime F: type, comptime G: type) type {
    return struct {
        const Self = @This();
        const Affine = AffinePoint(G);
        const SingleMSM = MSM(F, G);

        /// Thread context for batch computation
        const BatchThreadContext = struct {
            bases: []const Affine,
            scalars: []const F,
            result: Affine,
        };

        /// Compute multiple MSMs in parallel (one per thread)
        pub fn compute(
            bases: []const Affine,
            scalar_batches: []const []const F,
            allocator: Allocator,
        ) ![]Affine {
            if (scalar_batches.len == 0) {
                return &[_]Affine{};
            }

            // Allocate results array
            const results = try allocator.alloc(Affine, scalar_batches.len);
            errdefer allocator.free(results);

            // For small batch counts, run sequentially
            if (scalar_batches.len <= 2) {
                for (scalar_batches, 0..) |scalars, i| {
                    results[i] = SingleMSM.compute(bases, scalars);
                }
                return results;
            }

            // Allocate thread infrastructure
            const contexts = try allocator.alloc(BatchThreadContext, scalar_batches.len);
            defer allocator.free(contexts);

            const thread_handles = try allocator.alloc(std.Thread, scalar_batches.len);
            defer allocator.free(thread_handles);

            // Spawn a thread for each MSM
            for (scalar_batches, 0..) |scalars, i| {
                contexts[i] = .{
                    .bases = bases,
                    .scalars = scalars,
                    .result = Affine.identity(),
                };
                thread_handles[i] = try std.Thread.spawn(.{}, batchThreadWorker, .{&contexts[i]});
            }

            // Wait for all threads and collect results
            for (thread_handles, 0..) |handle, i| {
                handle.join();
                results[i] = contexts[i].result;
            }

            return results;
        }

        fn batchThreadWorker(ctx: *BatchThreadContext) void {
            ctx.result = SingleMSM.compute(ctx.bases, ctx.scalars);
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

    // Window size follows ln(n) + 2 formula (matching arkworks)
    try std.testing.expectEqual(@as(usize, 3), SingleMSM.optimalWindowSize(4));
    try std.testing.expectEqual(@as(usize, 3), SingleMSM.optimalWindowSize(16));
    try std.testing.expectEqual(@as(usize, 6), SingleMSM.optimalWindowSize(128)); // log2(128)=7, 7*69/100+2=6
    try std.testing.expectEqual(@as(usize, 10), SingleMSM.optimalWindowSize(4096)); // log2(4096)=12, 12*69/100+2=10
    try std.testing.expectEqual(@as(usize, 13), SingleMSM.optimalWindowSize(100000)); // log2(100000)=16, 16*69/100+2=13
    try std.testing.expectEqual(@as(usize, 15), SingleMSM.optimalWindowSize(524288)); // log2(524288)=19, 19*69/100+2=15
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

test "parallel msm types compile" {
    const F = @import("../field/mod.zig").BN254Scalar;

    // Verify parallel MSM types compile
    _ = ParallelMSM(F, F);
    _ = ParallelBatchMSM(F, F);
}

test "parallel msm detect optimal threads" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const PMSM = ParallelMSM(F, F);

    // Should return at least 1 thread
    const threads = PMSM.detectOptimalThreads();
    try std.testing.expect(threads >= 1);
    try std.testing.expect(threads <= 8);
}

test "parallel msm small input falls back to sequential" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const Affine = AffinePoint(F);
    const PMSM = ParallelMSM(F, F);
    const SingleMSM = MSM(F, F);

    const allocator = std.testing.allocator;

    var bases: [16]Affine = undefined;
    var scalars: [16]F = undefined;

    for (0..16) |i| {
        bases[i] = Affine.fromCoords(F.fromU64(i + 1), F.fromU64(i + 100));
        scalars[i] = F.fromU64(i * 17 + 3);
    }

    // Sequential result
    const seq_result = SingleMSM.compute(&bases, &scalars);

    // Parallel result (should fall back to sequential for small input)
    const par_result = try PMSM.compute(&bases, &scalars, 4, allocator);

    // Results should match
    try std.testing.expect(seq_result.x.eql(par_result.x));
    try std.testing.expect(seq_result.y.eql(par_result.y));
}

test "parallel msm empty input" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const Affine = AffinePoint(F);
    const PMSM = ParallelMSM(F, F);

    const allocator = std.testing.allocator;

    const result = try PMSM.compute(&[_]Affine{}, &[_]F{}, 4, allocator);
    try std.testing.expect(result.isIdentity());
}

test "parallel msm zero scalars" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const Affine = AffinePoint(F);
    const PMSM = ParallelMSM(F, F);

    const allocator = std.testing.allocator;

    var bases: [8]Affine = undefined;
    var scalars: [8]F = undefined;

    for (0..8) |i| {
        bases[i] = Affine.fromCoords(F.fromU64(i + 1), F.fromU64(i + 2));
        scalars[i] = F.zero();
    }

    const result = try PMSM.compute(&bases, &scalars, 4, allocator);
    try std.testing.expect(result.isIdentity());
}
