//! Generic extension field factories.
//!
//! `Fp2(BaseFp)` — quadratic extension Fp[u] / (u² + 1).
//!
//! Both BN254 and BLS12-381 use non-residue β = -1 for the quadratic
//! extension, so the generic factory is specialized for that case.
//! A future version can parameterize over β if needed for other curves.

/// Build a quadratic extension field Fp[u] / (u² + 1) over a base
/// field `BaseFp`. The base field must support: `zero, one, add, sub,
/// neg, mul, square, inverse, eql, isZero`.
pub fn Fp2(comptime BaseFp: type) type {
    return struct {
        c0: BaseFp,
        c1: BaseFp,

        const Self = @This();

        pub fn init(c0: BaseFp, c1: BaseFp) Self {
            return .{ .c0 = c0, .c1 = c1 };
        }

        pub fn zero() Self {
            return .{ .c0 = BaseFp.zero(), .c1 = BaseFp.zero() };
        }

        pub fn one() Self {
            return .{ .c0 = BaseFp.one(), .c1 = BaseFp.zero() };
        }

        pub fn eql(a: Self, b: Self) bool {
            return BaseFp.eql(a.c0, b.c0) and BaseFp.eql(a.c1, b.c1);
        }

        pub fn isZero(self: Self) bool {
            return self.c0.isZero() and self.c1.isZero();
        }

        pub fn add(a: Self, b: Self) Self {
            return .{
                .c0 = BaseFp.add(a.c0, b.c0),
                .c1 = BaseFp.add(a.c1, b.c1),
            };
        }

        pub fn sub(a: Self, b: Self) Self {
            return .{
                .c0 = BaseFp.sub(a.c0, b.c0),
                .c1 = BaseFp.sub(a.c1, b.c1),
            };
        }

        pub fn neg(a: Self) Self {
            return .{ .c0 = BaseFp.neg(a.c0), .c1 = BaseFp.neg(a.c1) };
        }

        /// Whether the base field exposes sumOfProducts (fused mul-accumulate).
        const has_sum_of_products = @hasDecl(BaseFp, "sumOfProducts");
        /// Whether the base field exposes addNoReduce (lazy reduction).
        const has_add_no_reduce = @hasDecl(BaseFp, "addNoReduce");

        /// Multiplication: (a₀ + a₁u)(b₀ + b₁u) = (a₀b₀ - a₁b₁) + (a₀b₁ + a₁b₀)u
        /// When BaseFp has sumOfProducts, uses fused path (2 reductions
        /// instead of 3). Otherwise falls back to Karatsuba.
        pub fn mul(a: Self, b: Self) Self {
            if (comptime has_sum_of_products) {
                const neg_a1 = BaseFp.neg(a.c1);
                return .{
                    .c0 = BaseFp.sumOfProducts(.{ a.c0, neg_a1 }, .{ b.c0, b.c1 }),
                    .c1 = BaseFp.sumOfProducts(.{ a.c0, a.c1 }, .{ b.c1, b.c0 }),
                };
            }
            const a0b0 = BaseFp.mul(a.c0, b.c0);
            const a1b1 = BaseFp.mul(a.c1, b.c1);
            const a0_plus_a1 = BaseFp.add(a.c0, a.c1);
            const b0_plus_b1 = BaseFp.add(b.c0, b.c1);
            const cross = BaseFp.mul(a0_plus_a1, b0_plus_b1);
            return .{
                .c0 = BaseFp.sub(a0b0, a1b1),
                .c1 = BaseFp.sub(BaseFp.sub(cross, a0b0), a1b1),
            };
        }

        /// Squaring: (a + bu)² = (a²-b²) + 2ab·u
        /// Uses addNoReduce for (a+b) when available (saves 1 reduction).
        pub fn square(a: Self) Self {
            const a_plus_b = if (comptime has_add_no_reduce)
                BaseFp.addNoReduce(a.c0, a.c1)
            else
                BaseFp.add(a.c0, a.c1);
            const a_minus_b = BaseFp.sub(a.c0, a.c1);
            const c0 = BaseFp.mul(a_plus_b, a_minus_b);
            const ab = BaseFp.mul(a.c0, a.c1);
            return .{ .c0 = c0, .c1 = BaseFp.add(ab, ab) };
        }

        /// Conjugate: a + bu → a - bu
        pub fn conjugate(a: Self) Self {
            return .{ .c0 = a.c0, .c1 = BaseFp.neg(a.c1) };
        }

        /// Inverse: (a + bu)⁻¹ = (a - bu) / (a² + b²)
        pub fn inverse(a: Self) ?Self {
            if (a.isZero()) return null;
            const norm = BaseFp.add(BaseFp.square(a.c0), BaseFp.square(a.c1));
            const norm_inv = norm.inverse() orelse return null;
            return .{
                .c0 = BaseFp.mul(a.c0, norm_inv),
                .c1 = BaseFp.mul(BaseFp.neg(a.c1), norm_inv),
            };
        }

        /// Non-optional inverse (returns zero for zero).
        pub fn inv(a: Self) Self {
            return a.inverse() orelse zero();
        }

        /// Scalar multiplication by a base field element.
        pub fn scalarMul(a: Self, s: BaseFp) Self {
            return .{
                .c0 = BaseFp.mul(a.c0, s),
                .c1 = BaseFp.mul(a.c1, s),
            };
        }

        /// Multiply by the non-residue u: (a + bu) * u = -b + au
        pub fn mulByNonResidue(a: Self) Self {
            return .{ .c0 = BaseFp.neg(a.c1), .c1 = a.c0 };
        }

        /// Double
        pub fn double(a: Self) Self {
            return .{
                .c0 = BaseFp.add(a.c0, a.c0),
                .c1 = BaseFp.add(a.c1, a.c1),
            };
        }
    };
}

/// Build a cubic extension field Fp2[v] / (v³ − ξ) where ξ is the
/// cubic non-residue in Fp2. The `mulByXiFn` parameter is a comptime
/// function that multiplies an Fp2 element by ξ.
///
/// BN254: ξ = 9 + u  → shift-add (4 Fp additions per component)
/// BLS12-381: ξ = 1 + u → (c0-c1) + (c0+c1)·u
pub fn Fp6(comptime Fp2Type: type, comptime mulByXiFn: fn (Fp2Type) Fp2Type) type {
    return struct {
        c0: Fp2Type,
        c1: Fp2Type,
        c2: Fp2Type,

        const Self = @This();

        pub fn zero() Self {
            return .{ .c0 = Fp2Type.zero(), .c1 = Fp2Type.zero(), .c2 = Fp2Type.zero() };
        }

        pub fn one() Self {
            return .{ .c0 = Fp2Type.one(), .c1 = Fp2Type.zero(), .c2 = Fp2Type.zero() };
        }

        pub fn eql(a: Self, b: Self) bool {
            return Fp2Type.eql(a.c0, b.c0) and Fp2Type.eql(a.c1, b.c1) and Fp2Type.eql(a.c2, b.c2);
        }

        pub fn add(a: Self, b: Self) Self {
            return .{
                .c0 = Fp2Type.add(a.c0, b.c0),
                .c1 = Fp2Type.add(a.c1, b.c1),
                .c2 = Fp2Type.add(a.c2, b.c2),
            };
        }

        pub fn sub(a: Self, b: Self) Self {
            return .{
                .c0 = Fp2Type.sub(a.c0, b.c0),
                .c1 = Fp2Type.sub(a.c1, b.c1),
                .c2 = Fp2Type.sub(a.c2, b.c2),
            };
        }

        pub fn neg(a: Self) Self {
            return .{ .c0 = Fp2Type.neg(a.c0), .c1 = Fp2Type.neg(a.c1), .c2 = Fp2Type.neg(a.c2) };
        }

        /// Multiply an Fp2 element by the cubic non-residue ξ.
        pub fn mulByXi(x: Fp2Type) Fp2Type {
            return mulByXiFn(x);
        }

        /// Karatsuba multiplication in Fp6.
        pub fn mul(a: Self, b: Self) Self {
            const v0 = Fp2Type.mul(a.c0, b.c0);
            const v1 = Fp2Type.mul(a.c1, b.c1);
            const v2 = Fp2Type.mul(a.c2, b.c2);

            const c1_plus_c2 = Fp2Type.add(a.c1, a.c2);
            const d1_plus_d2 = Fp2Type.add(b.c1, b.c2);
            const t0 = mulByXiFn(Fp2Type.sub(Fp2Type.sub(Fp2Type.mul(c1_plus_c2, d1_plus_d2), v1), v2));
            const new_c0 = Fp2Type.add(v0, t0);

            const c0_plus_c1 = Fp2Type.add(a.c0, a.c1);
            const d0_plus_d1 = Fp2Type.add(b.c0, b.c1);
            const t1 = Fp2Type.sub(Fp2Type.sub(Fp2Type.mul(c0_plus_c1, d0_plus_d1), v0), v1);
            const new_c1 = Fp2Type.add(t1, mulByXiFn(v2));

            const c0_plus_c2 = Fp2Type.add(a.c0, a.c2);
            const d0_plus_d2 = Fp2Type.add(b.c0, b.c2);
            const t2 = Fp2Type.sub(Fp2Type.sub(Fp2Type.mul(c0_plus_c2, d0_plus_d2), v0), v2);
            const new_c2 = Fp2Type.add(t2, v1);

            return .{ .c0 = new_c0, .c1 = new_c1, .c2 = new_c2 };
        }

        /// Chung-Hasan SQ2 squaring.
        pub fn square(a: Self) Self {
            const s0 = Fp2Type.square(a.c0);
            const ab = Fp2Type.mul(a.c0, a.c1);
            const s1 = Fp2Type.add(ab, ab);
            const s2 = Fp2Type.square(Fp2Type.add(Fp2Type.sub(a.c0, a.c1), a.c2));
            const bc = Fp2Type.mul(a.c1, a.c2);
            const s3 = Fp2Type.add(bc, bc);
            const s4 = Fp2Type.square(a.c2);

            return .{
                .c0 = Fp2Type.add(s0, mulByXiFn(s3)),
                .c1 = Fp2Type.add(s1, mulByXiFn(s4)),
                .c2 = Fp2Type.sub(Fp2Type.add(Fp2Type.add(s1, s2), s3), Fp2Type.add(s0, s4)),
            };
        }

        pub fn inverse(a: Self) ?Self {
            const c0_sq = Fp2Type.square(a.c0);
            const c1_sq = Fp2Type.square(a.c1);
            const c2_sq = Fp2Type.square(a.c2);
            const c0c1 = Fp2Type.mul(a.c0, a.c1);
            const c0c2 = Fp2Type.mul(a.c0, a.c2);
            const c1c2 = Fp2Type.mul(a.c1, a.c2);

            const a0 = Fp2Type.sub(c0_sq, mulByXiFn(c1c2));
            const a1 = Fp2Type.sub(mulByXiFn(c2_sq), c0c1);
            const a2 = Fp2Type.sub(c1_sq, c0c2);

            const tmp = mulByXiFn(Fp2Type.add(Fp2Type.mul(a.c1, a2), Fp2Type.mul(a.c2, a1)));
            const norm = Fp2Type.add(Fp2Type.mul(a.c0, a0), tmp);

            const norm_inv = norm.inverse() orelse return null;

            return .{
                .c0 = Fp2Type.mul(a0, norm_inv),
                .c1 = Fp2Type.mul(a1, norm_inv),
                .c2 = Fp2Type.mul(a2, norm_inv),
            };
        }

        /// Multiply by v (coefficient shift): (c0 + c1·v + c2·v²) · v = ξ·c2 + c0·v + c1·v²
        pub fn mulByV(f: Self) Self {
            return .{
                .c0 = mulByXiFn(f.c2),
                .c1 = f.c0,
                .c2 = f.c1,
            };
        }
    };
}

/// Build a quadratic extension Fp6[w] / (w² − v).
/// `Fp6MulByV` is used for w² = v → coefficient shift in Fp6.
pub fn Fp12(comptime Fp6Type: type, comptime Fp2Type: type) type {
    return struct {
        c0: Fp6Type,
        c1: Fp6Type,

        const Self = @This();

        pub fn zero() Self {
            return .{ .c0 = Fp6Type.zero(), .c1 = Fp6Type.zero() };
        }

        pub fn one() Self {
            return .{ .c0 = Fp6Type.one(), .c1 = Fp6Type.zero() };
        }

        pub fn eql(a: Self, b: Self) bool {
            return Fp6Type.eql(a.c0, b.c0) and Fp6Type.eql(a.c1, b.c1);
        }

        pub fn isOne(self: Self) bool {
            return self.eql(Self.one());
        }

        pub fn add(a: Self, b: Self) Self {
            return .{ .c0 = Fp6Type.add(a.c0, b.c0), .c1 = Fp6Type.add(a.c1, b.c1) };
        }

        pub fn sub(a: Self, b: Self) Self {
            return .{ .c0 = Fp6Type.sub(a.c0, b.c0), .c1 = Fp6Type.sub(a.c1, b.c1) };
        }

        pub fn neg(a: Self) Self {
            return .{ .c0 = Fp6Type.neg(a.c0), .c1 = Fp6Type.neg(a.c1) };
        }

        /// Karatsuba: (a + bw)(c + dw) = (ac + bd·v) + ((a+b)(c+d) - ac - bd)w
        pub fn mul(a: Self, b: Self) Self {
            const ac = Fp6Type.mul(a.c0, b.c0);
            const bd = Fp6Type.mul(a.c1, b.c1);
            const cross = Fp6Type.sub(Fp6Type.sub(
                Fp6Type.mul(Fp6Type.add(a.c0, a.c1), Fp6Type.add(b.c0, b.c1)),
                ac,
            ), bd);
            return .{
                .c0 = Fp6Type.add(ac, Fp6Type.mulByV(bd)),
                .c1 = cross,
            };
        }

        /// Complex squaring.
        pub fn square(a: Self) Self {
            const ab = Fp6Type.mul(a.c0, a.c1);
            const c1_new = Fp6Type.add(ab, ab);

            const a_plus_b = Fp6Type.add(a.c0, a.c1);
            const bv = Fp6Type.mulByV(a.c1);
            const a_plus_bv = Fp6Type.add(a.c0, bv);
            const abv = Fp6Type.mulByV(ab);
            const c0_new = Fp6Type.sub(Fp6Type.sub(Fp6Type.mul(a_plus_b, a_plus_bv), ab), abv);

            return .{ .c0 = c0_new, .c1 = c1_new };
        }

        /// Conjugate: a + bw → a - bw
        pub fn conjugate(a: Self) Self {
            return .{ .c0 = a.c0, .c1 = Fp6Type.neg(a.c1) };
        }

        pub fn inverse(a: Self) ?Self {
            const a_squared = Fp6Type.mul(a.c0, a.c0);
            const b_squared_v = Fp6Type.mulByV(Fp6Type.mul(a.c1, a.c1));
            const norm = Fp6Type.sub(a_squared, b_squared_v);
            const norm_inv = norm.inverse() orelse return null;
            return .{
                .c0 = Fp6Type.mul(a.c0, norm_inv),
                .c1 = Fp6Type.mul(Fp6Type.neg(a.c1), norm_inv),
            };
        }

        /// Cyclotomic squaring (Granger-Scott) for unitary Fp12.
        pub fn cyclotomicSquare(a: Self) Self {
            const r0 = a.c0.c0;
            const r4 = a.c0.c1;
            const r3 = a.c0.c2;
            const r2 = a.c1.c0;
            const r1 = a.c1.c1;
            const r5 = a.c1.c2;

            const xi = Fp6Type.mulByXi;

            const tmp01 = Fp2Type.mul(r0, r1);
            const t0 = Fp2Type.sub(Fp2Type.sub(Fp2Type.mul(Fp2Type.add(r0, r1), Fp2Type.add(xi(r1), r0)), tmp01), xi(tmp01));
            const t1 = Fp2Type.add(tmp01, tmp01);

            const tmp23 = Fp2Type.mul(r2, r3);
            const t2 = Fp2Type.sub(Fp2Type.sub(Fp2Type.mul(Fp2Type.add(r2, r3), Fp2Type.add(xi(r3), r2)), tmp23), xi(tmp23));
            const t3 = Fp2Type.add(tmp23, tmp23);

            const tmp45 = Fp2Type.mul(r4, r5);
            const t4 = Fp2Type.sub(Fp2Type.sub(Fp2Type.mul(Fp2Type.add(r4, r5), Fp2Type.add(xi(r5), r4)), tmp45), xi(tmp45));
            const t5 = Fp2Type.add(tmp45, tmp45);

            const z0 = Fp2Type.add(Fp2Type.add(Fp2Type.sub(t0, r0), Fp2Type.sub(t0, r0)), t0);
            const z1 = Fp2Type.add(Fp2Type.add(Fp2Type.add(t1, r1), Fp2Type.add(t1, r1)), t1);
            const xi_t5 = xi(t5);
            const z2 = Fp2Type.add(Fp2Type.add(Fp2Type.add(xi_t5, r2), Fp2Type.add(xi_t5, r2)), xi_t5);
            const z3 = Fp2Type.add(Fp2Type.add(Fp2Type.sub(t4, r3), Fp2Type.sub(t4, r3)), t4);
            const z4 = Fp2Type.add(Fp2Type.add(Fp2Type.sub(t2, r4), Fp2Type.sub(t2, r4)), t2);
            const z5 = Fp2Type.add(Fp2Type.add(Fp2Type.add(t3, r5), Fp2Type.add(t3, r5)), t3);

            return .{
                .c0 = .{ .c0 = z0, .c1 = z4, .c2 = z3 },
                .c1 = .{ .c0 = z2, .c1 = z1, .c2 = z5 },
            };
        }
    };
}

// =========================================================================
// Tests
// =========================================================================

const testing = @import("std").testing;
const bn254 = @import("bn254/mod.zig");
const BN254Fp2 = Fp2(bn254.Fp);

test "Generic Fp2(BN254): zero + one" {
    const z = BN254Fp2.zero();
    const o = BN254Fp2.one();
    try testing.expect(BN254Fp2.eql(BN254Fp2.add(z, o), o));
}

test "Generic Fp2(BN254): mul commutativity" {
    const a = BN254Fp2.init(bn254.Fp.fromU64(3), bn254.Fp.fromU64(7));
    const b = BN254Fp2.init(bn254.Fp.fromU64(11), bn254.Fp.fromU64(13));
    try testing.expect(BN254Fp2.eql(BN254Fp2.mul(a, b), BN254Fp2.mul(b, a)));
}

test "Generic Fp2(BN254): a * a^-1 = 1" {
    const a = BN254Fp2.init(bn254.Fp.fromU64(42), bn254.Fp.fromU64(99));
    const a_inv = BN254Fp2.inverse(a) orelse return error.SkipZigTest;
    try testing.expect(BN254Fp2.eql(BN254Fp2.mul(a, a_inv), BN254Fp2.one()));
}

test "Generic Fp2(BN254): square matches mul(a,a)" {
    const a = BN254Fp2.init(bn254.Fp.fromU64(7), bn254.Fp.fromU64(11));
    try testing.expect(BN254Fp2.eql(BN254Fp2.square(a), BN254Fp2.mul(a, a)));
}

test "Generic Fp2(BN254): conjugate identity a * conj(a) = norm" {
    const a = BN254Fp2.init(bn254.Fp.fromU64(3), bn254.Fp.fromU64(5));
    const prod = BN254Fp2.mul(a, BN254Fp2.conjugate(a));
    // norm = a₀² + a₁² (real number — imaginary part is zero)
    try testing.expect(prod.c1.isZero());
}

// -- Fp6 / Fp12 tests (BN254) --

/// BN254's cubic non-residue ξ = 9 + u. Multiply Fp2 by 9+u using shift-add.
fn bn254MulByXi(x: BN254Fp2) BN254Fp2 {
    const a = x.c0;
    const a2 = bn254.Fp.add(a, a);
    const a4 = bn254.Fp.add(a2, a2);
    const a8 = bn254.Fp.add(a4, a4);
    const a9 = bn254.Fp.add(a8, a);
    const b = x.c1;
    const b2 = bn254.Fp.add(b, b);
    const b4 = bn254.Fp.add(b2, b2);
    const b8 = bn254.Fp.add(b4, b4);
    const b9 = bn254.Fp.add(b8, b);
    return BN254Fp2.init(bn254.Fp.sub(a9, b), bn254.Fp.add(a, b9));
}

const BN254Fp6 = Fp6(BN254Fp2, bn254MulByXi);
const BN254Fp12 = Fp12(BN254Fp6, BN254Fp2);

test "Generic Fp6(BN254): one is multiplicative identity" {
    const a = BN254Fp6{ .c0 = BN254Fp2.init(bn254.Fp.fromU64(3), bn254.Fp.fromU64(7)), .c1 = BN254Fp2.init(bn254.Fp.fromU64(11), bn254.Fp.fromU64(0)), .c2 = BN254Fp2.init(bn254.Fp.fromU64(5), bn254.Fp.fromU64(2)) };
    try testing.expect(BN254Fp6.eql(BN254Fp6.mul(a, BN254Fp6.one()), a));
}

test "Generic Fp6(BN254): square matches mul(a,a)" {
    const a = BN254Fp6{ .c0 = BN254Fp2.init(bn254.Fp.fromU64(3), bn254.Fp.fromU64(7)), .c1 = BN254Fp2.init(bn254.Fp.fromU64(11), bn254.Fp.fromU64(13)), .c2 = BN254Fp2.init(bn254.Fp.fromU64(5), bn254.Fp.fromU64(2)) };
    try testing.expect(BN254Fp6.eql(BN254Fp6.square(a), BN254Fp6.mul(a, a)));
}

test "Generic Fp6(BN254): a * a^-1 = 1" {
    const a = BN254Fp6{ .c0 = BN254Fp2.init(bn254.Fp.fromU64(3), bn254.Fp.fromU64(7)), .c1 = BN254Fp2.init(bn254.Fp.fromU64(11), bn254.Fp.fromU64(13)), .c2 = BN254Fp2.init(bn254.Fp.fromU64(5), bn254.Fp.fromU64(2)) };
    const a_inv = BN254Fp6.inverse(a) orelse return error.SkipZigTest;
    try testing.expect(BN254Fp6.eql(BN254Fp6.mul(a, a_inv), BN254Fp6.one()));
}

test "Generic Fp12(BN254): one is multiplicative identity" {
    const fp6_a = BN254Fp6{ .c0 = BN254Fp2.init(bn254.Fp.fromU64(3), bn254.Fp.fromU64(7)), .c1 = BN254Fp2.init(bn254.Fp.fromU64(11), bn254.Fp.fromU64(0)), .c2 = BN254Fp2.init(bn254.Fp.fromU64(5), bn254.Fp.fromU64(2)) };
    const a = BN254Fp12{ .c0 = fp6_a, .c1 = BN254Fp6.one() };
    try testing.expect(BN254Fp12.eql(BN254Fp12.mul(a, BN254Fp12.one()), a));
}

test "Generic Fp12(BN254): square matches mul(a,a)" {
    const fp6_a = BN254Fp6{ .c0 = BN254Fp2.init(bn254.Fp.fromU64(3), bn254.Fp.fromU64(7)), .c1 = BN254Fp2.init(bn254.Fp.fromU64(11), bn254.Fp.fromU64(13)), .c2 = BN254Fp2.init(bn254.Fp.fromU64(5), bn254.Fp.fromU64(2)) };
    const fp6_b = BN254Fp6{ .c0 = BN254Fp2.init(bn254.Fp.fromU64(17), bn254.Fp.fromU64(19)), .c1 = BN254Fp2.zero(), .c2 = BN254Fp2.init(bn254.Fp.fromU64(23), bn254.Fp.fromU64(0)) };
    const a = BN254Fp12{ .c0 = fp6_a, .c1 = fp6_b };
    try testing.expect(BN254Fp12.eql(BN254Fp12.square(a), BN254Fp12.mul(a, a)));
}

test "Generic Fp12(BN254): a * a^-1 = 1" {
    const fp6_a = BN254Fp6{ .c0 = BN254Fp2.init(bn254.Fp.fromU64(3), bn254.Fp.fromU64(7)), .c1 = BN254Fp2.init(bn254.Fp.fromU64(11), bn254.Fp.fromU64(13)), .c2 = BN254Fp2.init(bn254.Fp.fromU64(5), bn254.Fp.fromU64(2)) };
    const fp6_b = BN254Fp6{ .c0 = BN254Fp2.init(bn254.Fp.fromU64(17), bn254.Fp.fromU64(19)), .c1 = BN254Fp2.init(bn254.Fp.fromU64(1), bn254.Fp.fromU64(0)), .c2 = BN254Fp2.init(bn254.Fp.fromU64(23), bn254.Fp.fromU64(29)) };
    const a = BN254Fp12{ .c0 = fp6_a, .c1 = fp6_b };
    const a_inv = BN254Fp12.inverse(a) orelse return error.SkipZigTest;
    try testing.expect(BN254Fp12.eql(BN254Fp12.mul(a, a_inv), BN254Fp12.one()));
}

// Cross-validate Fp6 against OG
test "Generic Fp6(BN254): mul matches OG Fp6.mul" {
    const OGFp6 = @import("../field/extensions.zig").Fp6;
    const OGFp2 = @import("../field/extensions.zig").Fp2;
    const OGFp = @import("../field/mod.zig").BN254BaseField;

    const og_a = OGFp6{ .c0 = OGFp2.init(OGFp.fromU64(3), OGFp.fromU64(7)), .c1 = OGFp2.init(OGFp.fromU64(11), OGFp.fromU64(13)), .c2 = OGFp2.init(OGFp.fromU64(5), OGFp.fromU64(2)) };
    const og_b = OGFp6{ .c0 = OGFp2.init(OGFp.fromU64(17), OGFp.fromU64(19)), .c1 = OGFp2.init(OGFp.fromU64(23), OGFp.fromU64(29)), .c2 = OGFp2.init(OGFp.fromU64(31), OGFp.fromU64(37)) };
    const og_result = og_a.mul(og_b);

    const gen_a = BN254Fp6{ .c0 = BN254Fp2.init(bn254.Fp.fromU64(3), bn254.Fp.fromU64(7)), .c1 = BN254Fp2.init(bn254.Fp.fromU64(11), bn254.Fp.fromU64(13)), .c2 = BN254Fp2.init(bn254.Fp.fromU64(5), bn254.Fp.fromU64(2)) };
    const gen_b = BN254Fp6{ .c0 = BN254Fp2.init(bn254.Fp.fromU64(17), bn254.Fp.fromU64(19)), .c1 = BN254Fp2.init(bn254.Fp.fromU64(23), bn254.Fp.fromU64(29)), .c2 = BN254Fp2.init(bn254.Fp.fromU64(31), bn254.Fp.fromU64(37)) };
    const gen_result = BN254Fp6.mul(gen_a, gen_b);

    try testing.expectEqual(gen_result.c0.c0.limbs, og_result.c0.c0.limbs);
    try testing.expectEqual(gen_result.c0.c1.limbs, og_result.c0.c1.limbs);
    try testing.expectEqual(gen_result.c1.c0.limbs, og_result.c1.c0.limbs);
    try testing.expectEqual(gen_result.c2.c0.limbs, og_result.c2.c0.limbs);
}

// Cross-validate against the OG BN254 Fp2
test "Generic Fp2(BN254): mul matches OG extensions.Fp2.mul" {
    const OGFp2 = @import("../field/extensions.zig").Fp2;
    const OGFp = @import("../field/mod.zig").BN254BaseField;

    const og_a = OGFp2.init(OGFp.fromU64(3), OGFp.fromU64(7));
    const og_b = OGFp2.init(OGFp.fromU64(11), OGFp.fromU64(13));
    const og_result = og_a.mul(og_b);

    const gen_a = BN254Fp2.init(bn254.Fp.fromU64(3), bn254.Fp.fromU64(7));
    const gen_b = BN254Fp2.init(bn254.Fp.fromU64(11), bn254.Fp.fromU64(13));
    const gen_result = BN254Fp2.mul(gen_a, gen_b);

    // BN254BaseField is now an alias for curves.bn254.Fp, so the limbs
    // should be directly comparable.
    try testing.expectEqual(gen_result.c0.limbs, og_result.c0.limbs);
    try testing.expectEqual(gen_result.c1.limbs, og_result.c1.limbs);
}
