//! BN254 optimal ate pairing backed by fiat-crypto verified field arithmetic.
//!
//! Same algorithms as field/pairing.zig but using FiatFp as the base field.
//! Extension fields (Fp2/Fp6/Fp12) are built compositionally on verified Fp.

const std = @import("std");
const fiat_ext = @import("extensions.zig");
const FiatFp = fiat_ext.FiatFp;
const Fp2 = fiat_ext.Fp2;
const Fp6 = fiat_ext.Fp6;
const Fp12 = fiat_ext.Fp12;

// ============================================================================
// Constants
// ============================================================================

const ATE_LOOP_COUNT: [65]i2 = .{
    0, 0,  0, 1,  0, 1, 0, -1, 0, 0,  -1, 0, 0,  0, 1,  0,
    0, -1, 0, -1, 0, 0, 0, 1,  0, -1, 0,  0, 0,  0, -1, 0,
    0, 1,  0, -1, 0, 0, 1, 0,  0, 0,  0,  0, -1, 0, 0,  -1,
    0, 1,  0, -1, 0, 0, 0, -1, 0, -1, 0,  0, 0,  1, 0,  1,
    1,
};

const X_IS_NEGATIVE: bool = false;
const BN_X: u64 = 4965661367192848881;

// ============================================================================
// Helper constructors
// ============================================================================

fn fp2FromLimbs(limbs: [8]u64) Fp2 {
    const c0 = (FiatFp{ .limbs = .{ limbs[0], limbs[1], limbs[2], limbs[3] } }).toMontgomery();
    const c1 = (FiatFp{ .limbs = .{ limbs[4], limbs[5], limbs[6], limbs[7] } }).toMontgomery();
    return Fp2.init(c0, c1);
}

fn fpFromLimbs(limbs: [4]u64) FiatFp {
    return (FiatFp{ .limbs = limbs }).toMontgomery();
}

fn fp2ScalarMul(a: Fp2, s: FiatFp) Fp2 {
    return Fp2.init(FiatFp.mul(a.c0, s), FiatFp.mul(a.c1, s));
}

// ============================================================================
// Frobenius coefficients
// ============================================================================

const FROBENIUS_GAMMA11: [8]u64 = .{ 0xD60B35DADCC9E470, 0x5C521E08292F2176, 0xE8B99FDD76E68B60, 0x1284B71C2865A7DF, 0xCA5CF05F80F362AC, 0x747992778EEEC7E5, 0xA6327CFE12150B8E, 0x246996F3B4FAE7E6 };
const FROBENIUS_GAMMA12: [8]u64 = .{ 0x99E39557176F553D, 0xB78CC310C2C3330C, 0x4C0BEC3CF559B143, 0x2FB347984F7911F7, 0x1665D51C640FCBA2, 0x32AE2A1D0B7C9DCE, 0x4BA4CC8BD75A0794, 0x16C9E55061EBAE20 };
const FROBENIUS_GAMMA13: [8]u64 = .{ 0xDC54014671A0135A, 0xDBAAE0EDA9C95998, 0xDC5EC698B6E2F9B9, 0x063CF305489AF5DC, 0x82D37F632623B0E3, 0x21807DC98FA25BD2, 0x0704B5A7EC796F2B, 0x07C03CBCAC41049A };
const FROBENIUS_GAMMA14: [8]u64 = .{ 0x848A1F55921EA762, 0xD33365F7BE94EC72, 0x80F3C0B75A181E84, 0x05B54F5E64EEA801, 0xC13B4711CD2B8126, 0x3685D2EA1BDEC763, 0x9F3A80B03B0B1C92, 0x2C145EDBE7FD8AEE };
const FROBENIUS_GAMMA15: [8]u64 = .{ 0x2EA2C810EAB7692F, 0x425C459B55AA1BD3, 0xE93A3661A4353FF4, 0x0183C1E74F798649, 0x24C6B8EE6E0C2C4B, 0xB080CB99678E2AC0, 0xA27FB246C7729F7D, 0x12ACF2CA76FD0675 };
const FROBENIUS_GAMMA21: [4]u64 = .{ 0xE4BD44E5607CFD49, 0xC28F069FBB966E3D, 0x5E6DD9E7E0ACCCB0, 0x30644E72E131A029 };
const FROBENIUS_GAMMA22: [4]u64 = .{ 0xE4BD44E5607CFD48, 0xC28F069FBB966E3D, 0x5E6DD9E7E0ACCCB0, 0x30644E72E131A029 };
const FROBENIUS_GAMMA23: [4]u64 = .{ 0x3C208C16D87CFD46, 0x97816A916871CA8D, 0xB85045B68181585D, 0x30644E72E131A029 };
const FROBENIUS_GAMMA24: [4]u64 = .{ 0x5763473177FFFFFE, 0xD4F263F1ACDB5C4F, 0x59E26BCEA0D48BAC, 0x0000000000000000 };
const FROBENIUS_GAMMA25: [4]u64 = .{ 0x5763473177FFFFFF, 0xD4F263F1ACDB5C4F, 0x59E26BCEA0D48BAC, 0x0000000000000000 };
const FROBENIUS_GAMMA31: [8]u64 = .{ 0xE86F7D391ED4A67F, 0x894CB38DBE55D24A, 0xEFE9608CD0ACAA90, 0x19DC81CFCC82E4BB, 0x7694AA2BF4C0C101, 0x7F03A5E397D439EC, 0x06CBEEE33576139D, 0x00ABF8B60BE77D73 };
const FROBENIUS_GAMMA32: [8]u64 = .{ 0x7B746EE87BDCFB6D, 0x805FFD3D5D6942D3, 0xBAFF1C77959F25AC, 0x0856E078B755EF0A, 0x380CAB2BAAA586DE, 0x0FDF31BF98FF2631, 0xA9F30E6DEC26094F, 0x04F1DE41B3D1766F };
const FROBENIUS_GAMMA33: [8]u64 = .{ 0x5FCC8AD066DCE9ED, 0xBBD689A3BEA870F4, 0xDBF17F1DCA9E5EA3, 0x2A275B6D9896AA4C, 0xB94D0CB3B2594C64, 0x7600ECC7D8CF6EBA, 0xB14B900E9507E932, 0x28A411B634F09B8F };
const FROBENIUS_GAMMA34: [8]u64 = .{ 0x0E1A92BC3CCBF066, 0xE633094575B06BCB, 0x19BEE0F7B5B2444E, 0x0BC58C6611C08DAB, 0x5FE3ED9D730C239F, 0xA44A9E08737F96E5, 0xFEB0F6EF0CD21D04, 0x23D5E999E1910A12 };
const FROBENIUS_GAMMA35: [8]u64 = .{ 0xEBDE847076261B43, 0x2ED68098967C84A5, 0x711699FA3B4D3F69, 0x13C49044952C0905, 0x1F25041384282499, 0x3E2DDAEA20028021, 0x9FB1B2282A48633D, 0x16DB366A59B1DD0B };

// ============================================================================
// Frobenius endomorphisms on Fp12
// ============================================================================

pub fn frobenius(self: Fp12) Fp12 {
    const g11 = fp2FromLimbs(FROBENIUS_GAMMA11);
    const g12 = fp2FromLimbs(FROBENIUS_GAMMA12);
    const g13 = fp2FromLimbs(FROBENIUS_GAMMA13);
    const g14 = fp2FromLimbs(FROBENIUS_GAMMA14);
    const g15 = fp2FromLimbs(FROBENIUS_GAMMA15);

    return Fp12{
        .c0 = Fp6{
            .c0 = self.c0.c0.conjugate(),
            .c1 = Fp2.mul(self.c0.c1.conjugate(), g12),
            .c2 = Fp2.mul(self.c0.c2.conjugate(), g14),
        },
        .c1 = Fp6{
            .c0 = Fp2.mul(self.c1.c0.conjugate(), g11),
            .c1 = Fp2.mul(self.c1.c1.conjugate(), g13),
            .c2 = Fp2.mul(self.c1.c2.conjugate(), g15),
        },
    };
}

pub fn frobenius2(self: Fp12) Fp12 {
    const g21 = fpFromLimbs(FROBENIUS_GAMMA21);
    const g22 = fpFromLimbs(FROBENIUS_GAMMA22);
    const g23 = fpFromLimbs(FROBENIUS_GAMMA23);
    const g24 = fpFromLimbs(FROBENIUS_GAMMA24);
    const g25 = fpFromLimbs(FROBENIUS_GAMMA25);

    return Fp12{
        .c0 = Fp6{
            .c0 = self.c0.c0,
            .c1 = fp2ScalarMul(self.c0.c1, g22),
            .c2 = fp2ScalarMul(self.c0.c2, g24),
        },
        .c1 = Fp6{
            .c0 = fp2ScalarMul(self.c1.c0, g21),
            .c1 = fp2ScalarMul(self.c1.c1, g23),
            .c2 = fp2ScalarMul(self.c1.c2, g25),
        },
    };
}

pub fn frobenius3(self: Fp12) Fp12 {
    const g31 = fp2FromLimbs(FROBENIUS_GAMMA31);
    const g32 = fp2FromLimbs(FROBENIUS_GAMMA32);
    const g33 = fp2FromLimbs(FROBENIUS_GAMMA33);
    const g34 = fp2FromLimbs(FROBENIUS_GAMMA34);
    const g35 = fp2FromLimbs(FROBENIUS_GAMMA35);

    return Fp12{
        .c0 = Fp6{
            .c0 = self.c0.c0.conjugate(),
            .c1 = Fp2.mul(self.c0.c1.conjugate(), g32),
            .c2 = Fp2.mul(self.c0.c2.conjugate(), g34),
        },
        .c1 = Fp6{
            .c0 = Fp2.mul(self.c1.c0.conjugate(), g31),
            .c1 = Fp2.mul(self.c1.c1.conjugate(), g33),
            .c2 = Fp2.mul(self.c1.c2.conjugate(), g35),
        },
    };
}

// ============================================================================
// Sparse Fp12 multiplication helpers
// ============================================================================

fn fp6MulBy01(f: Fp6, c0: Fp2, c1: Fp2) Fp6 {
    const a_a = Fp2.mul(f.c0, c0);
    const b_b = Fp2.mul(f.c1, c1);

    var t1 = Fp2.mul(c1, Fp2.add(f.c1, f.c2));
    t1 = Fp2.sub(t1, b_b);
    t1 = fiat_ext.fiatMulByXi(t1);
    t1 = Fp2.add(t1, a_a);

    var t3 = Fp2.mul(c0, Fp2.add(f.c0, f.c2));
    t3 = Fp2.sub(t3, a_a);
    t3 = Fp2.add(t3, b_b);

    var t2 = Fp2.mul(Fp2.add(c0, c1), Fp2.add(f.c0, f.c1));
    t2 = Fp2.sub(t2, a_a);
    t2 = Fp2.sub(t2, b_b);

    return Fp6{ .c0 = t1, .c1 = t2, .c2 = t3 };
}

fn fp12MulBy034(f: Fp12, c0: Fp2, c3: Fp2, c4: Fp2) Fp12 {
    const a = Fp6{
        .c0 = Fp2.mul(f.c0.c0, c0),
        .c1 = Fp2.mul(f.c0.c1, c0),
        .c2 = Fp2.mul(f.c0.c2, c0),
    };
    const b = fp6MulBy01(f.c1, c3, c4);

    const f_sum = Fp6{
        .c0 = Fp2.add(f.c0.c0, f.c1.c0),
        .c1 = Fp2.add(f.c0.c1, f.c1.c1),
        .c2 = Fp2.add(f.c0.c2, f.c1.c2),
    };
    const e = fp6MulBy01(f_sum, Fp2.add(c0, c3), c4);

    const c1_new = Fp6{
        .c0 = Fp2.sub(Fp2.sub(e.c0, a.c0), b.c0),
        .c1 = Fp2.sub(Fp2.sub(e.c1, a.c1), b.c1),
        .c2 = Fp2.sub(Fp2.sub(e.c2, a.c2), b.c2),
    };

    const bv = Fp6.mulByV(b);
    const c0_new = Fp6{
        .c0 = Fp2.add(a.c0, bv.c0),
        .c1 = Fp2.add(a.c1, bv.c1),
        .c2 = Fp2.add(a.c2, bv.c2),
    };

    return Fp12{ .c0 = c0_new, .c1 = c1_new };
}

// ============================================================================
// G2 Point types
// ============================================================================

pub const G2Point = struct {
    x: Fp2,
    y: Fp2,
    infinity: bool,

    pub fn identity() G2Point {
        return .{ .x = Fp2.zero(), .y = Fp2.one(), .infinity = true };
    }

    pub fn neg(self: G2Point) G2Point {
        if (self.infinity) return self;
        return .{ .x = self.x, .y = Fp2.neg(self.y), .infinity = false };
    }

    pub fn generator() G2Point {
        const x0 = FiatFp.fromBytes(&[_]u8{
            0xed, 0xf6, 0x92, 0xd9, 0x5c, 0xbd, 0xde, 0x46,
            0xdd, 0xda, 0x5e, 0xf7, 0xd4, 0x22, 0x43, 0x67,
            0x79, 0x44, 0x5c, 0x5e, 0x66, 0x00, 0x6a, 0x42,
            0x76, 0x1e, 0x1f, 0x12, 0xef, 0xde, 0x00, 0x18,
        });
        const x1 = FiatFp.fromBytes(&[_]u8{
            0xc2, 0x12, 0xf3, 0xae, 0xb7, 0x85, 0xe4, 0x97,
            0x12, 0xe7, 0xa9, 0x35, 0x33, 0x49, 0xaa, 0xf1,
            0x25, 0x5d, 0xfb, 0x31, 0xb7, 0xbf, 0x60, 0x72,
            0x3a, 0x48, 0x0d, 0x92, 0x93, 0x93, 0x8e, 0x19,
        });
        const y0 = FiatFp.fromBytes(&[_]u8{
            0xaa, 0x7d, 0xfa, 0x66, 0x01, 0xcc, 0xe6, 0x4c,
            0x7b, 0xd3, 0x43, 0x0c, 0x69, 0xe7, 0xd1, 0xe3,
            0x8f, 0x40, 0xcb, 0x8d, 0x80, 0x71, 0xab, 0x4a,
            0xeb, 0x6d, 0x8c, 0xdb, 0xa5, 0x5e, 0xc8, 0x12,
        });
        const y1 = FiatFp.fromBytes(&[_]u8{
            0x5b, 0x97, 0x22, 0xd1, 0xdc, 0xda, 0xac, 0x55,
            0xf3, 0x8e, 0xb3, 0x70, 0x33, 0x31, 0x4b, 0xbc,
            0x95, 0x33, 0x0c, 0x69, 0xad, 0x99, 0x9e, 0xec,
            0x75, 0xf0, 0x5f, 0x58, 0xd0, 0x89, 0x06, 0x09,
        });
        return .{ .x = Fp2.init(x0, x1), .y = Fp2.init(y0, y1), .infinity = false };
    }
};

/// G1 point (coordinates in Fp).
pub const G1Point = struct {
    x: FiatFp,
    y: FiatFp,
    infinity: bool,

    pub fn identity() G1Point {
        return .{ .x = FiatFp.zero(), .y = FiatFp.one(), .infinity = true };
    }

    pub fn generator() G1Point {
        return .{ .x = FiatFp.fromU64(1), .y = FiatFp.fromU64(2), .infinity = false };
    }

    pub fn neg(self: G1Point) G1Point {
        if (self.infinity) return self;
        return .{ .x = self.x, .y = FiatFp.neg(self.y), .infinity = false };
    }
};

const EllCoeff = struct { c0: Fp2, c1: Fp2, c2: Fp2 };

fn twistB() Fp2 {
    const b0 = (FiatFp{ .limbs = .{ 0x3267e6dc24a138e5, 0xb5b4c5e559dbefa3, 0x81be18991be06ac3, 0x2b149d40ceb8aaae } }).toMontgomery();
    const b1 = (FiatFp{ .limbs = .{ 0xe4a2bd0685c315d2, 0xa74fa084e52d1852, 0xcd2cafadeed8fdf4, 0x009713b03af0fed4 } }).toMontgomery();
    return Fp2.init(b0, b1);
}

fn twistMulByQX() Fp2 {
    const c0 = (FiatFp{ .limbs = .{ 0x99e39557176f553d, 0xb78cc310c2c3330c, 0x4c0bec3cf559b143, 0x2fb347984f7911f7 } }).toMontgomery();
    const c1 = (FiatFp{ .limbs = .{ 0x1665d51c640fcba2, 0x32ae2a1d0b7c9dce, 0x4ba4cc8bd75a0794, 0x16c9e55061ebae20 } }).toMontgomery();
    return Fp2.init(c0, c1);
}

fn twistMulByQY() Fp2 {
    const c0 = (FiatFp{ .limbs = .{ 0xdc54014671a0135a, 0xdbaae0eda9c95998, 0xdc5ec698b6e2f9b9, 0x063cf305489af5dc } }).toMontgomery();
    const c1 = (FiatFp{ .limbs = .{ 0x82d37f632623b0e3, 0x21807dc98fa25bd2, 0x0704b5a7ec796f2b, 0x07c03cbcac41049a } }).toMontgomery();
    return Fp2.init(c0, c1);
}

fn mulByChar(p: G2Point) G2Point {
    if (p.infinity) return p;
    return .{
        .x = Fp2.mul(p.x.conjugate(), twistMulByQX()),
        .y = Fp2.mul(p.y.conjugate(), twistMulByQY()),
        .infinity = false,
    };
}

// ============================================================================
// G2 projective (homogeneous) for Miller loop
// ============================================================================

const G2HomProjective = struct {
    x: Fp2,
    y: Fp2,
    z: Fp2,

    fn fromAffine(p: G2Point) G2HomProjective {
        if (p.infinity) return .{ .x = Fp2.zero(), .y = Fp2.one(), .z = Fp2.zero() };
        return .{ .x = p.x, .y = p.y, .z = Fp2.one() };
    }

    fn double_in_place(self: *G2HomProjective, two_inv: FiatFp) EllCoeff {
        var a = Fp2.mul(self.x, self.y);
        a = fp2ScalarMul(a, two_inv);

        const b = Fp2.square(self.y);
        const c = Fp2.square(self.z);
        const three_c = Fp2.add(Fp2.add(c, c), c);
        const e = Fp2.mul(three_c, twistB());
        const f = Fp2.add(Fp2.add(e, e), e);
        var g = Fp2.add(b, f);
        g = fp2ScalarMul(g, two_inv);
        const h = Fp2.sub(Fp2.square(Fp2.add(self.y, self.z)), Fp2.add(b, c));
        const i = Fp2.sub(e, b);
        const j = Fp2.square(self.x);
        const e_sq = Fp2.square(e);

        self.x = Fp2.mul(a, Fp2.sub(b, f));
        self.y = Fp2.sub(Fp2.square(g), Fp2.add(Fp2.add(e_sq, e_sq), e_sq));
        self.z = Fp2.mul(b, h);

        return .{
            .c0 = Fp2.neg(h),
            .c1 = Fp2.add(Fp2.add(j, j), j),
            .c2 = i,
        };
    }

    fn add_in_place(self: *G2HomProjective, q: G2Point) EllCoeff {
        const theta = Fp2.sub(self.y, Fp2.mul(q.y, self.z));
        const lambda = Fp2.sub(self.x, Fp2.mul(q.x, self.z));
        const c = Fp2.square(theta);
        const d = Fp2.square(lambda);
        const e = Fp2.mul(lambda, d);
        const f = Fp2.mul(self.z, c);
        const gg = Fp2.mul(self.x, d);
        const h = Fp2.sub(Fp2.add(e, f), Fp2.add(gg, gg));

        self.x = Fp2.mul(lambda, h);
        self.y = Fp2.sub(Fp2.mul(theta, Fp2.sub(gg, h)), Fp2.mul(e, self.y));
        self.z = Fp2.mul(self.z, e);

        return .{
            .c0 = lambda,
            .c1 = Fp2.neg(theta),
            .c2 = Fp2.sub(Fp2.mul(theta, q.x), Fp2.mul(lambda, q.y)),
        };
    }
};

// ============================================================================
// G1 scalar multiplication (double-and-add, for test purposes)
// ============================================================================

fn g1ScalarMul(p: G1Point, scalar: u64) G1Point {
    if (scalar == 0) return G1Point.identity();
    if (scalar == 1) return p;

    // Projective coordinates for doubling
    var rx = FiatFp.zero();
    var ry = FiatFp.one();
    var rz = FiatFp.zero();
    var px = p.x;
    var py = p.y;
    var pz = FiatFp.one();
    var s = scalar;

    while (s > 0) {
        if (s & 1 == 1) {
            // R = R + P (projective add)
            if (rz.isZero()) {
                rx = px;
                ry = py;
                rz = pz;
            } else {
                const uu1 = FiatFp.mul(ry, pz);
                const uu2 = FiatFp.mul(py, rz);
                const vv1 = FiatFp.mul(rx, pz);
                const vv2 = FiatFp.mul(px, rz);
                if (FiatFp.eql(vv1, vv2)) {
                    if (FiatFp.eql(uu1, uu2)) {
                        // Double
                        const xx = FiatFp.square(px);
                        const zz = FiatFp.square(pz);
                        const w = FiatFp.add(FiatFp.add(xx, xx), xx); // 3x^2 (a=0)
                        const s2 = FiatFp.mul(FiatFp.mul(py, pz), FiatFp.fromU64(2));
                        const ss = FiatFp.square(s2);
                        const sss = FiatFp.mul(s2, ss);
                        _ = zz;
                        const rr = FiatFp.mul(py, s2);
                        const rrr = FiatFp.square(rr);
                        const b2 = FiatFp.sub(FiatFp.square(FiatFp.add(px, rr)), FiatFp.add(xx, rrr));
                        const h2 = FiatFp.sub(FiatFp.square(w), FiatFp.add(b2, b2));
                        rx = FiatFp.mul(h2, s2);
                        ry = FiatFp.sub(FiatFp.mul(w, FiatFp.sub(b2, h2)), FiatFp.add(rrr, rrr));
                        rz = sss;
                    } else {
                        rx = FiatFp.zero();
                        ry = FiatFp.one();
                        rz = FiatFp.zero();
                    }
                } else {
                    const uu = FiatFp.sub(uu1, uu2);
                    const vv = FiatFp.sub(vv1, vv2);
                    const vvsq = FiatFp.square(vv);
                    const vv3 = FiatFp.mul(vv, vvsq);
                    const v1vv2 = FiatFp.mul(vv1, vvsq);
                    const a2 = FiatFp.sub(FiatFp.sub(FiatFp.mul(FiatFp.square(uu), FiatFp.mul(rz, pz)), vv3), FiatFp.add(v1vv2, v1vv2));
                    rx = FiatFp.mul(vv, a2);
                    ry = FiatFp.sub(FiatFp.mul(uu, FiatFp.sub(v1vv2, a2)), FiatFp.mul(vv3, uu1));
                    rz = FiatFp.mul(vv3, FiatFp.mul(rz, pz));
                }
            }
        }
        // P = 2P
        if (s > 1) {
            const xx = FiatFp.square(px);
            const w = FiatFp.add(FiatFp.add(xx, xx), xx); // 3x^2 (a=0)
            const s2 = FiatFp.mul(FiatFp.mul(py, pz), FiatFp.fromU64(2));
            const ss = FiatFp.square(s2);
            const sss = FiatFp.mul(s2, ss);
            const rr = FiatFp.mul(py, s2);
            const rrr = FiatFp.square(rr);
            const b2 = FiatFp.sub(FiatFp.square(FiatFp.add(px, rr)), FiatFp.add(xx, rrr));
            const h2 = FiatFp.sub(FiatFp.square(w), FiatFp.add(b2, b2));
            px = FiatFp.mul(h2, s2);
            py = FiatFp.sub(FiatFp.mul(w, FiatFp.sub(b2, h2)), FiatFp.add(rrr, rrr));
            pz = sss;
        }
        s >>= 1;
    }

    if (rz.isZero()) return G1Point.identity();
    const rz_inv = FiatFp.inverse(rz) orelse return G1Point.identity();
    return .{
        .x = FiatFp.mul(rx, rz_inv),
        .y = FiatFp.mul(ry, rz_inv),
        .infinity = false,
    };
}

fn g2ScalarMul(p: G2Point, scalar: u64) G2Point {
    if (scalar == 0) return G2Point.identity();
    if (scalar == 1) return p;

    // Simple double-and-add in affine (G2 scalars are small in test vectors)
    var result = G2Point.identity();
    var base = p;
    var s = scalar;
    while (s > 0) {
        if (s & 1 == 1) {
            result = g2AffineAdd(result, base);
        }
        if (s > 1) base = g2AffineDouble(base);
        s >>= 1;
    }
    return result;
}

fn g2AffineDouble(p: G2Point) G2Point {
    if (p.infinity) return p;
    // λ = 3x² / 2y (curve a=0)
    const xx = Fp2.square(p.x);
    const three_xx = Fp2.add(Fp2.add(xx, xx), xx);
    const two_y = Fp2.add(p.y, p.y);
    const lambda = Fp2.mul(three_xx, Fp2.inverse(two_y) orelse return G2Point.identity());
    const x3 = Fp2.sub(Fp2.sub(Fp2.square(lambda), p.x), p.x);
    const y3 = Fp2.sub(Fp2.mul(lambda, Fp2.sub(p.x, x3)), p.y);
    return .{ .x = x3, .y = y3, .infinity = false };
}

fn g2AffineAdd(p: G2Point, q: G2Point) G2Point {
    if (p.infinity) return q;
    if (q.infinity) return p;
    if (Fp2.eql(p.x, q.x)) {
        if (Fp2.eql(p.y, q.y)) return g2AffineDouble(p);
        return G2Point.identity();
    }
    const lambda = Fp2.mul(Fp2.sub(q.y, p.y), Fp2.inverse(Fp2.sub(q.x, p.x)) orelse return G2Point.identity());
    const x3 = Fp2.sub(Fp2.sub(Fp2.square(lambda), p.x), q.x);
    const y3 = Fp2.sub(Fp2.mul(lambda, Fp2.sub(p.x, x3)), p.y);
    return .{ .x = x3, .y = y3, .infinity = false };
}

// ============================================================================
// Miller loop
// ============================================================================

pub fn millerLoop(p: G1Point, q: G2Point) Fp12 {
    if (p.infinity or q.infinity) return Fp12.one();

    const two_inv = FiatFp.inverse(FiatFp.fromU64(2)) orelse return Fp12.one();
    var r = G2HomProjective.fromAffine(q);
    const neg_q = q.neg();
    var f = Fp12.one();

    var idx: usize = ATE_LOOP_COUNT.len - 1;
    while (idx >= 1) : (idx -= 1) {
        if (idx != ATE_LOOP_COUNT.len - 1) f = Fp12.square(f);

        const coeffs_dbl = r.double_in_place(two_inv);
        f = fp12MulBy034(f, fp2ScalarMul(coeffs_dbl.c0, p.y), fp2ScalarMul(coeffs_dbl.c1, p.x), coeffs_dbl.c2);

        const bit = ATE_LOOP_COUNT[idx - 1];
        if (bit == 1) {
            const coeffs_add = r.add_in_place(q);
            f = fp12MulBy034(f, fp2ScalarMul(coeffs_add.c0, p.y), fp2ScalarMul(coeffs_add.c1, p.x), coeffs_add.c2);
        } else if (bit == -1) {
            const coeffs_add = r.add_in_place(neg_q);
            f = fp12MulBy034(f, fp2ScalarMul(coeffs_add.c0, p.y), fp2ScalarMul(coeffs_add.c1, p.x), coeffs_add.c2);
        }
    }

    if (X_IS_NEGATIVE) f = Fp12.conjugate(f);

    // Final Frobenius steps
    const q1 = mulByChar(q);
    const coeffs_q1 = r.add_in_place(q1);
    f = fp12MulBy034(f, fp2ScalarMul(coeffs_q1.c0, p.y), fp2ScalarMul(coeffs_q1.c1, p.x), coeffs_q1.c2);

    var q2 = mulByChar(q1);
    q2.y = Fp2.neg(q2.y);
    const coeffs_q2 = r.add_in_place(q2);
    f = fp12MulBy034(f, fp2ScalarMul(coeffs_q2.c0, p.y), fp2ScalarMul(coeffs_q2.c1, p.x), coeffs_q2.c2);

    return f;
}

// ============================================================================
// Final exponentiation
// ============================================================================

fn expByX(f: Fp12) Fp12 {
    var result = Fp12.one();
    var base = f;
    var exp = BN_X;
    while (exp > 0) {
        if (exp & 1 == 1) result = Fp12.mul(result, base);
        base = Fp12.cyclotomicSquare(base);
        exp >>= 1;
    }
    return result;
}

fn expByNegX(f: Fp12) Fp12 {
    return Fp12.conjugate(expByX(f));
}

pub fn finalExponentiation(f: Fp12) Fp12 {
    if (Fp12.eql(f, Fp12.zero())) return Fp12.one();

    // Easy part: f^((p^6-1)(p^2+1))
    var f1 = Fp12.conjugate(f);
    const f2 = Fp12.inverse(f) orelse return Fp12.one();
    var r = Fp12.mul(f1, f2); // f^(p^6 - 1)
    f1 = r;
    r = frobenius2(r); // r^(p^2)
    r = Fp12.mul(r, f1); // f^((p^6-1)(p^2+1))

    // Hard part (arkworks algorithm)
    const y0 = expByNegX(r);
    const y1 = Fp12.cyclotomicSquare(y0);
    const y2 = Fp12.cyclotomicSquare(y1);
    var y3 = Fp12.mul(y2, y1);
    const y4 = expByNegX(y3);
    const y5 = Fp12.cyclotomicSquare(y4);
    var y6 = expByNegX(y5);
    y3 = Fp12.conjugate(y3);
    y6 = Fp12.conjugate(y6);
    const y7 = Fp12.mul(y6, y4);
    var y8 = Fp12.mul(y7, y3);
    const y9 = Fp12.mul(y8, y1);
    const y10 = Fp12.mul(y8, y4);
    const y11 = Fp12.mul(y10, r);
    const y12 = frobenius(y9);
    const y13 = Fp12.mul(y12, y11);
    y8 = frobenius2(y8);
    const y14 = Fp12.mul(y8, y13);
    const r_inv = Fp12.conjugate(r);
    var y15 = Fp12.mul(r_inv, y9);
    y15 = frobenius3(y15);
    return Fp12.mul(y15, y14);
}

// ============================================================================
// Top-level pairing function
// ============================================================================

pub fn pairingFp(p: G1Point, q: G2Point) Fp12 {
    if (p.infinity or q.infinity) return Fp12.one();
    return finalExponentiation(millerLoop(p, q));
}

// ============================================================================
// Tests
// ============================================================================

fn readFixture(relative_path: []const u8) ![]u8 {
    const dir = std.fs.cwd().openDir("../../testdata/zolt-arith-diff", .{}) catch
        return error.FixtureNotFound;
    return dir.readFileAlloc(std.testing.allocator, relative_path, 4 * 1024 * 1024);
}

fn parseHexBytes(comptime N: usize, hex: []const u8) ![N]u8 {
    if (hex.len != N * 2) return error.InvalidHexLength;
    var out: [N]u8 = undefined;
    _ = try std.fmt.hexToBytes(&out, hex);
    return out;
}

fn parseFp12HexLE(text: []const u8) !Fp12 {
    const bytes = try parseHexBytes(384, text);
    var result: Fp12 = undefined;
    inline for (0..12) |idx| {
        const fp_bytes: *const [32]u8 = bytes[idx * 32 ..][0..32];
        var limbs: [4]u64 = undefined;
        inline for (0..4) |i| {
            limbs[i] = std.mem.readInt(u64, fp_bytes[i * 8 ..][0..8], .little);
        }
        const val = (FiatFp{ .limbs = limbs }).toMontgomery();
        // Layout: c0.c0.c0, c0.c0.c1, c0.c1.c0, c0.c1.c1, c0.c2.c0, c0.c2.c1, c1.c0.c0, ...
        const fp6_idx = idx / 6;
        const fp2_idx = (idx % 6) / 2;
        const fp_idx = idx % 2;
        const fp6_ptr = if (fp6_idx == 0) &result.c0 else &result.c1;
        const fp2_ptr = switch (fp2_idx) {
            0 => &fp6_ptr.c0,
            1 => &fp6_ptr.c1,
            else => &fp6_ptr.c2,
        };
        if (fp_idx == 0) fp2_ptr.c0 = val else fp2_ptr.c1 = val;
    }
    return result;
}

test "fiat pairing: generator_cases.txt" {
    const text = try readFixture("pairing/generator_cases.txt");
    defer std.testing.allocator.free(text);
    var lines = std.mem.splitScalar(u8, text, '\n');
    var count: usize = 0;

    while (lines.next()) |raw_line| {
        const trimmed = std.mem.trim(u8, raw_line, " \t\r");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;

        var parts = std.mem.splitScalar(u8, trimmed, '|');
        _ = parts.next(); // name
        const g1_scalar_str = std.mem.trim(u8, parts.next() orelse continue, " \t\r");
        const g2_scalar_str = std.mem.trim(u8, parts.next() orelse continue, " \t\r");
        const expected_hex = std.mem.trim(u8, parts.next() orelse continue, " \t\r");

        const g1_scalar = try std.fmt.parseInt(u64, g1_scalar_str, 10);
        const g2_scalar = try std.fmt.parseInt(u64, g2_scalar_str, 10);
        const expected = try parseFp12HexLE(expected_hex);

        const g1 = g1ScalarMul(G1Point.generator(), g1_scalar);
        const g2 = g2ScalarMul(G2Point.generator(), g2_scalar);
        const result = pairingFp(g1, g2);

        try std.testing.expect(Fp12.eql(result, expected));
        count += 1;
    }
    try std.testing.expect(count >= 4);
}
