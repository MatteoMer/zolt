#include "fp_common.h"

// ── XYZZ point representation for bucket accumulation ───────────────────────

struct XYZZ {
    FpBase x;
    FpBase y;
    FpBase zz;
    FpBase zzz;
};

// ── XYZZ point arithmetic ──────────────────────────────────────────────────

static XYZZ xyzz_identity() {
    return { fpb_zero(), fpb_zero(), fpb_zero(), fpb_zero() };
}

static bool xyzz_is_empty(XYZZ p) {
    return fpb_is_zero(p.zz);
}

static XYZZ xyzz_from_affine(FpBase px, FpBase py) {
    return { px, py, fpb_one(), fpb_one() };
}

// Mixed addition: XYZZ + affine → XYZZ (madd-2008-s, 7M+2S)
static XYZZ xyzz_add_affine(XYZZ self, FpBase px, FpBase py) {
    if (xyzz_is_empty(self)) {
        return xyzz_from_affine(px, py);
    }

    FpBase U2 = fpb_mul(px, self.zz);
    FpBase S2 = fpb_mul(py, self.zzz);
    FpBase P = fpb_sub(U2, self.x);
    FpBase R = fpb_sub(S2, self.y);

    if (fpb_is_zero(P)) {
        if (fpb_is_zero(R)) {
            FpBase A = fpb_square(self.x);
            FpBase B = fpb_square(self.y);
            FpBase C = fpb_square(B);
            FpBase xpb = fpb_add(self.x, B);
            FpBase half_D = fpb_sub(fpb_sub(fpb_square(xpb), A), C);
            FpBase D = fpb_double(half_D);
            FpBase E = fpb_add(fpb_double(A), A);
            FpBase FF = fpb_square(E);
            FpBase X3 = fpb_sub(fpb_sub(FF, D), D);
            FpBase eight_C = fpb_double(fpb_double(fpb_double(C)));
            FpBase Y3 = fpb_sub(fpb_mul(E, fpb_sub(D, X3)), eight_C);
            FpBase four_B = fpb_double(fpb_double(B));
            FpBase ZZ3 = fpb_mul(four_B, self.zz);
            FpBase eight_Y_B = fpb_mul(fpb_double(fpb_double(fpb_double(self.y))), B);
            FpBase ZZZ3 = fpb_mul(eight_Y_B, self.zzz);
            return { X3, Y3, ZZ3, ZZZ3 };
        } else {
            return xyzz_identity();
        }
    }

    FpBase PP = fpb_square(P);
    FpBase PPP = fpb_mul(P, PP);
    FpBase Q = fpb_mul(self.x, PP);
    FpBase X3 = fpb_sub(fpb_sub(fpb_sub(fpb_square(R), PPP), Q), Q);
    FpBase Y3 = fpb_sub(fpb_mul(R, fpb_sub(Q, X3)), fpb_mul(self.y, PPP));
    FpBase ZZ3 = fpb_mul(self.zz, PP);
    FpBase ZZZ3 = fpb_mul(self.zzz, PPP);

    return { X3, Y3, ZZ3, ZZZ3 };
}

// Add two XYZZ points
static XYZZ xyzz_add(XYZZ self, XYZZ other) {
    if (xyzz_is_empty(self)) return other;
    if (xyzz_is_empty(other)) return self;

    FpBase U1 = fpb_mul(self.x, other.zz);
    FpBase U2 = fpb_mul(other.x, self.zz);
    FpBase S1 = fpb_mul(self.y, other.zzz);
    FpBase S2 = fpb_mul(other.y, self.zzz);
    FpBase P = fpb_sub(U2, U1);
    FpBase R = fpb_sub(S2, S1);

    if (fpb_is_zero(P)) {
        if (fpb_is_zero(R)) {
            FpBase A = fpb_square(self.x);
            FpBase B = fpb_square(self.y);
            FpBase C = fpb_square(B);
            FpBase xpb = fpb_add(self.x, B);
            FpBase half_D = fpb_sub(fpb_sub(fpb_square(xpb), A), C);
            FpBase D = fpb_double(half_D);
            FpBase E = fpb_add(fpb_double(A), A);
            FpBase FF = fpb_square(E);
            FpBase X3 = fpb_sub(fpb_sub(FF, D), D);
            FpBase eight_C = fpb_double(fpb_double(fpb_double(C)));
            FpBase Y3 = fpb_sub(fpb_mul(E, fpb_sub(D, X3)), eight_C);
            FpBase four_B = fpb_double(fpb_double(B));
            FpBase ZZ3 = fpb_mul(four_B, self.zz);
            FpBase eight_Y_B = fpb_mul(fpb_double(fpb_double(fpb_double(self.y))), B);
            FpBase ZZZ3 = fpb_mul(eight_Y_B, self.zzz);
            return { X3, Y3, ZZ3, ZZZ3 };
        } else {
            return xyzz_identity();
        }
    }

    FpBase PP = fpb_square(P);
    FpBase PPP = fpb_mul(P, PP);
    FpBase Q = fpb_mul(U1, PP);
    FpBase X3 = fpb_sub(fpb_sub(fpb_sub(fpb_square(R), PPP), Q), Q);
    FpBase Y3 = fpb_sub(fpb_mul(R, fpb_sub(Q, X3)), fpb_mul(S1, PPP));
    FpBase ZZ3 = fpb_mul(fpb_mul(self.zz, other.zz), PP);
    FpBase ZZZ3 = fpb_mul(fpb_mul(self.zzz, other.zzz), PPP);

    return { X3, Y3, ZZ3, ZZZ3 };
}

// ── MSM parameters ──────────────────────────────────────────────────────────

struct MsmParams {
    uint32_t num_points;
    uint32_t num_windows;
    uint32_t num_buckets;
    uint32_t num_rows;
};

// ── Fused bucket accumulate + running-sum reduce ────────────────────────────
//
// One thread per (row, window) pair.
// Each thread:
//   1. Iterates over all points, accumulates into buckets in device memory
//   2. Performs running-sum reduction across buckets
//   3. Writes one XYZZ result (the window sum for this row)
//
// Grid size: num_rows * num_windows

kernel void msm_accumulate_reduce(
    device const FpBase*  points    [[buffer(0)]],
    device const int16_t* digits    [[buffer(1)]],
    device FpBase*        buckets   [[buffer(2)]],
    device FpBase*        results   [[buffer(3)]],
    constant MsmParams&   params    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint num_points = params.num_points;
    uint num_windows = params.num_windows;
    uint num_buckets = params.num_buckets;

    uint row = tid / num_windows;
    uint win = tid % num_windows;
    if (row >= params.num_rows) return;

    uint bucket_offset = tid * num_buckets * 4;

    // Initialize buckets to identity (zz = 0)
    for (uint b = 0; b < num_buckets; b++) {
        uint base = bucket_offset + b * 4;
        buckets[base + 0] = fpb_zero();
        buckets[base + 1] = fpb_zero();
        buckets[base + 2] = fpb_zero();
        buckets[base + 3] = fpb_zero();
    }

    uint digit_offset = (row * num_windows + win) * num_points;

    // Phase 1: Bucket accumulation
    for (uint i = 0; i < num_points; i++) {
        int16_t d = digits[digit_offset + i];
        if (d == 0) continue;

        FpBase px = points[i * 2];
        FpBase py = points[i * 2 + 1];

        // Skip identity points
        if (fpb_is_zero(px) && fpb_is_zero(py)) continue;

        uint bidx;
        if (d > 0) {
            bidx = (uint)(d - 1);
        } else {
            bidx = (uint)(-d - 1);
            py = fpb_neg(py);
        }

        uint base = bucket_offset + bidx * 4;
        XYZZ bucket = { buckets[base], buckets[base + 1], buckets[base + 2], buckets[base + 3] };
        bucket = xyzz_add_affine(bucket, px, py);
        buckets[base + 0] = bucket.x;
        buckets[base + 1] = bucket.y;
        buckets[base + 2] = bucket.zz;
        buckets[base + 3] = bucket.zzz;
    }

    // Phase 2: Running-sum reduction (buckets high → low)
    XYZZ running_sum = xyzz_identity();
    XYZZ window_sum = xyzz_identity();

    for (uint b = num_buckets; b > 0; b--) {
        uint base = bucket_offset + (b - 1) * 4;
        XYZZ bkt = { buckets[base], buckets[base + 1], buckets[base + 2], buckets[base + 3] };
        running_sum = xyzz_add(running_sum, bkt);
        window_sum = xyzz_add(window_sum, running_sum);
    }

    uint result_base = tid * 4;
    results[result_base + 0] = window_sum.x;
    results[result_base + 1] = window_sum.y;
    results[result_base + 2] = window_sum.zz;
    results[result_base + 3] = window_sum.zzz;
}

// ── Test kernels ────────────────────────────────────────────────────────────

kernel void fp_base_mul_test(
    device const FpBase* a [[buffer(0)]],
    device const FpBase* b [[buffer(1)]],
    device FpBase*       c [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    c[tid] = fpb_mul(a[tid], b[tid]);
}

kernel void fp_base_add_test(
    device const FpBase* a [[buffer(0)]],
    device const FpBase* b [[buffer(1)]],
    device FpBase*       c [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    c[tid] = fpb_add(a[tid], b[tid]);
}
