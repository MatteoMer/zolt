#include "field_common.h"

// ── Polynomial binding and reduction kernels for sumcheck ───────────────────

// bindLow: out[i] = evals[2i] + r * (evals[2i+1] - evals[2i])
// Matches DensePolynomial.bindLow — binds the lowest-index variable.
// Dispatch n/2 threads.
kernel void poly_bind_low(
    device const Fp256* evals [[buffer(0)]],
    device Fp256*       out   [[buffer(1)]],
    constant Fp256&     r     [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    Fp256 lo = evals[2 * tid];
    Fp256 hi = evals[2 * tid + 1];
    Fp256 diff = fp_sub(hi, lo);
    out[tid] = fp_add(lo, fp_mul(diff, r));
}

// bindFirst: out[i] = one_minus_r * evals[i] + r * evals[i + half]
// Matches DensePolynomial.bindFirst — binds the highest-index variable.
// Dispatch n/2 threads.
kernel void poly_bind_first(
    device const Fp256* evals       [[buffer(0)]],
    device Fp256*       out         [[buffer(1)]],
    constant Fp256&     r           [[buffer(2)]],
    constant Fp256&     one_minus_r [[buffer(3)]],
    constant uint&      half_n      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    Fp256 lo = fp_mul(evals[tid], one_minus_r);
    Fp256 hi = fp_mul(evals[tid + half_n], r);
    out[tid] = fp_add(lo, hi);
}

// Parallel tree reduction: sum field elements within each threadgroup,
// write one partial sum per threadgroup. CPU sums the partials.
//
// Uses threadgroup shared memory for the reduction tree.
// Threadgroup size should be a power of 2 (typically 256).
kernel void field_reduce_sum(
    device const Fp256* input        [[buffer(0)]],
    device Fp256*       partial_sums [[buffer(1)]],
    constant uint&      n            [[buffer(2)]],
    uint tid     [[thread_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Load element (or zero if out of bounds)
    threadgroup Fp256 shared[256];
    shared[lid] = (tid < n) ? input[tid] : fp_zero();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] = fp_add(shared[lid], shared[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Threadgroup leader writes partial sum
    if (lid == 0) {
        partial_sums[gid] = shared[0];
    }
}

// ── In-place variants (zero-copy, persistent GPU buffers) ───────────────────

// bindLow in-place: reads from [2*tid] and [2*tid+1], writes to [tid].
// Safe because write index < read indices for all tid > 0.
kernel void poly_bind_low_inplace(
    device Fp256*   evals [[buffer(0)]],
    constant Fp256& r     [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    Fp256 lo = evals[2 * tid];
    Fp256 hi = evals[2 * tid + 1];
    Fp256 diff = fp_sub(hi, lo);
    evals[tid] = fp_add(lo, fp_mul(diff, r));
}

// bindFirst in-place: reads from [tid] and [tid + half], writes to [tid].
// Safe because each thread only writes to its own index.
kernel void poly_bind_first_inplace(
    device Fp256*       evals       [[buffer(0)]],
    constant Fp256&     r           [[buffer(1)]],
    constant Fp256&     one_minus_r [[buffer(2)]],
    constant uint&      half_n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    Fp256 lo = fp_mul(evals[tid], one_minus_r);
    Fp256 hi = fp_mul(evals[tid + half_n], r);
    evals[tid] = fp_add(lo, hi);
}

// ── Fused Stage 2 kernel: product sumcheck round ────────────────────────────
//
// Each thread handles one group g from interleaved left/right polynomials:
//   p0    = left[2g] * right[2g]
//   slope = (left[2g+1] - left[2g]) * (right[2g+1] - right[2g])
//   eq_w  = E_out[g >> xin_bits] * E_in[g & xin_mask]
//   contribution = (p0 * eq_w, slope * eq_w)
// Then threadgroup reduction sums contributions to partial sums.
kernel void product_sumcheck_round(
    device const Fp256* left         [[buffer(0)]],
    device const Fp256* right        [[buffer(1)]],
    device const Fp256* e_out        [[buffer(2)]],
    device const Fp256* e_in         [[buffer(3)]],
    device Fp256*       partial_t0   [[buffer(4)]],
    device Fp256*       partial_tinf [[buffer(5)]],
    constant uint&      num_groups   [[buffer(6)]],
    constant uint&      xin_bits     [[buffer(7)]],
    constant uint&      e_in_len     [[buffer(8)]],
    constant uint&      e_out_len    [[buffer(9)]],
    uint tid     [[thread_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    Fp256 my_t0 = fp_zero();
    Fp256 my_tinf = fp_zero();

    if (tid < num_groups) {
        uint g = tid;
        Fp256 l_lo = left[2 * g];
        Fp256 l_hi = left[2 * g + 1];
        Fp256 r_lo = right[2 * g];
        Fp256 r_hi = right[2 * g + 1];

        Fp256 p0 = fp_mul(l_lo, r_lo);
        Fp256 slope = fp_mul(fp_sub(l_hi, l_lo), fp_sub(r_hi, r_lo));

        // Compute eq weight with bounds checking (match CPU behavior)
        uint xin_mask = (e_in_len > 1) ? (e_in_len - 1) : 0;
        uint x_in = g & xin_mask;
        uint x_out = g >> xin_bits;

        bool in_bounds = (x_out < e_out_len) && (e_in_len <= 1 || x_in < e_in_len);
        if (in_bounds) {
            Fp256 eq_w = fp_mul(e_out[x_out],
                                (e_in_len <= 1) ? fp_one() : e_in[x_in]);
            my_t0 = fp_mul(p0, eq_w);
            my_tinf = fp_mul(slope, eq_w);
        }
    }

    // Threadgroup reduction for t0
    threadgroup Fp256 sh_t0[256];
    threadgroup Fp256 sh_tinf[256];
    sh_t0[lid] = my_t0;
    sh_tinf[lid] = my_tinf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sh_t0[lid] = fp_add(sh_t0[lid], sh_t0[lid + s]);
            sh_tinf[lid] = fp_add(sh_tinf[lid], sh_tinf[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partial_t0[gid] = sh_t0[0];
        partial_tinf[gid] = sh_tinf[0];
    }
}

// ── Fused Stage 1 kernel: bind Az/Bz + reduce products ─────────────────────
//
// Each thread at index i:
//   az_out[i] = az[2i] + r * (az[2i+1] - az[2i])   (bindLow)
//   bz_out[i] = bz[2i] + r * (bz[2i+1] - bz[2i])   (bindLow)
//   t0_contrib = az_lo * bz_lo                        (product at r=0)
//   tinf_contrib = (az_hi - az_lo) * (bz_hi - bz_lo) (slope product)
// Threadgroup reduction sums t0 and tinf.
kernel void outer_bind_reduce(
    device const Fp256* az_in    [[buffer(0)]],
    device const Fp256* bz_in    [[buffer(1)]],
    device Fp256*       az_out   [[buffer(2)]],
    device Fp256*       bz_out   [[buffer(3)]],
    device Fp256*       partial  [[buffer(4)]],  // [2] per threadgroup (t0, tinf)
    constant Fp256&     r        [[buffer(5)]],
    constant uint&      half_n   [[buffer(6)]],
    uint tid     [[thread_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    Fp256 my_t0 = fp_zero();
    Fp256 my_tinf = fp_zero();

    if (tid < half_n) {
        Fp256 az_lo = az_in[2 * tid];
        Fp256 az_hi = az_in[2 * tid + 1];
        Fp256 bz_lo = bz_in[2 * tid];
        Fp256 bz_hi = bz_in[2 * tid + 1];

        // bindLow outputs
        Fp256 az_diff = fp_sub(az_hi, az_lo);
        Fp256 bz_diff = fp_sub(bz_hi, bz_lo);
        az_out[tid] = fp_add(az_lo, fp_mul(az_diff, r));
        bz_out[tid] = fp_add(bz_lo, fp_mul(bz_diff, r));

        // Product contributions for t_prime
        my_t0 = fp_mul(az_lo, bz_lo);
        my_tinf = fp_mul(az_diff, bz_diff);
    }

    // Threadgroup reduction
    threadgroup Fp256 sh_t0[256];
    threadgroup Fp256 sh_tinf[256];
    sh_t0[lid] = my_t0;
    sh_tinf[lid] = my_tinf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sh_t0[lid] = fp_add(sh_t0[lid], sh_t0[lid + s]);
            sh_tinf[lid] = fp_add(sh_tinf[lid], sh_tinf[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partial[gid * 2] = sh_t0[0];
        partial[gid * 2 + 1] = sh_tinf[0];
    }
}

// ── Dot product: Σ a[i] * b[i] with threadgroup reduction ──────────────
//
// GPU equivalent of CPU's UnreducedProductAccum pattern. Each thread
// computes one fp_mul (fully reduced), then threadgroup tree reduction
// sums the products. Partial sums are written per threadgroup for CPU
// to finalize.

kernel void field_dot_product(
    device const Fp256* a        [[buffer(0)]],
    device const Fp256* b        [[buffer(1)]],
    device Fp256*       partials [[buffer(2)]],
    constant uint&      n        [[buffer(3)]],
    uint tid     [[thread_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    Fp256 val = (tid < n) ? fp_mul(a[tid], b[tid]) : fp_zero();

    threadgroup Fp256 shared[256];
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] = fp_add(shared[lid], shared[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partials[gid] = shared[0];
    }
}
