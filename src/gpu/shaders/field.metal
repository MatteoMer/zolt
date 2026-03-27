#include "field_common.h"

// ── Element-wise batch kernels ──────────────────────────────────────────────

kernel void field_mul_batch(
    device const Fp256* a [[buffer(0)]],
    device const Fp256* b [[buffer(1)]],
    device Fp256*       c [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    c[tid] = fp_mul(a[tid], b[tid]);
}

kernel void field_add_batch(
    device const Fp256* a [[buffer(0)]],
    device const Fp256* b [[buffer(1)]],
    device Fp256*       c [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    c[tid] = fp_add(a[tid], b[tid]);
}

kernel void field_sub_batch(
    device const Fp256* a [[buffer(0)]],
    device const Fp256* b [[buffer(1)]],
    device Fp256*       c [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    c[tid] = fp_sub(a[tid], b[tid]);
}

kernel void field_neg_batch(
    device const Fp256* a [[buffer(0)]],
    device Fp256*       b [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    b[tid] = fp_neg(a[tid]);
}
