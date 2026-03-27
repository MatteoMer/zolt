#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const uint32_t* a [[buffer(0)]],
    device const uint32_t* b [[buffer(1)]],
    device uint32_t*       c [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    c[tid] = a[tid] + b[tid];
}
