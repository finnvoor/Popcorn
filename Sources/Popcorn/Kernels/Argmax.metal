#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

// Single-threaded argmax — kept for the rowCount > 1 case, where each thread
// owns its own row.
kernel void argmax(
    device const float* x       [[ buffer(0) ]],
    device int*         indices [[ buffer(1) ]],
    constant ArgmaxConstants& p [[ buffer(2) ]],
    uint row [[ thread_position_in_grid ]]
) {
    if (row >= p.rows) return;

    device const float* xrow = x + row * p.N;
    float bestV = xrow[0];
    int bestI = 0;
    for (uint i = 1; i < p.N; ++i) {
        float v = xrow[i];
        if (v > bestV) { bestV = v; bestI = int(i); }
    }
    indices[row] = bestI;
}

// Parallel argmax: one threadgroup per row, with simdgroup + threadgroup
// reductions across N. Designed for large N (e.g. argmax over a vocabulary).
kernel void argmax_row(
    device const float* x       [[ buffer(0) ]],
    device int*         indices [[ buffer(1) ]],
    constant ArgmaxConstants& p [[ buffer(2) ]],
    uint  row      [[ threadgroup_position_in_grid ]],
    uint  tid      [[ thread_position_in_threadgroup ]],
    uint  tg_size  [[ threads_per_threadgroup ]],
    ushort lane    [[ thread_index_in_simdgroup ]],
    ushort simd_id [[ simdgroup_index_in_threadgroup ]]
) {
    if (row >= p.rows) return;
    device const float* xrow = x + row * p.N;

    float bestV = -INFINITY;
    int   bestI = 0;
    for (uint i = tid; i < p.N; i += tg_size) {
        float v = xrow[i];
        if (v > bestV) { bestV = v; bestI = int(i); }
    }

    // Simdgroup reduction.
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        float otherV = simd_shuffle_down(bestV, offset);
        int   otherI = simd_shuffle_down(bestI, offset);
        if (otherV > bestV) { bestV = otherV; bestI = otherI; }
    }

    threadgroup float partialV[32];
    threadgroup int   partialI[32];
    if (lane == 0) {
        partialV[simd_id] = bestV;
        partialI[simd_id] = bestI;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint num_simds = (tg_size + 31u) >> 5;
        bestV = (lane < num_simds) ? partialV[lane] : -INFINITY;
        bestI = (lane < num_simds) ? partialI[lane] : 0;
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float otherV = simd_shuffle_down(bestV, offset);
            int   otherI = simd_shuffle_down(bestI, offset);
            if (otherV > bestV) { bestV = otherV; bestI = otherI; }
        }
        if (lane == 0) {
            indices[row] = bestI;
        }
    }
}
