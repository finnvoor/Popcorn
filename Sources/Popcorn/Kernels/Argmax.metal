#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

#define ARGMAX_NO_INDEX 0x7fffffff

static inline bool argmax_is_better(float candidateV, int candidateI, float bestV, int bestI) {
    return candidateV > bestV || (candidateV == bestV && candidateI < bestI);
}

// Single-threaded argmax. This is still the best path when there are enough
// small rows to fill the GPU without paying per-row reduction overhead.
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
        if (argmax_is_better(v, int(i), bestV, bestI)) { bestV = v; bestI = int(i); }
    }
    indices[row] = bestI;
}

// Parallel argmax: one threadgroup per row. Threads stripe across N, reduce
// within each simdgroup, then reduce the simdgroup winners through threadgroup
// memory. The host selects the threadgroup width from the device limit, N, and
// row count so this kernel scales from small batches to large vocabularies.
kernel void argmax_row(
    device const float* x       [[ buffer(0) ]],
    device int*         indices [[ buffer(1) ]],
    constant ArgmaxConstants& p [[ buffer(2) ]],
    uint row [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tgSize [[ threads_per_threadgroup ]],
    ushort lane [[ thread_index_in_simdgroup ]],
    ushort simdID [[ simdgroup_index_in_threadgroup ]]
) {
    if (row >= p.rows) return;

    device const float* xrow = x + row * p.N;
    float bestV = -INFINITY;
    int bestI = ARGMAX_NO_INDEX;
    for (uint i = tid; i < p.N; i += tgSize) {
        float v = xrow[i];
        if (argmax_is_better(v, int(i), bestV, bestI)) { bestV = v; bestI = int(i); }
    }

    for (ushort offset = 16; offset > 0; offset >>= 1) {
        float otherV = simd_shuffle_down(bestV, offset);
        int otherI = simd_shuffle_down(bestI, offset);
        if (argmax_is_better(otherV, otherI, bestV, bestI)) { bestV = otherV; bestI = otherI; }
    }

    threadgroup float partialV[32];
    threadgroup int partialI[32];
    if (lane == 0) {
        partialV[simdID] = bestV;
        partialI[simdID] = bestI;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdID == 0) {
        uint simdCount = (tgSize + 31u) >> 5;
        bestV = lane < simdCount ? partialV[lane] : -INFINITY;
        bestI = lane < simdCount ? partialI[lane] : ARGMAX_NO_INDEX;
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float otherV = simd_shuffle_down(bestV, offset);
            int otherI = simd_shuffle_down(bestI, offset);
            if (argmax_is_better(otherV, otherI, bestV, bestI)) { bestV = otherV; bestI = otherI; }
        }
        if (lane == 0) {
            indices[row] = bestI;
        }
    }
}
