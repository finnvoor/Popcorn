#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

kernel void topk(
    device const float* x       [[ buffer(0) ]],   
    device float*       values  [[ buffer(1) ]],   
    device int*         indices [[ buffer(2) ]],   
    constant TopKConstants& p [[ buffer(3) ]],
    uint row [[ thread_position_in_grid ]]
) {
    if (row >= p.rows) return;

    constexpr uint MAX_K = 32;
    float bestV[MAX_K];
    int   bestI[MAX_K];
    uint  K = min(p.K, MAX_K);

    for (uint k = 0; k < K; ++k) {
        bestV[k] = -INFINITY;
        bestI[k] = -1;
    }

    device const float* xrow = x + row * p.E;
    for (uint e = 0; e < p.E; ++e) {
        float v = xrow[e];
        
        if (v > bestV[K - 1]) {
            uint pos = K - 1;
            while (pos > 0 && v > bestV[pos - 1]) {
                bestV[pos] = bestV[pos - 1];
                bestI[pos] = bestI[pos - 1];
                pos -= 1;
            }
            bestV[pos] = v;
            bestI[pos] = int(e);
        }
    }

    for (uint k = 0; k < K; ++k) {
        values[row * p.K + k] = bestV[k];
        indices[row * p.K + k] = bestI[k];
    }
}
