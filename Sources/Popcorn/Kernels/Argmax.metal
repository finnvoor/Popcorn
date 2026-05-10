#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

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
