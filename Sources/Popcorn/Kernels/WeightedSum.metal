#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

kernel void weighted_sum(
    device const float* contrib [[ buffer(0) ]],   
    device const float* weights [[ buffer(1) ]],   
    device float*       out     [[ buffer(2) ]],   
    constant WeightedSumConstants& p [[ buffer(3) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    uint r = tid.x;
    uint h = tid.y;
    if (r >= p.rows || h >= p.H) return;

    device const float* cBase = contrib + r * p.K * p.H;
    device const float* wRow  = weights + r * p.K;

    float acc = 0.0f;
    for (uint k = 0; k < p.K; ++k) {
        acc += wRow[k] * cBase[k * p.H + h];
    }
    out[r * p.H + h] = acc;
}
