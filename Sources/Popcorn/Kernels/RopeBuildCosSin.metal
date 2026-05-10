#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

kernel void rope_build_cos_sin(
    device const int*   positions [[ buffer(0) ]],   
    device const float* inv_freq  [[ buffer(1) ]],   
    device float*       cos_out   [[ buffer(2) ]],   
    device float*       sin_out   [[ buffer(3) ]],   
    constant RopeBuildCosSinConstants& p [[ buffer(4) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    uint t = tid.x;
    uint i = tid.y;
    if (t >= p.T || i >= p.Hd2) return;

    float angle = float(positions[t]) * inv_freq[i];
    uint idx = t * p.Hd2 + i;
    cos_out[idx] = cos(angle) * p.scaling;
    sin_out[idx] = sin(angle) * p.scaling;
}
