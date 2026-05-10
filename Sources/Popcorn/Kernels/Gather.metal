#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

kernel void gather(
    device const float* table [[ buffer(0) ]],
    device const int*   idx   [[ buffer(1) ]],
    device float*       out   [[ buffer(2) ]],
    constant GatherConstants& p [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= p.count) return;
    out[id] = table[uint(idx[id])];
}
