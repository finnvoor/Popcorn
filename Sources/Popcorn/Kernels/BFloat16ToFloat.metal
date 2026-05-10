#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

kernel void bfloat16_to_float(
    device const ushort* in  [[ buffer(0) ]],
    device float*        out [[ buffer(1) ]],
    constant BFloat16ToFloatConstants& p [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= p.count) return;
    uint bits = uint(in[id]) << 16;
    out[id] = as_type<float>(bits);
}
