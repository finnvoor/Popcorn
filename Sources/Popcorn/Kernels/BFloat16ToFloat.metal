#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

kernel void bfloat16_to_float(
    device const ushort* in  [[ buffer(0) ]],
    device float*        out [[ buffer(1) ]],
    constant BFloat16ToFloatConstants& p [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint base = id * 4u;
    if (base >= p.count) return;
    if (base + 4u <= p.count) {
        ushort4 s = ((device const ushort4*)in)[id];
        float4 v = as_type<float4>(uint4(s) << 16);
        ((device float4*)out)[id] = v;
    } else {
        for (uint i = base; i < p.count; ++i) {
            out[i] = as_type<float>(uint(in[i]) << 16);
        }
    }
}
