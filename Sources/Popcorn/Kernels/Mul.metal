#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename T>
kernel void mul_typed(
    device const T* a [[ buffer(0) ]],
    device const T* b [[ buffer(1) ]],
    device T* out [[ buffer(2) ]],
    constant MulConstants& p [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint base = id * 4u;
    if (base >= p.count) return;
    if (base + 4u <= p.count) {
        float4 va = popcorn_load4(a, id);
        float4 vb = popcorn_load4(b, id);
        popcorn_store4(out, id, va * vb);
    } else {
        for (uint i = base; i < p.count; ++i) {
            popcorn_store(out, i, popcorn_load(a, i) * popcorn_load(b, i));
        }
    }
}

POPCORN_INSTANTIATE_KERNEL("mul", mul_typed, float)
POPCORN_INSTANTIATE_KERNEL("mul_f16", mul_typed, half)
POPCORN_INSTANTIATE_KERNEL("mul_bf16", mul_typed, ushort)
