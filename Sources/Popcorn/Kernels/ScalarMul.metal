#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename T>
kernel void scalar_mul_typed(
    device const T* x [[ buffer(0) ]],
    device T* out [[ buffer(1) ]],
    constant ScalarMulConstants& p [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint base = id * 4u;
    if (base >= p.count) return;
    if (base + 4u <= p.count) {
        float4 v = popcorn_load4(x, id);
        popcorn_store4(out, id, v * p.scalar);
    } else {
        for (uint i = base; i < p.count; ++i) {
            popcorn_store(out, i, popcorn_load(x, i) * p.scalar);
        }
    }
}

POPCORN_INSTANTIATE_KERNEL("scalar_mul", scalar_mul_typed, float)
POPCORN_INSTANTIATE_KERNEL("scalar_mul_f16", scalar_mul_typed, half)
POPCORN_INSTANTIATE_KERNEL("scalar_mul_bf16", scalar_mul_typed, ushort)
