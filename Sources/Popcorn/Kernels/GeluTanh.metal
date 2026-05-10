#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename X, typename O>
kernel void gelu_tanh_typed(
    device const X* x [[ buffer(0) ]],
    device O* out [[ buffer(1) ]],
    constant GeluTanhConstants& p [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= p.count) return;
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float v = popcorn_load(x, id);
    float inner = clamp(k0 * (v + k1 * v * v * v), -20.0f, 20.0f);
    popcorn_store(out, id, 0.5f * v * (1.0f + tanh(inner)));
}

POPCORN_INSTANTIATE_KERNEL("gelu_tanh", gelu_tanh_typed, float, float)
POPCORN_INSTANTIATE_KERNEL("gelu_tanh_bf16", gelu_tanh_typed, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("gelu_tanh_f16", gelu_tanh_typed, half, half)
