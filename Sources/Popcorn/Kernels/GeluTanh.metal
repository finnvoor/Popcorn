#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

inline float gelu_tanh_scalar(float v) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float inner = clamp(k0 * (v + k1 * v * v * v), -20.0f, 20.0f);
    return 0.5f * v * (1.0f + tanh(inner));
}

template <typename X, typename O>
kernel void gelu_tanh_typed(
    device const X* x [[ buffer(0) ]],
    device O* out [[ buffer(1) ]],
    constant GeluTanhConstants& p [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint base = id * 4u;
    if (base >= p.count) return;
    if (base + 4u <= p.count) {
        float4 v = popcorn_load4(x, id);
        float4 r;
        r.x = gelu_tanh_scalar(v.x);
        r.y = gelu_tanh_scalar(v.y);
        r.z = gelu_tanh_scalar(v.z);
        r.w = gelu_tanh_scalar(v.w);
        popcorn_store4(out, id, r);
    } else {
        for (uint i = base; i < p.count; ++i) {
            popcorn_store(out, i, gelu_tanh_scalar(popcorn_load(x, i)));
        }
    }
}

POPCORN_INSTANTIATE_KERNEL("gelu_tanh", gelu_tanh_typed, float, float)
POPCORN_INSTANTIATE_KERNEL("gelu_tanh_bf16", gelu_tanh_typed, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("gelu_tanh_f16", gelu_tanh_typed, half, half)
