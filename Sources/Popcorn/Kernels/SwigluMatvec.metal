#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

// Fused SwiGLU matvec for the t=1 decode path of a SwiGLU MLP:
//   out[n] = gelu_tanh(dot(x, gate[n,:])) * dot(x, up[n,:])
// Gate and up share x and have the same K x N shape (transposed weights).
template <typename X, typename W, typename O>
kernel void swiglu_matvec_typed(
    device const X* x       [[ buffer(0) ]],
    device const W* wg      [[ buffer(1) ]],
    device const W* wu      [[ buffer(2) ]],
    device O*       out     [[ buffer(3) ]],
    constant MatvecConstants& p [[ buffer(4) ]],
    uint gid  [[ thread_position_in_grid ]],
    ushort lane [[ thread_index_in_simdgroup ]]
) {
    uint block = gid >> 5;
    uint row0 = block * 4;
    if (row0 >= p.N) return;

    const uint K = p.K;
    const bool v1 = row0 + 1 < p.N;
    const bool v2 = row0 + 2 < p.N;
    const bool v3 = row0 + 3 < p.N;

    float4 g0v = 0, g1v = 0, g2v = 0, g3v = 0;
    float4 u0v = 0, u1v = 0, u2v = 0, u3v = 0;

    if ((K & 3u) == 0u) {
        const uint K4 = K >> 2;
        const uint b0 = (row0 + 0) * K4;
        const uint b1 = (row0 + 1) * K4;
        const uint b2 = (row0 + 2) * K4;
        const uint b3 = (row0 + 3) * K4;
        for (uint k4 = lane; k4 < K4; k4 += 32) {
            float4 xv = popcorn_load4(x, k4);
            g0v += xv * popcorn_load4(wg, b0 + k4);
            u0v += xv * popcorn_load4(wu, b0 + k4);
            if (v1) { g1v += xv * popcorn_load4(wg, b1 + k4); u1v += xv * popcorn_load4(wu, b1 + k4); }
            if (v2) { g2v += xv * popcorn_load4(wg, b2 + k4); u2v += xv * popcorn_load4(wu, b2 + k4); }
            if (v3) { g3v += xv * popcorn_load4(wg, b3 + k4); u3v += xv * popcorn_load4(wu, b3 + k4); }
        }
    } else {
        const uint b0 = (row0 + 0) * K;
        const uint b1 = (row0 + 1) * K;
        const uint b2 = (row0 + 2) * K;
        const uint b3 = (row0 + 3) * K;
        for (uint k = lane; k < K; k += 32) {
            float xv = popcorn_load(x, k);
            g0v.x += xv * popcorn_load(wg, b0 + k); u0v.x += xv * popcorn_load(wu, b0 + k);
            if (v1) { g1v.x += xv * popcorn_load(wg, b1 + k); u1v.x += xv * popcorn_load(wu, b1 + k); }
            if (v2) { g2v.x += xv * popcorn_load(wg, b2 + k); u2v.x += xv * popcorn_load(wu, b2 + k); }
            if (v3) { g3v.x += xv * popcorn_load(wg, b3 + k); u3v.x += xv * popcorn_load(wu, b3 + k); }
        }
    }

    float g0 = simd_sum(g0v.x + g0v.y + g0v.z + g0v.w);
    float u0 = simd_sum(u0v.x + u0v.y + u0v.z + u0v.w);
    float g1 = simd_sum(g1v.x + g1v.y + g1v.z + g1v.w);
    float u1 = simd_sum(u1v.x + u1v.y + u1v.z + u1v.w);
    float g2 = simd_sum(g2v.x + g2v.y + g2v.z + g2v.w);
    float u2 = simd_sum(u2v.x + u2v.y + u2v.z + u2v.w);
    float g3 = simd_sum(g3v.x + g3v.y + g3v.z + g3v.w);
    float u3 = simd_sum(u3v.x + u3v.y + u3v.z + u3v.w);

    if (lane == 0) {
        const float k0 = 0.7978845608028654f;
        const float k1 = 0.044715f;
        float i0 = clamp(k0 * (g0 + k1 * g0 * g0 * g0), -20.0f, 20.0f);
        popcorn_store(out, row0 + 0, 0.5f * g0 * (1.0f + tanh(i0)) * u0);
        if (v1) { float i1 = clamp(k0 * (g1 + k1 * g1 * g1 * g1), -20.0f, 20.0f);
                  popcorn_store(out, row0 + 1, 0.5f * g1 * (1.0f + tanh(i1)) * u1); }
        if (v2) { float i2 = clamp(k0 * (g2 + k1 * g2 * g2 * g2), -20.0f, 20.0f);
                  popcorn_store(out, row0 + 2, 0.5f * g2 * (1.0f + tanh(i2)) * u2); }
        if (v3) { float i3 = clamp(k0 * (g3 + k1 * g3 * g3 * g3), -20.0f, 20.0f);
                  popcorn_store(out, row0 + 3, 0.5f * g3 * (1.0f + tanh(i3)) * u3); }
    }
}

POPCORN_INSTANTIATE_KERNEL("swiglu_matvec_bf16_bf16_bf16", swiglu_matvec_typed, ushort, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("swiglu_matvec_bf16_bf16_f32",  swiglu_matvec_typed, ushort, ushort, float)
POPCORN_INSTANTIATE_KERNEL("swiglu_matvec_f32_bf16_f32",   swiglu_matvec_typed, float,  ushort, float)
