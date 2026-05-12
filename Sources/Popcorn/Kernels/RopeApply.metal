#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename X, typename T, typename O>
kernel void rope_apply_typed(
    device const X* x [[ buffer(0) ]],
    device const T* cos_tab [[ buffer(1) ]],
    device const T* sin_tab [[ buffer(2) ]],
    device O* out [[ buffer(3) ]],
    constant RopeApplyConstants& p [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint Hd2_4 = (p.Hd2 + 3u) / 4u;
    uint total = p.B * p.T * p.Nh * Hd2_4;
    if (id >= total) return;

    uint iChunk = id % Hd2_4;
    uint rest = id / Hd2_4;
    uint h = rest % p.Nh;
    rest = rest / p.Nh;
    uint t = rest % p.T;
    uint b = rest / p.T;

    uint iBase = iChunk * 4u;
    uint Hd = 2u * p.Hd2;
    uint headBase = ((b * p.T + t) * p.Nh + h) * Hd;
    uint cosBase = t * p.Hd2;

    if (iBase + 4u <= p.Hd2 && (p.Hd2 & 3u) == 0u) {
        float4 x1 = popcorn_load4(x,        (headBase + iBase) >> 2);
        float4 x2 = popcorn_load4(x,        (headBase + p.Hd2 + iBase) >> 2);
        float4 c  = popcorn_load4(cos_tab,  (cosBase + iBase) >> 2);
        float4 s  = popcorn_load4(sin_tab,  (cosBase + iBase) >> 2);
        popcorn_store4(out, (headBase + iBase) >> 2,            x1 * c - x2 * s);
        popcorn_store4(out, (headBase + p.Hd2 + iBase) >> 2,    x1 * s + x2 * c);
    } else {
        uint limit = min(iBase + 4u, p.Hd2);
        for (uint i = iBase; i < limit; ++i) {
            float x1s = popcorn_load(x, headBase + i);
            float x2s = popcorn_load(x, headBase + p.Hd2 + i);
            float cs  = popcorn_load(cos_tab, cosBase + i);
            float ss  = popcorn_load(sin_tab, cosBase + i);
            popcorn_store(out, headBase + i,           x1s * cs - x2s * ss);
            popcorn_store(out, headBase + p.Hd2 + i,   x1s * ss + x2s * cs);
        }
    }
}

POPCORN_INSTANTIATE_KERNEL("rope_apply", rope_apply_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("rope_apply_bf16", rope_apply_typed, ushort, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("rope_apply_f32_tables_bf16", rope_apply_typed, ushort, float, ushort)
