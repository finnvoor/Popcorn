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
    uint total = p.B * p.T * p.Nh * p.Hd2;
    if (id >= total) return;

    uint i = id % p.Hd2;
    uint rest = id / p.Hd2;
    uint h = rest % p.Nh;
    rest = rest / p.Nh;
    uint t = rest % p.T;
    uint b = rest / p.T;

    uint Hd = 2u * p.Hd2;
    uint headBase = ((b * p.T + t) * p.Nh + h) * Hd;
    float x1 = popcorn_load(x, headBase + i);
    float x2 = popcorn_load(x, headBase + p.Hd2 + i);
    float c = popcorn_load(cos_tab, t * p.Hd2 + i);
    float s = popcorn_load(sin_tab, t * p.Hd2 + i);

    popcorn_store(out, headBase + i, x1 * c - x2 * s);
    popcorn_store(out, headBase + p.Hd2 + i, x1 * s + x2 * c);
}

POPCORN_INSTANTIATE_KERNEL("rope_apply", rope_apply_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("rope_apply_bf16", rope_apply_typed, ushort, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("rope_apply_f32_tables_bf16", rope_apply_typed, ushort, float, ushort)
