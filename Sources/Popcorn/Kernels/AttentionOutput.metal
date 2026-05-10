#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename SType, typename VType, typename OType>
kernel void attention_output_typed(
    device const SType* scores [[ buffer(0) ]],
    device const VType* V [[ buffer(1) ]],
    device OType* out [[ buffer(2) ]],
    constant AttentionOutputConstants& p [[ buffer(3) ]],
    uint3 tid [[ thread_position_in_grid ]]
) {
    uint b = tid.x, hq = tid.y, qd = tid.z;
    if (b >= p.B || hq >= p.Nq || qd >= p.Sq * p.Hd) return;

    uint q = qd / p.Hd;
    uint d = qd % p.Hd;
    uint hkv = hq * p.Nkv / p.Nq;
    uint sbase = (((b * p.Nq + hq) * p.Sq) + q) * p.Sk;
    uint vbase = ((b * p.Nkv + hkv) * p.Sk) * p.Hd;

    float acc = 0.0f;
    for (uint k = 0; k < p.Sk; ++k) acc += popcorn_load(scores, sbase + k) * popcorn_load(V, vbase + k * p.Hd + d);
    popcorn_store(out, (((b * p.Nq + hq) * p.Sq) + q) * p.Hd + d, acc);
}

POPCORN_INSTANTIATE_KERNEL("attention_output", attention_output_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("attention_output_bf16", attention_output_typed, ushort, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("attention_output_bf16_to_f32", attention_output_typed, ushort, ushort, float)
POPCORN_INSTANTIATE_KERNEL("attention_output_f32_bf16_to_bf16", attention_output_typed, float, ushort, ushort)
