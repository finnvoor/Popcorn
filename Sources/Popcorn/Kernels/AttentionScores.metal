#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename QType, typename KType, typename OType>
kernel void attention_scores_typed(
    device const QType* Q [[ buffer(0) ]],
    device const KType* K [[ buffer(1) ]],
    device OType* scores [[ buffer(2) ]],
    constant AttentionScoresConstants& p [[ buffer(3) ]],
    uint3 tid [[ thread_position_in_grid ]]
) {
    uint b = tid.x, hq = tid.y, qk = tid.z;
    if (b >= p.B || hq >= p.Nq || qk >= p.Sq * p.Sk) return;

    uint q = qk / p.Sk;
    uint k = qk % p.Sk;
    uint hkv = hq * p.Nkv / p.Nq;
    uint qbase = (((b * p.Nq + hq) * p.Sq) + q) * p.Hd;
    uint kbase = (((b * p.Nkv + hkv) * p.Sk) + k) * p.Hd;

    float acc = 0.0f;
    for (uint d = 0; d < p.Hd; ++d) acc += popcorn_load(Q, qbase + d) * popcorn_load(K, kbase + d);
    popcorn_store(scores, (((b * p.Nq + hq) * p.Sq) + q) * p.Sk + k, acc);
}

POPCORN_INSTANTIATE_KERNEL("attention_scores", attention_scores_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("attention_scores_bf16", attention_scores_typed, ushort, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("attention_scores_bf16_to_f32", attention_scores_typed, ushort, ushort, float)
