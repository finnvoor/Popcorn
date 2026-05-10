#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename QType, typename KType, typename OType>
kernel void attention_scores_softmax_typed(
    device const QType* Q [[ buffer(0) ]],
    device const KType* K [[ buffer(1) ]],
    device OType* probs [[ buffer(2) ]],
    constant AttentionScoresSoftmaxConstants& p [[ buffer(3) ]],
    uint row [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    threadgroup float scores_cache[1024];
    threadgroup float partial[1024];

    uint q_row = row;
    uint q = q_row % p.Sq;
    q_row /= p.Sq;
    uint hq = q_row % p.Nq;
    q_row /= p.Nq;
    uint b = q_row;
    if (b >= p.B) return;

    uint hkv = hq * p.Nkv / p.Nq;
    uint qbase = (((b * p.Nq + hq) * p.Sq) + q) * p.Hd;
    uint outbase = (((b * p.Nq + hq) * p.Sq) + q) * p.Sk;
    int posQ = int(p.Sk) - int(p.Sq) + int(q);

    float local_max = -INFINITY;
    for (uint key = tid; key < p.Sk; key += tg_size) {
        int posK = int(key);
        bool allowed = posK <= posQ;
        if (p.slidingWindow >= 0) allowed = allowed && ((posQ - posK) < p.slidingWindow);

        float score = -INFINITY;
        if (allowed) {
            uint kbase = (((b * p.Nkv + hkv) * p.Sk) + key) * p.Hd;
            float acc = 0.0f;
            for (uint d = 0; d < p.Hd; ++d) {
                acc += popcorn_load(Q, qbase + d) * popcorn_load(K, kbase + d);
            }
            score = acc * p.scale;
        }
        if (key < 1024) scores_cache[key] = score;
        local_max = max(local_max, score);
    }

    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = partial[0];

    float local_sum = 0.0f;
    for (uint key = tid; key < p.Sk; key += tg_size) {
        float score;
        if (key < 1024) {
            score = scores_cache[key];
        } else {
            int posK = int(key);
            bool allowed = posK <= posQ;
            if (p.slidingWindow >= 0) allowed = allowed && ((posQ - posK) < p.slidingWindow);
            score = -INFINITY;
            if (allowed) {
                uint kbase = (((b * p.Nkv + hkv) * p.Sk) + key) * p.Hd;
                float acc = 0.0f;
                for (uint d = 0; d < p.Hd; ++d) {
                    acc += popcorn_load(Q, qbase + d) * popcorn_load(K, kbase + d);
                }
                score = acc * p.scale;
            }
        }
        local_sum += exp(score - row_max);
    }

    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / partial[0];

    for (uint key = tid; key < p.Sk; key += tg_size) {
        float score;
        if (key < 1024) {
            score = scores_cache[key];
        } else {
            int posK = int(key);
            bool allowed = posK <= posQ;
            if (p.slidingWindow >= 0) allowed = allowed && ((posQ - posK) < p.slidingWindow);
            score = -INFINITY;
            if (allowed) {
                uint kbase = (((b * p.Nkv + hkv) * p.Sk) + key) * p.Hd;
                float acc = 0.0f;
                for (uint d = 0; d < p.Hd; ++d) {
                    acc += popcorn_load(Q, qbase + d) * popcorn_load(K, kbase + d);
                }
                score = acc * p.scale;
            }
        }
        popcorn_store(probs, outbase + key, exp(score - row_max) * inv_sum);
    }
}

POPCORN_INSTANTIATE_KERNEL("attention_scores_softmax", attention_scores_softmax_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("attention_scores_softmax_bf16_to_f32", attention_scores_softmax_typed, ushort, ushort, float)
