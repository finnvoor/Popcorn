// FlashDecoding: split-KV variant of FlashAttention for decode (Sq=1).
//
// At decode the original FA launches only B*Nq threadgroups (8 for Gemma E2B),
// which under-utilizes Apple GPUs. We split each (b, hq) row across `P` KV
// partitions, run a partial-FA in B*Nq*P threadgroups, then a tiny reduce
// kernel merges the P partial softmax states into one output.
//
// Partial output layout (scratch buffer):
//   partial_O [B, Nq, P, Hd]  (float)
//   partial_m [B, Nq, P]      (float)
//   partial_l [B, Nq, P]      (float)
//
// Reference: Dao et al., "FlashDecoding", 2023. The merge step uses the same
// log-sum-exp combine used by FA across KV tiles, just with P pre-computed
// summaries instead of streaming.

#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

constant constexpr uint kFD_TG = 128;
constant constexpr uint kFD_Bc = 128;
constant constexpr uint kFD_MaxHd = 512;
constant constexpr uint kFD_SimdgroupCount = kFD_TG / 32;
constant constexpr uint kFD_SimdWidth = 32;

template <typename QType, typename KVType>
kernel void flash_decoding_partial_typed(
    device const QType* Q [[ buffer(0) ]],
    device const KVType* K [[ buffer(1) ]],
    device const KVType* V [[ buffer(2) ]],
    device float* partialO [[ buffer(3) ]],   // [B, Nq, P, Hd]
    device float* partialM [[ buffer(4) ]],   // [B, Nq, P]
    device float* partialL [[ buffer(5) ]],   // [B, Nq, P]
    constant FlashDecodingPartialConstants& p [[ buffer(6) ]],
    uint3 tgid3 [[ threadgroup_position_in_grid ]],
    uint3 tid3 [[ thread_position_in_threadgroup ]],
    uint sgid [[ simdgroup_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]]
) {
    uint tid = tid3.x;
    threadgroup float Q_sm[kFD_MaxHd];
    threadgroup float O_sm[kFD_MaxHd];
    threadgroup float S_sm[kFD_Bc];
    threadgroup float scratch[kFD_SimdgroupCount];
    threadgroup float m_l[2];

    uint b   = tgid3.x;
    uint hq  = tgid3.y;
    uint part = tgid3.z;
    if (b >= p.B || hq >= p.Nq || part >= p.P) return;

    uint Hd = p.Hd;
    uint Sk = p.Sk;
    uint hkv = hq * p.Nkv / p.Nq;
    uint qbase = ((b * p.Nq + hq)) * Hd; // Sq=1
    uint kv_head_base = (b * p.Nkv + hkv) * Sk;

    // Partition this (b, hq) of KV into P equal slices (last may be shorter).
    uint per = (Sk + p.P - 1u) / p.P;
    uint jLo = part * per;
    uint jHi = min(jLo + per, Sk);

    for (uint d = tid; d < Hd; d += kFD_TG) {
        Q_sm[d] = popcorn_load(Q, qbase + d) * p.scale;
        O_sm[d] = 0.0f;
    }
    if (tid == 0) { m_l[0] = -INFINITY; m_l[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // For Sq=1 the query corresponds to the last KV position (posQ = Sk-1).
    int posQ = int(Sk) - 1;

    // Quick exit if this partition has no allowed keys (entirely above posQ,
    // or outside sliding window). Saves time for partitions that start past
    // the causal frontier.
    if (jLo > uint(posQ)) {
        if (tid == 0) {
            partialM[(b * p.Nq + hq) * p.P + part] = -INFINITY;
            partialL[(b * p.Nq + hq) * p.P + part] = 0.0f;
        }
        for (uint d = tid; d < Hd; d += kFD_TG) {
            partialO[((b * p.Nq + hq) * p.P + part) * Hd + d] = 0.0f;
        }
        return;
    }

    for (uint jStart = jLo; jStart < jHi; jStart += kFD_Bc) {
        uint Bc_actual = min(kFD_Bc, jHi - jStart);

        for (uint sub = 0; sub < kFD_Bc; sub += kFD_SimdgroupCount) {
            uint i = sub + sgid;
            if (i >= Bc_actual) break;
            uint k = jStart + i;
            int posK = int(k);
            bool allowed = posK <= posQ;
            if (p.slidingWindow >= 0) allowed = allowed && ((posQ - posK) < p.slidingWindow);

            float acc = -INFINITY;
            if (allowed) {
                uint kbase = (kv_head_base + k) * Hd;
                float partial = 0.0f;
                for (uint d = lane; d < Hd; d += kFD_SimdWidth) {
                    partial += Q_sm[d] * popcorn_load(K, kbase + d);
                }
                acc = simd_sum(partial);
            }
            if (lane == 0) S_sm[i] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_max = -INFINITY;
        for (uint ii = tid; ii < Bc_actual; ii += kFD_TG) {
            local_max = max(local_max, S_sm[ii]);
        }
        local_max = simd_max(local_max);
        if (lane == 0) scratch[sgid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            float v = (lane < kFD_SimdgroupCount) ? scratch[lane] : -INFINITY;
            v = simd_max(v);
            if (lane == 0) {
                float m_old = m_l[0];
                float m_new = max(m_old, v);
                scratch[0] = m_new;
                scratch[1] = exp(m_old - m_new);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float m_new = scratch[0];
        float scale_old = scratch[1];

        float p_local = 0.0f;
        for (uint ii = tid; ii < Bc_actual; ii += kFD_TG) {
            float pv = exp(S_sm[ii] - m_new);
            S_sm[ii] = pv;
            p_local += pv;
        }
        p_local = simd_sum(p_local);
        if (lane == 0) scratch[sgid] = p_local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            float v = (lane < kFD_SimdgroupCount) ? scratch[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0) {
                m_l[1] = scale_old * m_l[1] + v;
                m_l[0] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < Hd; d += kFD_TG) {
            float acc = 0.0f;
            for (uint ii = 0; ii < Bc_actual; ++ii) {
                uint vbase = (kv_head_base + jStart + ii) * Hd;
                acc += S_sm[ii] * popcorn_load(V, vbase + d);
            }
            O_sm[d] = scale_old * O_sm[d] + acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial (m, l, O) — do NOT normalize by l here.
    if (tid == 0) {
        partialM[(b * p.Nq + hq) * p.P + part] = m_l[0];
        partialL[(b * p.Nq + hq) * p.P + part] = m_l[1];
    }
    uint outBase = ((b * p.Nq + hq) * p.P + part) * Hd;
    for (uint d = tid; d < Hd; d += kFD_TG) {
        partialO[outBase + d] = O_sm[d];
    }
}

POPCORN_INSTANTIATE_KERNEL("flash_decoding_partial_bf16", flash_decoding_partial_typed, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("flash_decoding_partial_f32",  flash_decoding_partial_typed, float,  float)

// Merge P partial (m, l, O) into the final O for each (b, hq).
// One threadgroup per (b, hq); threads parallelize across Hd.
template <typename OType>
kernel void flash_decoding_reduce_typed(
    device const float* partialO [[ buffer(0) ]],   // [B, Nq, P, Hd]
    device const float* partialM [[ buffer(1) ]],   // [B, Nq, P]
    device const float* partialL [[ buffer(2) ]],   // [B, Nq, P]
    device OType* O [[ buffer(3) ]],                // [B, Nq, 1, Hd]
    constant FlashDecodingReduceConstants& p [[ buffer(4) ]],
    uint3 tgid3 [[ threadgroup_position_in_grid ]],
    uint3 tid3 [[ thread_position_in_threadgroup ]]
) {
    uint tid = tid3.x;
    uint b = tgid3.x;
    uint hq = tgid3.y;
    if (b >= p.B || hq >= p.Nq) return;

    uint Hd = p.Hd;
    uint P = p.P;
    uint pmBase = (b * p.Nq + hq) * P;
    uint poBase = (b * p.Nq + hq) * P * Hd;
    uint outBase = (b * p.Nq + hq) * Hd;

    // 1. Find global max across partials.
    float m_global = -INFINITY;
    for (uint k = 0; k < P; ++k) {
        float m_k = partialM[pmBase + k];
        m_global = max(m_global, m_k);
    }

    // 2. Compute global l = sum_k exp(m_k - m_global) * l_k.
    float l_global = 0.0f;
    for (uint k = 0; k < P; ++k) {
        float m_k = partialM[pmBase + k];
        float l_k = partialL[pmBase + k];
        l_global += exp(m_k - m_global) * l_k;
    }
    float inv_l = 1.0f / l_global;

    // 3. For each output dim, combine partial O's.
    for (uint d = tid; d < Hd; d += kFD_TG) {
        float acc = 0.0f;
        for (uint k = 0; k < P; ++k) {
            float m_k = partialM[pmBase + k];
            float w = exp(m_k - m_global);
            acc += w * partialO[poBase + k * Hd + d];
        }
        popcorn_store(O, outBase + d, acc * inv_l);
    }
}

POPCORN_INSTANTIATE_KERNEL("flash_decoding_reduce_bf16", flash_decoding_reduce_typed, ushort)
POPCORN_INSTANTIATE_KERNEL("flash_decoding_reduce_f32",  flash_decoding_reduce_typed, float)
