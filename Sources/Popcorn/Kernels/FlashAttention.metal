// FlashAttention fused kernel.
//
// Computes O = softmax(scale * Q K^T + mask) V in a single pass without ever
// materializing the [Sq, Sk] attention matrix. One threadgroup is dispatched
// per (batch, query_head, query_row); the running softmax state (m, l) and the
// unnormalized output O live in threadgroup memory, and K/V are streamed from
// device memory in tiles of Bc rows. At the end of the KV loop the running
// stats are folded together with a single divide (the FA delayed-rescale).
//
// Reference: Dao, "FlashAttention", arXiv:2307.08691, Algorithm 1.
//
// Threadgroup layout:
//   TG threads = SG_COUNT * 32. We use TG=128 (= 4 simdgroups).
//   For QK^T the simdgroups parallelize across K rows of the Bc tile; the
//   32 lanes within a simdgroup parallelize the Hd-dimensional dot product
//   and combine via simd_sum.
//   For O += P @ V the threads parallelize across the head-dim Hd.

#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

constant constexpr uint kFA_TG = 128;
constant constexpr uint kFA_Bc = 128;
constant constexpr uint kFA_MaxHd = 512;
constant constexpr uint kFA_SimdgroupCount = kFA_TG / 32;
constant constexpr uint kFA_SimdWidth = 32;

template <typename QType, typename KVType, typename OType>
kernel void flash_attention_typed(
    device const QType* Q [[ buffer(0) ]],
    device const KVType* K [[ buffer(1) ]],
    device const KVType* V [[ buffer(2) ]],
    device OType* O [[ buffer(3) ]],
    constant FlashAttentionConstants& p [[ buffer(4) ]],
    uint3 tgid3 [[ threadgroup_position_in_grid ]],
    uint3 tid3 [[ thread_position_in_threadgroup ]],
    uint sgid [[ simdgroup_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]]
) {
    uint tid = tid3.x;
    threadgroup float Q_sm[kFA_MaxHd];
    threadgroup float O_sm[kFA_MaxHd];
    threadgroup float S_sm[kFA_Bc];
    threadgroup float scratch[kFA_SimdgroupCount];
    threadgroup float m_l[2];

    uint b  = tgid3.x;
    uint hq = tgid3.y;
    uint q  = tgid3.z;
    if (b >= p.B || hq >= p.Nq || q >= p.Sq) return;

    uint Hd = p.Hd;
    uint Sq = p.Sq;
    uint Sk = p.Sk;
    uint hkv = hq * p.Nkv / p.Nq;
    uint qbase = (((b * p.Nq + hq) * Sq) + q) * Hd;
    uint kv_head_base = (b * p.Nkv + hkv) * Sk;

    // Load Q*scale into threadgroup memory and zero the running O accumulator.
    for (uint d = tid; d < Hd; d += kFA_TG) {
        Q_sm[d] = popcorn_load(Q, qbase + d) * p.scale;
        O_sm[d] = 0.0f;
    }
    if (tid == 0) { m_l[0] = -INFINITY; m_l[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // posQ is the absolute KV-cache index of this query row. The query
    // appended last has posQ = Sk - 1; the first new query in this call has
    // posQ = Sk - Sq.
    int posQ = int(Sk) - int(Sq) + int(q);

    for (uint jStart = 0; jStart < Sk; jStart += kFA_Bc) {
        uint Bc_actual = min(kFA_Bc, Sk - jStart);

        // 1. S[i] = Q . K[jStart + i] (with mask).
        //    sgid = which K row inside this 4-row inner step.
        //    lane = which slice of the Hd dot product.
        for (uint sub = 0; sub < kFA_Bc; sub += kFA_SimdgroupCount) {
            uint i = sub + sgid;
            if (i >= Bc_actual) break;
            uint k = jStart + i;
            int posK = int(k);
            // maskKind: 0 = causal, 1 = causal + sliding window, 2 = bidirectional.
            bool allowed = (p.maskKind == 2u) || (posK <= posQ);
            if (p.maskKind == 1u) allowed = allowed && ((posQ - posK) < p.slidingWindow);

            float acc = -INFINITY;
            if (allowed) {
                uint kbase = (kv_head_base + k) * Hd;
                float partial = 0.0f;
                for (uint d = lane; d < Hd; d += kFA_SimdWidth) {
                    partial += Q_sm[d] * popcorn_load(K, kbase + d);
                }
                acc = simd_sum(partial);
            }
            if (lane == 0) S_sm[i] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Row max of S[0..Bc_actual) merged with running m.
        float local_max = -INFINITY;
        for (uint ii = tid; ii < Bc_actual; ii += kFA_TG) {
            local_max = max(local_max, S_sm[ii]);
        }
        local_max = simd_max(local_max);
        if (lane == 0) scratch[sgid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            float v = (lane < kFA_SimdgroupCount) ? scratch[lane] : -INFINITY;
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

        // 3. P[i] = exp(S[i] - m_new); accumulate row sum.
        float p_local = 0.0f;
        for (uint ii = tid; ii < Bc_actual; ii += kFA_TG) {
            float pv = exp(S_sm[ii] - m_new);
            S_sm[ii] = pv;
            p_local += pv;
        }
        p_local = simd_sum(p_local);
        if (lane == 0) scratch[sgid] = p_local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            float v = (lane < kFA_SimdgroupCount) ? scratch[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0) {
                m_l[1] = scale_old * m_l[1] + v;
                m_l[0] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. O[d] = scale_old * O[d] + sum_i P[i] * V[jStart + i, d]
        for (uint d = tid; d < Hd; d += kFA_TG) {
            float acc = 0.0f;
            for (uint ii = 0; ii < Bc_actual; ++ii) {
                uint vbase = (kv_head_base + jStart + ii) * Hd;
                acc += S_sm[ii] * popcorn_load(V, vbase + d);
            }
            O_sm[d] = scale_old * O_sm[d] + acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 5. Final rescale by 1/l and write to global memory.
    float inv_l = 1.0f / m_l[1];
    uint obase = (((b * p.Nq + hq) * Sq) + q) * Hd;
    for (uint d = tid; d < Hd; d += kFA_TG) {
        popcorn_store(O, obase + d, O_sm[d] * inv_l);
    }
}

POPCORN_INSTANTIATE_KERNEL("flash_attention",                 flash_attention_typed, float,  float,  float)
POPCORN_INSTANTIATE_KERNEL("flash_attention_bf16",            flash_attention_typed, ushort, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("flash_attention_bf16_to_f32",     flash_attention_typed, ushort, ushort, float)
POPCORN_INSTANTIATE_KERNEL("flash_attention_f32_bf16_to_bf16", flash_attention_typed, float,  ushort, ushort)
