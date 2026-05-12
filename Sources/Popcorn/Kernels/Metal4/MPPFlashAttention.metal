// MPP variant of FlashAttention for prefill (Sq > 1).
//
// Same online-softmax algorithm as FlashAttention.metal, but the two inner
// matmuls (QK^T and P @ V) are done via `mpp::tensor_ops::matmul2d`. Each
// threadgroup handles a tile of Br consecutive query rows for one (b, hq).
//
// On current Apple silicon, MPP `matmul2d` requires **all** operand tensors
// (left, right, destination) to live in *device* memory. Threadgroup-memory
// operands cause a `static_assert(__assert_false_v<destinationValueType>)`
// at compile time. So Q, K, V, S, T all come from device buffers; the only
// threadgroup-memory state is the per-row softmax accumulators (m, l) and the
// running output `O_sm`, which is updated by hand rather than by MPP.
//
// Scale handling: rather than pre-scaling Q (which would require an extra
// device scratch buffer for scaled Q), we apply `p.scale` to the matmul
// output `S` element-wise after each QK^T tile. This is mathematically
// identical because softmax is invariant under a constant additive offset
// and we just absorb the scale into the score values before the row max.
//
// Per-threadgroup device scratch slices:
//   sScratch[b, hq, q_tile, Br, Bc]  f32   (QK^T result -> softmax P)
//   tScratch[b, hq, q_tile, Br, Hd]  f32   (P @ V result, added into O_sm)

#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp::tensor_ops;

constant constexpr int kMPPFA_Bc = 64;
constant constexpr int kMPPFA_SimdgroupCount = 4;
constant constexpr int kMPPFA_TG = kMPPFA_SimdgroupCount * 32;
constant constexpr int kMPPFA_SimdWidth = 32;

template <typename Q_T, typename KV_T, typename O_T, int Br, int MaxHd>
kernel void mpp_flash_attention_typed(
    // NOTE: MPP `matmul2d` matches `__is_same_v<leftValueType, bfloat>` etc.
    // without stripping const. Declaring these as `device const Q_T*` makes
    // the tensor's value_type `const Q_T`, which fails every triple in the
    // MPP dispatch and hits the catch-all `Unsupported type` static_assert.
    // Pass them as non-const `device Q_T*` to match MPPMatmul's pattern.
    device Q_T* Q [[ buffer(0) ]],
    device KV_T* K [[ buffer(1) ]],
    device KV_T* V [[ buffer(2) ]],
    device O_T* O [[ buffer(3) ]],
    device float* sScratch [[ buffer(4) ]],
    device float* tScratch [[ buffer(5) ]],
    constant MPPFlashAttentionConstants& p [[ buffer(6) ]],
    uint3 tgid3 [[ threadgroup_position_in_grid ]],
    uint3 tid3 [[ thread_position_in_threadgroup ]],
    uint sgid [[ simdgroup_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]]
) {
    threadgroup float O_sm[Br * MaxHd];
    threadgroup float m_sm[Br];
    threadgroup float l_sm[Br];
    threadgroup float reduce_sm[kMPPFA_SimdgroupCount];
    threadgroup float reduce_scratch[2];

    uint tid = tid3.x;

    uint b   = tgid3.x;
    uint hq  = tgid3.y;
    uint qTile = tgid3.z;
    if (b >= p.B || hq >= p.Nq) return;

    uint Hd = p.Hd;
    uint Sq = p.Sq;
    uint Sk = p.Sk;
    uint qBase = qTile * uint(Br);
    if (qBase >= Sq) return;
    uint qRows = min(uint(Br), Sq - qBase);

    uint hkv = hq * p.Nkv / p.Nq;
    uint q_dev_base = ((b * p.Nq + hq) * Sq + qBase) * Hd;
    uint k_dev_base = (b * p.Nkv + hkv) * Sk * Hd;
    uint v_dev_base = k_dev_base;
    uint o_dev_base = q_dev_base;

    // Per-threadgroup scratch offsets.
    uint sBase = ((b * p.Nq + hq) * p.qTilesPerHead + qTile) * uint(Br) * uint(kMPPFA_Bc);
    uint tBase = ((b * p.Nq + hq) * p.qTilesPerHead + qTile) * uint(Br) * uint(MaxHd);

    // Zero O.
    for (uint idx = tid; idx < uint(Br) * Hd; idx += kMPPFA_TG) {
        O_sm[idx] = 0.0f;
    }
    if (tid < uint(Br)) {
        m_sm[tid] = -INFINITY;
        l_sm[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc_qk = matmul2d_descriptor(
        Br, kMPPFA_Bc, static_cast<int>(metal::dynamic_extent),
        false, true, false
    );
    matmul2d<desc_qk, execution_simdgroups<kMPPFA_SimdgroupCount>> op_qk;

    constexpr auto desc_pv = matmul2d_descriptor(
        Br, MaxHd, static_cast<int>(metal::dynamic_extent),
        false, false, false
    );
    matmul2d<desc_pv, execution_simdgroups<kMPPFA_SimdgroupCount>> op_pv;

    // Q view: row-major [Sq, Hd] for this (b, hq); slice rows [qBase, qBase+Br).
    auto mQ = tensor(Q + ((b * p.Nq + hq) * Sq) * Hd,
                     dextents<int, 2>{int(Hd), int(Sq)},
                     array<int, 2>{1, int(Hd)});
    auto tQ = mQ.slice(0, int(qBase));

    auto mK = tensor(K + k_dev_base, dextents<int, 2>{int(Hd), int(Sk)}, array<int, 2>{1, int(Hd)});
    auto mV = tensor(V + v_dev_base, dextents<int, 2>{int(Hd), int(Sk)}, array<int, 2>{1, int(Hd)});

    auto tS = tensor(sScratch + sBase,
                     dextents<int, 2>{kMPPFA_Bc, Br},
                     array<int, 2>{1, kMPPFA_Bc});
    auto tT = tensor(tScratch + tBase,
                     dextents<int, 2>{MaxHd, Br},
                     array<int, 2>{1, MaxHd});

    int posQBase = int(Sk) - int(Sq) + int(qBase);

    for (uint jStart = 0; jStart < Sk; jStart += kMPPFA_Bc) {
        uint Bc_actual = min(uint(kMPPFA_Bc), Sk - jStart);

        // QK^T -> S.
        auto tK = mK.slice(0, int(jStart));
        op_qk.run(tQ, tK, tS);
        threadgroup_barrier(mem_flags::mem_device);

        // Apply scale + mask, run online softmax per query row.
        for (uint r = 0; r < uint(Br); ++r) {
            if (r >= qRows) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                continue;
            }
            int posQ = posQBase + int(r);
            device float* sRow = sScratch + sBase + r * uint(kMPPFA_Bc);

            // 1. Apply scale + mask. Zero trailing columns beyond Bc_actual.
            for (uint j = tid; j < uint(kMPPFA_Bc); j += kMPPFA_TG) {
                if (j < Bc_actual) {
                    int posK = int(jStart + j);
                    bool allowed = posK <= posQ;
                    if (p.slidingWindow >= 0) allowed = allowed && ((posQ - posK) < p.slidingWindow);
                    sRow[j] = allowed ? (sRow[j] * p.scale) : -INFINITY;
                } else {
                    sRow[j] = -INFINITY;
                }
            }
            threadgroup_barrier(mem_flags::mem_device);

            // 2. Row max -> m_new + scale_old.
            float local_max = -INFINITY;
            for (uint j = tid; j < Bc_actual; j += kMPPFA_TG) {
                local_max = max(local_max, sRow[j]);
            }
            local_max = simd_max(local_max);
            if (lane == 0) reduce_sm[sgid] = local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0) {
                float v = (lane < kMPPFA_SimdgroupCount) ? reduce_sm[lane] : -INFINITY;
                v = simd_max(v);
                if (lane == 0) {
                    float m_old = m_sm[r];
                    float m_new = max(m_old, v);
                    reduce_scratch[0] = m_new;
                    reduce_scratch[1] = exp(m_old - m_new);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float m_new = reduce_scratch[0];
            float scale_old = reduce_scratch[1];

            // 3. P = exp(S - m_new); scale O row by scale_old; accumulate sum.
            float p_local = 0.0f;
            for (uint j = tid; j < uint(kMPPFA_Bc); j += kMPPFA_TG) {
                float s = sRow[j];
                float pv = (j < Bc_actual) ? exp(s - m_new) : 0.0f;
                sRow[j] = pv;
                p_local += pv;
            }
            for (uint d = tid; d < Hd; d += kMPPFA_TG) {
                O_sm[r * MaxHd + d] *= scale_old;
            }
            p_local = simd_sum(p_local);
            if (lane == 0) reduce_sm[sgid] = p_local;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0) {
                float v = (lane < kMPPFA_SimdgroupCount) ? reduce_sm[lane] : 0.0f;
                v = simd_sum(v);
                if (lane == 0) {
                    l_sm[r] = scale_old * l_sm[r] + v;
                    m_sm[r] = m_new;
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }

        // P @ V -> T.
        auto tV = mV.slice(0, int(jStart));
        op_pv.run(tS, tV, tT);
        threadgroup_barrier(mem_flags::mem_device);

        // O += T.
        for (uint idx = tid; idx < qRows * Hd; idx += kMPPFA_TG) {
            uint r = idx / Hd;
            uint d = idx - r * Hd;
            O_sm[r * MaxHd + d] += tScratch[tBase + r * MaxHd + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final rescale and write.
    for (uint idx = tid; idx < qRows * Hd; idx += kMPPFA_TG) {
        uint r = idx / Hd;
        uint d = idx - r * Hd;
        float inv_l = 1.0f / l_sm[r];
        O[o_dev_base + r * Hd + d] = O_T(O_sm[r * MaxHd + d] * inv_l);
    }
}

#define POPCORN_MPP_FA_INSTANTIATE(NAME, Q_T, KV_T, O_T, BR, MAXHD)              \
    template [[host_name(NAME)]] [[kernel]]                                       \
    decltype(mpp_flash_attention_typed<Q_T, KV_T, O_T, BR, MAXHD>)               \
        mpp_flash_attention_typed<Q_T, KV_T, O_T, BR, MAXHD>;

POPCORN_MPP_FA_INSTANTIATE("mpp_flash_attention_bf16_hd256",       bfloat, bfloat, bfloat, 8, 256)
POPCORN_MPP_FA_INSTANTIATE("mpp_flash_attention_bf16_hd512",       bfloat, bfloat, bfloat, 8, 512)
POPCORN_MPP_FA_INSTANTIATE("mpp_flash_attention_bf16_to_f32_hd256",bfloat, bfloat, float,  8, 256)
POPCORN_MPP_FA_INSTANTIATE("mpp_flash_attention_bf16_to_f32_hd512",bfloat, bfloat, float,  8, 512)
POPCORN_MPP_FA_INSTANTIATE("mpp_flash_attention_f32_hd256",        float,  float,  float,  8, 256)
POPCORN_MPP_FA_INSTANTIATE("mpp_flash_attention_f32_hd512",        float,  float,  float,  8, 512)

#endif
