#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

// Single-pass fused attention kernel for the decode path (query seq len = 1).
// One threadgroup per (batch, q_head, q_pos) tuple. All BN=32 simdgroups in
// the TG cooperatively stream through the K/V cache: each simdgroup handles
// a different key position in a round-robin pattern, accumulating a partial
// online softmax + weighted V sum. The simdgroups then merge via threadgroup
// memory and write the final output row.
//
// Replaces the AttentionScoresSoftmax + AttentionOutput dispatch pair (and
// the intermediate `probs` materialization) with one streaming kernel,
// matching the structure of MLX's `sdpa_vector`.
//
// Template parameters:
//   D = head dim (Q/K/V). Must be a multiple of 32 (BD).

template <typename QT, typename KVT, int D>
[[kernel,
  max_total_threads_per_threadgroup(32 * 32)]]
void attention_decode_fused_typed(
    const device QT* queries [[ buffer(0) ]],
    const device KVT* keys    [[ buffer(1) ]],
    const device KVT* values  [[ buffer(2) ]],
    device QT* out            [[ buffer(3) ]],
    constant AttentionDecodeFusedConstants& p [[ buffer(4) ]],
    uint3 tid       [[ threadgroup_position_in_grid ]],
    uint  simd_gid  [[ simdgroup_index_in_threadgroup ]],
    uint  simd_lid  [[ thread_index_in_simdgroup ]]
) {
    constexpr int BN = 32;           // simdgroups per TG, one per key in a wave
    constexpr int BD = 32;           // simdgroup width (lanes)
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread  = D / BD;

    typedef float U;

    thread U q[qk_per_thread];
    thread U o[v_per_thread];

    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    // Decode (b, q_head) — q_seq_idx is implicit 0 (decode T=1).
    // tid.y encodes b * Nq + q_head; grid.x is the single 1024-thread TG.
    uint b_qhead = tid.y;
    uint b       = b_qhead / p.Nq;
    uint q_head  = b_qhead % p.Nq;
    uint kv_head = q_head * p.Nkv / p.Nq;

    // Per-head bases. Layout: Q is [B, Nq, T=1, Hd]; K, V are [B, Nkv, Sk, Hd].
    uint q_base = ((b * p.Nq + q_head) * p.Hd);
    uint kv_head_base = ((b * p.Nkv + kv_head) * p.Sk * p.Hd);

    queries += q_base + simd_lid * qk_per_thread;
    keys    += kv_head_base + simd_gid * p.Hd + simd_lid * qk_per_thread;
    values  += kv_head_base + simd_gid * p.Hd + simd_lid * v_per_thread;
    // After cross-simdgroup reduction the output write is keyed by simd_gid
    // (each simdgroup owns one chunk of v_per_thread outputs).
    out     += q_base + simd_gid * v_per_thread;

    // Load query (with scale baked in).
    for (int i = 0; i < qk_per_thread; ++i) {
        q[i] = U(p.scale) * popcorn_load(queries, uint(i));
    }
    for (int i = 0; i < v_per_thread; ++i) {
        o[i] = 0;
    }

    U max_score = -INFINITY;
    U sum_exp   = 0;

    // Sliding window: only keys with (posQ - k) < slidingWindow are allowed.
    // For decode, posQ = Sk - 1 (last position).
    int posQ = int(p.Sk) - 1;
    int sw   = p.slidingWindow;

    int key_stride = BN * int(p.Hd);

    for (int k = int(simd_gid); k < int(p.Sk); k += BN) {
        bool allowed = (k <= posQ);
        if (sw >= 0) allowed = allowed && ((posQ - k) < sw);
        if (allowed) {
            // QK dot.
            U score = 0;
            for (int i = 0; i < qk_per_thread; ++i) {
                score += q[i] * popcorn_load(keys, uint(i));
            }
            score = simd_sum(score);

            // Online softmax update.
            U new_max = max(max_score, score);
            U factor  = fast::exp(max_score - new_max);
            U exp_s   = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp   = sum_exp * factor + exp_s;

            for (int i = 0; i < v_per_thread; ++i) {
                o[i] = o[i] * factor + exp_s * popcorn_load(values, uint(i));
            }
        }
        keys   += key_stride;
        values += key_stride;
    }

    // Cross-simdgroup merge.
    if (simd_lid == 0) {
        max_scores[simd_gid]     = max_score;
        sum_exp_scores[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane reads ONE simdgroup's max/sum from shared memory. After the
    // global simd_max+simd_sum across lanes, every lane has the global
    // max/sum; `factor` is per-lane (factor for original-simdgroup-index ==
    // simd_lid).
    U other_max = max_scores[simd_lid];
    U new_max   = simd_max(other_max);
    U factor    = fast::exp(max_scores[simd_lid] - new_max);
    U new_sum   = simd_sum(sum_exp_scores[simd_lid] * factor);

    // Reduce per-output-element across simdgroups using the MLX transpose
    // trick: 32x32 partials are stored as outputs[lane, sg] and read as
    // outputs[sg, lane], then simd_sum across lanes combines partials from
    // all source simdgroups for one output position.
    for (int i = 0; i < v_per_thread; ++i) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        U v = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        if (new_sum != 0) v /= new_sum;
        if (simd_lid == 0) {
            popcorn_store(out, uint(i), v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define POPCORN_INSTANTIATE_ATTN_DECODE(NAME, Q, KV, D) \
    template [[host_name(NAME)]] [[kernel]] \
    decltype(attention_decode_fused_typed<Q, KV, D>) attention_decode_fused_typed<Q, KV, D>;

POPCORN_INSTANTIATE_ATTN_DECODE("attention_decode_fused_bf16_D256", ushort, ushort, 256)
POPCORN_INSTANTIATE_ATTN_DECODE("attention_decode_fused_bf16_D512", ushort, ushort, 512)
