#include <metal_stdlib>
#include "../../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../../PopcornDTypes.h"
using namespace metal;

// One simdgroup (32 lanes) cooperatively computes 4 contiguous output rows for
// a single m-row of X. Each lane walks `groupSize / 32` quantized weights per
// group; the 32-wide partial sums are reduced with simd_sum at the end.
template <typename X, typename S, typename O, uint Bits, uint Group>
kernel void aq_matvec_simd4_typed(
    device const uint32_t* W [[ buffer(0) ]],
    device const S* Scales   [[ buffer(1) ]],
    device const S* Biases   [[ buffer(2) ]],
    device const X* Xp       [[ buffer(3) ]],
    device O* Yp             [[ buffer(4) ]],
    constant AffineQMatmulConstants& p [[ buffer(5) ]],
    uint2 gid [[ thread_position_in_grid ]],
    ushort lane [[ thread_index_in_simdgroup ]]
) {
    constexpr uint Rows = 4u;

    uint block = gid.x >> 5;
    uint row0 = block * Rows;
    uint m = gid.y;
    if (row0 >= p.N || m >= p.M) return;

    const uint K = p.K;
    const uint kGroups = p.kGroups;
    const uint wordsPerRow = p.wordsPerRow;
    const bool hasBias = p.hasBias != 0u;

    float accs[Rows] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint g = 0; g < kGroups; ++g) {
        float scales_r[Rows] = {0.0f, 0.0f, 0.0f, 0.0f};
        float biases_r[Rows] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
            uint row = row0 + r;
            if (row < p.N) {
                scales_r[r] = popcorn_load(Scales, row * kGroups + g);
                biases_r[r] = hasBias ? popcorn_load(Biases, row * kGroups + g) : 0.0f;
            }
        }

        uint kStart = g * Group;
        for (uint i = lane; i < Group; i += 32u) {
            uint k = kStart + i;
            float xv = popcorn_load(Xp, m * K + k);
            #pragma unroll
            for (uint r = 0; r < Rows; ++r) {
                uint row = row0 + r;
                if (row < p.N) {
                    uint q = popcorn_unpack_little_endian<uint32_t, Bits>(W + row * wordsPerRow, k);
                    accs[r] += xv * (float(q) * scales_r[r] + biases_r[r]);
                }
            }
        }
    }

    #pragma unroll
    for (uint r = 0; r < Rows; ++r) {
        float s = simd_sum(accs[r]);
        uint row = row0 + r;
        if (lane == 0 && row < p.N) {
            popcorn_store(Yp, m * p.N + row, s);
        }
    }
}

POPCORN_INSTANTIATE_KERNEL("aq_matvec_simd4_bf16_bf16_bf16_b4_g64", aq_matvec_simd4_typed, ushort, ushort, ushort, 4u, 64u)
POPCORN_INSTANTIATE_KERNEL("aq_matvec_simd4_bf16_bf16_f32_b4_g64",  aq_matvec_simd4_typed, ushort, ushort, float,  4u, 64u)
POPCORN_INSTANTIATE_KERNEL("aq_matvec_simd4_f32_bf16_f32_b4_g64",   aq_matvec_simd4_typed, float,  ushort, float,  4u, 64u)

// Decode-specialized quantized matvec. This uses the same algebraic idea as
// high-performance affine quant kernels without sharing their implementation:
//
//   sum(x * (q * scale + bias)) == scale * sum(x * q) + bias * sum(x)
//
// Each lane owns 16 contiguous K values, so for groupSize=64 four adjacent
// lanes share one affine group. One threadgroup has two SIMD groups, and each
// SIMD group computes four output rows.
template <typename X, typename S, typename O, uint Bits, uint Group>
kernel void aq_qmv_fast_typed(
    device const uint32_t* W [[ buffer(0) ]],
    device const S* Scales   [[ buffer(1) ]],
    device const S* Biases   [[ buffer(2) ]],
    device const X* Xp       [[ buffer(3) ]],
    device O* Yp             [[ buffer(4) ]],
    constant AffineQMatmulConstants& p [[ buffer(5) ]],
    uint3 tg [[ threadgroup_position_in_grid ]],
    ushort simd [[ simdgroup_index_in_threadgroup ]],
    ushort lane [[ thread_index_in_simdgroup ]]
) {
    constexpr uint perWord = 32u / Bits;
    constexpr uint valuesPerLane = 16u;
    constexpr uint wordsPerLane = valuesPerLane / perWord;
    constexpr uint blockK = valuesPerLane * 32u;
    constexpr uint rowsPerSIMD = 4u;
    constexpr uint simdGroupsPerTG = 2u;

    static_assert(Bits == 4u, "aq_qmv_fast_typed currently specializes the 4-bit path");
    static_assert(Group == 64u, "aq_qmv_fast_typed currently specializes group size 64");
    static_assert(valuesPerLane % perWord == 0u, "valuesPerLane must align to packed words");

    uint m = tg.x;
    uint row0 = tg.y * (rowsPerSIMD * simdGroupsPerTG) + uint(simd) * rowsPerSIMD;
    if (m >= p.M || row0 >= p.N) return;

    const bool hasBias = p.hasBias != 0u;
    float acc[rowsPerSIMD] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint kBlock = 0; kBlock < p.K; kBlock += blockK) {
        uint laneK = kBlock + uint(lane) * valuesPerLane;
        uint group = laneK / Group;

        float xv[valuesPerLane];
        float sumX = 0.0f;
        #pragma unroll
        for (uint j = 0; j < valuesPerLane; ++j) {
            uint k = laneK + j;
            float v = (k < p.K) ? popcorn_load(Xp, m * p.K + k) : 0.0f;
            xv[j] = v;
            sumX += v;
        }

        uint wordBase = laneK / perWord;
        #pragma unroll
        for (uint r = 0; r < rowsPerSIMD; ++r) {
            uint row = row0 + r;
            if (row < p.N && laneK < p.K) {
                float scale = popcorn_load(Scales, row * p.kGroups + group);
                float bias = hasBias ? popcorn_load(Biases, row * p.kGroups + group) : 0.0f;
                float qdot = 0.0f;

                #pragma unroll
                for (uint wi = 0; wi < wordsPerLane; ++wi) {
                    uint packed = W[row * p.wordsPerRow + wordBase + wi];
                    #pragma unroll
                    for (uint nibble = 0; nibble < perWord; ++nibble) {
                        uint q = popcorn_unpack_from_word_little_endian<Bits>(packed, nibble);
                        qdot += xv[wi * perWord + nibble] * float(q);
                    }
                }

                acc[r] += scale * qdot + bias * sumX;
            }
        }
    }

    #pragma unroll
    for (uint r = 0; r < rowsPerSIMD; ++r) {
        float s = simd_sum(acc[r]);
        uint row = row0 + r;
        if (lane == 0 && row < p.N) {
            popcorn_store(Yp, m * p.N + row, s);
        }
    }
}

POPCORN_INSTANTIATE_KERNEL("aq_qmv_fast_bf16_bf16_bf16_b4_g64", aq_qmv_fast_typed, ushort, ushort, ushort, 4u, 64u)
POPCORN_INSTANTIATE_KERNEL("aq_qmv_fast_bf16_bf16_f32_b4_g64",  aq_qmv_fast_typed, ushort, ushort, float,  4u, 64u)
POPCORN_INSTANTIATE_KERNEL("aq_qmv_fast_f32_bf16_f32_b4_g64",   aq_qmv_fast_typed, float,  ushort, float,  4u, 64u)
