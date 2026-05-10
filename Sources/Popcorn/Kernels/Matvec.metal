#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename X, typename W, typename O>
kernel void matvec_typed(
    device const X* x [[ buffer(0) ]],
    device const W* w [[ buffer(1) ]],
    device O* out [[ buffer(2) ]],
    constant MatvecConstants& p [[ buffer(3) ]],
    uint n [[ thread_position_in_grid ]]
) {
    if (n >= p.N) return;

    float acc = 0.0f;
    if (p.transposeW != 0) {
        
        uint base = n * p.K;
        for (uint k = 0; k < p.K; ++k) {
            acc += popcorn_load(x, k) * popcorn_load(w, base + k);
        }
    } else {
        
        for (uint k = 0; k < p.K; ++k) {
            acc += popcorn_load(x, k) * popcorn_load(w, k * p.N + n);
        }
    }
    popcorn_store(out, n, acc);
}

template <typename X, typename W, typename O, uint RowsPerThread = 1>
kernel void matvec_nk_simd_typed(
    device const X* x [[ buffer(0) ]],
    device const W* w [[ buffer(1) ]],
    device O* out [[ buffer(2) ]],
    constant MatvecConstants& p [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]],
    ushort lane [[ thread_index_in_simdgroup ]]
) {
    uint row = gid >> 5;
    if (row >= p.N) return;

    uint base = row * p.K;
    float acc = 0.0f;
    for (uint k = lane; k < p.K; k += 32) {
        acc += popcorn_load(x, k) * popcorn_load(w, base + k);
    }
    float sum = simd_sum(acc);
    if (lane == 0) {
        popcorn_store(out, row, sum);
    }
}

template <typename X, typename W, typename O>
kernel void matvec_nk_simd4_typed(
    device const X* x [[ buffer(0) ]],
    device const W* w [[ buffer(1) ]],
    device O* out [[ buffer(2) ]],
    constant MatvecConstants& p [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]],
    ushort lane [[ thread_index_in_simdgroup ]]
) {
    uint block = gid >> 5;
    uint row0 = block * 4;
    if (row0 >= p.N) return;

    const uint K = p.K;
    const bool row1_valid = row0 + 1 < p.N;
    const bool row2_valid = row0 + 2 < p.N;
    const bool row3_valid = row0 + 3 < p.N;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    if ((K & 3u) == 0u) {

        const uint K4 = K >> 2;
        const uint w4_base0 = (row0 + 0) * K4;
        const uint w4_base1 = (row0 + 1) * K4;
        const uint w4_base2 = (row0 + 2) * K4;
        const uint w4_base3 = (row0 + 3) * K4;

        float4 v0 = float4(0);
        float4 v1 = float4(0);
        float4 v2 = float4(0);
        float4 v3 = float4(0);

        for (uint k4 = lane; k4 < K4; k4 += 32) {
            float4 xv = popcorn_load4(x, k4);
            v0 += xv * popcorn_load4(w, w4_base0 + k4);
            if (row1_valid) v1 += xv * popcorn_load4(w, w4_base1 + k4);
            if (row2_valid) v2 += xv * popcorn_load4(w, w4_base2 + k4);
            if (row3_valid) v3 += xv * popcorn_load4(w, w4_base3 + k4);
        }

        acc0 = v0.x + v0.y + v0.z + v0.w;
        acc1 = v1.x + v1.y + v1.z + v1.w;
        acc2 = v2.x + v2.y + v2.z + v2.w;
        acc3 = v3.x + v3.y + v3.z + v3.w;
    } else {
        
        const uint w_base0 = (row0 + 0) * K;
        const uint w_base1 = (row0 + 1) * K;
        const uint w_base2 = (row0 + 2) * K;
        const uint w_base3 = (row0 + 3) * K;
        for (uint k = lane; k < K; k += 32) {
            float xv = popcorn_load(x, k);
            acc0 += xv * popcorn_load(w, w_base0 + k);
            if (row1_valid) acc1 += xv * popcorn_load(w, w_base1 + k);
            if (row2_valid) acc2 += xv * popcorn_load(w, w_base2 + k);
            if (row3_valid) acc3 += xv * popcorn_load(w, w_base3 + k);
        }
    }

    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    float sum2 = simd_sum(acc2);
    float sum3 = simd_sum(acc3);
    if (lane == 0) {
        popcorn_store(out, row0 + 0, sum0);
        if (row1_valid) popcorn_store(out, row0 + 1, sum1);
        if (row2_valid) popcorn_store(out, row0 + 2, sum2);
        if (row3_valid) popcorn_store(out, row0 + 3, sum3);
    }}

POPCORN_INSTANTIATE_KERNEL("matvec", matvec_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("matvec_f16", matvec_typed, float, half, float)
POPCORN_INSTANTIATE_KERNEL("matvec_bf16", matvec_typed, float, ushort, float)
POPCORN_INSTANTIATE_KERNEL("matvec_bf16_bf16_f32", matvec_typed, ushort, ushort, float)
POPCORN_INSTANTIATE_KERNEL("matvec_bf16_bf16_bf16", matvec_typed, ushort, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("matvec_nk_simd_f32_f32_f32", matvec_nk_simd4_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("matvec_nk_simd_f32_bf16_f32", matvec_nk_simd4_typed, float, ushort, float)
POPCORN_INSTANTIATE_KERNEL("matvec_nk_simd_bf16_bf16_f32", matvec_nk_simd4_typed, ushort, ushort, float)
POPCORN_INSTANTIATE_KERNEL("matvec_nk_simd_bf16_bf16_bf16", matvec_nk_simd4_typed, ushort, ushort, ushort)
