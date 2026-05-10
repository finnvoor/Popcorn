#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename X, typename O>
kernel void softmax_typed(
    device const X* x [[ buffer(0) ]],
    device O* out [[ buffer(1) ]],
    constant SoftmaxConstants& p [[ buffer(2) ]],
    uint row [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    threadgroup float partial[1024];
    uint base = row * p.N;

    float local_max = -INFINITY;
    for (uint i = tid; i < p.N; i += tg_size) local_max = max(local_max, popcorn_load(x, base + i));
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = partial[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < p.N; i += tg_size) local_sum += exp(popcorn_load(x, base + i) - row_max);
    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / partial[0];
    for (uint i = tid; i < p.N; i += tg_size) popcorn_store(out, base + i, exp(popcorn_load(x, base + i) - row_max) * inv);
}

POPCORN_INSTANTIATE_KERNEL("softmax", softmax_typed, float, float)
POPCORN_INSTANTIATE_KERNEL("softmax_bf16", softmax_typed, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("softmax_bf16_to_f32", softmax_typed, ushort, float)
POPCORN_INSTANTIATE_KERNEL("softmax_f32_to_bf16", softmax_typed, float, ushort)
