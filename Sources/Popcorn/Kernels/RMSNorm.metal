#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename X, typename W, typename O>
kernel void rmsnorm_typed(
    device const X* x [[ buffer(0) ]],
    device const W* weight [[ buffer(1) ]],
    device O* out [[ buffer(2) ]],
    constant RMSNormConstants& p [[ buffer(3) ]],
    uint row [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    threadgroup float partial[1024];
    uint base = row * p.H;

    float acc = 0.0f;
    for (uint i = tid; i < p.H; i += tg_size) {
        float v = popcorn_load(x, base + i);
        acc += v * v;
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(partial[0] / float(p.H) + p.eps);
    for (uint i = tid; i < p.H; i += tg_size) {
        float v = popcorn_load(x, base + i) * scale;
        if (p.hasWeight != 0) {
            float w = popcorn_load(weight, i);
            if (p.addOneToWeight != 0) w = 1.0f + w;
            v *= w;
        }
        popcorn_store(out, base + i, v);
    }
}

POPCORN_INSTANTIATE_KERNEL("rmsnorm", rmsnorm_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("rmsnorm_bf16", rmsnorm_typed, ushort, float, ushort)
POPCORN_INSTANTIATE_KERNEL("rmsnorm_f16", rmsnorm_typed, half, float, half)
