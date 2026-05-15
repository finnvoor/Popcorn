#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename X, typename W, typename O>
kernel void layernorm_typed(
    device const X* x [[ buffer(0) ]],
    device const W* weight [[ buffer(1) ]],
    device const W* bias [[ buffer(2) ]],
    device O* out [[ buffer(3) ]],
    constant LayerNormConstants& p [[ buffer(4) ]],
    uint row [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    threadgroup float partial[1024];
    uint base = row * p.H;

    // Pass 1: mean
    float sum = 0.0f;
    for (uint i = tid; i < p.H; i += tg_size) sum += popcorn_load(x, base + i);
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial[0] / float(p.H);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: variance
    float vsum = 0.0f;
    for (uint i = tid; i < p.H; i += tg_size) {
        float d = popcorn_load(x, base + i) - mean;
        vsum += d * d;
    }
    partial[tid] = vsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float invStd = rsqrt(partial[0] / float(p.H) + p.eps);

    for (uint i = tid; i < p.H; i += tg_size) {
        float v = (popcorn_load(x, base + i) - mean) * invStd;
        if (p.hasWeight != 0) v *= popcorn_load(weight, i);
        if (p.hasBias != 0)   v += popcorn_load(bias, i);
        popcorn_store(out, base + i, v);
    }
}

POPCORN_INSTANTIATE_KERNEL("layernorm",      layernorm_typed, float,  float, float)
POPCORN_INSTANTIATE_KERNEL("layernorm_f16",  layernorm_typed, half,   float, half)
POPCORN_INSTANTIATE_KERNEL("layernorm_bf16", layernorm_typed, ushort, float, ushort)
