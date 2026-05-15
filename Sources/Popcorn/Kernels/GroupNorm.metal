#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

// GroupNorm over [N, C, L]: per (n, g) compute mean/var across
// (C/groups) channels * L positions, then normalize + affine per channel.
// One threadgroup handles one (n, g) pair.
template <typename X, typename W, typename O>
kernel void groupnorm_typed(
    device const X* x      [[ buffer(0) ]],
    device const W* weight [[ buffer(1) ]],
    device const W* bias   [[ buffer(2) ]],
    device O*       out    [[ buffer(3) ]],
    constant GroupNormConstants& p [[ buffer(4) ]],
    uint3 pos [[ threadgroup_position_in_grid ]],
    uint3 tid3 [[ thread_position_in_threadgroup ]],
    uint3 tg_size3 [[ threads_per_threadgroup ]]
) {
    threadgroup float partial[1024];
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    uint n = pos.y;
    uint g = pos.x;
    uint cPerGroup = p.C / p.groups;
    uint cStart = g * cPerGroup;
    uint groupCount = cPerGroup * p.L;
    uint groupBase = n * p.C * p.L + cStart * p.L;

    // sum (init partial for *all* threads, not just those with work — the
    // tree reduction reads partial[tid+stride] even when tid+stride >= groupCount).
    float s = 0.0f;
    for (uint i = tid; i < groupCount; i += tg_size) s += popcorn_load(x, groupBase + i);
    partial[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial[0] / float(groupCount);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // var
    float v = 0.0f;
    for (uint i = tid; i < groupCount; i += tg_size) {
        float d = popcorn_load(x, groupBase + i) - mean;
        v += d * d;
    }
    partial[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float invStd = rsqrt(partial[0] / float(groupCount) + p.eps);

    for (uint i = tid; i < groupCount; i += tg_size) {
        uint cInGroup = i / p.L;          // 0..cPerGroup-1
        uint cAbs     = cStart + cInGroup;
        float nv = (popcorn_load(x, groupBase + i) - mean) * invStd;
        if (p.hasWeight != 0) nv *= popcorn_load(weight, cAbs);
        if (p.hasBias != 0)   nv += popcorn_load(bias, cAbs);
        popcorn_store(out, groupBase + i, nv);
    }
}

POPCORN_INSTANTIATE_KERNEL("groupnorm",      groupnorm_typed, float,  float, float)
POPCORN_INSTANTIATE_KERNEL("groupnorm_f16",  groupnorm_typed, half,   float, half)
POPCORN_INSTANTIATE_KERNEL("groupnorm_bf16", groupnorm_typed, ushort, float, ushort)
