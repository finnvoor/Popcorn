#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename T>
kernel void mul_typed(
    device const T* a [[ buffer(0) ]],
    device const T* b [[ buffer(1) ]],
    device T* out [[ buffer(2) ]],
    constant MulConstants& p [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= p.count) return;
    popcorn_store(out, id, popcorn_load(a, id) * popcorn_load(b, id));
}

POPCORN_INSTANTIATE_KERNEL("mul", mul_typed, float)
POPCORN_INSTANTIATE_KERNEL("mul_f16", mul_typed, half)
POPCORN_INSTANTIATE_KERNEL("mul_bf16", mul_typed, ushort)
