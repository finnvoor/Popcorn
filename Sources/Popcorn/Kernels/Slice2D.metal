#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename T>
kernel void slice2d_typed(
    device const T* src [[ buffer(0) ]],
    device T* out [[ buffer(1) ]],
    constant Slice2DConstants& p [[ buffer(2) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    uint r = tid.x;
    uint c = tid.y;
    if (r >= p.rowCount || c >= p.outColumnCount) return;
    popcorn_store(out, r * p.outColumnCount + c, popcorn_load(src, r * p.srcRowStride + p.srcColumnOffset + c));
}

POPCORN_INSTANTIATE_KERNEL("slice2d", slice2d_typed, float)
POPCORN_INSTANTIATE_KERNEL("slice2d_bf16", slice2d_typed, ushort)
POPCORN_INSTANTIATE_KERNEL("slice2d_f16", slice2d_typed, half)
