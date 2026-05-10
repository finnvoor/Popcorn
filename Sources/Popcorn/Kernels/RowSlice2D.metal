#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename T>
kernel void row_slice2d_typed(
    device const T* src [[ buffer(0) ]],
    device T* out [[ buffer(1) ]],
    constant RowSlice2DConstants& p [[ buffer(2) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    uint r = tid.x;
    uint c = tid.y;
    if (r >= p.rowCount || c >= p.columnCount) return;
    out[r * p.columnCount + c] = src[(p.rowOffset + r) * p.srcRowStride + c];
}

POPCORN_INSTANTIATE_KERNEL("row_slice2d", row_slice2d_typed, float)
POPCORN_INSTANTIATE_KERNEL("row_slice2d_f16", row_slice2d_typed, half)
POPCORN_INSTANTIATE_KERNEL("row_slice2d_bf16", row_slice2d_typed, ushort)
