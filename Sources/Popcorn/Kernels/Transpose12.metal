#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename T>
kernel void transpose12_typed(
    device const T* src [[ buffer(0) ]],
    device T* out [[ buffer(1) ]],
    constant Transpose12Constants& p [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint total = p.D0 * p.D1 * p.D2 * p.D3;
    if (id >= total) return;

    uint d3 = id % p.D3;
    uint t = id / p.D3;
    uint d2 = t % p.D2;
    t = t / p.D2;
    uint d1 = t % p.D1;
    uint d0 = t / p.D1;

    uint outIdx = ((d0 * p.D2 + d2) * p.D1 + d1) * p.D3 + d3;
    popcorn_store(out, outIdx, popcorn_load(src, id));
}

POPCORN_INSTANTIATE_KERNEL("transpose12", transpose12_typed, float)
POPCORN_INSTANTIATE_KERNEL("transpose12_bf16", transpose12_typed, ushort)
POPCORN_INSTANTIATE_KERNEL("transpose12_f16", transpose12_typed, half)
