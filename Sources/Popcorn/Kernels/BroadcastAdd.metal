#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename A, typename B, typename O>
kernel void broadcast_add_typed(
    device const A* a [[ buffer(0) ]],
    device const B* b [[ buffer(1) ]],
    device O* out [[ buffer(2) ]],
    constant BroadcastAddConstants& p [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= p.count) return;
    popcorn_store(out, id, popcorn_load(a, id) + popcorn_load(b, id % p.bCount));
}

POPCORN_INSTANTIATE_KERNEL("broadcast_add", broadcast_add_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("broadcast_add_bf16", broadcast_add_typed, ushort, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("broadcast_add_bf16_f32_to_f32", broadcast_add_typed, ushort, float, float)
