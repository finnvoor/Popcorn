#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

// Vec4 fast-path: requires p.bCount % 4 == 0 and p.count % 4 == 0.
// Picked in Swift when both alignment conditions hold (the common case for
// per-channel bias adds where bCount equals a model dim that is a multiple of 4).
template <typename A, typename B, typename O>
kernel void broadcast_add_vec4_typed(
    device const A* a [[ buffer(0) ]],
    device const B* b [[ buffer(1) ]],
    device O* out [[ buffer(2) ]],
    constant BroadcastAddConstants& p [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint count4 = p.count >> 2;
    if (id >= count4) return;
    uint base = id << 2;
    float4 va = popcorn_load4(a, id);
    float4 vb = popcorn_load4(b, (base % p.bCount) >> 2);
    popcorn_store4(out, id, va + vb);
}

// Scalar fallback for arbitrary (count, bCount).
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
POPCORN_INSTANTIATE_KERNEL("broadcast_add_vec4", broadcast_add_vec4_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("broadcast_add_vec4_bf16", broadcast_add_vec4_typed, ushort, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("broadcast_add_vec4_bf16_f32_to_f32", broadcast_add_vec4_typed, ushort, float, float)
