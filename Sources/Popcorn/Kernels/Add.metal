#include <metal_stdlib>
#include "../PopcornDTypes.h"
using namespace metal;

template <typename T>
kernel void add_typed(
    device const T* inA [[ buffer(0) ]],
    device const T* inB [[ buffer(1) ]],
    device T* out [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    popcorn_store(out, id, popcorn_load(inA, id) + popcorn_load(inB, id));
}

POPCORN_INSTANTIATE_KERNEL("add", add_typed, float)
POPCORN_INSTANTIATE_KERNEL("add_f16", add_typed, half)
POPCORN_INSTANTIATE_KERNEL("add_bf16", add_typed, ushort)
