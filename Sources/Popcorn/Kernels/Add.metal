#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename T>
kernel void add_typed(
    device const T* inA [[ buffer(0) ]],
    device const T* inB [[ buffer(1) ]],
    device T* out [[ buffer(2) ]],
    constant AddConstants& p [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint base = id * 4u;
    if (base >= p.count) return;
    if (base + 4u <= p.count) {
        float4 va = popcorn_load4(inA, id);
        float4 vb = popcorn_load4(inB, id);
        popcorn_store4(out, id, va + vb);
    } else {
        for (uint i = base; i < p.count; ++i) {
            popcorn_store(out, i, popcorn_load(inA, i) + popcorn_load(inB, i));
        }
    }
}

POPCORN_INSTANTIATE_KERNEL("add", add_typed, float)
POPCORN_INSTANTIATE_KERNEL("add_f16", add_typed, half)
POPCORN_INSTANTIATE_KERNEL("add_bf16", add_typed, ushort)
