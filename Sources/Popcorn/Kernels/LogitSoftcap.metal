#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename T>
kernel void logit_softcap_typed(
    device const T* x [[ buffer(0) ]],
    device T* out [[ buffer(1) ]],
    constant LogitSoftcapConstants& p [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= p.count) return;
    popcorn_store(out, id, tanh(popcorn_load(x, id) / p.cap) * p.cap);
}

POPCORN_INSTANTIATE_KERNEL("logit_softcap", logit_softcap_typed, float)
POPCORN_INSTANTIATE_KERNEL("logit_softcap_bf16", logit_softcap_typed, ushort)
