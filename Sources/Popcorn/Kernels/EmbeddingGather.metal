#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename T, typename O, typename Params>
kernel void embedding_gather_typed(
    device const int* ids [[ buffer(0) ]],
    device const T* table [[ buffer(1) ]],
    device O* out [[ buffer(2) ]],
    constant Params& p [[ buffer(3) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    uint n = tid.x;
    uint h = tid.y;
    if (n >= p.N || h >= p.H) return;
    uint row = uint(ids[n]);
    popcorn_store(out, n * p.H + h, popcorn_load(table, row * p.H + h));
}

POPCORN_INSTANTIATE_KERNEL("embedding_gather", embedding_gather_typed, float, float, EmbeddingGatherConstants)
POPCORN_INSTANTIATE_KERNEL("embedding_gather_f16", embedding_gather_typed, half, half, EmbeddingGatherConstants)
POPCORN_INSTANTIATE_KERNEL("embedding_gather_bf16", embedding_gather_typed, ushort, ushort, EmbeddingGatherConstants)
POPCORN_INSTANTIATE_KERNEL("embedding_gather_bf16_to_f32", embedding_gather_typed, ushort, float, EmbeddingGatherConstants)
