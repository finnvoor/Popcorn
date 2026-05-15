#include <metal_stdlib>
#include "../../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../../PopcornDTypes.h"
using namespace metal;

template <typename S, typename O, uint Bits, uint Group>
kernel void aq_embedding_gather_typed(
    device const int*      ids    [[ buffer(0) ]],
    device const uint32_t* W      [[ buffer(1) ]],
    device const S*        Scales [[ buffer(2) ]],
    device const S*        Biases [[ buffer(3) ]],
    device O*              out    [[ buffer(4) ]],
    constant AffineQEmbeddingGatherConstants& p [[ buffer(5) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    uint n = tid.x;
    uint h = tid.y;
    if (n >= p.T || h >= p.H) return;

    uint row = uint(ids[n]);
    uint g = h / Group;
    float scale = popcorn_load(Scales, row * p.kGroups + g);
    float bias = (p.hasBias != 0u) ? popcorn_load(Biases, row * p.kGroups + g) : 0.0f;
    uint q = popcorn_unpack_little_endian<uint32_t, Bits>(W + row * p.wordsPerRow, h);
    float v = float(q) * scale + bias;
    popcorn_store(out, n * p.H + h, v);
}

POPCORN_INSTANTIATE_KERNEL("aq_embedding_gather_bf16_bf16_b4_g64", aq_embedding_gather_typed, ushort, ushort, 4u, 64u)
POPCORN_INSTANTIATE_KERNEL("aq_embedding_gather_bf16_f32_b4_g64",  aq_embedding_gather_typed, ushort, float,  4u, 64u)
