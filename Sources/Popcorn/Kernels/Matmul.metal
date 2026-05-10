#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename A, typename B, typename C>
kernel void matmul_typed(
    device const A* Ap [[ buffer(0) ]],
    device const B* Bp [[ buffer(1) ]],
    device C*       Cp [[ buffer(2) ]],
    constant MatmulConstants& p [[ buffer(3) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    uint m = tid.x;
    uint n = tid.y;
    if (m >= p.M || n >= p.N) return;

    float acc = 0.0f;
    uint arow = m * p.K;
    if (p.transposeB != 0) {
        uint brow = n * p.K;
        for (uint k = 0; k < p.K; ++k) acc += popcorn_load(Ap, arow + k) * popcorn_load(Bp, brow + k);
    } else {
        for (uint k = 0; k < p.K; ++k) acc += popcorn_load(Ap, arow + k) * popcorn_load(Bp, k * p.N + n);
    }
    popcorn_store(Cp, m * p.N + n, acc);
}

POPCORN_INSTANTIATE_KERNEL("matmul", matmul_typed, float, float, float)
POPCORN_INSTANTIATE_KERNEL("matmul_f16", matmul_typed, float, half, float)
POPCORN_INSTANTIATE_KERNEL("matmul_bf16", matmul_typed, float, ushort, float)
POPCORN_INSTANTIATE_KERNEL("matmul_bf16_bf16_f32", matmul_typed, ushort, ushort, float)
POPCORN_INSTANTIATE_KERNEL("matmul_bf16_bf16_bf16", matmul_typed, ushort, ushort, ushort)
