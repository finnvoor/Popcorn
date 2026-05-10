#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

kernel void indexed_matmul(
    device const float* X       [[ buffer(0) ]],   
    device const float* W       [[ buffer(1) ]],   
    device const int*   exp_idx [[ buffer(2) ]],   
    device float*       out     [[ buffer(3) ]],   
    constant IndexedMatmulConstants& p [[ buffer(4) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    uint n = tid.x;
    uint m = tid.y;
    if (n >= p.N || m >= p.M) return;

    int e = exp_idx[n];
    device const float* wexp = W + uint(e) * p.K * p.M;
    device const float* xrow = X + n * p.K;

    float acc = 0.0f;
    if (p.transposeW != 0) {
        device const float* wrow = wexp + m * p.K;
        for (uint k = 0; k < p.K; ++k) {
            acc += xrow[k] * wrow[k];
        }
    } else {
        for (uint k = 0; k < p.K; ++k) {
            acc += xrow[k] * wexp[k * p.M + m];
        }
    }
    out[n * p.M + m] = acc;
}
