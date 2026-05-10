#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
using namespace metal;

kernel void attention_mask_build(
    device float* mask [[ buffer(0) ]],
    constant AttentionMaskBuildConstants& p [[ buffer(1) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    uint i = tid.x;
    uint j = tid.y;
    if (i >= p.Sq || j >= p.Sk) return;

    int posQ = int(p.Sk) - int(p.Sq) + int(i);
    int posK = int(j);
    bool allowed = (posK <= posQ);
    if (p.slidingWindow >= 0) {
        allowed = allowed && ((posQ - posK) < p.slidingWindow);
    }
    mask[i * p.Sk + j] = allowed ? 0.0f : -INFINITY;
}
