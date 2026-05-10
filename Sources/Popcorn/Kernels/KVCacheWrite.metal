#include <metal_stdlib>
#include "../../PopcornShaderTypes/PopcornKernelTypes.h"
#include "../PopcornDTypes.h"
using namespace metal;

template <typename S, typename C>
kernel void kv_cache_write_typed(
    device const S* src [[ buffer(0) ]],
    device C* cache [[ buffer(1) ]],
    constant KVCacheWriteConstants& p [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint total = p.B * p.Nkv * p.Snew * p.Hd;
    if (id >= total) return;

    uint d = id % p.Hd;
    uint rest = id / p.Hd;
    uint s = rest % p.Snew;
    rest = rest / p.Snew;
    uint h = rest % p.Nkv;
    uint b = rest / p.Nkv;

    uint dstS = p.offset + s;
    uint dstIdx = ((b * p.Nkv + h) * p.Smax + dstS) * p.Hd + d;
    popcorn_store(cache, dstIdx, popcorn_load(src, id));
}

POPCORN_INSTANTIATE_KERNEL("kv_cache_write", kv_cache_write_typed, float, float)
POPCORN_INSTANTIATE_KERNEL("kv_cache_write_bf16", kv_cache_write_typed, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("kv_cache_write_f32_to_bf16", kv_cache_write_typed, float, ushort)
