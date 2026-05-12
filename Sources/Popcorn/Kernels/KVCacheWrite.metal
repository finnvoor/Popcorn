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
    uint base = id * 4u;
    if (base >= total) return;

    // Decode the (b, h, s, d) of `base`.
    uint d = base % p.Hd;
    uint rest = base / p.Hd;
    uint s = rest % p.Snew;
    rest = rest / p.Snew;
    uint h = rest % p.Nkv;
    uint b = rest / p.Nkv;

    uint dstS = p.offset + s;
    uint dstBase = ((b * p.Nkv + h) * p.Smax + dstS) * p.Hd + d;

    // Fast path: 4-aligned within head dim, all 4 lanes valid.
    if (base + 4u <= total && d + 4u <= p.Hd) {
        float4 v = popcorn_load4(src, id);
        // dstBase is contiguous in d, but writes may not be 16B-aligned for the
        // float4 view; use the per-element store helper four times — the compiler
        // coalesces these into wider stores when alignment permits.
        // For best throughput, do a vector store when dstBase is aligned to 4.
        if ((dstBase & 3u) == 0u) {
            popcorn_store4(cache, dstBase >> 2, v);
        } else {
            popcorn_store(cache, dstBase + 0, v.x);
            popcorn_store(cache, dstBase + 1, v.y);
            popcorn_store(cache, dstBase + 2, v.z);
            popcorn_store(cache, dstBase + 3, v.w);
        }
        return;
    }

    // Scalar tail.
    uint limit = min(base + 4u, total);
    for (uint i = base; i < limit; ++i) {
        uint dd = i % p.Hd;
        uint rr = i / p.Hd;
        uint ss = rr % p.Snew;
        rr = rr / p.Snew;
        uint hh = rr % p.Nkv;
        uint bb = rr / p.Nkv;
        uint dstSi = p.offset + ss;
        uint dstIdx = ((bb * p.Nkv + hh) * p.Smax + dstSi) * p.Hd + dd;
        popcorn_store(cache, dstIdx, popcorn_load(src, i));
    }
}

POPCORN_INSTANTIATE_KERNEL("kv_cache_write", kv_cache_write_typed, float, float)
POPCORN_INSTANTIATE_KERNEL("kv_cache_write_bf16", kv_cache_write_typed, ushort, ushort)
POPCORN_INSTANTIATE_KERNEL("kv_cache_write_f32_to_bf16", kv_cache_write_typed, float, ushort)
