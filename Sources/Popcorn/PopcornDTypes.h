#pragma once

#include <metal_stdlib>
using namespace metal;

#define POPCORN_INSTANTIATE_KERNEL(name, func, ...) \
    template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

inline float popcorn_load(device const float* p, uint i) { return p[i]; }
inline float popcorn_load(device const half* p, uint i) { return float(p[i]); }
inline float popcorn_load(device const ushort* p, uint i) { return as_type<float>(uint(p[i]) << 16); }
inline float popcorn_load(device const char* p, uint i) { return float(p[i]); }
inline float popcorn_load(device const uchar* p, uint i) { return float(p[i]); }

inline float4 popcorn_load4(device const float* p, uint i4) {
    return ((device const float4*)p)[i4];
}
inline float4 popcorn_load4(device const half* p, uint i4) {
    return float4(((device const half4*)p)[i4]);
}
inline float4 popcorn_load4(device const ushort* p, uint i4) {
    ushort4 s = ((device const ushort4*)p)[i4];
    return as_type<float4>(uint4(s) << 16);
}

inline half popcorn_store_half(float x) { return half(x); }
inline float popcorn_store_float(float x) { return x; }

inline ushort popcorn_float_to_bf16(float x) {
    uint bits = as_type<uint>(x);
    uint lsb = (bits >> 16) & 1u;
    uint roundingBias = 0x7fffu + lsb;
    return ushort((bits + roundingBias) >> 16);
}

inline void popcorn_store(device float* p, uint i, float x) { p[i] = x; }
inline void popcorn_store(device half* p, uint i, float x) { p[i] = half(x); }
inline void popcorn_store(device ushort* p, uint i, float x) { p[i] = popcorn_float_to_bf16(x); }

template <uint bits>
inline int popcorn_load_packed_signed(device const uchar* p, uint i) {
    constexpr uint mask = (1u << bits) - 1u;
    constexpr uint perByte = 8u / bits;
    uint packed = p[i / perByte];
    uint u = (packed >> ((i % perByte) * bits)) & mask;
    return int(u) - int(1u << (bits - 1u));
}

template <uint bits>
inline float popcorn_load_quantized(device const uchar* values, device const float* scales, uint groupSize, uint i) {
    return float(popcorn_load_packed_signed<bits>(values, i)) * scales[i / groupSize];
}
