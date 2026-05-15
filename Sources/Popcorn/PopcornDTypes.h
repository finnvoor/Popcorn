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

inline void popcorn_store4(device float* p, uint i4, float4 v) {
    ((device float4*)p)[i4] = v;
}
inline void popcorn_store4(device half* p, uint i4, float4 v) {
    ((device half4*)p)[i4] = half4(v);
}
inline void popcorn_store4(device ushort* p, uint i4, float4 v) {
    ushort4 r;
    r.x = popcorn_float_to_bf16(v.x);
    r.y = popcorn_float_to_bf16(v.y);
    r.z = popcorn_float_to_bf16(v.z);
    r.w = popcorn_float_to_bf16(v.w);
    ((device ushort4*)p)[i4] = r;
}

template <typename Word, uint bits>
inline uint popcorn_unpack_little_endian(device const Word* packedValues, uint valueIndex) {
    constexpr uint storageBits = sizeof(Word) * 8u;
    static_assert(storageBits % bits == 0u, "bits must evenly divide the packed storage width");
    constexpr uint valuesPerWord = storageBits / bits;
    constexpr uint mask = (1u << bits) - 1u;
    Word packed = packedValues[valueIndex / valuesPerWord];
    return (uint(packed) >> ((valueIndex % valuesPerWord) * bits)) & mask;
}

template <uint bits>
inline uint popcorn_unpack_from_word_little_endian(uint packed, uint valueInWord) {
    constexpr uint mask = (1u << bits) - 1u;
    return (packed >> (valueInWord * bits)) & mask;
}

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
