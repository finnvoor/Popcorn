#include <metal_stdlib>
using namespace metal;

typedef struct {
    uint M;
    uint K;
    uint N;
    uint transposeB;
} MatmulConstants;

#if defined(__HAVE_TENSOR__)

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp::tensor_ops;

constant constexpr int kMPPMatmulSimdgroups = 4;

template <typename A, typename B, typename C, bool TransposeRight, int TileM, int TileN>
kernel void mpp_matmul_typed(
    device A* Ap [[ buffer(0) ]],
    device B* Bp [[ buffer(1) ]],
    device C* Cp [[ buffer(2) ]],
    constant MatmulConstants& p [[ buffer(3) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]]
) {
    constexpr auto desc = matmul2d_descriptor(
        TileM,
        TileN,
        static_cast<int>(metal::dynamic_extent),
         false,
         TransposeRight,
         false
    );
    matmul2d<desc, execution_simdgroups<kMPPMatmulSimdgroups>> op;

    int M = int(p.M);
    int K = int(p.K);
    int N = int(p.N);

    int m_off = int(tgid.y) * TileM;
    int n_off = int(tgid.x) * TileN;

    auto mA = tensor(Ap, dextents<int, 2>{K, M}, array<int, 2>{1, K});

    auto mD = tensor(Cp, dextents<int, 2>{N, M}, array<int, 2>{1, N});

    if (TransposeRight) {

        auto mB = tensor(Bp, dextents<int, 2>{K, N}, array<int, 2>{1, K});
        auto tA = mA.slice(0,     m_off);
        auto tB = mB.slice(0,     n_off);
        auto tD = mD.slice(n_off, m_off);
        op.run(tA, tB, tD);
    } else {
        
        auto mB = tensor(Bp, dextents<int, 2>{N, K}, array<int, 2>{1, N});
        auto tA = mA.slice(0,     m_off);
        auto tB = mB.slice(n_off, 0    );
        auto tD = mD.slice(n_off, m_off);
        op.run(tA, tB, tD);
    }
}

#define POPCORN_MPP_INSTANTIATE_TILE(NAME, A, B, C, T, TM, TN)                                  \
    template [[host_name(NAME)]] [[kernel]]                                                     \
    decltype(mpp_matmul_typed<A, B, C, T, TM, TN>) mpp_matmul_typed<A, B, C, T, TM, TN>;

#define POPCORN_MPP_INSTANTIATE(NAME, A, B, C, T)                                               \
    POPCORN_MPP_INSTANTIATE_TILE(NAME, A, B, C, T, 64, 64)

#define POPCORN_MPP_INSTANTIATE_TILES(NAME, A, B, C, T)                                         \
    POPCORN_MPP_INSTANTIATE_TILE(NAME "_m8",  A, B, C, T,  8, 64)                               \
    POPCORN_MPP_INSTANTIATE_TILE(NAME "_m16", A, B, C, T, 16, 64)                               \
    POPCORN_MPP_INSTANTIATE_TILE(NAME "_m32", A, B, C, T, 32, 64)                               \
    POPCORN_MPP_INSTANTIATE_TILE(NAME,        A, B, C, T, 64, 64)

POPCORN_MPP_INSTANTIATE("mpp_matmul_f32_f32_f32",       float,  float,  float,  false)
POPCORN_MPP_INSTANTIATE("mpp_matmul_f32_f32_f32_tb",    float,  float,  float,  true )
POPCORN_MPP_INSTANTIATE("mpp_matmul_f16_f16_f16",       half,   half,   half,   false)
POPCORN_MPP_INSTANTIATE("mpp_matmul_f16_f16_f16_tb",    half,   half,   half,   true )
POPCORN_MPP_INSTANTIATE("mpp_matmul_f16_f16_f32",       half,   half,   float,  false)
POPCORN_MPP_INSTANTIATE("mpp_matmul_f16_f16_f32_tb",    half,   half,   float,  true )
POPCORN_MPP_INSTANTIATE("mpp_matmul_bf16_bf16_f32",     bfloat, bfloat, float,  false)
POPCORN_MPP_INSTANTIATE_TILES("mpp_matmul_bf16_bf16_f32_tb",  bfloat, bfloat, float,  true )
POPCORN_MPP_INSTANTIATE("mpp_matmul_bf16_bf16_bf16",    bfloat, bfloat, bfloat, false)
POPCORN_MPP_INSTANTIATE_TILES("mpp_matmul_bf16_bf16_bf16_tb", bfloat, bfloat, bfloat, true )
POPCORN_MPP_INSTANTIATE("mpp_matmul_f32_bf16_f32",      float,  bfloat, float,  false)
POPCORN_MPP_INSTANTIATE("mpp_matmul_f32_bf16_f32_tb",   float,  bfloat, float,  true )
POPCORN_MPP_INSTANTIATE("mpp_matmul_bf16_f32_f32",      bfloat, float,  float,  false)
POPCORN_MPP_INSTANTIATE("mpp_matmul_bf16_f32_f32_tb",   bfloat, float,  float,  true )

#endif 
