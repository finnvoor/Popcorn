#include <metal_stdlib>
using namespace metal;

typedef struct {
    uint K;
    uint N;
    uint transposeW;
} MatvecConstants;

#if defined(__HAVE_TENSOR__)

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp::tensor_ops;

constant constexpr int kMPPMatvecTileM = 8;
constant constexpr int kMPPMatvecTileN = 64;
constant constexpr int kMPPMatvecSimdgroups = 1;

template <typename X, typename W, typename O, bool TransposeRight>
kernel void mpp_matvec_typed(
    device X* Xp [[ buffer(0) ]],
    device W* Wp [[ buffer(1) ]],
    device O* Op [[ buffer(2) ]],
    constant MatvecConstants& p [[ buffer(3) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]]
) {
    constexpr auto desc = matmul2d_descriptor(
        kMPPMatvecTileM,
        kMPPMatvecTileN,
        static_cast<int>(metal::dynamic_extent),
         false,
         TransposeRight,
         false
    );
    matmul2d<desc, execution_simdgroups<kMPPMatvecSimdgroups>> op;

    int K = int(p.K);
    int N = int(p.N);
    int n_off = int(tgid.x) * kMPPMatvecTileN;

    auto mX = tensor(Xp, dextents<int, 2>{K, kMPPMatvecTileM}, array<int, 2>{1, 0});

    auto mO = tensor(Op, dextents<int, 2>{N, kMPPMatvecTileM}, array<int, 2>{1, 0});

    if (TransposeRight) {
        
        auto mW = tensor(Wp, dextents<int, 2>{K, N}, array<int, 2>{1, K});
        auto tX = mX.slice(0,     0);
        auto tW = mW.slice(0,     n_off);
        auto tO = mO.slice(n_off, 0);
        op.run(tX, tW, tO);
    } else {
        
        auto mW = tensor(Wp, dextents<int, 2>{N, K}, array<int, 2>{1, N});
        auto tX = mX.slice(0,     0);
        auto tW = mW.slice(n_off, 0);
        auto tO = mO.slice(n_off, 0);
        op.run(tX, tW, tO);
    }
}

#define POPCORN_MPP_MATVEC_INSTANTIATE(NAME, X, W, O, T)                                \
    template [[host_name(NAME)]] [[kernel]]                                             \
    decltype(mpp_matvec_typed<X, W, O, T>) mpp_matvec_typed<X, W, O, T>;

POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_f32_f32_f32",       float,  float,  float,  false)
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_f32_f32_f32_tb",    float,  float,  float,  true )
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_f16_f16_f16",       half,   half,   half,   false)
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_f16_f16_f16_tb",    half,   half,   half,   true )
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_f16_f16_f32",       half,   half,   float,  false)
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_f16_f16_f32_tb",    half,   half,   float,  true )
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_bf16_bf16_f32",     bfloat, bfloat, float,  false)
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_bf16_bf16_f32_tb",  bfloat, bfloat, float,  true )
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_bf16_bf16_bf16",    bfloat, bfloat, bfloat, false)
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_bf16_bf16_bf16_tb", bfloat, bfloat, bfloat, true )
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_f32_bf16_f32",      float,  bfloat, float,  false)
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_f32_bf16_f32_tb",   float,  bfloat, float,  true )
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_bf16_f32_f32",      bfloat, float,  float,  false)
POPCORN_MPP_MATVEC_INSTANTIATE("mpp_matvec_bf16_f32_f32_tb",   bfloat, float,  float,  true )

#endif 
