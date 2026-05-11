#pragma once

#ifdef __METAL_VERSION__
typedef unsigned int uint32_t;
typedef int          int32_t;
#else
#include <stdint.h>
#endif

typedef struct {
    uint32_t N;
    uint32_t H;
} EmbeddingGatherConstants;

typedef struct {
    uint32_t count;
    float scalar;
} ScalarMulConstants;

typedef struct {
    uint32_t count;
} MulConstants;

typedef struct {
    uint32_t count;
    uint32_t bCount;
} BroadcastAddConstants;

typedef struct {
    uint32_t count;
} GeluTanhConstants;

typedef struct {
    uint32_t count;
} BFloat16ToFloatConstants;

typedef struct {
    uint32_t count;
    float cap;
} LogitSoftcapConstants;

typedef struct {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t transposeB;
} MatmulConstants;

typedef struct {
    uint32_t K;
    uint32_t N;
    uint32_t transposeW;
} MatvecConstants;

typedef struct {
    uint32_t N;        
    uint32_t K;        
    uint32_t M;        
    uint32_t transposeW;
} IndexedMatmulConstants;

typedef struct {
    uint32_t rows;
    uint32_t K;        
    uint32_t H;        
} WeightedSumConstants;

typedef struct {
    uint32_t H;
    uint32_t hasWeight;
    uint32_t addOneToWeight;
    float eps;
} RMSNormConstants;

typedef struct {
    uint32_t N;
} SoftmaxConstants;

typedef struct {
    uint32_t Sq;
    uint32_t Sk;
    int32_t  slidingWindow;   
} AttentionMaskBuildConstants;

typedef struct {
    uint32_t B;
    uint32_t Nq;
    uint32_t Nkv;
    uint32_t Sq;
    uint32_t Sk;
    uint32_t Hd;
} AttentionScoresConstants;

typedef struct {
    uint32_t B;
    uint32_t Nq;
    uint32_t Nkv;
    uint32_t Sq;
    uint32_t Sk;
    uint32_t Hd;
    int32_t  slidingWindow;   
    float    scale;
} AttentionScoresSoftmaxConstants;

typedef struct {
    uint32_t B;
    uint32_t Nq;
    uint32_t Nkv;
    uint32_t Sq;
    uint32_t Sk;
    uint32_t Hd;
} AttentionOutputConstants;

typedef struct {
    uint32_t B;
    uint32_t Nkv;
    uint32_t Snew;
    uint32_t Smax;
    uint32_t Hd;
    uint32_t offset;
} KVCacheWriteConstants;

typedef struct {
    uint32_t T;
    uint32_t Hd2;
    float    scaling;
} RopeBuildCosSinConstants;

typedef struct {
    uint32_t B;
    uint32_t T;
    uint32_t Nh;
    uint32_t Hd2;
} RopeApplyConstants;

typedef struct {
    uint32_t rows;
    uint32_t E;
    uint32_t K;
} TopKConstants;

typedef struct {
    uint32_t count;
} GatherConstants;

typedef struct {
    uint32_t rows;
    uint32_t N;
} ArgmaxConstants;

typedef struct {
    uint32_t rowCount;
    uint32_t outColumnCount;
    uint32_t srcRowStride;
    uint32_t srcColumnOffset;
} Slice2DConstants;

typedef struct {
    uint32_t rowCount;
    uint32_t columnCount;
    uint32_t srcRowStride;
    uint32_t rowOffset;
} RowSlice2DConstants;

typedef struct {
    uint32_t D0;
    uint32_t D1;
    uint32_t D2;
    uint32_t D3;
} Transpose12Constants;

typedef struct {
    uint32_t B;
    uint32_t Nq;
    uint32_t Nkv;
    uint32_t Sk;
    uint32_t Hd;
    int32_t  slidingWindow;
    float    scale;
} AttentionDecodeFusedConstants;
