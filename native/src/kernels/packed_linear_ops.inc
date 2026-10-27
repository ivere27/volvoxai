/* Shared CPU/ARM/WASM packed linear execution. Bounded activation tiles
 * reuse packed weights; SIMD128, SSE2 and NEON use exact integer reductions
 * for dynamic-I8 arithmetic, including groups with a short tail. */
#include "packed_linear.h"
#include <string.h>
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_PACKED_X86_AVX2 1
#define VX_PACKED_TARGET_AVX2 __attribute__((target("avx2")))
static inline int vx_packed_use_avx2(void) {
#if defined(VX_PACKED_FORCE_SSE2_TEST)
    if (VX_PACKED_FORCE_SSE2_TEST) return 0;
#endif
#if defined(VOLVOXAI_KERNEL_PLATFORM_H) && defined(VX_GEMM_F32_WASM_SIMD)
    return vx_kernel_platform()->has_avx2;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#endif
}
#else
#define VX_PACKED_X86_AVX2 0
#endif

/* Keep division and ties-to-even identical to the scalar arithmetic contract.
 * In particular, multiplying by a reciprocal can move half-way values across
 * a rounding boundary. The optional override is a private numerical control. */
#ifndef VX_PACKED_LINEAR_SIMD_QUANT
#define VX_PACKED_LINEAR_SIMD_QUANT 1
#endif
#ifndef VX_PACKED_LINEAR_GROUP32_UNROLL
#define VX_PACKED_LINEAR_GROUP32_UNROLL 1
#endif
#ifndef VX_PACKED_LINEAR_PARALLEL_MACS
#define VX_PACKED_LINEAR_PARALLEL_MACS 16384u
#endif

#if defined(__wasm_simd128__) && VX_PACKED_LINEAR_SIMD_QUANT
#define VX_PACKED_HAS_SIMD_QUANT 1
static int vx_packed_prepare_i8_wasm(const float* input,uint32_t k,
        unsigned char* prepared,int wide,float* scale) {
    v128_t maxima=wasm_f32x4_splat(0.0f),exponent=wasm_i32x4_splat(0x7f800000);
    uint32_t j=0;int finite=1;
    for(;j+4u<=k;j+=4u) {
        v128_t value=wasm_v128_load(input+j);
        if(wasm_v128_any_true(wasm_i32x4_eq(wasm_v128_and(value,exponent),exponent))) {
            finite=0;break;
        }
        maxima=wasm_f32x4_max(maxima,wasm_f32x4_abs(value));
    }
    float maximum=wasm_f32x4_extract_lane(maxima,0);
    float value=wasm_f32x4_extract_lane(maxima,1);if(value>maximum)maximum=value;
    value=wasm_f32x4_extract_lane(maxima,2);if(value>maximum)maximum=value;
    value=wasm_f32x4_extract_lane(maxima,3);if(value>maximum)maximum=value;
    for(;finite&&j<k;j++) {
        value=input[j];if(!vx_packed_finite(value)){finite=0;break;}
        float absolute=value<0.0f?-value:value;if(absolute>maximum)maximum=absolute;
    }
    *scale=maximum==0.0f?1.0f:maximum/127.0f;
    if(*scale<0x1p-126f)*scale=0x1p-126f;
    if(!finite) {
        __builtin_memset(prepared,0,(size_t)k*(wide?2u:1u));
    } else {
        v128_t divisor=wasm_f32x4_splat(*scale),zero=wasm_i32x4_splat(0);
        j=0;
        for(;j+4u<=k;j+=4u) {
            v128_t q=wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(
                wasm_f32x4_div(wasm_v128_load(input+j),divisor)));
            q=wasm_i32x4_min(wasm_i32x4_max(q,wasm_i32x4_splat(-127)),wasm_i32x4_splat(127));
            v128_t pairs=wasm_i16x8_narrow_i32x4(q,zero);
            if(wide)wasm_v128_store64_lane(prepared+j*2u,pairs,0);
            else wasm_v128_store32_lane(prepared+j,wasm_i8x16_narrow_i16x8(pairs,zero),0);
        }
        for(;j<k;j++) {
            int16_t q=(int16_t)vx_packed_round_i8(input[j]/(*scale));
            if(wide)__builtin_memcpy(prepared+j*2u,&q,sizeof(q));
            else prepared[j]=(unsigned char)(int8_t)q;
        }
    }
    /* Pair loads include one padded lane for odd K. */
    if(wide&&(k&1u)){prepared[k*2u]=0;prepared[k*2u+1u]=0;}
    return finite;
}
#define vx_packed_prepare_i8_simd vx_packed_prepare_i8_wasm
#elif defined(__SSE2__) && VX_PACKED_LINEAR_SIMD_QUANT
#define VX_PACKED_HAS_SIMD_QUANT 1
#if VX_PACKED_X86_AVX2
static VX_PACKED_TARGET_AVX2 int vx_packed_prepare_i8_avx2(
        const float* input,uint32_t k,unsigned char* prepared,int wide,float* scale) {
    __m256 maxima=_mm256_setzero_ps();
    const __m256i exponent=_mm256_set1_epi32(0x7f800000);
    const __m256 abs_mask=_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    uint32_t j=0;int finite=1;
    for(;j+8u<=k;j+=8u) {
        __m256 value=_mm256_loadu_ps(input+j);
        if(_mm256_movemask_epi8(_mm256_cmpeq_epi32(
                _mm256_and_si256(_mm256_castps_si256(value),exponent),exponent))) {
            finite=0;break;
        }
        maxima=_mm256_max_ps(maxima,_mm256_and_ps(value,abs_mask));
    }
    __m128 max128=_mm_max_ps(_mm256_castps256_ps128(maxima),_mm256_extractf128_ps(maxima,1));
    __m128 shuf=_mm_shuffle_ps(max128,max128,_MM_SHUFFLE(2,3,0,1));
    max128=_mm_max_ps(max128,shuf);
    shuf=_mm_shuffle_ps(max128,max128,_MM_SHUFFLE(1,0,3,2));
    max128=_mm_max_ps(max128,shuf);
    float maximum=_mm_cvtss_f32(max128);
    for(;finite&&j<k;j++) {
        float value=input[j];if(!vx_packed_finite(value)){finite=0;break;}
        float absolute=value<0.0f?-value:value;if(absolute>maximum)maximum=absolute;
    }
    *scale=maximum==0.0f?1.0f:maximum/127.0f;
    if(*scale<0x1p-126f)*scale=0x1p-126f;
    if(!finite) {
        __builtin_memset(prepared,0,(size_t)k*(wide?2u:1u));
    } else {
        const __m256 divisor=_mm256_set1_ps(*scale);
        const __m256 c127=_mm256_set1_ps(127.0f),cm127=_mm256_set1_ps(-127.0f);
        const __m256 half=_mm256_set1_ps(0.5f),neg_half=_mm256_set1_ps(-0.5f);
        j=0;
        for(;j+8u<=k;j+=8u) {
            __m256 v=_mm256_div_ps(_mm256_loadu_ps(input+j),divisor);
            v=_mm256_max_ps(_mm256_min_ps(v,c127),cm127);
            __m256i integer=_mm256_cvttps_epi32(v);
            __m256 rem=_mm256_sub_ps(v,_mm256_cvtepi32_ps(integer));
            __m256i odd=_mm256_srai_epi32(_mm256_slli_epi32(integer,31),31);
            __m256i inc=_mm256_or_si256(_mm256_castps_si256(_mm256_cmp_ps(rem,half,_CMP_GT_OQ)),
                _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(rem,half,_CMP_EQ_OQ)),odd));
            __m256i dec=_mm256_or_si256(_mm256_castps_si256(_mm256_cmp_ps(rem,neg_half,_CMP_LT_OQ)),
                _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(rem,neg_half,_CMP_EQ_OQ)),odd));
            __m256i q=_mm256_add_epi32(_mm256_sub_epi32(integer,inc),dec);
            __m128i pairs=_mm_packs_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
            if(wide)_mm_storeu_si128((__m128i*)(prepared+j*2u),pairs);
            else _mm_storel_epi64((__m128i*)(prepared+j),_mm_packs_epi16(pairs,pairs));
        }
        for(;j<k;j++) {
            int16_t q=(int16_t)vx_packed_round_i8(input[j]/(*scale));
            if(wide)__builtin_memcpy(prepared+j*2u,&q,sizeof(q));
            else prepared[j]=(unsigned char)(int8_t)q;
        }
    }
    if(wide&&(k&1u)){prepared[k*2u]=0;prepared[k*2u+1u]=0;}
    return finite;
}
#endif
static int vx_packed_prepare_i8_x86(const float* input,uint32_t k,
        unsigned char* prepared,int wide,float* scale) {
#if VX_PACKED_X86_AVX2
    if(vx_packed_use_avx2())
        return vx_packed_prepare_i8_avx2(input,k,prepared,wide,scale);
#endif
    __m128 maxima=_mm_setzero_ps();
    const __m128i exponent=_mm_set1_epi32(0x7f800000);
    const __m128 abs_mask=_mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    uint32_t j=0;int finite=1;
    for(;j+4u<=k;j+=4u) {
        __m128 value=_mm_loadu_ps(input+j);
        if(_mm_movemask_epi8(_mm_cmpeq_epi32(
                _mm_and_si128(_mm_castps_si128(value),exponent),exponent))) {
            finite=0;break;
        }
        maxima=_mm_max_ps(maxima,_mm_and_ps(value,abs_mask));
    }
    __m128 shuf=_mm_shuffle_ps(maxima,maxima,_MM_SHUFFLE(2,3,0,1));
    maxima=_mm_max_ps(maxima,shuf);
    shuf=_mm_shuffle_ps(maxima,maxima,_MM_SHUFFLE(1,0,3,2));
    maxima=_mm_max_ps(maxima,shuf);
    float maximum=_mm_cvtss_f32(maxima);
    for(;finite&&j<k;j++) {
        float value=input[j];if(!vx_packed_finite(value)){finite=0;break;}
        float absolute=value<0.0f?-value:value;if(absolute>maximum)maximum=absolute;
    }
    *scale=maximum==0.0f?1.0f:maximum/127.0f;
    if(*scale<0x1p-126f)*scale=0x1p-126f;
    if(!finite) {
        __builtin_memset(prepared,0,(size_t)k*(wide?2u:1u));
    } else {
        const __m128 divisor=_mm_set1_ps(*scale);
        const __m128 c127=_mm_set1_ps(127.0f),cm127=_mm_set1_ps(-127.0f);
        const __m128 half=_mm_set1_ps(0.5f),neg_half=_mm_set1_ps(-0.5f);
        j=0;
        for(;j+4u<=k;j+=4u) {
            __m128 v=_mm_div_ps(_mm_loadu_ps(input+j),divisor);
            v=_mm_max_ps(_mm_min_ps(v,c127),cm127);
            __m128i integer=_mm_cvttps_epi32(v);
            __m128 rem=_mm_sub_ps(v,_mm_cvtepi32_ps(integer));
            __m128i odd=_mm_srai_epi32(_mm_slli_epi32(integer,31),31);
            __m128i inc=_mm_or_si128(_mm_castps_si128(_mm_cmpgt_ps(rem,half)),
                _mm_and_si128(_mm_castps_si128(_mm_cmpeq_ps(rem,half)),odd));
            __m128i dec=_mm_or_si128(_mm_castps_si128(_mm_cmplt_ps(rem,neg_half)),
                _mm_and_si128(_mm_castps_si128(_mm_cmpeq_ps(rem,neg_half)),odd));
            __m128i q=_mm_add_epi32(_mm_sub_epi32(integer,inc),dec);
            __m128i pairs=_mm_packs_epi32(q,q);
            if(wide)_mm_storel_epi64((__m128i*)(prepared+j*2u),pairs);
            else {
                int32_t packed4=_mm_cvtsi128_si32(_mm_packs_epi16(pairs,pairs));
                __builtin_memcpy(prepared+j,&packed4,sizeof(packed4));
            }
        }
        for(;j<k;j++) {
            int16_t q=(int16_t)vx_packed_round_i8(input[j]/(*scale));
            if(wide)__builtin_memcpy(prepared+j*2u,&q,sizeof(q));
            else prepared[j]=(unsigned char)(int8_t)q;
        }
    }
    if(wide&&(k&1u)){prepared[k*2u]=0;prepared[k*2u+1u]=0;}
    return finite;
}
#define vx_packed_prepare_i8_wasm vx_packed_prepare_i8_x86
#define vx_packed_prepare_i8_simd vx_packed_prepare_i8_x86
#else
#define VX_PACKED_HAS_SIMD_QUANT 0
#endif


/* Expand sixteen signed I2 codes from exactly one offset-binary word. */
#if defined(__wasm_simd128__)
static inline v128_t vx_packed_i2_bytes(uint32_t word) {
    v128_t b=wasm_i32x4_make((int32_t)word,0,0,0),mask=wasm_i8x16_splat(3);
    v128_t q0=wasm_v128_and(b,mask),q1=wasm_v128_and(wasm_u8x16_shr(b,2),mask);
    v128_t q2=wasm_v128_and(wasm_u8x16_shr(b,4),mask),q3=wasm_u8x16_shr(b,6);
    v128_t lo=wasm_i8x16_shuffle(q0,q1,0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
    v128_t hi=wasm_i8x16_shuffle(q2,q3,0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
    return wasm_i8x16_sub(wasm_i16x8_shuffle(lo,hi,0,8,1,9,2,10,3,11),wasm_i8x16_splat(2));
}
#elif defined(__SSE2__)
static inline __m128i vx_packed_i2_bytes(uint32_t word) {
    __m128i b=_mm_cvtsi32_si128((int32_t)word),mask=_mm_set1_epi8(3);
    __m128i q0=_mm_and_si128(b,mask),q1=_mm_and_si128(_mm_srli_epi16(b,2),mask);
    __m128i q2=_mm_and_si128(_mm_srli_epi16(b,4),mask),q3=_mm_and_si128(_mm_srli_epi16(b,6),mask);
    return _mm_sub_epi8(_mm_unpacklo_epi16(_mm_unpacklo_epi8(q0,q1),_mm_unpacklo_epi8(q2,q3)),_mm_set1_epi8(2));
}
#endif

/* I3 is a dense row bitstream. Group boundaries are multiples of 32,
 * hence every supported group starts on a word boundary. */
static inline uint32_t vx_packed_words(uint32_t k,uint32_t bits) {
    return (k / 32u) * bits + ((k % 32u) * bits + 31u) / 32u;
}
static inline uint32_t vx_packed_i3_code(const uint32_t* words,uint32_t k) {
    uint64_t bit=(uint64_t)k*3u;uint32_t shift=(uint32_t)(bit%32u);
    uint32_t value=words[bit/32u]>>shift;
    if(shift>29u)value|=words[bit/32u+1u]<<(32u-shift);
    return value&7u;
}

#if defined(__wasm_simd128__)
/* Eight dense I3 codes occupy 24 bits. The tile starts at a multiple of
 * eight codes, so the only word offsets are 0, 8, 16 and 24. Read a second
 * word only when those 24 bits cross its boundary, including short rows.
 * Shuffling overlapping byte pairs and multiplying by powers of two aligns
 * each code with bits 13..15 of its I16 lane without variable lane shifts. */
static inline v128_t vx_packed_i3_i16x8(const uint32_t* p,uint32_t bit_offset) {
    uint32_t codes=p[0]>>bit_offset;
    if(bit_offset>8u)codes|=p[1]<<(32u-bit_offset);
    v128_t bytes=wasm_i32x4_splat((int32_t)codes);
    v128_t pairs=wasm_i8x16_shuffle(bytes,bytes,0,1,0,1,0,1,1,2,1,2,1,2,2,3,2,3);
    return wasm_i16x8_sub(wasm_u16x8_shr(wasm_i16x8_mul(pairs,
        wasm_i16x8_make(8192,1024,128,4096,512,64,2048,256)),13),wasm_i16x8_splat(4));
}
#elif defined(__SSE2__)
static inline __m128i vx_packed_i3_i16x8(const uint32_t* p,uint32_t bit_offset) {
    uint32_t codes=p[0]>>bit_offset;
    if(bit_offset>8u)codes|=p[1]<<(32u-bit_offset);
    __m128i d=_mm_cvtsi32_si128((int32_t)codes);
    __m128i u=_mm_unpacklo_epi8(d,_mm_srli_epi32(d,8));
    __m128i pairs=_mm_unpacklo_epi64(
        _mm_shufflelo_epi16(u,_MM_SHUFFLE(1,0,0,0)),
        _mm_shufflelo_epi16(u,_MM_SHUFFLE(2,2,1,1)));
    __m128i scaled=_mm_mullo_epi16(pairs,_mm_setr_epi16(8192,1024,128,4096,512,64,2048,256));
    return _mm_sub_epi16(_mm_srli_epi16(scaled,13),_mm_set1_epi16(4));
}
#endif

#if VX_PACKED_X86_AVX2
static VX_PACKED_TARGET_AVX2 int32_t vx_packed_dot_i8_avx2(
        const int8_t* activation, const uint32_t* packed,
        uint32_t count, uint32_t weight_bits) {
    uint32_t k = 0;
    __m256i sum256 = _mm256_setzero_si256();
    if (weight_bits == 3u) {
        for (; k + 16u <= count; k += 16u) {
            __m128i w_lo = vx_packed_i3_i16x8(packed + k * 3u / 32u, (k * 3u) % 32u);
            __m128i w_hi = vx_packed_i3_i16x8(packed + (k + 8u) * 3u / 32u, ((k + 8u) * 3u) % 32u);
            __m256i w = _mm256_set_m128i(w_hi, w_lo);
            __m256i a = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(activation + k)));
            sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(a, w));
        }
        __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(sum256),
                                    _mm256_extracti128_si256(sum256, 1));
        if (k + 8u <= count) {
            __m128i w = vx_packed_i3_i16x8(packed + k * 3u / 32u, (k * 3u) % 32u);
            __m128i a = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(activation + k)));
            sum = _mm_add_epi32(sum, _mm_madd_epi16(a, w));
            k += 8u;
        }
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
        int32_t total = _mm_cvtsi128_si32(sum);
        for (; k < count; k++)
            total += (int32_t)activation[k] * ((int32_t)vx_packed_i3_code(packed, k) - 4);
        return total;
    }
    for (; k + 32u <= count; k += 32u) {
        __m128i w0, w1;
        if (weight_bits == 2u) {
            w0 = vx_packed_i2_bytes(packed[k / 16u]);
            w1 = vx_packed_i2_bytes(packed[k / 16u + 1u]);
        } else if (weight_bits == 4u) {
            __m128i bytes = _mm_loadu_si128((const __m128i*)(packed + k / 8u));
            __m128i mask = _mm_set1_epi8(15), eight = _mm_set1_epi8(8);
            __m128i lo = _mm_and_si128(bytes, mask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
            w0 = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), eight);
            w1 = _mm_sub_epi8(_mm_unpackhi_epi8(lo, hi), eight);
        } else {
            __m128i sign = _mm_set1_epi8((char)0x80);
            w0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(packed + k / 4u)), sign);
            w1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(packed + k / 4u + 4u)), sign);
        }
        __m256i a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(activation + k)));
        __m256i a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(activation + k + 16u)));
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(a0, _mm256_cvtepi8_epi16(w0)));
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(a1, _mm256_cvtepi8_epi16(w1)));
    }
    if (k + 16u <= count) {
        __m128i w;
        if (weight_bits == 2u) w = vx_packed_i2_bytes(packed[k / 16u]);
        else if (weight_bits == 4u) {
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(packed + k / 8u));
            __m128i mask = _mm_set1_epi8(15);
            __m128i lo = _mm_and_si128(bytes, mask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
            w = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), _mm_set1_epi8(8));
        } else {
            w = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(packed + k / 4u)),
                              _mm_set1_epi8((char)0x80));
        }
        __m256i a = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(activation + k)));
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(a, _mm256_cvtepi8_epi16(w)));
        k += 16u;
    }
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(sum256),
                                _mm256_extracti128_si256(sum256, 1));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    int32_t total = _mm_cvtsi128_si32(sum);
    if (weight_bits == 2u) {
        for (; k < count; k++)
            total += (int32_t)activation[k] * ((int32_t)((packed[k / 16u] >> ((k % 16u) * 2u)) & 3u) - 2);
    } else if (weight_bits == 4u) {
        for (; k < count; k++)
            total += (int32_t)activation[k] * ((int32_t)((packed[k / 8u] >> ((k % 8u) * 4u)) & 15u) - 8);
    } else {
        for (; k < count; k++)
            total += (int32_t)activation[k] * ((int32_t)((packed[k / 4u] >> ((k % 4u) * 8u)) & 255u) - 128);
    }
    return total;
}
#endif

static int32_t vx_packed_dot_i8_baseline(const int8_t* activation,
                                       const uint32_t* packed, uint32_t count,
                                       uint32_t weight_bits) {
    int32_t total=0;
    uint32_t k=0;
    if(weight_bits==3u) {
#if defined(__wasm_simd128__)
        v128_t sum=wasm_i32x4_splat(0);
        for(;k+8u<=count;k+=8u) {
            v128_t w=vx_packed_i3_i16x8(packed+k*3u/32u,(k*3u)%32u);
            v128_t a=wasm_i16x8_extend_low_i8x16(wasm_v128_load64_zero(activation+k));
            sum=wasm_i32x4_add(sum,wasm_i32x4_dot_i16x8(a,w));
        }
        total=wasm_i32x4_extract_lane(sum,0)+wasm_i32x4_extract_lane(sum,1)+
              wasm_i32x4_extract_lane(sum,2)+wasm_i32x4_extract_lane(sum,3);
#elif defined(__SSE2__)
        __m128i sum=_mm_setzero_si128(),zero=_mm_setzero_si128();
        for(;k+8u<=count;k+=8u) {
            __m128i w=vx_packed_i3_i16x8(packed+k*3u/32u,(k*3u)%32u);
            __m128i b=_mm_loadl_epi64((const __m128i*)(activation+k));
            __m128i a=_mm_unpacklo_epi8(b,_mm_cmpgt_epi8(zero,b));
            sum=_mm_add_epi32(sum,_mm_madd_epi16(a,w));
        }
        sum=_mm_add_epi32(sum,_mm_srli_si128(sum,8));
        sum=_mm_add_epi32(sum,_mm_srli_si128(sum,4));
        total=_mm_cvtsi128_si32(sum);
#endif
        for(;k<count;k++)total+=(int32_t)activation[k]*((int32_t)vx_packed_i3_code(packed,k)-4);
        return total;
    }
#if defined(__wasm_simd128__)
    v128_t sum=wasm_i32x4_splat(0);
    for(;k+16u<=count;k+=16u) {
        v128_t w;
        if(weight_bits==2u) w=vx_packed_i2_bytes(packed[k/16u]);
        else if(weight_bits==4u) {
            v128_t bytes=wasm_v128_load64_zero(packed+k/8u);
            v128_t lo=wasm_v128_and(bytes,wasm_i8x16_splat(15));
            v128_t hi=wasm_u8x16_shr(bytes,4);
            w=wasm_i8x16_shuffle(lo,hi,0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
            w=wasm_i8x16_sub(w,wasm_i8x16_splat(8));
        } else w=wasm_i8x16_sub(wasm_v128_load(packed+k/4u),wasm_i8x16_splat(-128));
        v128_t a=wasm_v128_load(activation+k);
        sum=wasm_i32x4_add(sum,wasm_i32x4_dot_i16x8(
            wasm_i16x8_extend_low_i8x16(a),wasm_i16x8_extend_low_i8x16(w)));
        sum=wasm_i32x4_add(sum,wasm_i32x4_dot_i16x8(
            wasm_i16x8_extend_high_i8x16(a),wasm_i16x8_extend_high_i8x16(w)));
    }
    total=wasm_i32x4_extract_lane(sum,0)+wasm_i32x4_extract_lane(sum,1)+
          wasm_i32x4_extract_lane(sum,2)+wasm_i32x4_extract_lane(sum,3);
#elif defined(__ARM_NEON)
    if(weight_bits==2u) {
        for(;k<count;k++)total+=(int32_t)activation[k]*((int32_t)((packed[k/16u]>>((k%16u)*2u))&3u)-2);
        return total;
    }
    int32x4_t sum=vdupq_n_s32(0);
    for(;k+16u<=count;k+=16u) {
        int8x16_t w;
        if(weight_bits==4u) {
            uint8x8_t bytes=vld1_u8((const uint8_t*)(packed+k/8u));
            uint8x8x2_t halves=vzip_u8(vand_u8(bytes,vdup_n_u8(15)),vshr_n_u8(bytes,4));
            w=vreinterpretq_s8_u8(vsubq_u8(vcombine_u8(halves.val[0],halves.val[1]),vdupq_n_u8(8)));
        } else w=vreinterpretq_s8_u8(vsubq_u8(vld1q_u8((const uint8_t*)(packed+k/4u)),vdupq_n_u8(128)));
        int8x16_t a=vld1q_s8(activation+k);
        sum=vpadalq_s16(sum,vmull_s8(vget_low_s8(a),vget_low_s8(w)));
        sum=vpadalq_s16(sum,vmull_s8(vget_high_s8(a),vget_high_s8(w)));
    }
    int32_t lanes[4];vst1q_s32(lanes,sum);
    total=lanes[0]+lanes[1]+lanes[2]+lanes[3];
#elif defined(__SSE2__)
    /* SSE2 is part of the x86-64 baseline. Widen signed bytes before pmaddwd;
     * saturated byte multiply/add would lose valid I8 group dot products. */
    __m128i sum = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();
    for (; k + 16u <= count; k += 16u) {
        __m128i w;
        if (weight_bits == 2u) w=vx_packed_i2_bytes(packed[k/16u]);
        else if (weight_bits == 4u) {
            __m128i bytes = _mm_loadl_epi64((const __m128i*)(packed + k / 8u));
            __m128i mask = _mm_set1_epi8(15);
            __m128i lo = _mm_and_si128(bytes, mask);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
            w = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), _mm_set1_epi8(8));
        } else {
            w = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(packed + k / 4u)),
                              _mm_set1_epi8((char)0x80));
        }
        __m128i a = _mm_loadu_si128((const __m128i*)(activation + k));
        __m128i as = _mm_cmpgt_epi8(zero, a), ws = _mm_cmpgt_epi8(zero, w);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpacklo_epi8(a, as),
                                               _mm_unpacklo_epi8(w, ws)));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpackhi_epi8(a, as),
                                               _mm_unpackhi_epi8(w, ws)));
    }
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    total = _mm_cvtsi128_si32(sum);
#endif
    /* Constant shifts keep the scalar fallback vectorizable without requiring
     * a host ISA with per-lane variable shifts. */
    if(weight_bits==2u) {
        for(;k<count;k++)total+=(int32_t)activation[k]*((int32_t)((packed[k/16u]>>((k%16u)*2u))&3u)-2);
    } else if(weight_bits==4u) {
        for(;k<count;k++)total+=(int32_t)activation[k]*
            ((int32_t)((packed[k/8u]>>((k%8u)*4u))&15u)-8);
    } else {
        for(;k<count;k++)total+=(int32_t)activation[k]*
            ((int32_t)((packed[k/4u]>>((k%4u)*8u))&255u)-128);
    }
    return total;
}

static inline int32_t vx_packed_dot_i8(const int8_t* activation,
        const uint32_t* packed,uint32_t count,uint32_t weight_bits) {
#if VX_PACKED_X86_AVX2
    /* Short W3/W4 groups cannot amortize the wider reduction. Four-row AVX2
     * tiles and longer groups retain their separate dispatch. */
    if ((count > 32u || (weight_bits != 3u && weight_bits != 4u)) &&
        vx_packed_use_avx2())
        return vx_packed_dot_i8_avx2(activation,packed,count,weight_bits);
#endif
    return vx_packed_dot_i8_baseline(activation,packed,count,weight_bits);
}

#if defined(VOLVOXAI_KERNEL_THREAD_POOL_H) && \
    (defined(VX_GEMM_F32_WASM_SIMD) || defined(VX_MAX_THREADS))
#define VX_PACKED_HAS_THREAD_POOL 1
#else
#define VX_PACKED_HAS_THREAD_POOL 0
#endif

static inline int vx_packed_linear_threads(uint64_t macs, uint32_t tasks) {
#if VX_PACKED_HAS_THREAD_POOL
    if (tasks > 1u && macs >= VX_PACKED_LINEAR_PARALLEL_MACS) {
        int threads = vx_kernels_thread_count();
        return threads > 1 ? threads : 1;
    }
#else
    (void)macs;
    (void)tasks;
#endif
    return 1;
}

/* A bounded 4-row by 4-column tile. K*sizeof(float) scratch already reserved
 * by the caller holds four I8 rows. Only eight weights per column are unpacked
 * at a time (sixteen for I2); the model never acquires a dequantized weight matrix. SIMD lanes
 * represent output columns, avoiding a horizontal reduction for every group.
 * Integer dots finish before each ascending F32 group-scale accumulation. */
#if defined(__wasm_simd128__) || defined(__SSE2__)
#if defined(__wasm_simd128__)
typedef v128_t VxPackedI8Vec;
#define VX_PI_ZERO() wasm_i32x4_splat(0)
#define VX_PI_ADD(a,b) wasm_i32x4_add(a,b)
#define VX_PI_DOT(a,b) wasm_i32x4_dot_i16x8(a,b)
#define VX_PI_PAIR(a,i) wasm_i32x4_shuffle(a,a,i,i,i,i)
#define VX_PI_LO32(a,b) wasm_i32x4_shuffle(a,b,0,4,1,5)
#define VX_PI_HI32(a,b) wasm_i32x4_shuffle(a,b,2,6,3,7)
#define VX_PI_LO64(a,b) wasm_i64x2_shuffle(a,b,0,2)
#define VX_PI_HI64(a,b) wasm_i64x2_shuffle(a,b,1,3)
#define VX_PI_F32(a,b,c,d) wasm_f32x4_make(a,b,c,d)
#define VX_PI_SPLAT(a) wasm_f32x4_splat(a)
#define VX_PI_FLOAT(a) wasm_f32x4_convert_i32x4(a)
#define VX_PI_FADD(a,b) wasm_f32x4_add(a,b)
#define VX_PI_FMUL(a,b) wasm_f32x4_mul(a,b)
#define VX_PI_STORE(p,a) wasm_v128_store(p,a)
#define VX_PI_LOAD(p) wasm_v128_load(p)
#define VX_PI_PAIR_LOAD(p) wasm_v128_load32_splat(p)
static inline VxPackedI8Vec vx_packed_load_i8x8(const int8_t* p) {
    return wasm_i16x8_extend_low_i8x16(wasm_v128_load64_zero(p));
}
static inline VxPackedI8Vec vx_packed_load_weight8(const uint32_t* p,uint32_t bits,uint32_t bit_offset) {
    v128_t w;
    if(bits==2u) w=vx_packed_i2_bytes(*p>>bit_offset);
    else if(bits==3u)return vx_packed_i3_i16x8(p,bit_offset);
    else if(bits==4u) {
        v128_t bytes=wasm_v128_load32_zero(p);
        v128_t lo=wasm_v128_and(bytes,wasm_i8x16_splat(15));
        v128_t hi=wasm_u8x16_shr(bytes,4);
        w=wasm_i8x16_sub(wasm_i8x16_shuffle(lo,hi,0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23),
                          wasm_i8x16_splat(8));
    } else w=wasm_i8x16_sub(wasm_v128_load64_zero(p),wasm_i8x16_splat(-128));
    return wasm_i16x8_extend_low_i8x16(w);
}
#else
typedef __m128i VxPackedI8Vec;
#define VX_PI_ZERO() _mm_setzero_si128()
#define VX_PI_ADD(a,b) _mm_add_epi32(a,b)
#define VX_PI_DOT(a,b) _mm_madd_epi16(a,b)
#define VX_PI_PAIR(a,i) _mm_shuffle_epi32(a,_MM_SHUFFLE(i,i,i,i))
#define VX_PI_LO32(a,b) _mm_unpacklo_epi32(a,b)
#define VX_PI_HI32(a,b) _mm_unpackhi_epi32(a,b)
#define VX_PI_LO64(a,b) _mm_unpacklo_epi64(a,b)
#define VX_PI_HI64(a,b) _mm_unpackhi_epi64(a,b)
#define VX_PI_F32(a,b,c,d) _mm_castps_si128(_mm_setr_ps(a,b,c,d))
#define VX_PI_SPLAT(a) _mm_castps_si128(_mm_set1_ps(a))
#define VX_PI_FLOAT(a) _mm_castps_si128(_mm_cvtepi32_ps(a))
#define VX_PI_FADD(a,b) _mm_castps_si128(_mm_add_ps(_mm_castsi128_ps(a),_mm_castsi128_ps(b)))
#define VX_PI_FMUL(a,b) _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(a),_mm_castsi128_ps(b)))
#define VX_PI_STORE(p,a) _mm_storeu_si128((__m128i*)(p),a)
#define VX_PI_LOAD(p) _mm_loadu_si128((const __m128i*)(p))
#define VX_PI_PAIR_LOAD(p) _mm_set1_epi32((int32_t)vx_packed_panel_load_pair(p))
static inline VxPackedI8Vec vx_packed_load_i8x8(const int8_t* p) {
    __m128i b=_mm_loadl_epi64((const __m128i*)p);
    return _mm_unpacklo_epi8(b,_mm_cmpgt_epi8(_mm_setzero_si128(),b));
}
static inline VxPackedI8Vec vx_packed_load_weight8(const uint32_t* p,uint32_t bits,uint32_t bit_offset) {
    __m128i w;
    if(bits==2u) w=vx_packed_i2_bytes(*p>>bit_offset);
    else if(bits==3u) return vx_packed_i3_i16x8(p,bit_offset);
    else if(bits==4u) {
        __m128i b=_mm_cvtsi32_si128((int)*p),mask=_mm_set1_epi8(15);
        w=_mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(b,mask),
            _mm_and_si128(_mm_srli_epi16(b,4),mask)),_mm_set1_epi8(8));
    } else w=_mm_xor_si128(_mm_loadl_epi64((const __m128i*)p),_mm_set1_epi8((char)0x80));
    return _mm_unpacklo_epi8(w,_mm_cmpgt_epi8(_mm_setzero_si128(),w));
}
#endif

typedef struct { VxPackedI8Vec row[4]; } VxPackedI8Tile;

static inline void vx_packed_i2_accumulate8(VxPackedI8Tile* dot,const int8_t* a,uint32_t stride,
        uint32_t j,VxPackedI8Vec w0,VxPackedI8Vec w1,VxPackedI8Vec w2,VxPackedI8Vec w3) {
    VxPackedI8Vec lo01=VX_PI_LO32(w0,w1),hi01=VX_PI_HI32(w0,w1);
    VxPackedI8Vec lo23=VX_PI_LO32(w2,w3),hi23=VX_PI_HI32(w2,w3);
    VxPackedI8Vec pair0=VX_PI_LO64(lo01,lo23),pair1=VX_PI_HI64(lo01,lo23);
    VxPackedI8Vec pair2=VX_PI_LO64(hi01,hi23),pair3=VX_PI_HI64(hi01,hi23);
#define VX_I2_ROW(r) do { \
    VxPackedI8Vec av=vx_packed_load_i8x8(a+(uint64_t)(r)*stride+j); \
    dot->row[r]=VX_PI_ADD(dot->row[r],VX_PI_DOT(VX_PI_PAIR(av,0),pair0)); \
    dot->row[r]=VX_PI_ADD(dot->row[r],VX_PI_DOT(VX_PI_PAIR(av,1),pair1)); \
    dot->row[r]=VX_PI_ADD(dot->row[r],VX_PI_DOT(VX_PI_PAIR(av,2),pair2)); \
    dot->row[r]=VX_PI_ADD(dot->row[r],VX_PI_DOT(VX_PI_PAIR(av,3),pair3)); \
} while(0)
    VX_I2_ROW(0);VX_I2_ROW(1);VX_I2_ROW(2);VX_I2_ROW(3);
#undef VX_I2_ROW
}
static inline VxPackedI8Vec vx_packed_extend_low(VxPackedI8Vec w) {
#if defined(__wasm_simd128__)
    return wasm_i16x8_extend_low_i8x16(w);
#else
    return _mm_unpacklo_epi8(w,_mm_cmpgt_epi8(_mm_setzero_si128(),w));
#endif
}
static inline VxPackedI8Vec vx_packed_extend_high(VxPackedI8Vec w) {
#if defined(__wasm_simd128__)
    return wasm_i16x8_extend_high_i8x16(w);
#else
    return _mm_unpackhi_epi8(w,_mm_cmpgt_epi8(_mm_setzero_si128(),w));
#endif
}
static inline VxPackedI8Tile vx_packed_group_i2_16(const int8_t* a,uint32_t stride,
        const uint32_t* w,uint32_t words,uint32_t length) {
    VxPackedI8Tile dot={{VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO()}};
    uint32_t j=0;
    for(;j+16u<=length;j+=16u) {
        const uint32_t* p=w+j/16u;
        VxPackedI8Vec b0=vx_packed_i2_bytes(p[0]),b1=vx_packed_i2_bytes(p[words]);
        VxPackedI8Vec b2=vx_packed_i2_bytes(p[2u*words]),b3=vx_packed_i2_bytes(p[3u*words]);
        vx_packed_i2_accumulate8(&dot,a,stride,j,
            vx_packed_extend_low(b0),vx_packed_extend_low(b1),vx_packed_extend_low(b2),vx_packed_extend_low(b3));
        vx_packed_i2_accumulate8(&dot,a,stride,j+8u,
            vx_packed_extend_high(b0),vx_packed_extend_high(b1),vx_packed_extend_high(b2),vx_packed_extend_high(b3));
    }
    if(j+8u<=length) {
        const uint32_t* p=w+j/16u;
        vx_packed_i2_accumulate8(&dot,a,stride,j,
            vx_packed_load_weight8(p,2u,0),vx_packed_load_weight8(p+words,2u,0),
            vx_packed_load_weight8(p+2u*words,2u,0),vx_packed_load_weight8(p+3u*words,2u,0));
        j+=8u;
    }
    if(j<length) {
        int32_t tail[4][4]={{0}};
        for(;j<length;j++)for(uint32_t out=0;out<4;out++) {
            int32_t code=(int32_t)((w[out*words+j/16u]>>((j%16u)*2u))&3u)-2;
            for(uint32_t r=0;r<4;r++)tail[r][out]+=(int32_t)a[(uint64_t)r*stride+j]*code;
        }
        for(uint32_t r=0;r<4;r++) {
#if defined(__wasm_simd128__)
            dot.row[r]=VX_PI_ADD(dot.row[r],wasm_v128_load(tail[r]));
#else
            dot.row[r]=VX_PI_ADD(dot.row[r],_mm_loadu_si128((const __m128i*)tail[r]));
#endif
        }
    }
    return dot;
}

#if VX_PACKED_X86_AVX2
static VX_PACKED_TARGET_AVX2 VxPackedI8Tile vx_packed_group_i8_4x4_avx2(
        const int8_t* a,uint32_t stride,const uint32_t* w,
        uint32_t words,uint32_t length,uint32_t bits) {
    VxPackedI8Tile dot={{_mm_setzero_si128(),_mm_setzero_si128(),
                         _mm_setzero_si128(),_mm_setzero_si128()}};
    __m256i d0=_mm256_setzero_si256(),d1=_mm256_setzero_si256();
    __m256i d2=_mm256_setzero_si256(),d3=_mm256_setzero_si256();
    const __m256i shuf01=_mm256_setr_epi8(
        0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,
        4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7);
    const __m256i shuf23=_mm256_setr_epi8(
        8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11,
        12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15);
    uint32_t j=0,per_word=32u/bits;
    for(;j+8u<=length;j+=8u) {
        const uint32_t* p=w+j*bits/32u;
        uint32_t bit_offset=(j*bits)%32u;
        __m128i w0=vx_packed_load_weight8(p,bits,bit_offset);
        __m128i w1=vx_packed_load_weight8(p+words,bits,bit_offset);
        __m128i w2=vx_packed_load_weight8(p+2u*words,bits,bit_offset);
        __m128i w3=vx_packed_load_weight8(p+3u*words,bits,bit_offset);
        __m128i lo01=_mm_unpacklo_epi32(w0,w1),hi01=_mm_unpackhi_epi32(w0,w1);
        __m128i lo23=_mm_unpacklo_epi32(w2,w3),hi23=_mm_unpackhi_epi32(w2,w3);
        __m256i pair01=_mm256_set_m128i(_mm_unpackhi_epi64(lo01,lo23),_mm_unpacklo_epi64(lo01,lo23));
        __m256i pair23=_mm256_set_m128i(_mm_unpackhi_epi64(hi01,hi23),_mm_unpacklo_epi64(hi01,hi23));
#define VX_PI_AVX2_ROW(r, acc) do { \
        __m128i av=_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(a+(uint64_t)(r)*stride+j))); \
        __m256i av256=_mm256_broadcastsi128_si256(av); \
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(_mm256_shuffle_epi8(av256,shuf01),pair01)); \
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(_mm256_shuffle_epi8(av256,shuf23),pair23)); \
    } while(0)
        VX_PI_AVX2_ROW(0, d0);VX_PI_AVX2_ROW(1, d1);
        VX_PI_AVX2_ROW(2, d2);VX_PI_AVX2_ROW(3, d3);
#undef VX_PI_AVX2_ROW
    }
    dot.row[0]=_mm_add_epi32(_mm256_castsi256_si128(d0),_mm256_extracti128_si256(d0,1));
    dot.row[1]=_mm_add_epi32(_mm256_castsi256_si128(d1),_mm256_extracti128_si256(d1,1));
    dot.row[2]=_mm_add_epi32(_mm256_castsi256_si128(d2),_mm256_extracti128_si256(d2,1));
    dot.row[3]=_mm_add_epi32(_mm256_castsi256_si128(d3),_mm256_extracti128_si256(d3,1));
    if(j<length) {
        int32_t tail[4][4]={{0}};
        for(;j<length;j++)for(uint32_t out=0;out<4;out++) {
            int32_t code=(int32_t)(bits==3u?vx_packed_i3_code(w+out*words,j):
                (w[out*words+j/per_word]>>((j%per_word)*bits))&((1u<<bits)-1u))-(1<<(bits-1u));
            for(uint32_t r=0;r<4;r++)tail[r][out]+=(int32_t)a[(uint64_t)r*stride+j]*code;
        }
        for(uint32_t r=0;r<4;r++)
            dot.row[r]=_mm_add_epi32(dot.row[r],_mm_loadu_si128((const __m128i*)tail[r]));
    }
    return dot;
}
#endif

static inline VxPackedI8Tile vx_packed_group_i8_4x4(const int8_t* a,uint32_t stride,
        const uint32_t* w,uint32_t words,uint32_t length,uint32_t bits) {
    if(bits==2u)return vx_packed_group_i2_16(a,stride,w,words,length);
#if VX_PACKED_X86_AVX2
    if(vx_packed_use_avx2())
        return vx_packed_group_i8_4x4_avx2(a,stride,w,words,length,bits);
#endif
    VxPackedI8Tile dot={{VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO()}};
    uint32_t j=0,per_word=32u/bits;
    for(;j+8u<=length;j+=8u) {
        const uint32_t* p=w+j*bits/32u;
        uint32_t bit_offset=(j*bits)%32u;
        VxPackedI8Vec w0=vx_packed_load_weight8(p,bits,bit_offset),w1=vx_packed_load_weight8(p+words,bits,bit_offset);
        VxPackedI8Vec w2=vx_packed_load_weight8(p+2u*words,bits,bit_offset),w3=vx_packed_load_weight8(p+3u*words,bits,bit_offset);
        VxPackedI8Vec lo01=VX_PI_LO32(w0,w1),hi01=VX_PI_HI32(w0,w1);
        VxPackedI8Vec lo23=VX_PI_LO32(w2,w3),hi23=VX_PI_HI32(w2,w3);
        VxPackedI8Vec pair0=VX_PI_LO64(lo01,lo23),pair1=VX_PI_HI64(lo01,lo23);
        VxPackedI8Vec pair2=VX_PI_LO64(hi01,hi23),pair3=VX_PI_HI64(hi01,hi23);
#define VX_PI_ROW(r) do { \
        VxPackedI8Vec av=vx_packed_load_i8x8(a+(uint64_t)(r)*stride+j); \
        dot.row[r]=VX_PI_ADD(dot.row[r],VX_PI_DOT(VX_PI_PAIR(av,0),pair0)); \
        dot.row[r]=VX_PI_ADD(dot.row[r],VX_PI_DOT(VX_PI_PAIR(av,1),pair1)); \
        dot.row[r]=VX_PI_ADD(dot.row[r],VX_PI_DOT(VX_PI_PAIR(av,2),pair2)); \
        dot.row[r]=VX_PI_ADD(dot.row[r],VX_PI_DOT(VX_PI_PAIR(av,3),pair3)); \
    } while(0)
        VX_PI_ROW(0);VX_PI_ROW(1);VX_PI_ROW(2);VX_PI_ROW(3);
#undef VX_PI_ROW
    }
    if(j<length) {
        int32_t tail[4][4]={{0}};
        for(;j<length;j++)for(uint32_t out=0;out<4;out++) {
            int32_t code=(int32_t)(bits==3u?vx_packed_i3_code(w+out*words,j):
                (w[out*words+j/per_word]>>((j%per_word)*bits))&((1u<<bits)-1u))-(1<<(bits-1u));
            for(uint32_t r=0;r<4;r++)tail[r][out]+=(int32_t)a[(uint64_t)r*stride+j]*code;
        }
        for(uint32_t r=0;r<4;r++) {
#if defined(__wasm_simd128__)
            dot.row[r]=VX_PI_ADD(dot.row[r],wasm_v128_load(tail[r]));
#else
            dot.row[r]=VX_PI_ADD(dot.row[r],_mm_loadu_si128((const __m128i*)tail[r]));
#endif
        }
    }
    return dot;
}

typedef struct {
    const VxLLMOpArgs* a;
    const int8_t* prepared;
    const float* as;
    const int* finite;
    uint64_t row;
} VxPackedLinearRows4Ctx;

static void vx_packed_linear_i8_rows4_range(const VxPackedLinearRows4Ctx* ctx,
        uint32_t panel_begin,uint32_t panel_end) {
    const VxLLMOpArgs* a=ctx->a;
    uint32_t k=(uint32_t)a->key_dim,n=(uint32_t)a->value_dim,bits=a->groups?(uint32_t)a->groups:4u;
    uint32_t group_size=(uint32_t)a->rate;
    uint32_t words=vx_packed_words(k,bits),groups=(k+group_size-1u)/group_size;
    for(uint32_t panel_idx=panel_begin;panel_idx<panel_end;panel_idx++) {
        uint32_t out=panel_idx*4u;
        VxPackedI8Vec sum[4]={VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO()};
        for(uint32_t g=0;g<groups;g++) {
            uint32_t start=g*group_size,length=k-start<group_size?k-start:group_size;
            VxPackedI8Tile dot=vx_packed_group_i8_4x4(ctx->prepared+start,k,
                (const uint32_t*)a->x[1]+(uint64_t)out*words+(uint64_t)start*bits/32u,words,length,bits);
            const float* scale=a->x[2]+(uint64_t)out*groups+g;
            VxPackedI8Vec ws=VX_PI_F32(scale[0],scale[groups],scale[2u*groups],scale[3u*groups]);
            for(uint32_t r=0;r<4;r++)sum[r]=VX_PI_FADD(sum[r],
                VX_PI_FMUL(VX_PI_FLOAT(dot.row[r]),VX_PI_FMUL(VX_PI_SPLAT(ctx->as[r]),ws)));
        }
        VxPackedI8Vec bias=a->x[3]?VX_PI_F32(a->x[3][out],a->x[3][out+1],a->x[3][out+2],a->x[3][out+3]):VX_PI_ZERO();
        for(uint32_t r=0;r<4;r++) {
            VxPackedI8Vec value=ctx->finite[r]?VX_PI_FADD(sum[r],bias):VX_PI_SPLAT(__builtin_nanf(""));
            VX_PI_STORE(a->y[0]+(ctx->row+r)*n+out,value);
        }
    }
}

#if VX_PACKED_HAS_THREAD_POOL
static void vx_packed_linear_i8_rows4_worker(void* opaque,int begin,int end) {
    if(begin<end)
        vx_packed_linear_i8_rows4_range((const VxPackedLinearRows4Ctx*)opaque,
                                        (uint32_t)begin,(uint32_t)end);
}
#endif

static void vx_packed_linear_i8_rows4(const VxLLMOpArgs* a,float* scratch,uint64_t row) {
    uint32_t k=(uint32_t)a->key_dim,n=(uint32_t)a->value_dim,bits=a->groups?(uint32_t)a->groups:4u;
    uint32_t group_size=(uint32_t)a->rate;
    uint32_t words=vx_packed_words(k,bits),groups=(k+group_size-1u)/group_size;
    float as[4];int finite[4];int8_t* prepared=(int8_t*)scratch;
    for(uint32_t r=0;r<4;r++) {
#if VX_PACKED_HAS_SIMD_QUANT
        finite[r]=vx_packed_prepare_i8_simd(a->x[0]+(row+r)*k,k,
            (unsigned char*)prepared+(uint64_t)r*k,0,&as[r]);
#else
        const float* input=a->x[0]+(row+r)*k;float maximum=0.0f;finite[r]=1;
        for(uint32_t j=0;j<k;j++) {
            float value=input[j];
            if(!vx_packed_finite(value)){finite[r]=0;break;}
            float absolute=value<0.0f?-value:value;if(absolute>maximum)maximum=absolute;
        }
        as[r]=maximum==0.0f?1.0f:maximum/127.0f;
        if(as[r]<0x1p-126f)as[r]=0x1p-126f;
        for(uint32_t j=0;j<k;j++)prepared[(uint64_t)r*k+j]=finite[r]?
            (int8_t)vx_packed_round_i8(input[j]/as[r]):0;
#endif
    }
    uint32_t panel_count=n/4u;
    VxPackedLinearRows4Ctx ctx={a,prepared,as,finite,row};
    int threads=vx_packed_linear_threads((uint64_t)4u*k*(panel_count*4u),panel_count);
#if VX_PACKED_HAS_THREAD_POOL
    if(threads>1) {
        uint32_t target=(uint32_t)threads*4u;
        uint32_t grain=(panel_count+target-1u)/target;
        if(!grain)grain=1u;
        vx_kernels_parallel_for((int)panel_count,(int)grain,
                                vx_packed_linear_i8_rows4_worker,&ctx);
    } else
#endif
    {
        (void)threads;
        vx_packed_linear_i8_rows4_range(&ctx,0u,panel_count);
    }
    for(uint32_t out=panel_count*4u;out<n;out++)for(uint32_t r=0;r<4;r++) {
        float sum=0.0f;
        for(uint32_t g=0;g<groups;g++) {
            uint32_t start=g*group_size,length=k-start<group_size?k-start:group_size;
            int32_t dot=vx_packed_dot_i8(prepared+(uint64_t)r*k+start,
                (const uint32_t*)a->x[1]+(uint64_t)out*words+(uint64_t)start*bits/32u,length,bits);
            sum+=(float)dot*(as[r]*a->x[2][(uint64_t)out*groups+g]);
        }
        a->y[0][(row+r)*n+out]=finite[r]?sum+(a->x[3]?a->x[3][out]:0.0f):__builtin_nanf("");
    }
}

#if VX_PACKED_LINEAR_TILE_ROWS > 4 || VX_PACKED_LINEAR_TILE2
/* Scratch may be supplied as a declared float array by private callers. Byte
 * addressing and fixed-size builtin copies avoid changing that array's
 * effective type, including in freestanding builds. */
static inline uint32_t vx_packed_panel_load_pair(const void* source) {
    uint32_t value;__builtin_memcpy(&value,source,sizeof(value));return value;
}
static inline void vx_packed_panel_store_pair(void* destination,uint32_t value) {
    __builtin_memcpy(destination,&value,sizeof(value));
}

/* Decode sixteen I4 codes with one exact eight-byte load. The ordinary
 * four-row entry remains available as the unchanged small-work baseline. */
static inline VxPackedI8Vec vx_packed_i4_bytes16(const uint32_t* p) {
#if defined(__wasm_simd128__)
    v128_t bytes=wasm_v128_load64_zero(p);
    v128_t lo=wasm_v128_and(bytes,wasm_i8x16_splat(15));
    v128_t hi=wasm_u8x16_shr(bytes,4);
    return wasm_i8x16_sub(wasm_i8x16_shuffle(lo,hi,
        0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23),wasm_i8x16_splat(8));
#else
    __m128i bytes=_mm_loadl_epi64((const __m128i*)p),mask=_mm_set1_epi8(15);
    return _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(bytes,mask),
        _mm_and_si128(_mm_srli_epi16(bytes,4),mask)),_mm_set1_epi8(8));
#endif
}

static inline void vx_packed_panel_interleave8(unsigned char* panel,
        VxPackedI8Vec w0,VxPackedI8Vec w1,VxPackedI8Vec w2,VxPackedI8Vec w3) {
    VxPackedI8Vec lo01=VX_PI_LO32(w0,w1),hi01=VX_PI_HI32(w0,w1);
    VxPackedI8Vec lo23=VX_PI_LO32(w2,w3),hi23=VX_PI_HI32(w2,w3);
    VX_PI_STORE(panel,VX_PI_LO64(lo01,lo23));
    VX_PI_STORE(panel+16,VX_PI_HI64(lo01,lo23));
    VX_PI_STORE(panel+32,VX_PI_LO64(hi01,hi23));
    VX_PI_STORE(panel+48,VX_PI_HI64(hi01,hi23));
}

#if VX_PACKED_X86_AVX2
static VX_PACKED_TARGET_AVX2 void vx_packed_group_panel_i4_16_avx2(
        unsigned char* panel,const uint32_t* p,uint32_t words) {
    __m256i w0=_mm256_cvtepi8_epi16(vx_packed_i4_bytes16(p));
    __m256i w1=_mm256_cvtepi8_epi16(vx_packed_i4_bytes16(p+words));
    __m256i w2=_mm256_cvtepi8_epi16(vx_packed_i4_bytes16(p+2u*words));
    __m256i w3=_mm256_cvtepi8_epi16(vx_packed_i4_bytes16(p+3u*words));
    __m256i lo01=_mm256_unpacklo_epi32(w0,w1),hi01=_mm256_unpackhi_epi32(w0,w1);
    __m256i lo23=_mm256_unpacklo_epi32(w2,w3),hi23=_mm256_unpackhi_epi32(w2,w3);
    __m256i p0=_mm256_unpacklo_epi64(lo01,lo23),p1=_mm256_unpackhi_epi64(lo01,lo23);
    __m256i p2=_mm256_unpacklo_epi64(hi01,hi23),p3=_mm256_unpackhi_epi64(hi01,hi23);
    _mm256_storeu_si256((__m256i*)(panel+0),_mm256_permute2x128_si256(p0,p1,0x20));
    _mm256_storeu_si256((__m256i*)(panel+32),_mm256_permute2x128_si256(p2,p3,0x20));
    _mm256_storeu_si256((__m256i*)(panel+64),_mm256_permute2x128_si256(p0,p1,0x31));
    _mm256_storeu_si256((__m256i*)(panel+96),_mm256_permute2x128_si256(p2,p3,0x31));
}
#endif

/* [K pairs, four output columns, two signed I16 values]. The largest panel
 * is 256*4*2 bytes regardless of M, N or the total weight-bank extent. */
static void vx_packed_group_panel(unsigned char* panel,const uint32_t* w,
        uint32_t words,uint32_t length,uint32_t bits) {
    uint32_t j=0;
#if VX_PACKED_X86_AVX2
    const int use_avx2=length>32u && vx_packed_use_avx2();
#endif
    if(bits==4u)for(;j+16u<=length;j+=16u) {
        const uint32_t* p=w+j/8u;
#if VX_PACKED_X86_AVX2
        if(use_avx2) {
            vx_packed_group_panel_i4_16_avx2(panel+j*8u,p,words);
            continue;
        }
#endif
        VxPackedI8Vec b0=vx_packed_i4_bytes16(p),b1=vx_packed_i4_bytes16(p+words);
        VxPackedI8Vec b2=vx_packed_i4_bytes16(p+2u*words),b3=vx_packed_i4_bytes16(p+3u*words);
        vx_packed_panel_interleave8(panel+j*8u,vx_packed_extend_low(b0),
            vx_packed_extend_low(b1),vx_packed_extend_low(b2),vx_packed_extend_low(b3));
        vx_packed_panel_interleave8(panel+(j+8u)*8u,vx_packed_extend_high(b0),
            vx_packed_extend_high(b1),vx_packed_extend_high(b2),vx_packed_extend_high(b3));
    }
    for(;j+8u<=length;j+=8u) {
        const uint32_t* p=w+j*bits/32u;uint32_t offset=j*bits%32u;
        vx_packed_panel_interleave8(panel+j*8u,
            vx_packed_load_weight8(p,bits,offset),vx_packed_load_weight8(p+words,bits,offset),
            vx_packed_load_weight8(p+2u*words,bits,offset),vx_packed_load_weight8(p+3u*words,bits,offset));
    }
    for(;j<length;j+=2u) {
        int16_t pair[8]={0};
        for(uint32_t out=0;out<4u;out++)for(uint32_t part=0;part<2u&&j+part<length;part++) {
            uint32_t position=j+part;
            uint32_t code=bits==3u?vx_packed_i3_code(w+out*words,position):
                (w[out*words+position/(32u/bits)]>>((position%(32u/bits))*bits))&((1u<<bits)-1u);
            pair[out*2u+part]=(int16_t)((int32_t)code-(1<<(bits-1u)));
        }
        VX_PI_STORE(panel+j*8u,VX_PI_LOAD(pair));
    }
}

#if (defined(__wasm_simd128__) || defined(__SSE2__)) && VX_PACKED_LINEAR_GROUP32_UNROLL
/* Complete W3/W4 group32 microtiles need neither a row-count branch for each
 * pair nor a sixteen-iteration loop. Keep four exact integer dots live and
 * unroll four adjacent pairs; F32 storage-group accumulation stays outside. */
static inline void vx_packed_group32_panel_4x4(const unsigned char* prepared,
        uint32_t stride,const unsigned char* panel,VxPackedI8Vec* dot) {
    const unsigned char* a0=prepared;
    const unsigned char* a1=a0+stride;
    const unsigned char* a2=a1+stride;
    const unsigned char* a3=a2+stride;
    VxPackedI8Vec d0=VX_PI_ZERO(),d1=VX_PI_ZERO(),d2=VX_PI_ZERO(),d3=VX_PI_ZERO();
#define VX_PI_PANEL_PAIR(i) do { \
    VxPackedI8Vec w=VX_PI_LOAD(panel+(i)*16u); \
    d0=VX_PI_ADD(d0,VX_PI_DOT(VX_PI_PAIR_LOAD(a0+(i)*4u),w)); \
    d1=VX_PI_ADD(d1,VX_PI_DOT(VX_PI_PAIR_LOAD(a1+(i)*4u),w)); \
    d2=VX_PI_ADD(d2,VX_PI_DOT(VX_PI_PAIR_LOAD(a2+(i)*4u),w)); \
    d3=VX_PI_ADD(d3,VX_PI_DOT(VX_PI_PAIR_LOAD(a3+(i)*4u),w)); \
} while(0)
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for(uint32_t pair=0;pair<16u;pair+=4u) {
        VX_PI_PANEL_PAIR(pair);VX_PI_PANEL_PAIR(pair+1u);
        VX_PI_PANEL_PAIR(pair+2u);VX_PI_PANEL_PAIR(pair+3u);
    }
#undef VX_PI_PANEL_PAIR
    dot[0]=d0;dot[1]=d1;dot[2]=d2;dot[3]=d3;
}
#if defined(__SSE2__)
/* A two-slot batch must not pay a variable-row loop for every input pair.
 * Read exactly its two prepared rows, including in a final two-row tile. */
static inline void vx_packed_group32_panel_2x4(const unsigned char* prepared,
        uint32_t stride,const unsigned char* panel,VxPackedI8Vec* dot) {
    const unsigned char* a0=prepared;
    const unsigned char* a1=a0+stride;
    VxPackedI8Vec d0=VX_PI_ZERO(),d1=VX_PI_ZERO();
#define VX_PI_PANEL_PAIR2(i) do { \
    VxPackedI8Vec w=VX_PI_LOAD(panel+(i)*16u); \
    d0=VX_PI_ADD(d0,VX_PI_DOT(VX_PI_PAIR_LOAD(a0+(i)*4u),w)); \
    d1=VX_PI_ADD(d1,VX_PI_DOT(VX_PI_PAIR_LOAD(a1+(i)*4u),w)); \
} while(0)
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for(uint32_t pair=0;pair<16u;pair+=4u) {
        VX_PI_PANEL_PAIR2(pair);VX_PI_PANEL_PAIR2(pair+1u);
        VX_PI_PANEL_PAIR2(pair+2u);VX_PI_PANEL_PAIR2(pair+3u);
    }
#undef VX_PI_PANEL_PAIR2
    dot[0]=d0;dot[1]=d1;
}
#endif
#endif

#if VX_PACKED_X86_AVX2
static VX_PACKED_TARGET_AVX2 void vx_packed_panel_accum_avx2(
        const unsigned char* prepared,uint32_t pairs,
        uint32_t start,uint32_t length,uint32_t base,uint32_t count,
        const unsigned char* panel,__m128i* dot) {
    const uint32_t total_pairs=(length+1u)/2u;
    const __m256i shuf=_mm256_setr_epi8(
        0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,
        4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7);
    uint32_t pair=0;
    if(count==4u) {
        const unsigned char* a0=prepared+((uint64_t)base*pairs+start/2u)*sizeof(uint32_t);
        const unsigned char* a1=a0+(uint64_t)pairs*sizeof(uint32_t);
        const unsigned char* a2=a1+(uint64_t)pairs*sizeof(uint32_t);
        const unsigned char* a3=a2+(uint64_t)pairs*sizeof(uint32_t);
        __m256i d0=_mm256_setzero_si256(),d1=_mm256_setzero_si256();
        __m256i d2=_mm256_setzero_si256(),d3=_mm256_setzero_si256();
        for(;pair+2u<=total_pairs;pair+=2u) {
            __m256i w=_mm256_loadu_si256((const __m256i*)(panel+pair*16u));
            int64_t p0,p1,p2,p3;
            __builtin_memcpy(&p0,a0+pair*4u,sizeof(p0));
            __builtin_memcpy(&p1,a1+pair*4u,sizeof(p1));
            __builtin_memcpy(&p2,a2+pair*4u,sizeof(p2));
            __builtin_memcpy(&p3,a3+pair*4u,sizeof(p3));
            d0=_mm256_add_epi32(d0,_mm256_madd_epi16(_mm256_shuffle_epi8(_mm256_set1_epi64x(p0),shuf),w));
            d1=_mm256_add_epi32(d1,_mm256_madd_epi16(_mm256_shuffle_epi8(_mm256_set1_epi64x(p1),shuf),w));
            d2=_mm256_add_epi32(d2,_mm256_madd_epi16(_mm256_shuffle_epi8(_mm256_set1_epi64x(p2),shuf),w));
            d3=_mm256_add_epi32(d3,_mm256_madd_epi16(_mm256_shuffle_epi8(_mm256_set1_epi64x(p3),shuf),w));
        }
        dot[0]=_mm_add_epi32(_mm256_castsi256_si128(d0),_mm256_extracti128_si256(d0,1));
        dot[1]=_mm_add_epi32(_mm256_castsi256_si128(d1),_mm256_extracti128_si256(d1,1));
        dot[2]=_mm_add_epi32(_mm256_castsi256_si128(d2),_mm256_extracti128_si256(d2,1));
        dot[3]=_mm_add_epi32(_mm256_castsi256_si128(d3),_mm256_extracti128_si256(d3,1));
    } else {
        __m256i d[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),
                      _mm256_setzero_si256(),_mm256_setzero_si256()};
        for(;pair+2u<=total_pairs;pair+=2u) {
            __m256i w=_mm256_loadu_si256((const __m256i*)(panel+pair*16u));
            for(uint32_t r=0;r<count;r++) {
                const unsigned char* value=prepared+
                    ((uint64_t)(base+r)*pairs+start/2u+pair)*sizeof(uint32_t);
                int64_t p;
                __builtin_memcpy(&p,value,sizeof(p));
                d[r]=_mm256_add_epi32(d[r],
                    _mm256_madd_epi16(_mm256_shuffle_epi8(_mm256_set1_epi64x(p),shuf),w));
            }
        }
        for(uint32_t r=0;r<count;r++)
            dot[r]=_mm_add_epi32(_mm256_castsi256_si128(d[r]),_mm256_extracti128_si256(d[r],1));
    }
    for(;pair<total_pairs;pair++) {
        __m128i weight=_mm_loadu_si128((const __m128i*)(panel+pair*16u));
        for(uint32_t r=0;r<count;r++) {
            const unsigned char* value=prepared+
                ((uint64_t)(base+r)*pairs+start/2u+pair)*sizeof(uint32_t);
            dot[r]=_mm_add_epi32(dot[r],
                _mm_madd_epi16(_mm_set1_epi32((int32_t)vx_packed_panel_load_pair(value)),weight));
        }
    }
}
#endif

typedef struct {
    const VxLLMOpArgs* a;
    const unsigned char* prepared;
    const float* as;
    const int* finite;
    uint64_t first_row;
    uint32_t rows;
} VxPackedLinearPanelCtx;

static void vx_packed_linear_i8_rows_panel_range(const VxPackedLinearPanelCtx* ctx,
        unsigned char* panel,uint32_t panel_begin,uint32_t panel_end) {
    const VxLLMOpArgs* a=ctx->a;
    uint32_t k=(uint32_t)a->key_dim,n=(uint32_t)a->value_dim,bits=a->groups?(uint32_t)a->groups:4u;
    uint32_t group_size=(uint32_t)a->rate,pairs=k/2u+k%2u;
    uint32_t words=vx_packed_words(k,bits),groups=(k+group_size-1u)/group_size;
    uint32_t rows=ctx->rows;
#if VX_PACKED_X86_AVX2
    /* Group32 SSE2 microtiles avoid the AVX2 horizontal combine and leave
     * the decoded panel reusable for both two-row decode and larger tiles. */
    const int use_avx2=!(group_size==32u && (bits==3u || bits==4u)) &&
        vx_packed_use_avx2();
#endif
    for(uint32_t panel_idx=panel_begin;panel_idx<panel_end;panel_idx++) {
        uint32_t out=panel_idx*4u;
        VxPackedI8Vec sum[VX_PACKED_LINEAR_TILE_ROWS];
        for(uint32_t r=0;r<rows;r++)sum[r]=VX_PI_ZERO();
        for(uint32_t g=0;g<groups;g++) {
            uint32_t start=g*group_size,length=k-start<group_size?k-start:group_size;
            vx_packed_group_panel(panel,(const uint32_t*)a->x[1]+
                (uint64_t)out*words+(uint64_t)start*bits/32u,words,length,bits);
            const float* scale=a->x[2]+(uint64_t)out*groups+g;
            VxPackedI8Vec ws=VX_PI_F32(scale[0],scale[groups],scale[2u*groups],scale[3u*groups]);
            /* Four live integer accumulators keep the register demand bounded
             * even when many row microtiles share this decoded group. */
            for(uint32_t base=0;base<rows;base+=4u) {
                uint32_t count=rows-base<4u?rows-base:4u;
                VxPackedI8Vec dot[4]={VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO()};
#if VX_PACKED_X86_AVX2
                if(use_avx2)
                    vx_packed_panel_accum_avx2(ctx->prepared,pairs,start,length,base,count,panel,dot);
                else
#endif
#if (defined(__wasm_simd128__) || defined(__SSE2__)) && VX_PACKED_LINEAR_GROUP32_UNROLL
#if defined(__SSE2__)
                if((bits==3u || bits==4u) && group_size==32u && length==32u && count==2u)
                    vx_packed_group32_panel_2x4(ctx->prepared+
                        ((uint64_t)base*pairs+start/2u)*sizeof(uint32_t),
                        pairs*sizeof(uint32_t),panel,dot);
                else
#endif
                if((bits==3u || bits==4u) && group_size==32u && length==32u && count==4u)
                    vx_packed_group32_panel_4x4(ctx->prepared+
                        ((uint64_t)base*pairs+start/2u)*sizeof(uint32_t),
                        pairs*sizeof(uint32_t),panel,dot);
                else
#endif
                for(uint32_t pair=0;pair<(length+1u)/2u;pair++) {
                    VxPackedI8Vec weight=VX_PI_LOAD(panel+pair*16u);
                    for(uint32_t r=0;r<count;r++) {
                        const unsigned char* value=ctx->prepared+
                            ((uint64_t)(base+r)*pairs+start/2u+pair)*sizeof(uint32_t);
                        dot[r]=VX_PI_ADD(dot[r],VX_PI_DOT(VX_PI_PAIR_LOAD(value),weight));
                    }
                }
                for(uint32_t r=0;r<count;r++)sum[base+r]=VX_PI_FADD(sum[base+r],
                    VX_PI_FMUL(VX_PI_FLOAT(dot[r]),VX_PI_FMUL(VX_PI_SPLAT(ctx->as[base+r]),ws)));
            }
        }
        VxPackedI8Vec bias=a->x[3]?VX_PI_F32(a->x[3][out],a->x[3][out+1],a->x[3][out+2],a->x[3][out+3]):VX_PI_ZERO();
        for(uint32_t r=0;r<rows;r++) {
            VxPackedI8Vec value=ctx->finite[r]?VX_PI_FADD(sum[r],bias):VX_PI_SPLAT(__builtin_nanf(""));
            VX_PI_STORE(a->y[0]+(ctx->first_row+r)*n+out,value);
        }
    }
}

#if VX_PACKED_HAS_THREAD_POOL
static void vx_packed_linear_i8_rows_panel_worker(void* opaque,int begin,int end) {
    if(begin<end) {
        unsigned char local_panel[VX_PACKED_MAX_GROUP_SIZE*8u] __attribute__((aligned(16)));
        vx_packed_linear_i8_rows_panel_range((const VxPackedLinearPanelCtx*)opaque,
            local_panel,(uint32_t)begin,(uint32_t)end);
    }
}
#endif

static void vx_packed_linear_i8_rows_panel(const VxLLMOpArgs* a,float* scratch,
        uint64_t first_row,uint32_t rows) {
    uint32_t k=(uint32_t)a->key_dim,n=(uint32_t)a->value_dim,bits=a->groups?(uint32_t)a->groups:4u;
    uint32_t group_size=(uint32_t)a->rate,pairs=k/2u+k%2u;
    uint32_t words=vx_packed_words(k,bits),groups=(k+group_size-1u)/group_size;
    unsigned char* prepared=(unsigned char*)scratch;
    unsigned char* panel=prepared+((((uint64_t)rows*pairs+3u)&~UINT64_C(3))*sizeof(uint32_t));
    float as[VX_PACKED_LINEAR_TILE_ROWS];int finite[VX_PACKED_LINEAR_TILE_ROWS];
    for(uint32_t r=0;r<rows;r++) {
#if VX_PACKED_HAS_SIMD_QUANT
        finite[r]=vx_packed_prepare_i8_simd(a->x[0]+(first_row+r)*k,k,
            prepared+(uint64_t)r*pairs*sizeof(uint32_t),1,&as[r]);
#else
        const float* input=a->x[0]+(first_row+r)*k;float maximum=0.0f;finite[r]=1;
        for(uint32_t j=0;j<k;j++) {
            float value=input[j];
            if(!vx_packed_finite(value)){finite[r]=0;break;}
            float absolute=value<0.0f?-value:value;if(absolute>maximum)maximum=absolute;
        }
        as[r]=maximum==0.0f?1.0f:maximum/127.0f;
        if(as[r]<0x1p-126f)as[r]=0x1p-126f;
        for(uint32_t pair=0;pair<pairs;pair++) {
            uint32_t j=pair*2u;
            int32_t lo=finite[r]?vx_packed_round_i8(input[j]/as[r]):0;
            int32_t hi=finite[r]&&j+1u<k?vx_packed_round_i8(input[j+1u]/as[r]):0;
            uint32_t value=(uint16_t)(int16_t)lo|((uint32_t)(uint16_t)(int16_t)hi<<16u);
            vx_packed_panel_store_pair(prepared+((uint64_t)r*pairs+pair)*sizeof(uint32_t),value);
        }
#endif
    }
    uint32_t panel_count=n/4u;
    VxPackedLinearPanelCtx ctx={a,prepared,as,finite,first_row,rows};
    int threads=vx_packed_linear_threads((uint64_t)rows*k*(panel_count*4u),panel_count);
#if VX_PACKED_HAS_THREAD_POOL
    if(threads>1) {
        uint32_t target=(uint32_t)threads*4u;
        uint32_t grain=(panel_count+target-1u)/target;
        if(!grain)grain=1u;
        vx_kernels_parallel_for((int)panel_count,(int)grain,
                                vx_packed_linear_i8_rows_panel_worker,&ctx);
    } else
#endif
    {
        (void)threads;
        vx_packed_linear_i8_rows_panel_range(&ctx,panel,0u,panel_count);
    }
    for(uint32_t out=panel_count*4u;out<n;out++)for(uint32_t r=0;r<rows;r++) {
        float sum=0.0f;
        for(uint32_t g=0;g<groups;g++) {
            uint32_t start=g*group_size,end=k-start<group_size?k:start+group_size;int32_t dot=0;
            for(uint32_t j=start;j<end;j++) {
                uint32_t packed=vx_packed_panel_load_pair(prepared+((uint64_t)r*pairs+j/2u)*sizeof(uint32_t));
                int32_t activation=(int16_t)(packed>>((j%2u)*16u));
                const uint32_t* weight=(const uint32_t*)a->x[1]+(uint64_t)out*words;
                uint32_t code=bits==3u?vx_packed_i3_code(weight,j):
                    (weight[j/(32u/bits)]>>((j%(32u/bits))*bits))&((1u<<bits)-1u);
                dot+=activation*((int32_t)code-(1<<(bits-1u)));
            }
            sum+=(float)dot*(as[r]*a->x[2][(uint64_t)out*groups+g]);
        }
        a->y[0][(first_row+r)*n+out]=finite[r]?sum+(a->x[3]?a->x[3][out]:0.0f):__builtin_nanf("");
    }
}
#endif
#undef VX_PI_ZERO
#undef VX_PI_ADD
#undef VX_PI_DOT
#undef VX_PI_PAIR
#undef VX_PI_LO32
#undef VX_PI_HI32
#undef VX_PI_LO64
#undef VX_PI_HI64
#undef VX_PI_F32
#undef VX_PI_SPLAT
#undef VX_PI_FLOAT
#undef VX_PI_FADD
#undef VX_PI_FMUL
#undef VX_PI_STORE
#undef VX_PI_LOAD
#undef VX_PI_PAIR_LOAD
#endif

static inline float vx_packed_dot_w4_table_scalar(float sum, const float* scratch,
                                                  const uint32_t* weight,
                                                  const float* decoded, uint32_t length) {
    uint32_t j = 0;
    for (; j + 8u <= length; j += 8u) {
        uint32_t w = weight[j / 8u];
        sum += scratch[j + 0] * decoded[(w >> 0) & 15u];
        sum += scratch[j + 1] * decoded[(w >> 4) & 15u];
        sum += scratch[j + 2] * decoded[(w >> 8) & 15u];
        sum += scratch[j + 3] * decoded[(w >> 12) & 15u];
        sum += scratch[j + 4] * decoded[(w >> 16) & 15u];
        sum += scratch[j + 5] * decoded[(w >> 20) & 15u];
        sum += scratch[j + 6] * decoded[(w >> 24) & 15u];
        sum += scratch[j + 7] * decoded[(w >> 28) & 15u];
    }
    for (; j < length; j++) {
        uint32_t nibble = (weight[j / 8u] >> ((j % 8u) * 4u)) & 15u;
        sum += scratch[j] * decoded[nibble];
    }
    return sum;
}

static inline float vx_packed_dot_w8a16_scalar(float sum, const float* scratch,
                                               const uint32_t* weight,
                                               float group_scale, uint32_t length) {
    uint32_t j = 0;
    for (; j + 4u <= length; j += 4u) {
        uint32_t w = weight[j / 4u];
        int32_t q0 = (int32_t)((w >> 0) & 255u) - 128;
        int32_t q1 = (int32_t)((w >> 8) & 255u) - 128;
        int32_t q2 = (int32_t)((w >> 16) & 255u) - 128;
        int32_t q3 = (int32_t)((w >> 24) & 255u) - 128;
        sum += scratch[j + 0] * vx_packed_half((float)q0 * group_scale);
        sum += scratch[j + 1] * vx_packed_half((float)q1 * group_scale);
        sum += scratch[j + 2] * vx_packed_half((float)q2 * group_scale);
        sum += scratch[j + 3] * vx_packed_half((float)q3 * group_scale);
    }
    for (; j < length; j++) {
        int32_t q = (int32_t)((weight[j / 4u] >> ((j % 4u) * 8u)) & 255u) - 128;
        sum += scratch[j] * vx_packed_half((float)q * group_scale);
    }
    return sum;
}

static inline float vx_packed_dot_w8a32_scalar(float sum, const float* scratch,
                                               const uint32_t* weight,
                                               float group_scale, uint32_t length) {
    uint32_t j = 0;
    for (; j + 4u <= length; j += 4u) {
        uint32_t w = weight[j / 4u];
        int32_t q0 = (int32_t)((w >> 0) & 255u) - 128;
        int32_t q1 = (int32_t)((w >> 8) & 255u) - 128;
        int32_t q2 = (int32_t)((w >> 16) & 255u) - 128;
        int32_t q3 = (int32_t)((w >> 24) & 255u) - 128;
        sum += scratch[j + 0] * ((float)q0 * group_scale);
        sum += scratch[j + 1] * ((float)q1 * group_scale);
        sum += scratch[j + 2] * ((float)q2 * group_scale);
        sum += scratch[j + 3] * ((float)q3 * group_scale);
    }
    for (; j < length; j++) {
        int32_t q = (int32_t)((weight[j / 4u] >> ((j % 4u) * 8u)) & 255u) - 128;
        sum += scratch[j] * ((float)q * group_scale);
    }
    return sum;
}

typedef struct {
    const VxLLMOpArgs* a;
    const float* scratch;
    uint64_t row;
    uint32_t begin;
    float activation_scale;
    int finite;
} VxPackedLinearSingleRowCtx;

#if VX_PACKED_X86_AVX2
/* Choose the short-group dot once per output range. A capability/length
 * branch inside every group costs more than it saves for single-row decode. */
static void vx_packed_linear_i8_group32_single_row_range(
        const VxPackedLinearSingleRowCtx* ctx,uint32_t first,uint32_t last) {
    const VxLLMOpArgs* a=ctx->a;
    uint32_t k=(uint32_t)a->key_dim,n=(uint32_t)a->value_dim;
    uint32_t bits=a->groups?(uint32_t)a->groups:4u;
    uint32_t words=vx_packed_words(k,bits),groups=k/32u+(k%32u!=0);
    for(uint32_t out=ctx->begin+first;out<ctx->begin+last;out++) {
        float sum=0.0f;
        if(!ctx->finite){a->y[0][ctx->row*n+out]=__builtin_nanf("");continue;}
        const uint32_t* weight=(const uint32_t*)a->x[1]+(uint64_t)out*words;
        const float* scale=a->x[2]+(uint64_t)out*groups;
        for(uint32_t group=0;group<groups;group++) {
            uint32_t start=group*32u,length=k-start<32u?k-start:32u;
            int32_t dot=vx_packed_dot_i8_baseline((const int8_t*)ctx->scratch+start,
                weight+(uint64_t)start*bits/32u,length,bits);
            sum+=(float)dot*(ctx->activation_scale*scale[group]);
        }
        a->y[0][ctx->row*n+out]=sum+(a->x[3]?a->x[3][out]:0.0f);
    }
}
#endif

static void vx_packed_linear_single_row_range(const VxPackedLinearSingleRowCtx* ctx,
        uint32_t col_offset_begin,uint32_t col_offset_end) {
    const VxLLMOpArgs* a=ctx->a;
    const float* scratch=ctx->scratch;
    const uint32_t k=(uint32_t)a->key_dim,n=(uint32_t)a->value_dim;
    const uint32_t weight_bits=a->groups?(uint32_t)a->groups:4u;
    const uint32_t group_size=(uint32_t)a->rate,per_word=32u/weight_bits;
    const uint32_t words=vx_packed_words(k,weight_bits),groups=k/group_size+(k%group_size!=0);
    const uint64_t row=ctx->row;
    const float activation_scale=ctx->activation_scale;
    const int finite=ctx->finite;
#if VX_PACKED_X86_AVX2
    if(a->normalize==8 && group_size==32u && (weight_bits==3u || weight_bits==4u)) {
        vx_packed_linear_i8_group32_single_row_range(ctx,col_offset_begin,col_offset_end);
        return;
    }
#endif
    for(uint32_t output=ctx->begin+col_offset_begin;output<ctx->begin+col_offset_end;output++) {
        float sum=0.0f;
        if(!finite){a->y[0][row*n+output]=__builtin_nanf("");continue;}
        const uint32_t* weight=(const uint32_t*)a->x[1]+(uint64_t)output*words;
        const float* scale=a->x[2]+(uint64_t)output*groups;
        for(uint32_t group=0;group<groups;group++) {
            uint32_t start=group*group_size,length=k-start<group_size?k-start:group_size;
            if(a->normalize==8) {
                /* Complete the integer storage group before applying its
                 * scale, even when a SIMD tile is shorter than the group. */
                int32_t dot=vx_packed_dot_i8((const int8_t*)scratch+start,weight+(uint64_t)start*weight_bits/32u,length,weight_bits);
                sum+=(float)dot*(activation_scale*scale[group]);
            } else if(weight_bits==2u||weight_bits==3u) {
                /* Preserve ascending-K accumulation and the selected
                 * operand rounding exactly, including F16 overflow. */
                float decoded[8];
                for(int q=0;q<(1<<weight_bits);q++) {
                    float value=(float)(q-(1<<(weight_bits-1u)))*scale[group];
                    decoded[q]=a->normalize==32?value:vx_packed_half(value);
                }
                for(uint32_t j=0;j<length;j++) {
                    uint32_t code=weight_bits==3u?vx_packed_i3_code(weight,start+j):
                        (weight[(start+j)/16u]>>(((start+j)%16u)*2u))&3u;
                    sum+=scratch[start+j]*decoded[code];
                }
            } else if(weight_bits==4u) {
                /* A group has only sixteen possible rounded weights. */
                float decoded[16];
                if(a->normalize==32) {
                    for(int q=0;q<16;q++)decoded[q]=(float)(q-8)*scale[group];
                } else {
                    for(int q=0;q<16;q++)decoded[q]=vx_packed_half((float)(q-8)*scale[group]);
                }
                sum = vx_packed_dot_w4_table_scalar(sum,scratch+start,weight+start/per_word,decoded,length);
            } else if(a->normalize==32) {
                sum = vx_packed_dot_w8a32_scalar(sum,scratch+start,weight+start/per_word,scale[group],length);
            } else {
                sum = vx_packed_dot_w8a16_scalar(sum,scratch+start,weight+start/per_word,scale[group],length);
            }
        }
        a->y[0][row*n+output]=sum+(a->x[3]?a->x[3][output]:0.0f);
    }
}

#if VX_PACKED_HAS_THREAD_POOL
static void vx_packed_linear_single_row_worker(void* opaque,int begin,int end) {
    if(begin<end)
        vx_packed_linear_single_row_range((const VxPackedLinearSingleRowCtx*)opaque,
                                          (uint32_t)begin,(uint32_t)end);
}
#endif

static int vx_packed_linear_forward(const VxLLMOpArgs* a,float* scratch) {
    const uint32_t k=(uint32_t)a->key_dim,n=(uint32_t)a->value_dim;
    const uint32_t weight_bits=a->groups?(uint32_t)a->groups:4u;
    if((weight_bits!=2u&&weight_bits!=3u&&weight_bits!=4u&&weight_bits!=8u)||!vx_packed_group_size_supported(a->rate))return -1;
    const uint64_t rows=(uint64_t)a->batch*a->sequence;
    uint64_t first=a->first_unit,units=a->unit_count?a->unit_count:rows*n;
    if(!scratch||!k||!n||(a->normalize!=8&&a->normalize!=16&&a->normalize!=32)||
       first>rows*n||units>rows*n-first)return -1;
    for(uint64_t row=first/n;row<(first+units+n-1u)/n;row++) {
#if defined(__wasm_simd128__) || defined(__SSE2__)
        /* Only complete rows belong to this tile. Partial work, short shapes
         * and single-token decode retain their existing direct path. */
        if(a->normalize==8 && k>=32u && n>=4u && row*n>=first) {
            uint32_t tile=vx_packed_linear_tile_rows((first+units-row*n)/n);
#if VX_PACKED_LINEAR_TILE_ROWS > 4 || VX_PACKED_LINEAR_TILE2
            if(vx_packed_linear_panel_eligible(k,n,weight_bits) && (tile>4u || tile==2u)) {
                vx_packed_linear_i8_rows_panel(a,scratch,row,tile);row+=tile-1u;continue;
            }
#endif
            if(tile>=4u) {
                vx_packed_linear_i8_rows4(a,scratch,row);row+=3u;continue;
            }
        }
#endif
        const float* input=a->x[0]+row*k;
        float maximum=0.0f;int finite=1;
#if VX_PACKED_HAS_SIMD_QUANT
        float activation_scale;
        if(a->normalize==8)finite=vx_packed_prepare_i8_simd(input,k,
            (unsigned char*)scratch,0,&activation_scale);
        else {
#endif
        for(uint32_t j=0;j<k;j++) {
            float value=input[j];
            if(!vx_packed_finite(value)){finite=0;break;}
            float absolute=value<0.0f?-value:value;
            if(absolute>maximum)maximum=absolute;
        }
#if !VX_PACKED_HAS_SIMD_QUANT
        float activation_scale;
#endif
        activation_scale=maximum==0.0f?1.0f:maximum/127.0f;
        if(activation_scale<0x1p-126f)activation_scale=0x1p-126f;
        if(finite)for(uint32_t j=0;j<k;j++) {
            if(a->normalize==32)scratch[j]=input[j];
            else if(a->normalize==16)scratch[j]=vx_packed_half(input[j]);
            else ((int8_t*)scratch)[j]=(int8_t)vx_packed_round_i8(input[j]/activation_scale);
        }
#if VX_PACKED_HAS_SIMD_QUANT
        }
#endif
        uint32_t begin=first>row*n?(uint32_t)(first-row*n):0;
        uint32_t end=first+units<(row+1u)*n?(uint32_t)(first+units-row*n):n;
        uint32_t col_count=end-begin;
        VxPackedLinearSingleRowCtx ctx={a,scratch,row,begin,activation_scale,finite};
        int threads=vx_packed_linear_threads((uint64_t)col_count*k,col_count);
#if VX_PACKED_HAS_THREAD_POOL
        if(threads>1) {
            uint32_t target=(uint32_t)threads*4u;
            uint32_t grain=(col_count+target-1u)/target;
            if(!grain)grain=1u;
            vx_kernels_parallel_for((int)col_count,(int)grain,
                                    vx_packed_linear_single_row_worker,&ctx);
        } else
#endif
        {
            (void)threads;
            vx_packed_linear_single_row_range(&ctx,0u,col_count);
        }
    }
    return 0;
}
