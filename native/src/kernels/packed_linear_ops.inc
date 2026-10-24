/* Shared CPU/ARM/WASM packed linear execution. Bounded activation tiles
 * reuse packed weights; SIMD128, SSE2 and NEON use exact integer reductions
 * for dynamic-I8 arithmetic, including groups with a short tail. */
#include "packed_linear.h"
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
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

static int32_t vx_packed_dot_i8(const int8_t* activation,
                               const uint32_t* packed, uint32_t count,
                               uint32_t weight_bits) {
    int32_t total=0;
    uint32_t k=0;
    if(weight_bits==3u) {
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
static inline VxPackedI8Vec vx_packed_load_i8x8(const int8_t* p) {
    return wasm_i16x8_extend_low_i8x16(wasm_v128_load64_zero(p));
}
static inline VxPackedI8Vec vx_packed_load_weight8(const uint32_t* p,uint32_t bits,uint32_t bit_offset) {
    v128_t w;
    if(bits==2u) w=vx_packed_i2_bytes(*p>>bit_offset);
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
static inline VxPackedI8Vec vx_packed_load_i8x8(const int8_t* p) {
    __m128i b=_mm_loadl_epi64((const __m128i*)p);
    return _mm_unpacklo_epi8(b,_mm_cmpgt_epi8(_mm_setzero_si128(),b));
}
static inline VxPackedI8Vec vx_packed_load_weight8(const uint32_t* p,uint32_t bits,uint32_t bit_offset) {
    __m128i w;
    if(bits==2u) w=vx_packed_i2_bytes(*p>>bit_offset);
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

static inline VxPackedI8Tile vx_packed_group_i8_4x4(const int8_t* a,uint32_t stride,
        const uint32_t* w,uint32_t words,uint32_t length,uint32_t bits) {
    if(bits==2u)return vx_packed_group_i2_16(a,stride,w,words,length);
    VxPackedI8Tile dot={{VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO()}};
    uint32_t j=0,per_word=32u/bits;
    for(;j+8u<=length;j+=8u) {
        const uint32_t* p=w+j/per_word;
        uint32_t bit_offset=(j%per_word)*bits;
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
            int32_t code=(int32_t)((w[out*words+j/per_word]>>((j%per_word)*bits))&((1u<<bits)-1u))-(1<<(bits-1u));
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

static void vx_packed_linear_i8_rows4(const VxLLMOpArgs* a,float* scratch,uint64_t row) {
    uint32_t k=(uint32_t)a->key_dim,n=(uint32_t)a->value_dim,bits=a->groups?(uint32_t)a->groups:4u;
    uint32_t group_size=(uint32_t)a->rate,per_word=32u/bits;
    uint32_t words=(k+per_word-1u)/per_word,groups=(k+group_size-1u)/group_size;
    float as[4];int finite[4];int8_t* prepared=(int8_t*)scratch;
    for(uint32_t r=0;r<4;r++) {
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
    }
    uint32_t out=0;
    for(;out+4u<=n;out+=4u) {
        VxPackedI8Vec sum[4]={VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO(),VX_PI_ZERO()};
        for(uint32_t g=0;g<groups;g++) {
            uint32_t start=g*group_size,length=k-start<group_size?k-start:group_size;
            VxPackedI8Tile dot=vx_packed_group_i8_4x4(prepared+start,k,
                (const uint32_t*)a->x[1]+(uint64_t)out*words+start/per_word,words,length,bits);
            const float* scale=a->x[2]+(uint64_t)out*groups+g;
            VxPackedI8Vec ws=VX_PI_F32(scale[0],scale[groups],scale[2u*groups],scale[3u*groups]);
            for(uint32_t r=0;r<4;r++)sum[r]=VX_PI_FADD(sum[r],
                VX_PI_FMUL(VX_PI_FLOAT(dot.row[r]),VX_PI_FMUL(VX_PI_SPLAT(as[r]),ws)));
        }
        VxPackedI8Vec bias=a->x[3]?VX_PI_F32(a->x[3][out],a->x[3][out+1],a->x[3][out+2],a->x[3][out+3]):VX_PI_ZERO();
        for(uint32_t r=0;r<4;r++) {
            VxPackedI8Vec value=finite[r]?VX_PI_FADD(sum[r],bias):VX_PI_SPLAT(__builtin_nanf(""));
            VX_PI_STORE(a->y[0]+(row+r)*n+out,value);
        }
    }
    for(;out<n;out++)for(uint32_t r=0;r<4;r++) {
        float sum=0.0f;
        for(uint32_t g=0;g<groups;g++) {
            uint32_t start=g*group_size,length=k-start<group_size?k-start:group_size;
            int32_t dot=vx_packed_dot_i8(prepared+(uint64_t)r*k+start,
                (const uint32_t*)a->x[1]+(uint64_t)out*words+start/per_word,length,bits);
            sum+=(float)dot*(as[r]*a->x[2][(uint64_t)out*groups+g]);
        }
        a->y[0][(row+r)*n+out]=finite[r]?sum+(a->x[3]?a->x[3][out]:0.0f):__builtin_nanf("");
    }
}
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

static int vx_packed_linear_forward(const VxLLMOpArgs* a,float* scratch) {
    const uint32_t k=(uint32_t)a->key_dim,n=(uint32_t)a->value_dim;
    const uint32_t weight_bits=a->groups?(uint32_t)a->groups:4u;
    if((weight_bits!=2u&&weight_bits!=3u&&weight_bits!=4u&&weight_bits!=8u)||!vx_packed_group_size_supported(a->rate))return -1;
    const uint32_t group_size=(uint32_t)a->rate;
    const uint32_t per_word=32u/weight_bits;
    const uint32_t words=vx_packed_words(k,weight_bits),groups=k/group_size+(k%group_size!=0);
    const uint64_t rows=(uint64_t)a->batch*a->sequence;
    uint64_t first=a->first_unit,units=a->unit_count?a->unit_count:rows*n;
    if(!scratch||!k||!n||(a->normalize!=8&&a->normalize!=16&&a->normalize!=32)||
       first>rows*n||units>rows*n-first)return -1;
    for(uint64_t row=first/n;row<(first+units+n-1u)/n;row++) {
#if defined(__wasm_simd128__) || defined(__SSE2__)
        /* Only complete rows belong to this tile. Partial work, short shapes
         * and single-token decode retain their existing direct path. */
        if(weight_bits!=3u && a->normalize==8 && k>=32u && n>=4u && row*n>=first &&
           (first+units-row*n)/n>=4u) {
            vx_packed_linear_i8_rows4(a,scratch,row);row+=3u;continue;
        }
#endif
        const float* input=a->x[0]+row*k;
        float maximum=0.0f;int finite=1;
        for(uint32_t j=0;j<k;j++) {
            float value=input[j];
            if(!vx_packed_finite(value)){finite=0;break;}
            float absolute=value<0.0f?-value:value;
            if(absolute>maximum)maximum=absolute;
        }
        float activation_scale=maximum==0.0f?1.0f:maximum/127.0f;
        if(activation_scale<0x1p-126f)activation_scale=0x1p-126f;
        if(finite)for(uint32_t j=0;j<k;j++) {
            if(a->normalize==32)scratch[j]=input[j];
            else if(a->normalize==16)scratch[j]=vx_packed_half(input[j]);
            else ((int8_t*)scratch)[j]=(int8_t)vx_packed_round_i8(input[j]/activation_scale);
        }
        uint32_t begin=first>row*n?(uint32_t)(first-row*n):0;
        uint32_t end=first+units<(row+1u)*n?(uint32_t)(first+units-row*n):n;
        for(uint32_t output=begin;output<end;output++) {
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
    return 0;
}
