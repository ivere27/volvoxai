/* Shared CPU/ARM/WASM packed linear execution. Only one activation row is
 * prepared at a time; weights always remain packed. SIMD128 and NEON use
 * exact integer reductions for W4A8, including groups with a short tail. */
#include "packed_linear.h"
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static int32_t vx_packed_dot_i8(const int8_t* activation,
                               const uint32_t* packed, uint32_t count,
                               uint32_t weight_bits) {
    int32_t total=0;
    uint32_t k=0;
#if defined(__wasm_simd128__)
    v128_t sum=wasm_i32x4_splat(0);
    for(;k+16u<=count;k+=16u) {
        v128_t w;
        if(weight_bits==4u) {
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
#endif
    /* Constant shifts keep the scalar fallback vectorizable without requiring
     * a host ISA with per-lane variable shifts. */
    if(weight_bits==4u) {
        for(;k<count;k++)total+=(int32_t)activation[k]*
            ((int32_t)((packed[k/8u]>>((k%8u)*4u))&15u)-8);
    } else {
        for(;k<count;k++)total+=(int32_t)activation[k]*
            ((int32_t)((packed[k/4u]>>((k%4u)*8u))&255u)-128);
    }
    return total;
}

static int vx_packed_linear_forward(const VxLLMOpArgs* a,float* scratch) {
    const uint32_t k=(uint32_t)a->key_dim,n=(uint32_t)a->value_dim;
    const uint32_t weight_bits=a->groups?(uint32_t)a->groups:4u;
    if(weight_bits!=4u&&weight_bits!=8u)return -1;
    const uint32_t per_word=32u/weight_bits;
    const uint32_t words=k/per_word+(k%per_word!=0),groups=k/128u+(k%128u!=0);
    const uint64_t rows=(uint64_t)a->batch*a->sequence;
    uint64_t first=a->first_unit,units=a->unit_count?a->unit_count:rows*n;
    if(!scratch||!k||!n||a->rate!=128||(a->normalize!=8&&a->normalize!=16)||
       first>rows*n||units>rows*n-first)return -1;
    for(uint64_t row=first/n;row<(first+units+n-1u)/n;row++) {
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
            if(a->normalize==16)scratch[j]=vx_packed_half(input[j]);
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
                uint32_t start=group*128u,length=k-start<128u?k-start:128u;
                if(a->normalize==8) {
                    int32_t dot=vx_packed_dot_i8((const int8_t*)scratch+start,weight+start/per_word,length,weight_bits);
                    sum+=(float)dot*(activation_scale*scale[group]);
                } else if(weight_bits==4u) {
                    /* A group has only sixteen possible rounded weights. */
                    float decoded[16];
                    for(int q=0;q<16;q++)decoded[q]=vx_packed_half((float)(q-8)*scale[group]);
                    for(uint32_t j=start;j<start+length;j++) {
                        uint32_t nibble=(weight[j/8u]>>((j%8u)*4u))&15u;
                        sum+=scratch[j]*decoded[nibble];
                    }
                } else {
                    for(uint32_t j=start;j<start+length;j++) {
                        int32_t q=(int32_t)((weight[j/4u]>>((j%4u)*8u))&255u)-128;
                        sum+=scratch[j]*vx_packed_half((float)q*scale[group]);
                    }
                }
            }
            a->y[0][row*n+output]=sum+(a->x[3]?a->x[3][output]:0.0f);
        }
    }
    return 0;
}
