/* Correctness-first F32 decoder primitives. Each recurrent value column has
 * independent state. Attention recomputes scalar scores so its memory usage
 * is independent of context length and it needs no unplanned workspace. */
#include "../packed_launch_config.h"
static VX_CUDA_DEVICE float vx_llm_cuda_log(float x) {
    float y;
    asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y * 0.6931471805599453f;
}
#define VX_LLM_CUDA_MATH 1
#define VX_LLM_INLINE static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE
#define expf vx_exp
#define logf vx_llm_cuda_log
#define sqrtf vx_sqrt
#define fabsf vx_abs
#include "../../../kernels/llm_math.h"
#undef expf
#undef logf
#undef sqrtf
#undef fabsf
#undef VX_LLM_INLINE
#undef VX_LLM_CUDA_MATH

static VX_CUDA_DEVICE float vx_llm_index_score(const VxLLMOpArgs* a,int b,int t,int j) {
    float score=0.0f;
    const float* key=a->x[6]+((vx_u64)b*a->keys+j)*a->index_dim;
    for(int h=0;h<a->index_heads;h++) {
        const float* q=a->x[5]+(((vx_u64)b*a->sequence+t)*a->index_heads+h)*a->index_dim;
        score+=vx_max(vx_llm_dot(q,key,a->index_dim),0.0f)*a->x[7][((vx_u64)b*a->sequence+t)*a->index_heads+h];
    }
    return score/vx_sqrt((float)a->index_heads*a->index_dim);
}
static VX_CUDA_DEVICE int vx_llm_index_selected(const VxLLMOpArgs* a,int b,int t,int j,int ready) {
    if(a->x[5]&&!a->index_heads) return ((const int32_t*)a->x[5])[((vx_u64)b*a->sequence+t)*a->keys+j]!=0;
    if(!a->x[5]||a->top_k<=0||a->top_k>=ready) return 1;
    float own=vx_llm_index_score(a,b,t,j);
    int rank=0;
    for(int other=0;other<ready;other++) {
        float score=vx_llm_index_score(a,b,t,other);
        rank+=(score>own||(score==own&&other<j));
        if(rank>=a->top_k) return 0;
    }
    return 1;
}
static VX_CUDA_DEVICE void vx_llm_attention_column(const VxLLMOpArgs* a,vx_u64 id) {
    int c=(int)(id%a->value_dim);vx_u64 unit=id/a->value_dim;
    int h=(int)(unit%a->heads);vx_u64 row=unit/a->heads;
    int t=(int)(row%a->sequence),b=(int)(row/a->sequence),kh=h/(a->heads/a->kv_heads);
    const float* q=a->x[0]+unit*a->key_dim;
    int ready=0;
    if(a->x[3]) {
        int position=a->positions?a->positions[t]:t+a->query_offset;
        ready=position>=0&&position<2147483647?(position+1)/a->rate:0;
        if(ready>a->keys/a->rate) ready=a->keys/a->rate;
    }
    float maximum=a->x[4]?a->x[4][h]:-__builtin_inff();
    for(int j=0;j<a->keys;j++) if(vx_llm_attention_keep(a,b,t,j)) {
        const float* k=a->x[1]+(((vx_u64)b*a->keys+j)*a->kv_heads+kh)*a->key_dim;
        maximum=vx_max(maximum,vx_llm_dot(q,k,a->key_dim)*a->scale);
    }
    for(int j=0;j<ready;j++) if(vx_llm_index_selected(a,b,t,j,ready)) {
        const float* k=a->x[3]+((vx_u64)b*a->keys+j)*a->key_dim;
        maximum=vx_max(maximum,vx_llm_dot(q,k,a->key_dim)*a->scale);
    }
    if(maximum==-__builtin_inff()) {a->y[0][id]=0.0f;return;}
    float denominator=a->x[4]?vx_exp(a->x[4][h]-maximum):0.0f,total=0.0f;
    for(int j=0;j<a->keys;j++) if(vx_llm_attention_keep(a,b,t,j)) {
        const float* k=a->x[1]+(((vx_u64)b*a->keys+j)*a->kv_heads+kh)*a->key_dim;
        const float* v=a->x[2]+(((vx_u64)b*a->keys+j)*a->kv_heads+kh)*a->value_dim;
        float p=vx_exp(vx_llm_dot(q,k,a->key_dim)*a->scale-maximum);
        denominator+=p;total+=p*v[c];
    }
    for(int j=0;j<ready;j++) if(vx_llm_index_selected(a,b,t,j,ready)) {
        const float* k=a->x[3]+((vx_u64)b*a->keys+j)*a->key_dim;
        float p=vx_exp(vx_llm_dot(q,k,a->key_dim)*a->scale-maximum);
        denominator+=p;total+=p*k[c];
    }
    a->y[0][id]=denominator>0.0f?total/denominator:0.0f;
}
static VX_CUDA_DEVICE void vx_llm_delta_column(const VxLLMOpArgs* a,vx_u64 id) {
    int c=(int)(id%a->value_dim);vx_u64 unit=id/a->value_dim;
    int h=(int)(unit%a->heads),b=(int)(unit/a->heads),kh=h/(a->heads/a->kv_heads);
    float state[512];
    for(int d=0;d<a->key_dim;d++) state[d]=a->x[7]?a->x[7][(unit*a->key_dim+d)*a->value_dim+c]:0.0f;
    for(int t=0;t<a->sequence;t++) {
        vx_u64 row=(vx_u64)b*a->sequence+t;
        const float* q=a->x[0]+(row*a->kv_heads+kh)*a->key_dim;
        const float* k=a->x[1]+(row*a->kv_heads+kh)*a->key_dim;
        float qn=a->normalize?1.0f/vx_sqrt(vx_llm_dot(q,q,a->key_dim)+a->eps):1.0f;
        float kn=a->normalize?1.0f/vx_sqrt(vx_llm_dot(k,k,a->key_dim)+a->eps):1.0f;
        float beta=vx_llm_sigmoid(a->x[4][row*a->heads+h]);
        float decay=vx_exp(-vx_exp(a->x[5][h])*vx_llm_softplus(a->x[3][row*a->heads+h]+a->x[6][h]));
        float prediction=0.0f;
        for(int d=0;d<a->key_dim;d++) {state[d]*=decay;prediction+=state[d]*k[d]*kn;}
        float correction=(a->x[2][(row*a->heads+h)*a->value_dim+c]-prediction)*beta,value=0.0f;
        for(int d=0;d<a->key_dim;d++) {state[d]+=k[d]*kn*correction;value+=state[d]*q[d]*qn*a->scale;}
        a->y[0][(row*a->heads+h)*a->value_dim+c]=value;
    }
    if(a->y[1]) for(int d=0;d<a->key_dim;d++) a->y[1][(unit*a->key_dim+d)*a->value_dim+c]=state[d];
}

static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE unsigned short vx_packed_cuda_half_bits(float value) {
    unsigned short half;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(half) : "f"(value));
    return half;
}
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE float vx_packed_cuda_half(float value) {
    unsigned short half=vx_packed_cuda_half_bits(value);
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(half));
    return result;
}

static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE float vx_packed_cuda_broadcast(float value) {
    float result;
    asm("shfl.sync.idx.b32 %0, %1, 0, 31, -1;" : "=f"(result) : "f"(value));
    return result;
}
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE int vx_packed_cuda_quantize(float value,float scale) {
    float quotient;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(quotient) : "f"(value),"f"(scale));
    float bounded=vx_max(-127.0f,vx_min(127.0f,quotient));
    int result;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(result) : "f"(bounded));
    return result;
}
struct VX_CUDA_ALIGN(8) VxCudaUint2 {
    vx_u32 x;
    vx_u32 y;
};
struct VX_CUDA_ALIGN(16) VxCudaUint4 {
    vx_u32 x;
    vx_u32 y;
    vx_u32 z;
    vx_u32 w;
};
template <bool EightBit>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE int vx_packed_cuda_weight_bits(
        const VxLLMOpArgs* a,vx_u64 column,vx_u32 k) {
    if(EightBit) {
        vx_u32 word=((const vx_u32*)a->x[1])[column*(((vx_u32)a->key_dim+3u)/4u)+k/4u];
        return (int)((word>>((k&3u)*8u))&255u)-128;
    }
    vx_u32 word=((const vx_u32*)a->x[1])[column*(((vx_u32)a->key_dim+7u)/8u)+k/8u];
    return (int)((word>>((k&7u)*4u))&15u)-8;
}
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE int vx_packed_cuda_weight(
        const VxLLMOpArgs* a,vx_u64 column,vx_u32 k) {
    return a->groups==8?vx_packed_cuda_weight_bits<true>(a,column,k):
        vx_packed_cuda_weight_bits<false>(a,column,k);
}
/* Duplicate each packed byte, then place its two nibbles in adjacent
 * bytes. Adding 0x78 cannot carry between bytes; XOR maps 0..15 to -8..7. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE vx_u32 vx_packed_cuda_unpack_w4_low(vx_u32 word) {
    vx_u32 duplicated;
    asm("prmt.b32 %0, %1, 0, 0x1100;" : "=r"(duplicated) : "r"(word));
    vx_u32 expanded=(duplicated&0x000f000fu)|((duplicated>>4u)&0x0f000f00u);
    return (expanded+0x78787878u)^0x80808080u;
}
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE vx_u32 vx_packed_cuda_unpack_w4_high(vx_u32 word) {
    vx_u32 duplicated;
    asm("prmt.b32 %0, %1, 0, 0x3322;" : "=r"(duplicated) : "r"(word));
    vx_u32 expanded=(duplicated&0x000f000fu)|((duplicated>>4u)&0x0f000f00u);
    return (expanded+0x78787878u)^0x80808080u;
}
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE vx_u32 vx_packed_cuda_weight4(
        const VxLLMOpArgs* a,vx_u64 column,vx_u32 k) {
    if(k>=(vx_u32)a->key_dim)return 0u;
    vx_u32 shift=a->groups==8?2u:3u,per_word=1u<<shift;
    vx_u32 word=((const vx_u32*)a->x[1])[column*(((vx_u32)a->key_dim+per_word-1u)>>shift)+(k>>shift)];
    if(a->groups==8)return word^0x80808080u;
    return (k&4u)?vx_packed_cuda_unpack_w4_high(word):vx_packed_cuda_unpack_w4_low(word);
}
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE int vx_packed_cuda_dot4(vx_u32 x,vx_u32 w) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, 0;" : "=r"(result) : "r"(x),"r"(w));
    return result;
#else
    int result=0;
    for(vx_u32 i=0u;i<4u;i++) {
        int a=(int)((x>>(i*8u))&255u),b=(int)((w>>(i*8u))&255u);
        result+=(a>=128?a-256:a)*(b>=128?b-256:b);
    }
    return result;
#endif
}
/* Matrix tiles and the bounded A8 row cache share one dynamic arena. Its
 * size is selected by the host; decode without a cache pays no shared cost. */
extern VX_CUDA_SHARED vx_u32 vx_packed_tile_shared[];
/* Prepare each input row once in the context's bounded workspace. Rows are
 * padded to K128; x[5] holds one scale (A8) or invalid flag (A16) per row.
 * A separate launch makes these words visible to every output tile without
 * changing the group's dot products or the order of F32 accumulation. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_packed_cuda_prepare(const VxLLMOpArgs* a) {
    vx_u64 row=blockIdx.x,rows=(vx_u64)a->batch*a->sequence;
    if(row>=rows)return;
    vx_u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5u;
    const float* input=a->x[0]+row*(vx_u32)a->key_dim;
    float maximum=0.0f;
    for(vx_u32 k=threadIdx.x;k<(vx_u32)a->key_dim;k+=256u) {
        float value=input[k];
        maximum=!(value<=3.402823466e38f && value>=-3.402823466e38f)?
            __builtin_inff():vx_max(maximum,vx_abs(value));
    }
    for(vx_u32 delta=16u;delta;delta>>=1u)
        maximum=vx_max(maximum,vx_cuda_gemv_shuffle_down(maximum,delta));
    float* maxima=(float*)vx_packed_tile_shared;
    if(!lane)maxima[warp]=maximum;
    vx_sync_threads();
    if(!warp) {
        maximum=lane<8u?maxima[lane]:0.0f;
        for(vx_u32 delta=16u;delta;delta>>=1u)
            maximum=vx_max(maximum,vx_cuda_gemv_shuffle_down(maximum,delta));
        if(!lane)maxima[0]=maximum;
    }
    vx_sync_threads();
    maximum=maxima[0];
    vx_u32 groups=((vx_u32)a->key_dim+127u)/128u;
    bool invalid=maximum==__builtin_inff();
    if(a->normalize==8) {
        float scale=invalid?maximum:maximum>0.0f?
            vx_max(maximum/127.0f,1.1754943508222875e-38f):1.0f;
        if(!threadIdx.x)((float*)a->x[5])[row]=scale;
        vx_u32* output=(vx_u32*)a->x[4]+row*groups*32u;
        for(vx_u32 word=threadIdx.x;word<groups*32u;word+=256u) {
            vx_u32 packed=0u;
            for(vx_u32 i=0u;i<4u;i++) {
                vx_u32 k=word*4u+i;
                int value=!invalid && k<(vx_u32)a->key_dim?
                    vx_packed_cuda_quantize(input[k],scale):0;
                packed|=((vx_u32)value&255u)<<(i*8u);
            }
            output[word]=packed;
        }
    } else {
        if(!threadIdx.x)((vx_u32*)a->x[5])[row]=(vx_u32)invalid;
        vx_u32* output=(vx_u32*)a->x[4]+row*groups*64u;
        for(vx_u32 word=threadIdx.x;word<groups*64u;word+=256u) {
            vx_u32 k=word*2u;
            float x0=!invalid && k<(vx_u32)a->key_dim?input[k]:0.0f;
            float x1=!invalid && k+1u<(vx_u32)a->key_dim?input[k+1u]:0.0f;
            output[word]=(vx_u32)vx_packed_cuda_half_bits(x0)|
                ((vx_u32)vx_packed_cuda_half_bits(x1)<<16u);
        }
    }
}
/* One CTA reuses a quantized 128-value activation group across 32 output
 * columns. The base route needs 160 static shared bytes. With at most 16KiB
 * of dynamic storage, the row is quantized once and group barriers disappear.
 * Each eight-lane subgroup owns one column. Four integer dots per lane
 * cover a group of 128; the subgroup sum is exact, and scales accumulate
 * in ascending group order, just as on the scalar path. */
template <bool Prepared, bool PairGroups = false>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_packed_cuda_a8_columns(const VxLLMOpArgs* a) {
    VX_CUDA_SHARED float maxima[8];
    VX_CUDA_SHARED vx_u32 activation_words[32];
    vx_u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5u;
    vx_u64 columns=(vx_u32)a->value_dim,column_tiles=(columns+31u)/32u;
    vx_u64 row=a->first_unit/columns+(vx_u64)blockIdx.x/column_tiles;
    vx_u32 sub_lane=lane&7u;
    vx_u64 column=((vx_u64)blockIdx.x%column_tiles)*32u+warp*4u+(lane>>3u);
    const float* input=a->x[0]+row*(vx_u32)a->key_dim;
    const bool prepared=Prepared;
    float maximum=0.0f;
    if(!prepared) {
        for(vx_u32 k=threadIdx.x;k<(vx_u32)a->key_dim;k+=256u) {
            float value=input[k];
            maximum=!(value<=3.402823466e38f && value>=-3.402823466e38f)?
                __builtin_inff():vx_max(maximum,vx_abs(value));
        }
        for(vx_u32 delta=16u;delta;delta>>=1u)maximum=vx_max(maximum,vx_cuda_gemv_shuffle_down(maximum,delta));
        if(!lane)maxima[warp]=maximum;
        vx_sync_threads();
        if(!warp) {
            maximum=lane<8u?maxima[lane]:0.0f;
            for(vx_u32 delta=16u;delta;delta>>=1u)maximum=vx_max(maximum,vx_cuda_gemv_shuffle_down(maximum,delta));
            if(!lane)maxima[0]=maximum;
        }
        vx_sync_threads();
        maximum=maxima[0];
    }
    float activation_scale=prepared?a->x[5][row]:maximum>0.0f?
        vx_max(maximum/127.0f,1.1754943508222875e-38f):1.0f;
    if(activation_scale==__builtin_inff()) {
        if(!sub_lane && column<columns) {
            union {vx_u32 bits;float value;} nan;nan.bits=0x7fc00000u;
            a->y[0][row*columns+column]=nan.value;
        }
        return;
    }
    float sum=0.0f;
    vx_u32 groups=((vx_u32)a->key_dim+127u)/128u;
    bool cached=prepared || a->window==4;
    if(cached) {
        if(!prepared) {
            for(vx_u32 k=threadIdx.x;k<groups*128u;k+=256u) {
                vx_u32 value=(vx_u32)(k<(vx_u32)a->key_dim?vx_packed_cuda_quantize(input[k],activation_scale):0)&255u;
                for(vx_u32 shift=1u;shift<4u;shift<<=1u) {
                    vx_u32 other;
                    asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(value),"r"(shift));
                    value|=other<<(shift*8u);
                }
                if(!(lane&3u))vx_packed_tile_shared[k>>2u]=value;
            }
            vx_sync_threads();
        }
        const vx_u32* cached_words=prepared?
            ((const vx_u32*)a->x[4])+row*groups*32u:vx_packed_tile_shared;
        vx_u32 full_groups=(vx_u32)a->key_dim>>7u;
        vx_u32 word_base=sub_lane*4u;
        if(a->groups==4) {
            vx_u64 stride=((vx_u32)a->key_dim+7u)>>3u;
            const vx_u32* w_row=column<columns?((const vx_u32*)a->x[1])+column*stride:0;
            const float* s_row=column<columns?a->x[2]+column*groups:0;
            vx_u32 group=0u;
            /* The scalar decode entry overlaps independent integer groups.
             * Other callers keep the established loop and compiler resources. */
            if(PairGroups)for(;group+3u<full_groups;group+=4u) {
                int dot0=0,dot1=0,dot2=0,dot3=0;
                float scale0=0.0f,scale1=0.0f,scale2=0.0f,scale3=0.0f;
                if(column<columns) {
                    VxCudaUint4 act0=((const VxCudaUint4*)(cached_words+(group+0u)*32u))[sub_lane];
                    VxCudaUint4 act1=((const VxCudaUint4*)(cached_words+(group+1u)*32u))[sub_lane];
                    VxCudaUint4 act2=((const VxCudaUint4*)(cached_words+(group+2u)*32u))[sub_lane];
                    VxCudaUint4 act3=((const VxCudaUint4*)(cached_words+(group+3u)*32u))[sub_lane];
                    const vx_u32* wp=w_row+group*16u+sub_lane*2u;
                    vx_u32 w0=wp[0],w1=wp[1];
                    vx_u32 w2=wp[16],w3=wp[17];
                    vx_u32 w4=wp[32],w5=wp[33];
                    vx_u32 w6=wp[48],w7=wp[49];
                    if(!sub_lane) {
                        scale0=activation_scale*s_row[group+0u];
                        scale1=activation_scale*s_row[group+1u];
                        scale2=activation_scale*s_row[group+2u];
                        scale3=activation_scale*s_row[group+3u];
                    }
                    dot0=vx_packed_cuda_dot4(act0.x,vx_packed_cuda_unpack_w4_low(w0));
                    dot1=vx_packed_cuda_dot4(act1.x,vx_packed_cuda_unpack_w4_low(w2));
                    dot2=vx_packed_cuda_dot4(act2.x,vx_packed_cuda_unpack_w4_low(w4));
                    dot3=vx_packed_cuda_dot4(act3.x,vx_packed_cuda_unpack_w4_low(w6));
                    dot0+=vx_packed_cuda_dot4(act0.y,vx_packed_cuda_unpack_w4_high(w0));
                    dot1+=vx_packed_cuda_dot4(act1.y,vx_packed_cuda_unpack_w4_high(w2));
                    dot2+=vx_packed_cuda_dot4(act2.y,vx_packed_cuda_unpack_w4_high(w4));
                    dot3+=vx_packed_cuda_dot4(act3.y,vx_packed_cuda_unpack_w4_high(w6));
                    dot0+=vx_packed_cuda_dot4(act0.z,vx_packed_cuda_unpack_w4_low(w1));
                    dot1+=vx_packed_cuda_dot4(act1.z,vx_packed_cuda_unpack_w4_low(w3));
                    dot2+=vx_packed_cuda_dot4(act2.z,vx_packed_cuda_unpack_w4_low(w5));
                    dot3+=vx_packed_cuda_dot4(act3.z,vx_packed_cuda_unpack_w4_low(w7));
                    dot0+=vx_packed_cuda_dot4(act0.w,vx_packed_cuda_unpack_w4_high(w1));
                    dot1+=vx_packed_cuda_dot4(act1.w,vx_packed_cuda_unpack_w4_high(w3));
                    dot2+=vx_packed_cuda_dot4(act2.w,vx_packed_cuda_unpack_w4_high(w5));
                    dot3+=vx_packed_cuda_dot4(act3.w,vx_packed_cuda_unpack_w4_high(w7));
                }
                for(vx_u32 delta=4u;delta;delta>>=1u) {
                    int other0,other1,other2,other3;
                    asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=r"(other0) : "r"(dot0),"r"(delta));
                    asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=r"(other1) : "r"(dot1),"r"(delta));
                    asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=r"(other2) : "r"(dot2),"r"(delta));
                    asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=r"(other3) : "r"(dot3),"r"(delta));
                    dot0+=other0;dot1+=other1;dot2+=other2;dot3+=other3;
                }
                /* Integer reductions overlap; FP32 group sums stay ascending. */
                if(!sub_lane && column<columns) {
                    sum+=(float)dot0*scale0;
                    sum+=(float)dot1*scale1;
                    sum+=(float)dot2*scale2;
                    sum+=(float)dot3*scale3;
                }
            }
            for(;group<full_groups;group++) {
                int dot=0;
                if(column<columns) {
                    VxCudaUint4 act=((const VxCudaUint4*)(cached_words+group*32u))[sub_lane];
                    const vx_u32* wp=w_row+group*16u+sub_lane*2u;
                    vx_u32 w0=wp[0],w1=wp[1];
                    dot=vx_packed_cuda_dot4(act.x,vx_packed_cuda_unpack_w4_low(w0))
                       +vx_packed_cuda_dot4(act.y,vx_packed_cuda_unpack_w4_high(w0))
                       +vx_packed_cuda_dot4(act.z,vx_packed_cuda_unpack_w4_low(w1))
                       +vx_packed_cuda_dot4(act.w,vx_packed_cuda_unpack_w4_high(w1));
                }
                for(vx_u32 delta=4u;delta;delta>>=1u) {
                    int other;
                    asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(dot),"r"(delta));
                    dot+=other;
                }
                if(!sub_lane && column<columns)sum+=(float)dot*(activation_scale*s_row[group]);
            }
        } else {
            vx_u64 stride=((vx_u32)a->key_dim+3u)>>2u;
            const vx_u32* w_row=column<columns?((const vx_u32*)a->x[1])+column*stride:0;
            const float* s_row=column<columns?a->x[2]+column*groups:0;
            for(vx_u32 group=0u;group<full_groups;group++) {
                int dot=0;
                if(column<columns) {
                    VxCudaUint4 act=((const VxCudaUint4*)(cached_words+group*32u))[sub_lane];
                    const vx_u32* wp=w_row+group*32u+sub_lane*4u;
                    dot=vx_packed_cuda_dot4(act.x,wp[0]^0x80808080u)
                       +vx_packed_cuda_dot4(act.y,wp[1]^0x80808080u)
                       +vx_packed_cuda_dot4(act.z,wp[2]^0x80808080u)
                       +vx_packed_cuda_dot4(act.w,wp[3]^0x80808080u);
                }
                for(vx_u32 delta=4u;delta;delta>>=1u) {
                    int other;
                    asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(dot),"r"(delta));
                    dot+=other;
                }
                if(!sub_lane && column<columns)sum+=(float)dot*(activation_scale*s_row[group]);
            }
        }
        for(vx_u32 group=full_groups;group<groups;group++) {
            int dot=0;
            if(column<columns) {
                for(vx_u32 chunk=0u;chunk<4u;chunk++) {
                    vx_u32 word=word_base+chunk;
                    vx_u32 activation=cached_words[group*32u+word];
                    dot+=vx_packed_cuda_dot4(activation,vx_packed_cuda_weight4(a,column,group*128u+word*4u));
                }
            }
            for(vx_u32 delta=4u;delta;delta>>=1u) {
                int other;
                asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(dot),"r"(delta));
                dot+=other;
            }
            if(!sub_lane && column<columns)sum+=(float)dot*(activation_scale*a->x[2][column*groups+group]);
        }
    } else {
        for(vx_u32 group=0u;group<groups;group++) {
            if(threadIdx.x<128u) {
                vx_u32 k=group*128u+threadIdx.x;
                vx_u32 value=(vx_u32)(k<(vx_u32)a->key_dim?vx_packed_cuda_quantize(input[k],activation_scale):0)&255u;
                for(vx_u32 shift=1u;shift<4u;shift<<=1u) {
                    vx_u32 other;
                    asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(value),"r"(shift));
                    value|=other<<(shift*8u);
                }
                if(!(lane&3u))activation_words[threadIdx.x>>2u]=value;
            }
            vx_sync_threads();
            int dot=0;
            if(column<columns) {
                for(vx_u32 chunk=0u;chunk<4u;chunk++) {
                    vx_u32 word=sub_lane*4u+chunk;
                    vx_u32 activation=activation_words[word];
                    dot+=vx_packed_cuda_dot4(activation,vx_packed_cuda_weight4(a,column,group*128u+word*4u));
                }
            }
            for(vx_u32 delta=4u;delta;delta>>=1u) {
                int other;
                asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(dot),"r"(delta));
                dot+=other;
            }
            if(!sub_lane && column<columns)sum+=(float)dot*(activation_scale*a->x[2][column*groups+group]);
            vx_sync_threads();
        }
    }
    if(!sub_lane && column<columns)
        a->y[0][row*columns+column]=sum+(a->x[3]?a->x[3][column]:0.0f);
}
/* A 16x32 (or adaptive 16x64 / 16x128) tile reuses each group of quantized
 * activations across column warps. Eight integer MMA operations form each exact
 * group-128 dot; only then are scales applied, in the same ascending order as
 * the scalar route. Shared storage is fixed at 2368 bytes and independent of
 * the model shape. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_packed_cuda_a8_tile(const VxLLMOpArgs* a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    vx_u32* activation_words=vx_packed_tile_shared;
    float* row_scale=(float*)(vx_packed_tile_shared+576u);
    vx_u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5u;
    vx_u64 rows=(vx_u64)a->batch*a->sequence,columns=(vx_u32)a->value_dim;
    vx_u32 tile_cols=a->heads?(vx_u32)a->heads:
        vx_cuda_packed_linear_tile_columns(rows,(vx_u32)columns,8u,0u);
    vx_u64 column_tiles=(columns+tile_cols-1u)/tile_cols;
    vx_u64 row_base=(vx_u64)blockIdx.x/column_tiles*16u;
    if(row_base>=rows)return;
    vx_u64 column_base=((vx_u64)blockIdx.x%column_tiles)*tile_cols;
    for(vx_u32 r=0u;r<2u;r++) {
        vx_u64 row=row_base+warp*2u+r;
        if(a->x[4]) {
            if(!lane)row_scale[warp*2u+r]=row<rows?a->x[5][row]:1.0f;
            continue;
        }
        float maximum=0.0f;
        if(row<rows)for(vx_u32 k=lane;k<(vx_u32)a->key_dim;k+=32u) {
            float value=a->x[0][row*(vx_u32)a->key_dim+k];
            maximum=!(value<=3.402823466e38f && value>=-3.402823466e38f)?
                __builtin_inff():vx_max(maximum,vx_abs(value));
        }
        for(vx_u32 delta=16u;delta;delta>>=1u)maximum=vx_max(maximum,vx_cuda_gemv_shuffle_down(maximum,delta));
        if(!lane) {
            float scale=maximum==__builtin_inff()?maximum:
                maximum>0.0f?vx_max(maximum/127.0f,1.1754943508222875e-38f):1.0f;
            row_scale[warp*2u+r]=scale;
        }
    }
    vx_sync_threads();
    vx_u32 group=lane>>2u,sub=lane&3u,pair=sub*2u;
    vx_u32 groups=((vx_u32)a->key_dim+127u)/128u;
    vx_u32 shift=a->groups==8?2u:3u,per_word=1u<<shift;
    vx_u64 w_stride=((vx_u32)a->key_dim+per_word-1u)>>shift;
    if(tile_cols==32u) {
        vx_u32 tile_row=(warp>>2u)*8u;
        vx_u64 row=row_base+tile_row+group,column=column_base+(warp&3u)*8u+pair;
        vx_u64 wcol=column_base+(warp&3u)*8u+group;
        float activation_scale=row_scale[tile_row+group];
        float sum0=0.0f,sum1=0.0f;
        for(vx_u32 g=0u;g<groups;g++) {
            /* A thread packs four adjacent values, avoiding shared byte writes. */
            for(vx_u32 item=threadIdx.x;item<512u;item+=256u) {
                vx_u32 local_row=item>>5u,word_idx=item&31u,k=g*128u+word_idx*4u;
                vx_u64 input_row=row_base+local_row;
                float scale=row_scale[local_row];
                vx_u32 packed=0u;
                if(a->x[4]) {
                    if(input_row<rows)packed=((const vx_u32*)a->x[4])[
                        input_row*groups*32u+g*32u+word_idx];
                } else if(input_row<rows && scale!=__builtin_inff()) {
                    for(vx_u32 i=0u;i<4u;i++) {
                        int value=k+i<(vx_u32)a->key_dim?
                            vx_packed_cuda_quantize(a->x[0][input_row*(vx_u32)a->key_dim+k+i],scale):0;
                        packed|=((vx_u32)value&255u)<<(i*8u);
                    }
                }
                activation_words[local_row*36u+word_idx]=packed;
            }
            vx_sync_threads();
            int dot0=0,dot1=0;
            bool full_k=(g*128u+128u<=(vx_u32)a->key_dim);
            if(full_k && column_base+(warp&3u)*8u+8u<=columns) {
                const vx_u32* wp=((const vx_u32*)a->x[1])+wcol*w_stride+((g*128u)>>shift);
                #pragma unroll 2
                for(vx_u32 chunk=0u;chunk<8u;chunk++) {
                    vx_u32 word_off=chunk*4u+sub;
                    vx_u32 av=activation_words[(tile_row+group)*36u+word_off];
                    vx_u32 raw=wp[(word_off*4u)>>shift];
                    vx_u32 bv=a->groups==8?(raw^0x80808080u):
                        ((sub&1u)?vx_packed_cuda_unpack_w4_high(raw):vx_packed_cuda_unpack_w4_low(raw));
                    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                        "{%0, %1}, {%2}, {%3}, {%0, %1};"
                        : "+r"(dot0),"+r"(dot1) : "r"(av),"r"(bv));
                }
            } else {
                for(vx_u32 chunk=0u;chunk<8u;chunk++) {
                    vx_u32 word_off=chunk*4u+sub,k=g*128u+word_off*4u;
                    vx_u32 av=activation_words[(tile_row+group)*36u+word_off];
                    vx_u32 bv=wcol<columns?vx_packed_cuda_weight4(a,wcol,k):0u;
                    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                        "{%0, %1}, {%2}, {%3}, {%0, %1};"
                        : "+r"(dot0),"+r"(dot1) : "r"(av),"r"(bv));
                }
            }
            if(column<columns)sum0+=(float)dot0*(activation_scale*a->x[2][column*groups+g]);
            if(column+1u<columns)sum1+=(float)dot1*(activation_scale*a->x[2][(column+1u)*groups+g]);
            vx_sync_threads();
        }
        union {vx_u32 bits;float value;} nan;nan.bits=0x7fc00000u;
        if(row<rows) {
            if(column<columns)a->y[0][row*columns+column]=activation_scale==__builtin_inff()?nan.value:
                sum0+(a->x[3]?a->x[3][column]:0.0f);
            if(column+1u<columns)a->y[0][row*columns+column+1u]=activation_scale==__builtin_inff()?nan.value:
                sum1+(a->x[3]?a->x[3][column+1u]:0.0f);
        }
        return;
    }
    float act_scale0=row_scale[group],act_scale1=row_scale[group+8u];
    float s00_0=0.0f,s00_1=0.0f,s01_0=0.0f,s01_1=0.0f;
    float s10_0=0.0f,s10_1=0.0f,s11_0=0.0f,s11_1=0.0f;
    vx_u32 col_blocks=tile_cols>>6u;
    for(vx_u32 g=0u;g<groups;g++) {
        for(vx_u32 item=threadIdx.x;item<512u;item+=256u) {
            vx_u32 local_row=item>>5u,word_idx=item&31u,k=g*128u+word_idx*4u;
            vx_u64 input_row=row_base+local_row;
            float scale=row_scale[local_row];
            vx_u32 packed=0u;
            if(a->x[4]) {
                if(input_row<rows)packed=((const vx_u32*)a->x[4])[
                    input_row*groups*32u+g*32u+word_idx];
            } else if(input_row<rows && scale!=__builtin_inff()) {
                for(vx_u32 i=0u;i<4u;i++) {
                    int value=k+i<(vx_u32)a->key_dim?
                        vx_packed_cuda_quantize(a->x[0][input_row*(vx_u32)a->key_dim+k+i],scale):0;
                    packed|=((vx_u32)value&255u)<<(i*8u);
                }
            }
            activation_words[local_row*36u+word_idx]=packed;
        }
        vx_sync_threads();
        bool full_k=(g*128u+128u<=(vx_u32)a->key_dim);
        #pragma unroll
        for(vx_u32 cb=0u;cb<2u;cb++) {
            if(cb>=col_blocks)break;
            vx_u64 col=column_base+cb*64u+warp*8u+pair;
            vx_u64 wcol=column_base+cb*64u+warp*8u+group;
            int d0_0=0,d0_1=0,d1_0=0,d1_1=0;
            if(full_k && column_base+cb*64u+warp*8u+8u<=columns) {
                const vx_u32* wp=((const vx_u32*)a->x[1])+wcol*w_stride+((g*128u)>>shift);
                #pragma unroll 2
                for(vx_u32 chunk=0u;chunk<8u;chunk++) {
                    vx_u32 word_off=chunk*4u+sub;
                    vx_u32 av0=activation_words[group*36u+word_off];
                    vx_u32 av1=activation_words[(group+8u)*36u+word_off];
                    vx_u32 raw0=wp[(word_off*4u)>>shift];
                    vx_u32 bv0=a->groups==8?(raw0^0x80808080u):
                        ((sub&1u)?vx_packed_cuda_unpack_w4_high(raw0):vx_packed_cuda_unpack_w4_low(raw0));
                    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                        "{%0, %1}, {%2}, {%3}, {%0, %1};"
                        : "+r"(d0_0),"+r"(d0_1) : "r"(av0),"r"(bv0));
                    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                        "{%0, %1}, {%2}, {%3}, {%0, %1};"
                        : "+r"(d1_0),"+r"(d1_1) : "r"(av1),"r"(bv0));
                }
            } else {
                for(vx_u32 chunk=0u;chunk<8u;chunk++) {
                    vx_u32 word_off=chunk*4u+sub,k=g*128u+word_off*4u;
                    vx_u32 av0=activation_words[group*36u+word_off];
                    vx_u32 av1=activation_words[(group+8u)*36u+word_off];
                    vx_u32 bv0=wcol<columns?vx_packed_cuda_weight4(a,wcol,k):0u;
                    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                        "{%0, %1}, {%2}, {%3}, {%0, %1};"
                        : "+r"(d0_0),"+r"(d0_1) : "r"(av0),"r"(bv0));
                    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                        "{%0, %1}, {%2}, {%3}, {%0, %1};"
                        : "+r"(d1_0),"+r"(d1_1) : "r"(av1),"r"(bv0));
                }
            }
            if(col<columns) {
                float s=a->x[2][col*groups+g];
                float v0=(float)d0_0*(act_scale0*s),v1=(float)d1_0*(act_scale1*s);
                if(cb==0u) {s00_0+=v0;s10_0+=v1;} else {s01_0+=v0;s11_0+=v1;}
            }
            if(col+1u<columns) {
                float s=a->x[2][(col+1u)*groups+g];
                float v0=(float)d0_1*(act_scale0*s),v1=(float)d1_1*(act_scale1*s);
                if(cb==0u) {s00_1+=v0;s10_1+=v1;} else {s01_1+=v0;s11_1+=v1;}
            }
        }
        vx_sync_threads();
    }
    union {vx_u32 bits;float value;} nan;nan.bits=0x7fc00000u;
    vx_u64 row0=row_base+group,row1=row_base+group+8u;
    #pragma unroll
    for(vx_u32 cb=0u;cb<2u;cb++) {
        if(cb>=col_blocks)break;
        vx_u64 col=column_base+cb*64u+warp*8u+pair;
        float r0c0=cb==0u?s00_0:s01_0,r0c1=cb==0u?s00_1:s01_1;
        float r1c0=cb==0u?s10_0:s11_0,r1c1=cb==0u?s10_1:s11_1;
        if(row0<rows) {
            if(col<columns)a->y[0][row0*columns+col]=act_scale0==__builtin_inff()?nan.value:r0c0+(a->x[3]?a->x[3][col]:0.0f);
            if(col+1u<columns)a->y[0][row0*columns+col+1u]=act_scale0==__builtin_inff()?nan.value:r0c1+(a->x[3]?a->x[3][col+1u]:0.0f);
        }
        if(row1<rows) {
            if(col<columns)a->y[0][row1*columns+col]=act_scale1==__builtin_inff()?nan.value:r1c0+(a->x[3]?a->x[3][col]:0.0f);
            if(col+1u<columns)a->y[0][row1*columns+col+1u]=act_scale1==__builtin_inff()?nan.value:r1c1+(a->x[3]?a->x[3][col+1u]:0.0f);
        }
    }
#endif
}
template <bool EightBit>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_packed_cuda_column_a16(
        const VxLLMOpArgs* a,vx_u64 unit,vx_u64 row,vx_u64 column) {
    vx_u32 lane=threadIdx.x&31u;
    const float* input=a->x[0]+row*(vx_u32)a->key_dim;
    vx_u32 groups=((vx_u32)a->key_dim+127u)/128u;
    vx_u32 full_groups=(vx_u32)a->key_dim>>7u;
    const vx_u32 word_shift=EightBit?2u:3u,bits=EightBit?8u:4u,mask=EightBit?255u:15u;
    const int offset=EightBit?128:8;
    vx_u64 w_stride=((vx_u32)a->key_dim+(1u<<word_shift)-1u)>>word_shift;
    const vx_u32* w_row=((const vx_u32*)a->x[1])+column*w_stride;
    const float* s_row=a->x[2]+column*groups;
    vx_u32 shift=(lane&((1u<<word_shift)-1u))*bits;
    vx_u32 lane_word=lane>>word_shift;
    vx_u32 step_words=32u>>word_shift;
    vx_u32 invalid=0u;
    float sum=0.0f;
    for(vx_u32 g=0u;g<full_groups;g++) {
        float scale=s_row[g];
        vx_u32 k0=g*128u+lane;
        const vx_u32* wp=w_row+g*(128u>>word_shift)+lane_word;
        float x0=input[k0],x1=input[k0+32u],x2=input[k0+64u],x3=input[k0+96u];
        vx_u32 w0=wp[0],w1=wp[step_words],w2=wp[step_words*2u],w3=wp[step_words*3u];
        invalid|=!(x0<=3.402823466e38f && x0>=-3.402823466e38f);
        invalid|=!(x1<=3.402823466e38f && x1>=-3.402823466e38f);
        invalid|=!(x2<=3.402823466e38f && x2>=-3.402823466e38f);
        invalid|=!(x3<=3.402823466e38f && x3>=-3.402823466e38f);
        sum+=vx_packed_cuda_half(x0)*vx_packed_cuda_half((float)((int)((w0>>shift)&mask)-offset)*scale);
        sum+=vx_packed_cuda_half(x1)*vx_packed_cuda_half((float)((int)((w1>>shift)&mask)-offset)*scale);
        sum+=vx_packed_cuda_half(x2)*vx_packed_cuda_half((float)((int)((w2>>shift)&mask)-offset)*scale);
        sum+=vx_packed_cuda_half(x3)*vx_packed_cuda_half((float)((int)((w3>>shift)&mask)-offset)*scale);
    }
    for(vx_u32 k=full_groups*128u+lane;k<(vx_u32)a->key_dim;k+=32u) {
        float x_val=input[k];
        invalid|=!(x_val<=3.402823466e38f && x_val>=-3.402823466e38f);
        vx_u32 word=w_row[k>>word_shift];
        int q=(int)((word>>shift)&mask)-offset;
        float weight=vx_packed_cuda_half((float)q*s_row[k>>7u]);
        sum+=vx_packed_cuda_half(x_val)*weight;
    }
    for(vx_u32 delta=16u;delta;delta>>=1u) {
        sum+=vx_cuda_gemv_shuffle_down(sum,delta);
        vx_u32 other_inv;
        asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other_inv) : "r"(invalid),"r"(delta));
        invalid|=other_inv;
    }
    if(!lane) {
        if(invalid) {
            union {vx_u32 bits;float value;} nan;nan.bits=0x7fc00000u;
            a->y[0][unit]=nan.value;
        } else {
            a->y[0][unit]=sum+(a->x[3]?a->x[3][column]:0.0f);
        }
    }
}
/* A warp owns one output and reads packed K words with contiguous lanes. A8
 * reduces each INT32 group exactly; A16 reduces FP32 products of RN-even half
 * operands. No full-size dequantized weight or request workspace is created. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_packed_cuda_column(const VxLLMOpArgs* a) {
    vx_u32 lane=threadIdx.x&31u;
    vx_u64 unit=(vx_u64)blockIdx.x*2u+(threadIdx.x>>5u);
    vx_u64 count=(vx_u64)a->batch*(vx_u32)a->sequence*(vx_u32)a->value_dim;
    vx_u64 first=a->first_unit,units=a->unit_count?a->unit_count:count;
    if(unit>=units || first>count || units>count-first)return;
    unit+=first;
    vx_u64 row=unit/(vx_u32)a->value_dim,column=unit%(vx_u32)a->value_dim;
    if(a->normalize!=8) {
        if(a->groups==8)vx_packed_cuda_column_a16<true>(a,unit,row,column);
        else vx_packed_cuda_column_a16<false>(a,unit,row,column);
        return;
    }
    const float* input=a->x[0]+row*(vx_u32)a->key_dim;
    vx_u32 groups=(a->key_dim+127u)/128u;
    float maximum=0.0f;
    for(vx_u32 k=lane;k<(vx_u32)a->key_dim;k+=32u) {
        float value=input[k];
        if(!(value<=3.402823466e38f && value>=-3.402823466e38f))maximum=__builtin_inff();
        else maximum=vx_max(maximum,vx_abs(value));
    }
    for(vx_u32 delta=16u;delta;delta>>=1u)maximum=vx_max(maximum,vx_cuda_gemv_shuffle_down(maximum,delta));
    maximum=vx_packed_cuda_broadcast(maximum);
    if(maximum==__builtin_inff()) {
        if(!lane) {union {vx_u32 bits;float value;} nan;nan.bits=0x7fc00000u;a->y[0][unit]=nan.value;}
        return;
    }
    float sum=0.0f;
    float activation_scale=maximum>0.0f?vx_max(maximum/127.0f,1.1754943508222875e-38f):1.0f;
    for(vx_u32 group=0u;group<groups;group++) {
        int dot=0;
        vx_u32 end=(group+1u)*128u;
        if(end>(vx_u32)a->key_dim)end=(vx_u32)a->key_dim;
        for(vx_u32 k=group*128u+lane;k<end;k+=32u)
            dot+=vx_packed_cuda_quantize(input[k],activation_scale)*vx_packed_cuda_weight(a,column,k);
        for(vx_u32 delta=16u;delta;delta>>=1u) {
            int other;
            asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(dot),"r"(delta));
            dot+=other;
        }
        if(!lane)sum+=(float)dot*(activation_scale*a->x[2][column*groups+group]);
    }
    if(!lane)a->y[0][unit]=sum+(a->x[3]?a->x[3][column]:0.0f);
}
/* Four warps share RN-even half activations for a 16x32 (or adaptive 16x64)
 * output tile. A 68-word row stride avoids shared-memory bank conflicts; 4416
 * dynamic shared bytes include row validity. Each group-128 accumulation is
 * rounded to F32 before the next group, preserving the previous matrix
 * arithmetic. */
template <bool Prepared>
static VX_CUDA_DEVICE void vx_packed_cuda_a16_shared_tile(const VxLLMOpArgs* a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    vx_u32* activation_words=vx_packed_tile_shared;
    vx_u32* invalid_rows=vx_packed_tile_shared+1088u;
    vx_u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5u,group=lane>>2u,sub=lane&3u,pair=sub*2u;
    vx_u64 rows=(vx_u64)a->batch*a->sequence,columns=(vx_u32)a->value_dim;
    vx_u32 tile_cols=a->heads?(vx_u32)a->heads:
        vx_cuda_packed_linear_tile_columns(rows,(vx_u32)columns,16u,0u);
    vx_u64 column_tiles=(columns+tile_cols-1u)/tile_cols;
    vx_u64 row_base=(vx_u64)blockIdx.x/column_tiles*16u;
    if(row_base>=rows)return;
    vx_u64 column_base=((vx_u64)blockIdx.x%column_tiles)*tile_cols+warp*8u;
    for(vx_u32 r=0u;r<4u;r++) {
        vx_u64 row=row_base+warp*4u+r;
        if(Prepared) {
            if(!lane)invalid_rows[warp*4u+r]=row<rows?((const vx_u32*)a->x[5])[row]:0u;
            continue;
        }
        vx_u32 invalid=0u;
        if(row<rows)for(vx_u32 k=lane;k<(vx_u32)a->key_dim;k+=32u) {
            float value=a->x[0][row*(vx_u32)a->key_dim+k];
            invalid|=!(value<=3.402823466e38f && value>=-3.402823466e38f);
        }
        for(vx_u32 delta=16u;delta;delta>>=1u) {
            vx_u32 other;
            asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(invalid),"r"(delta));
            invalid|=other;
        }
        if(!lane)invalid_rows[warp*4u+r]=invalid;
    }
    vx_sync_threads();
    vx_u32 groups=((vx_u32)a->key_dim+127u)/128u;
    vx_u32 word_shift=a->groups==8?2u:3u,bits=a->groups==8?8u:4u,mask=a->groups==8?255u:15u;
    int offset=a->groups==8?128:8;
    vx_u64 weight_stride=((vx_u32)a->key_dim+(1u<<word_shift)-1u)>>word_shift;
    vx_u32 shift=(pair&((1u<<word_shift)-1u))*bits;
    vx_u32 col_blocks=tile_cols>>5u;
    float s0_0=0.0f,s0_1=0.0f,s0_2=0.0f,s0_3=0.0f;
    float s1_0=0.0f,s1_1=0.0f,s1_2=0.0f,s1_3=0.0f;
    for(vx_u32 g=0u;g<groups;g++) {
        for(vx_u32 item=threadIdx.x;item<1024u;item+=128u) {
            vx_u64 row=row_base+item/64u;
            vx_u32 k=g*128u+(item&63u)*2u;
            vx_u32 packed=0u;
            if(Prepared) {
                if(row<rows)packed=((const vx_u32*)a->x[4])[row*groups*64u+g*64u+(item&63u)];
            } else {
                float x0=row<rows && k<(vx_u32)a->key_dim?a->x[0][row*(vx_u32)a->key_dim+k]:0.0f;
                float x1=row<rows && k+1u<(vx_u32)a->key_dim?a->x[0][row*(vx_u32)a->key_dim+k+1u]:0.0f;
                packed=(vx_u32)vx_packed_cuda_half_bits(x0)|((vx_u32)vx_packed_cuda_half_bits(x1)<<16u);
            }
            activation_words[(item/64u)*68u+(item&63u)]=packed;
        }
        vx_sync_threads();
        bool full_k=(g*128u+128u<=(vx_u32)a->key_dim);
        #pragma unroll
        for(vx_u32 cb=0u;cb<2u;cb++) {
            if(cb>=col_blocks)break;
            vx_u64 weight_column=column_base+cb*32u+group;
            float scale=weight_column<columns?a->x[2][weight_column*groups+g]:0.0f;
            float p0=0.0f,p1=0.0f,p2=0.0f,p3=0.0f;
            if(full_k && column_base+cb*32u+8u<=columns) {
                const vx_u32* wp=((const vx_u32*)a->x[1])+weight_column*weight_stride+((g*128u)>>word_shift);
                for(vx_u32 chunk=0u;chunk<16u;chunk++) {
                    vx_u32 av0=activation_words[group*68u+chunk*4u+sub];
                    vx_u32 av1=activation_words[(group+8u)*68u+chunk*4u+sub];
                    vx_u32 word=wp[(chunk*8u+pair)>>word_shift]>>shift;
                    float w0=(float)((int)(word&mask)-offset)*scale;
                    float w1=(float)((int)((word>>bits)&mask)-offset)*scale;
                    vx_u32 bv=(vx_u32)vx_packed_cuda_half_bits(w0)|((vx_u32)vx_packed_cuda_half_bits(w1)<<16u);
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                        : "+f"(p0),"+f"(p1),"+f"(p2),"+f"(p3)
                        : "r"(av0),"r"(av1),"r"(bv));
                }
            } else {
                for(vx_u32 chunk=0u;chunk<16u;chunk++) {
                    vx_u32 k=g*128u+chunk*8u+pair;
                    vx_u32 av0=activation_words[group*68u+chunk*4u+sub];
                    vx_u32 av1=activation_words[(group+8u)*68u+chunk*4u+sub];
                    vx_u32 word=weight_column<columns && k<(vx_u32)a->key_dim?
                        ((const vx_u32*)a->x[1])[weight_column*weight_stride+(k>>word_shift)]:0u;
                    word>>=shift;
                    float w0=k<(vx_u32)a->key_dim?(float)((int)(word&mask)-offset)*scale:0.0f;
                    float w1=k+1u<(vx_u32)a->key_dim?(float)((int)((word>>bits)&mask)-offset)*scale:0.0f;
                    vx_u32 bv=(vx_u32)vx_packed_cuda_half_bits(w0)|((vx_u32)vx_packed_cuda_half_bits(w1)<<16u);
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                        : "+f"(p0),"+f"(p1),"+f"(p2),"+f"(p3)
                        : "r"(av0),"r"(av1),"r"(bv));
                }
            }
            if(cb==0u) {s0_0+=p0;s0_1+=p1;s0_2+=p2;s0_3+=p3;}
            else       {s1_0+=p0;s1_1+=p1;s1_2+=p2;s1_3+=p3;}
        }
        vx_sync_threads();
    }
    union {vx_u32 bits;float value;} nan;nan.bits=0x7fc00000u;
    #pragma unroll
    for(vx_u32 cb=0u;cb<2u;cb++) {
        if(cb>=col_blocks)break;
        vx_u64 col=column_base+cb*32u+pair;
        float c00=cb==0u?s0_0:s1_0,c01=cb==0u?s0_1:s1_1;
        float c10=cb==0u?s0_2:s1_2,c11=cb==0u?s0_3:s1_3;
        vx_u64 row0=row_base+group,row1=row_base+group+8u;
        if(row0<rows) {
            if(col<columns)a->y[0][row0*columns+col]=invalid_rows[group]?nan.value:c00+(a->x[3]?a->x[3][col]:0.0f);
            if(col+1u<columns)a->y[0][row0*columns+col+1u]=invalid_rows[group]?nan.value:c01+(a->x[3]?a->x[3][col+1u]:0.0f);
        }
        if(row1<rows) {
            if(col<columns)a->y[0][row1*columns+col]=invalid_rows[group+8u]?nan.value:c10+(a->x[3]?a->x[3][col]:0.0f);
            if(col+1u<columns)a->y[0][row1*columns+col+1u]=invalid_rows[group+8u]?nan.value:c11+(a->x[3]?a->x[3][col+1u]:0.0f);
        }
    }
#endif
}
/* Prepared matrix entries keep their register footprint independent of the
 * general LLM dispatcher. A8 reuses each signed weight fragment across 16,
 * 32 or 64 token rows; integer dots and ascending K128 scale accumulation
 * retain the PackedLinear numerical contract. A16 reuses each rounded half
 * weight pair across two M16 fragments, retaining its K8/group sum order.
 * The host selects only qualified geometry and provides bounded scratch.
 * Older compiled targets retain the existing dispatcher. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_packed_a8_m16(VxLLMOpArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const vx_u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5u;
    const vx_u32 group=lane>>2u,sub=lane&3u;
    const vx_u32 rows=a.sequence,columns=a.value_dim,k=a.key_dim,groups=(k+127u)/128u;
    const vx_u32 column_tiles=(columns+63u)/64u;
    const vx_u32 row_base=blockIdx.x/column_tiles*16u,column_base=blockIdx.x%column_tiles*64u;
    const vx_u32 wcol=column_base+warp*8u+group;
    const vx_u32 column=column_base+warp*8u+sub*2u;
    const vx_u32 shift=a.groups==8?2u:3u,wstride=(k+(1u<<shift)-1u)>>shift;
    vx_u32* act=vx_packed_tile_shared;
    float s00=0.0f;
    float s01=0.0f;
    float s02=0.0f;
    float s03=0.0f;
    float as00=row_base+0u+group<rows?a.x[5][row_base+0u+group]:1.0f;
    float as01=row_base+8u+group<rows?a.x[5][row_base+8u+group]:1.0f;
    for(vx_u32 g=0u;g<groups;g++) {
        for(vx_u32 i=threadIdx.x;i<512u;i+=256u) {
            vx_u32 row=i>>5u,word=i&31u;
            act[row*36u+word]=row_base+row<rows?((const vx_u32*)a.x[4])[
                (vx_u64)(row_base+row)*groups*32u+g*32u+word]:0u;
        }
        vx_sync_threads();
        int d00=0,d01=0,d02=0,d03=0;
        #pragma unroll 2
        for(vx_u32 chunk=0u;chunk<4u;chunk++) {
            vx_u32 word=chunk*8u+sub;
            vx_u32 b0=0u,b1=0u;
            if(wcol<columns && g*128u+128u<=k) {
                const vx_u32* wp=(const vx_u32*)a.x[1]+(vx_u64)wcol*wstride+(g*128u>>shift);
                vx_u32 raw0=wp[word*4u>>shift],raw1=wp[(word+4u)*4u>>shift];
                b0=a.groups==8?(raw0^0x80808080u):(sub&1u?vx_packed_cuda_unpack_w4_high(raw0):vx_packed_cuda_unpack_w4_low(raw0));
                b1=a.groups==8?(raw1^0x80808080u):(sub&1u?vx_packed_cuda_unpack_w4_high(raw1):vx_packed_cuda_unpack_w4_low(raw1));
            } else if(wcol<columns) {
                b0=vx_packed_cuda_weight4(&a,wcol,g*128u+word*4u);
                b1=vx_packed_cuda_weight4(&a,wcol,g*128u+(word+4u)*4u);
            }
            {
                vx_u32 a0=act[(group+0u)*36u+word],a1=act[(group+8u)*36u+word];
                vx_u32 a2=act[(group+0u)*36u+word+4u],a3=act[(group+8u)*36u+word+4u];
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(d00),"+r"(d01),"+r"(d02),"+r"(d03)
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        if(column+0u<columns) {
            float ws=a.x[2][(vx_u64)(column+0u)*groups+g];
            s00=vx_add_rn(s00,vx_mul_rn((float)d00,vx_mul_rn(as00,ws)));
            s02=vx_add_rn(s02,vx_mul_rn((float)d02,vx_mul_rn(as01,ws)));
        }
        if(column+1u<columns) {
            float ws=a.x[2][(vx_u64)(column+1u)*groups+g];
            s01=vx_add_rn(s01,vx_mul_rn((float)d01,vx_mul_rn(as00,ws)));
            s03=vx_add_rn(s03,vx_mul_rn((float)d03,vx_mul_rn(as01,ws)));
        }
        vx_sync_threads();
    }
    union {vx_u32 bits;float value;} nan;nan.bits=0x7fc00000u;
    if(row_base+0u+group<rows && column+0u<columns)
        a.y[0][(vx_u64)(row_base+0u+group)*columns+column+0u]=
        as00==__builtin_inff()?nan.value:vx_add_rn(s00,a.x[3]?a.x[3][column+0u]:0.0f);
    if(row_base+0u+group<rows && column+1u<columns)
        a.y[0][(vx_u64)(row_base+0u+group)*columns+column+1u]=
        as00==__builtin_inff()?nan.value:vx_add_rn(s01,a.x[3]?a.x[3][column+1u]:0.0f);
    if(row_base+8u+group<rows && column+0u<columns)
        a.y[0][(vx_u64)(row_base+8u+group)*columns+column+0u]=
        as01==__builtin_inff()?nan.value:vx_add_rn(s02,a.x[3]?a.x[3][column+0u]:0.0f);
    if(row_base+8u+group<rows && column+1u<columns)
        a.y[0][(vx_u64)(row_base+8u+group)*columns+column+1u]=
        as01==__builtin_inff()?nan.value:vx_add_rn(s03,a.x[3]?a.x[3][column+1u]:0.0f);
#endif
}


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_packed_cuda_stage_a8_async(
        const VxLLMOpArgs* a, vx_u32 row_base, vx_u32 g, vx_u32 groups,
        vx_u32* stage) {
    for(vx_u32 v=threadIdx.x;v<512u;v+=256u) {
        vx_u32 row=v>>3u, word=(v&7u)*4u;
        vx_u32 valid=row_base+row<(vx_u32)a->sequence?16u:0u;
        const vx_u32* source=(const vx_u32*)a->x[4]+
            (valid?((vx_u64)(row_base+row)*groups*32u+g*32u+word):0u);
        vx_u32* destination=stage+row*36u+word;
        vx_u64 shared_address;
        asm volatile("cvta.to.shared.u64 %0, %1;"
            : "=l"(shared_address): "l"(destination));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"((vx_u32)shared_address), "l"(source), "r"(valid): "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
#endif

extern "C" VX_CUDA_GLOBAL void vx_cuda_packed_a8_m64(VxLLMOpArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const vx_u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5u;
    const vx_u32 group=lane>>2u,sub=lane&3u;
    const vx_u32 rows=a.sequence,columns=a.value_dim,k=a.key_dim,groups=(k+127u)/128u;
    const vx_u32 column_tiles=(columns+63u)/64u;
    const vx_u32 row_base=blockIdx.x/column_tiles*64u,column_base=blockIdx.x%column_tiles*64u;
    const vx_u32 wcol=column_base+warp*8u+group;
    const vx_u32 column=column_base+warp*8u+sub*2u;
    const vx_u32 shift=a.groups==8?2u:3u,wstride=(k+(1u<<shift)-1u)>>shift;
    vx_u32* act=vx_packed_tile_shared;
    float s00=0.0f;
    float s01=0.0f;
    float s02=0.0f;
    float s03=0.0f;
    float as00=row_base+0u+group<rows?a.x[5][row_base+0u+group]:1.0f;
    float as01=row_base+8u+group<rows?a.x[5][row_base+8u+group]:1.0f;
    float s10=0.0f;
    float s11=0.0f;
    float s12=0.0f;
    float s13=0.0f;
    float as10=row_base+16u+group<rows?a.x[5][row_base+16u+group]:1.0f;
    float as11=row_base+24u+group<rows?a.x[5][row_base+24u+group]:1.0f;
    float s20=0.0f;
    float s21=0.0f;
    float s22=0.0f;
    float s23=0.0f;
    float as20=row_base+32u+group<rows?a.x[5][row_base+32u+group]:1.0f;
    float as21=row_base+40u+group<rows?a.x[5][row_base+40u+group]:1.0f;
    float s30=0.0f;
    float s31=0.0f;
    float s32=0.0f;
    float s33=0.0f;
    float as30=row_base+48u+group<rows?a.x[5][row_base+48u+group]:1.0f;
    float as31=row_base+56u+group<rows?a.x[5][row_base+56u+group]:1.0f;
    vx_packed_cuda_stage_a8_async(&a,row_base,0u,groups,vx_packed_tile_shared);
    for(vx_u32 g=0u;g<groups;g++) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        vx_sync_threads();
        act=vx_packed_tile_shared+(g&1u)*2304u;
        if(g+1u<groups)
            vx_packed_cuda_stage_a8_async(&a,row_base,g+1u,groups,
                vx_packed_tile_shared+((g+1u)&1u)*2304u);
        int d00=0,d01=0,d02=0,d03=0;
        int d10=0,d11=0,d12=0,d13=0;
        int d20=0,d21=0,d22=0,d23=0;
        int d30=0,d31=0,d32=0,d33=0;
        #pragma unroll 2
        for(vx_u32 chunk=0u;chunk<4u;chunk++) {
            vx_u32 word=chunk*8u+sub;
            vx_u32 b0=0u,b1=0u;
            if(wcol<columns && g*128u+128u<=k) {
                const vx_u32* wp=(const vx_u32*)a.x[1]+(vx_u64)wcol*wstride+(g*128u>>shift);
                vx_u32 raw0=wp[word*4u>>shift],raw1=wp[(word+4u)*4u>>shift];
                b0=a.groups==8?(raw0^0x80808080u):(sub&1u?vx_packed_cuda_unpack_w4_high(raw0):vx_packed_cuda_unpack_w4_low(raw0));
                b1=a.groups==8?(raw1^0x80808080u):(sub&1u?vx_packed_cuda_unpack_w4_high(raw1):vx_packed_cuda_unpack_w4_low(raw1));
            } else if(wcol<columns) {
                b0=vx_packed_cuda_weight4(&a,wcol,g*128u+word*4u);
                b1=vx_packed_cuda_weight4(&a,wcol,g*128u+(word+4u)*4u);
            }
            {
                vx_u32 a0=act[(group+0u)*36u+word],a1=act[(group+8u)*36u+word];
                vx_u32 a2=act[(group+0u)*36u+word+4u],a3=act[(group+8u)*36u+word+4u];
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(d00),"+r"(d01),"+r"(d02),"+r"(d03)
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
            {
                vx_u32 a0=act[(group+16u)*36u+word],a1=act[(group+24u)*36u+word];
                vx_u32 a2=act[(group+16u)*36u+word+4u],a3=act[(group+24u)*36u+word+4u];
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(d10),"+r"(d11),"+r"(d12),"+r"(d13)
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
            {
                vx_u32 a0=act[(group+32u)*36u+word],a1=act[(group+40u)*36u+word];
                vx_u32 a2=act[(group+32u)*36u+word+4u],a3=act[(group+40u)*36u+word+4u];
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(d20),"+r"(d21),"+r"(d22),"+r"(d23)
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
            {
                vx_u32 a0=act[(group+48u)*36u+word],a1=act[(group+56u)*36u+word];
                vx_u32 a2=act[(group+48u)*36u+word+4u],a3=act[(group+56u)*36u+word+4u];
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(d30),"+r"(d31),"+r"(d32),"+r"(d33)
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        if(column+0u<columns) {
            float ws=a.x[2][(vx_u64)(column+0u)*groups+g];
            s00=vx_add_rn(s00,vx_mul_rn((float)d00,vx_mul_rn(as00,ws)));
            s02=vx_add_rn(s02,vx_mul_rn((float)d02,vx_mul_rn(as01,ws)));
            s10=vx_add_rn(s10,vx_mul_rn((float)d10,vx_mul_rn(as10,ws)));
            s12=vx_add_rn(s12,vx_mul_rn((float)d12,vx_mul_rn(as11,ws)));
            s20=vx_add_rn(s20,vx_mul_rn((float)d20,vx_mul_rn(as20,ws)));
            s22=vx_add_rn(s22,vx_mul_rn((float)d22,vx_mul_rn(as21,ws)));
            s30=vx_add_rn(s30,vx_mul_rn((float)d30,vx_mul_rn(as30,ws)));
            s32=vx_add_rn(s32,vx_mul_rn((float)d32,vx_mul_rn(as31,ws)));
        }
        if(column+1u<columns) {
            float ws=a.x[2][(vx_u64)(column+1u)*groups+g];
            s01=vx_add_rn(s01,vx_mul_rn((float)d01,vx_mul_rn(as00,ws)));
            s03=vx_add_rn(s03,vx_mul_rn((float)d03,vx_mul_rn(as01,ws)));
            s11=vx_add_rn(s11,vx_mul_rn((float)d11,vx_mul_rn(as10,ws)));
            s13=vx_add_rn(s13,vx_mul_rn((float)d13,vx_mul_rn(as11,ws)));
            s21=vx_add_rn(s21,vx_mul_rn((float)d21,vx_mul_rn(as20,ws)));
            s23=vx_add_rn(s23,vx_mul_rn((float)d23,vx_mul_rn(as21,ws)));
            s31=vx_add_rn(s31,vx_mul_rn((float)d31,vx_mul_rn(as30,ws)));
            s33=vx_add_rn(s33,vx_mul_rn((float)d33,vx_mul_rn(as31,ws)));
        }
    }
    union {vx_u32 bits;float value;} nan;nan.bits=0x7fc00000u;
    if(row_base+0u+group<rows && column+0u<columns)
        a.y[0][(vx_u64)(row_base+0u+group)*columns+column+0u]=
        as00==__builtin_inff()?nan.value:vx_add_rn(s00,a.x[3]?a.x[3][column+0u]:0.0f);
    if(row_base+0u+group<rows && column+1u<columns)
        a.y[0][(vx_u64)(row_base+0u+group)*columns+column+1u]=
        as00==__builtin_inff()?nan.value:vx_add_rn(s01,a.x[3]?a.x[3][column+1u]:0.0f);
    if(row_base+8u+group<rows && column+0u<columns)
        a.y[0][(vx_u64)(row_base+8u+group)*columns+column+0u]=
        as01==__builtin_inff()?nan.value:vx_add_rn(s02,a.x[3]?a.x[3][column+0u]:0.0f);
    if(row_base+8u+group<rows && column+1u<columns)
        a.y[0][(vx_u64)(row_base+8u+group)*columns+column+1u]=
        as01==__builtin_inff()?nan.value:vx_add_rn(s03,a.x[3]?a.x[3][column+1u]:0.0f);
    if(row_base+16u+group<rows && column+0u<columns)
        a.y[0][(vx_u64)(row_base+16u+group)*columns+column+0u]=
        as10==__builtin_inff()?nan.value:vx_add_rn(s10,a.x[3]?a.x[3][column+0u]:0.0f);
    if(row_base+16u+group<rows && column+1u<columns)
        a.y[0][(vx_u64)(row_base+16u+group)*columns+column+1u]=
        as10==__builtin_inff()?nan.value:vx_add_rn(s11,a.x[3]?a.x[3][column+1u]:0.0f);
    if(row_base+24u+group<rows && column+0u<columns)
        a.y[0][(vx_u64)(row_base+24u+group)*columns+column+0u]=
        as11==__builtin_inff()?nan.value:vx_add_rn(s12,a.x[3]?a.x[3][column+0u]:0.0f);
    if(row_base+24u+group<rows && column+1u<columns)
        a.y[0][(vx_u64)(row_base+24u+group)*columns+column+1u]=
        as11==__builtin_inff()?nan.value:vx_add_rn(s13,a.x[3]?a.x[3][column+1u]:0.0f);
    if(row_base+32u+group<rows && column+0u<columns)
        a.y[0][(vx_u64)(row_base+32u+group)*columns+column+0u]=
        as20==__builtin_inff()?nan.value:vx_add_rn(s20,a.x[3]?a.x[3][column+0u]:0.0f);
    if(row_base+32u+group<rows && column+1u<columns)
        a.y[0][(vx_u64)(row_base+32u+group)*columns+column+1u]=
        as20==__builtin_inff()?nan.value:vx_add_rn(s21,a.x[3]?a.x[3][column+1u]:0.0f);
    if(row_base+40u+group<rows && column+0u<columns)
        a.y[0][(vx_u64)(row_base+40u+group)*columns+column+0u]=
        as21==__builtin_inff()?nan.value:vx_add_rn(s22,a.x[3]?a.x[3][column+0u]:0.0f);
    if(row_base+40u+group<rows && column+1u<columns)
        a.y[0][(vx_u64)(row_base+40u+group)*columns+column+1u]=
        as21==__builtin_inff()?nan.value:vx_add_rn(s23,a.x[3]?a.x[3][column+1u]:0.0f);
    if(row_base+48u+group<rows && column+0u<columns)
        a.y[0][(vx_u64)(row_base+48u+group)*columns+column+0u]=
        as30==__builtin_inff()?nan.value:vx_add_rn(s30,a.x[3]?a.x[3][column+0u]:0.0f);
    if(row_base+48u+group<rows && column+1u<columns)
        a.y[0][(vx_u64)(row_base+48u+group)*columns+column+1u]=
        as30==__builtin_inff()?nan.value:vx_add_rn(s31,a.x[3]?a.x[3][column+1u]:0.0f);
    if(row_base+56u+group<rows && column+0u<columns)
        a.y[0][(vx_u64)(row_base+56u+group)*columns+column+0u]=
        as31==__builtin_inff()?nan.value:vx_add_rn(s32,a.x[3]?a.x[3][column+0u]:0.0f);
    if(row_base+56u+group<rows && column+1u<columns)
        a.y[0][(vx_u64)(row_base+56u+group)*columns+column+1u]=
        as31==__builtin_inff()?nan.value:vx_add_rn(s33,a.x[3]?a.x[3][column+1u]:0.0f);
#endif
}

/* Reuse each rounded-half weight across four M16 fragments. Preserve the
 * original K8 MMA sequence and the ascending K128 group additions. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_packed_a16_m64(VxLLMOpArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const vx_u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5u,group=lane>>2u,sub=lane&3u,pair=sub*2u;
    const vx_u32 rows=a.sequence,columns=a.value_dim,kdim=a.key_dim,groups=(kdim+127u)/128u;
    const vx_u32 ct=(columns+63u)/64u,rb=blockIdx.x/ct*64u,cb=blockIdx.x%ct*64u+warp*8u;
    const vx_u32 word_shift=a.groups==8?2u:3u,bits=a.groups==8?8u:4u,mask=a.groups==8?255u:15u;
    const int offset=a.groups==8?128:8;
    const vx_u32 stride=(kdim+(1u<<word_shift)-1u)>>word_shift,shift=(pair&((1u<<word_shift)-1u))*bits;
    vx_u32* act=vx_packed_tile_shared;
    float s0_0_0=0.0f;
    float s0_0_1=0.0f;
    float s0_0_2=0.0f;
    float s0_0_3=0.0f;
    float s0_1_0=0.0f;
    float s0_1_1=0.0f;
    float s0_1_2=0.0f;
    float s0_1_3=0.0f;
    float s0_2_0=0.0f;
    float s0_2_1=0.0f;
    float s0_2_2=0.0f;
    float s0_2_3=0.0f;
    float s0_3_0=0.0f;
    float s0_3_1=0.0f;
    float s0_3_2=0.0f;
    float s0_3_3=0.0f;
    float s1_0_0=0.0f;
    float s1_0_1=0.0f;
    float s1_0_2=0.0f;
    float s1_0_3=0.0f;
    float s1_1_0=0.0f;
    float s1_1_1=0.0f;
    float s1_1_2=0.0f;
    float s1_1_3=0.0f;
    float s1_2_0=0.0f;
    float s1_2_1=0.0f;
    float s1_2_2=0.0f;
    float s1_2_3=0.0f;
    float s1_3_0=0.0f;
    float s1_3_1=0.0f;
    float s1_3_2=0.0f;
    float s1_3_3=0.0f;
    for(vx_u32 g=0u;g<groups;g++) {
        for(vx_u32 i=threadIdx.x;i<4096u;i+=128u) {
            vx_u32 row=rb+i/64u;
            act[(i/64u)*68u+(i&63u)]=row<rows?((const vx_u32*)a.x[4])[(vx_u64)row*groups*64u+g*64u+(i&63u)]:0u;
        }
        vx_sync_threads();
        {
            vx_u32 wc=cb+0u+group;
            float scale=wc<columns?a.x[2][(vx_u64)wc*groups+g]:0.0f;
            float p00=0.0f,p01=0.0f,p02=0.0f,p03=0.0f;
            float p10=0.0f,p11=0.0f,p12=0.0f,p13=0.0f;
            float p20=0.0f,p21=0.0f,p22=0.0f,p23=0.0f;
            float p30=0.0f,p31=0.0f,p32=0.0f,p33=0.0f;
            for(vx_u32 chunk=0u;chunk<16u;chunk++) {
                vx_u32 k=g*128u+chunk*8u+pair;
                vx_u32 word=wc<columns && k<kdim?((const vx_u32*)a.x[1])[(vx_u64)wc*stride+(k>>word_shift)]:0u;
                word>>=shift;
                float w0=k<kdim?(float)((int)(word&mask)-offset)*scale:0.0f;
                float w1=k+1u<kdim?(float)((int)((word>>bits)&mask)-offset)*scale:0.0f;
                vx_u32 bv=(vx_u32)vx_packed_cuda_half_bits(w0)|((vx_u32)vx_packed_cuda_half_bits(w1)<<16u);
                {
                    vx_u32 av0=act[(group+0u)*68u+chunk*4u+sub],av1=act[(group+8u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p00),"+f"(p01),"+f"(p02),"+f"(p03):"r"(av0),"r"(av1),"r"(bv));
                }
                {
                    vx_u32 av0=act[(group+16u)*68u+chunk*4u+sub],av1=act[(group+24u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p10),"+f"(p11),"+f"(p12),"+f"(p13):"r"(av0),"r"(av1),"r"(bv));
                }
                {
                    vx_u32 av0=act[(group+32u)*68u+chunk*4u+sub],av1=act[(group+40u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p20),"+f"(p21),"+f"(p22),"+f"(p23):"r"(av0),"r"(av1),"r"(bv));
                }
                {
                    vx_u32 av0=act[(group+48u)*68u+chunk*4u+sub],av1=act[(group+56u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p30),"+f"(p31),"+f"(p32),"+f"(p33):"r"(av0),"r"(av1),"r"(bv));
                }
            }
            s0_0_0=vx_add_rn(s0_0_0,p00);
            s0_0_1=vx_add_rn(s0_0_1,p01);
            s0_0_2=vx_add_rn(s0_0_2,p02);
            s0_0_3=vx_add_rn(s0_0_3,p03);
            s0_1_0=vx_add_rn(s0_1_0,p10);
            s0_1_1=vx_add_rn(s0_1_1,p11);
            s0_1_2=vx_add_rn(s0_1_2,p12);
            s0_1_3=vx_add_rn(s0_1_3,p13);
            s0_2_0=vx_add_rn(s0_2_0,p20);
            s0_2_1=vx_add_rn(s0_2_1,p21);
            s0_2_2=vx_add_rn(s0_2_2,p22);
            s0_2_3=vx_add_rn(s0_2_3,p23);
            s0_3_0=vx_add_rn(s0_3_0,p30);
            s0_3_1=vx_add_rn(s0_3_1,p31);
            s0_3_2=vx_add_rn(s0_3_2,p32);
            s0_3_3=vx_add_rn(s0_3_3,p33);
        }
        {
            vx_u32 wc=cb+32u+group;
            float scale=wc<columns?a.x[2][(vx_u64)wc*groups+g]:0.0f;
            float p00=0.0f,p01=0.0f,p02=0.0f,p03=0.0f;
            float p10=0.0f,p11=0.0f,p12=0.0f,p13=0.0f;
            float p20=0.0f,p21=0.0f,p22=0.0f,p23=0.0f;
            float p30=0.0f,p31=0.0f,p32=0.0f,p33=0.0f;
            for(vx_u32 chunk=0u;chunk<16u;chunk++) {
                vx_u32 k=g*128u+chunk*8u+pair;
                vx_u32 word=wc<columns && k<kdim?((const vx_u32*)a.x[1])[(vx_u64)wc*stride+(k>>word_shift)]:0u;
                word>>=shift;
                float w0=k<kdim?(float)((int)(word&mask)-offset)*scale:0.0f;
                float w1=k+1u<kdim?(float)((int)((word>>bits)&mask)-offset)*scale:0.0f;
                vx_u32 bv=(vx_u32)vx_packed_cuda_half_bits(w0)|((vx_u32)vx_packed_cuda_half_bits(w1)<<16u);
                {
                    vx_u32 av0=act[(group+0u)*68u+chunk*4u+sub],av1=act[(group+8u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p00),"+f"(p01),"+f"(p02),"+f"(p03):"r"(av0),"r"(av1),"r"(bv));
                }
                {
                    vx_u32 av0=act[(group+16u)*68u+chunk*4u+sub],av1=act[(group+24u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p10),"+f"(p11),"+f"(p12),"+f"(p13):"r"(av0),"r"(av1),"r"(bv));
                }
                {
                    vx_u32 av0=act[(group+32u)*68u+chunk*4u+sub],av1=act[(group+40u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p20),"+f"(p21),"+f"(p22),"+f"(p23):"r"(av0),"r"(av1),"r"(bv));
                }
                {
                    vx_u32 av0=act[(group+48u)*68u+chunk*4u+sub],av1=act[(group+56u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p30),"+f"(p31),"+f"(p32),"+f"(p33):"r"(av0),"r"(av1),"r"(bv));
                }
            }
            s1_0_0=vx_add_rn(s1_0_0,p00);
            s1_0_1=vx_add_rn(s1_0_1,p01);
            s1_0_2=vx_add_rn(s1_0_2,p02);
            s1_0_3=vx_add_rn(s1_0_3,p03);
            s1_1_0=vx_add_rn(s1_1_0,p10);
            s1_1_1=vx_add_rn(s1_1_1,p11);
            s1_1_2=vx_add_rn(s1_1_2,p12);
            s1_1_3=vx_add_rn(s1_1_3,p13);
            s1_2_0=vx_add_rn(s1_2_0,p20);
            s1_2_1=vx_add_rn(s1_2_1,p21);
            s1_2_2=vx_add_rn(s1_2_2,p22);
            s1_2_3=vx_add_rn(s1_2_3,p23);
            s1_3_0=vx_add_rn(s1_3_0,p30);
            s1_3_1=vx_add_rn(s1_3_1,p31);
            s1_3_2=vx_add_rn(s1_3_2,p32);
            s1_3_3=vx_add_rn(s1_3_3,p33);
        }
        vx_sync_threads();
    }
    union {vx_u32 bits;float value;} nan;nan.bits=0x7fc00000u;
    if(rb+0u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+0u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+0u+group]?nan.value:
            vx_add_rn(s0_0_0,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+0u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+0u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+0u+group]?nan.value:
            vx_add_rn(s0_0_1,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+8u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+8u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+8u+group]?nan.value:
            vx_add_rn(s0_0_2,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+8u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+8u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+8u+group]?nan.value:
            vx_add_rn(s0_0_3,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+16u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+16u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+16u+group]?nan.value:
            vx_add_rn(s0_1_0,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+16u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+16u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+16u+group]?nan.value:
            vx_add_rn(s0_1_1,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+24u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+24u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+24u+group]?nan.value:
            vx_add_rn(s0_1_2,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+24u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+24u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+24u+group]?nan.value:
            vx_add_rn(s0_1_3,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+32u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+32u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+32u+group]?nan.value:
            vx_add_rn(s0_2_0,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+32u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+32u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+32u+group]?nan.value:
            vx_add_rn(s0_2_1,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+40u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+40u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+40u+group]?nan.value:
            vx_add_rn(s0_2_2,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+40u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+40u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+40u+group]?nan.value:
            vx_add_rn(s0_2_3,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+48u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+48u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+48u+group]?nan.value:
            vx_add_rn(s0_3_0,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+48u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+48u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+48u+group]?nan.value:
            vx_add_rn(s0_3_1,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+56u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+56u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+56u+group]?nan.value:
            vx_add_rn(s0_3_2,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+56u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+56u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+56u+group]?nan.value:
            vx_add_rn(s0_3_3,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+0u+group<rows && cb+32u+pair<columns)
        a.y[0][(vx_u64)(rb+0u+group)*columns+cb+32u+pair]=((const vx_u32*)a.x[5])[rb+0u+group]?nan.value:
            vx_add_rn(s1_0_0,a.x[3]?a.x[3][cb+32u+pair]:0.0f);
    if(rb+0u+group<rows && cb+33u+pair<columns)
        a.y[0][(vx_u64)(rb+0u+group)*columns+cb+33u+pair]=((const vx_u32*)a.x[5])[rb+0u+group]?nan.value:
            vx_add_rn(s1_0_1,a.x[3]?a.x[3][cb+33u+pair]:0.0f);
    if(rb+8u+group<rows && cb+32u+pair<columns)
        a.y[0][(vx_u64)(rb+8u+group)*columns+cb+32u+pair]=((const vx_u32*)a.x[5])[rb+8u+group]?nan.value:
            vx_add_rn(s1_0_2,a.x[3]?a.x[3][cb+32u+pair]:0.0f);
    if(rb+8u+group<rows && cb+33u+pair<columns)
        a.y[0][(vx_u64)(rb+8u+group)*columns+cb+33u+pair]=((const vx_u32*)a.x[5])[rb+8u+group]?nan.value:
            vx_add_rn(s1_0_3,a.x[3]?a.x[3][cb+33u+pair]:0.0f);
    if(rb+16u+group<rows && cb+32u+pair<columns)
        a.y[0][(vx_u64)(rb+16u+group)*columns+cb+32u+pair]=((const vx_u32*)a.x[5])[rb+16u+group]?nan.value:
            vx_add_rn(s1_1_0,a.x[3]?a.x[3][cb+32u+pair]:0.0f);
    if(rb+16u+group<rows && cb+33u+pair<columns)
        a.y[0][(vx_u64)(rb+16u+group)*columns+cb+33u+pair]=((const vx_u32*)a.x[5])[rb+16u+group]?nan.value:
            vx_add_rn(s1_1_1,a.x[3]?a.x[3][cb+33u+pair]:0.0f);
    if(rb+24u+group<rows && cb+32u+pair<columns)
        a.y[0][(vx_u64)(rb+24u+group)*columns+cb+32u+pair]=((const vx_u32*)a.x[5])[rb+24u+group]?nan.value:
            vx_add_rn(s1_1_2,a.x[3]?a.x[3][cb+32u+pair]:0.0f);
    if(rb+24u+group<rows && cb+33u+pair<columns)
        a.y[0][(vx_u64)(rb+24u+group)*columns+cb+33u+pair]=((const vx_u32*)a.x[5])[rb+24u+group]?nan.value:
            vx_add_rn(s1_1_3,a.x[3]?a.x[3][cb+33u+pair]:0.0f);
    if(rb+32u+group<rows && cb+32u+pair<columns)
        a.y[0][(vx_u64)(rb+32u+group)*columns+cb+32u+pair]=((const vx_u32*)a.x[5])[rb+32u+group]?nan.value:
            vx_add_rn(s1_2_0,a.x[3]?a.x[3][cb+32u+pair]:0.0f);
    if(rb+32u+group<rows && cb+33u+pair<columns)
        a.y[0][(vx_u64)(rb+32u+group)*columns+cb+33u+pair]=((const vx_u32*)a.x[5])[rb+32u+group]?nan.value:
            vx_add_rn(s1_2_1,a.x[3]?a.x[3][cb+33u+pair]:0.0f);
    if(rb+40u+group<rows && cb+32u+pair<columns)
        a.y[0][(vx_u64)(rb+40u+group)*columns+cb+32u+pair]=((const vx_u32*)a.x[5])[rb+40u+group]?nan.value:
            vx_add_rn(s1_2_2,a.x[3]?a.x[3][cb+32u+pair]:0.0f);
    if(rb+40u+group<rows && cb+33u+pair<columns)
        a.y[0][(vx_u64)(rb+40u+group)*columns+cb+33u+pair]=((const vx_u32*)a.x[5])[rb+40u+group]?nan.value:
            vx_add_rn(s1_2_3,a.x[3]?a.x[3][cb+33u+pair]:0.0f);
    if(rb+48u+group<rows && cb+32u+pair<columns)
        a.y[0][(vx_u64)(rb+48u+group)*columns+cb+32u+pair]=((const vx_u32*)a.x[5])[rb+48u+group]?nan.value:
            vx_add_rn(s1_3_0,a.x[3]?a.x[3][cb+32u+pair]:0.0f);
    if(rb+48u+group<rows && cb+33u+pair<columns)
        a.y[0][(vx_u64)(rb+48u+group)*columns+cb+33u+pair]=((const vx_u32*)a.x[5])[rb+48u+group]?nan.value:
            vx_add_rn(s1_3_1,a.x[3]?a.x[3][cb+33u+pair]:0.0f);
    if(rb+56u+group<rows && cb+32u+pair<columns)
        a.y[0][(vx_u64)(rb+56u+group)*columns+cb+32u+pair]=((const vx_u32*)a.x[5])[rb+56u+group]?nan.value:
            vx_add_rn(s1_3_2,a.x[3]?a.x[3][cb+32u+pair]:0.0f);
    if(rb+56u+group<rows && cb+33u+pair<columns)
        a.y[0][(vx_u64)(rb+56u+group)*columns+cb+33u+pair]=((const vx_u32*)a.x[5])[rb+56u+group]?nan.value:
            vx_add_rn(s1_3_3,a.x[3]?a.x[3][cb+33u+pair]:0.0f);
#endif
}

/* Eight column warps retain the K8/group128 arithmetic with fewer live sums. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_packed_a16_m64_warps8(VxLLMOpArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const vx_u32 lane=threadIdx.x&31u,warp=threadIdx.x>>5u,group=lane>>2u,sub=lane&3u,pair=sub*2u;
    const vx_u32 rows=a.sequence,columns=a.value_dim,kdim=a.key_dim,groups=(kdim+127u)/128u;
    const vx_u32 ct=(columns+63u)/64u,rb=blockIdx.x/ct*64u,cb=blockIdx.x%ct*64u+warp*8u;
    const vx_u32 word_shift=a.groups==8?2u:3u,bits=a.groups==8?8u:4u,mask=a.groups==8?255u:15u;
    const int offset=a.groups==8?128:8;
    const vx_u32 stride=(kdim+(1u<<word_shift)-1u)>>word_shift,shift=(pair&((1u<<word_shift)-1u))*bits;
    vx_u32* act=vx_packed_tile_shared;
    float s0_0_0=0.0f;
    float s0_0_1=0.0f;
    float s0_0_2=0.0f;
    float s0_0_3=0.0f;
    float s0_1_0=0.0f;
    float s0_1_1=0.0f;
    float s0_1_2=0.0f;
    float s0_1_3=0.0f;
    float s0_2_0=0.0f;
    float s0_2_1=0.0f;
    float s0_2_2=0.0f;
    float s0_2_3=0.0f;
    float s0_3_0=0.0f;
    float s0_3_1=0.0f;
    float s0_3_2=0.0f;
    float s0_3_3=0.0f;
    for(vx_u32 g=0u;g<groups;g++) {
        for(vx_u32 i=threadIdx.x;i<4096u;i+=256u) {
            vx_u32 row=rb+i/64u;
            act[(i/64u)*68u+(i&63u)]=row<rows?((const vx_u32*)a.x[4])[(vx_u64)row*groups*64u+g*64u+(i&63u)]:0u;
        }
        vx_sync_threads();
        {
            vx_u32 wc=cb+0u+group;
            float scale=wc<columns?a.x[2][(vx_u64)wc*groups+g]:0.0f;
            float p00=0.0f,p01=0.0f,p02=0.0f,p03=0.0f;
            float p10=0.0f,p11=0.0f,p12=0.0f,p13=0.0f;
            float p20=0.0f,p21=0.0f,p22=0.0f,p23=0.0f;
            float p30=0.0f,p31=0.0f,p32=0.0f,p33=0.0f;
            for(vx_u32 chunk=0u;chunk<16u;chunk++) {
                vx_u32 k=g*128u+chunk*8u+pair;
                vx_u32 word=wc<columns && k<kdim?((const vx_u32*)a.x[1])[(vx_u64)wc*stride+(k>>word_shift)]:0u;
                word>>=shift;
                float w0=k<kdim?(float)((int)(word&mask)-offset)*scale:0.0f;
                float w1=k+1u<kdim?(float)((int)((word>>bits)&mask)-offset)*scale:0.0f;
                vx_u32 bv=(vx_u32)vx_packed_cuda_half_bits(w0)|((vx_u32)vx_packed_cuda_half_bits(w1)<<16u);
                {
                    vx_u32 av0=act[(group+0u)*68u+chunk*4u+sub],av1=act[(group+8u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p00),"+f"(p01),"+f"(p02),"+f"(p03):"r"(av0),"r"(av1),"r"(bv));
                }
                {
                    vx_u32 av0=act[(group+16u)*68u+chunk*4u+sub],av1=act[(group+24u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p10),"+f"(p11),"+f"(p12),"+f"(p13):"r"(av0),"r"(av1),"r"(bv));
                }
                {
                    vx_u32 av0=act[(group+32u)*68u+chunk*4u+sub],av1=act[(group+40u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p20),"+f"(p21),"+f"(p22),"+f"(p23):"r"(av0),"r"(av1),"r"(bv));
                }
                {
                    vx_u32 av0=act[(group+48u)*68u+chunk*4u+sub],av1=act[(group+56u)*68u+chunk*4u+sub];
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(p30),"+f"(p31),"+f"(p32),"+f"(p33):"r"(av0),"r"(av1),"r"(bv));
                }
            }
            s0_0_0=vx_add_rn(s0_0_0,p00);
            s0_0_1=vx_add_rn(s0_0_1,p01);
            s0_0_2=vx_add_rn(s0_0_2,p02);
            s0_0_3=vx_add_rn(s0_0_3,p03);
            s0_1_0=vx_add_rn(s0_1_0,p10);
            s0_1_1=vx_add_rn(s0_1_1,p11);
            s0_1_2=vx_add_rn(s0_1_2,p12);
            s0_1_3=vx_add_rn(s0_1_3,p13);
            s0_2_0=vx_add_rn(s0_2_0,p20);
            s0_2_1=vx_add_rn(s0_2_1,p21);
            s0_2_2=vx_add_rn(s0_2_2,p22);
            s0_2_3=vx_add_rn(s0_2_3,p23);
            s0_3_0=vx_add_rn(s0_3_0,p30);
            s0_3_1=vx_add_rn(s0_3_1,p31);
            s0_3_2=vx_add_rn(s0_3_2,p32);
            s0_3_3=vx_add_rn(s0_3_3,p33);
        }

        vx_sync_threads();
    }
    union {vx_u32 bits;float value;} nan;nan.bits=0x7fc00000u;
    if(rb+0u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+0u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+0u+group]?nan.value:
            vx_add_rn(s0_0_0,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+0u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+0u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+0u+group]?nan.value:
            vx_add_rn(s0_0_1,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+8u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+8u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+8u+group]?nan.value:
            vx_add_rn(s0_0_2,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+8u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+8u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+8u+group]?nan.value:
            vx_add_rn(s0_0_3,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+16u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+16u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+16u+group]?nan.value:
            vx_add_rn(s0_1_0,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+16u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+16u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+16u+group]?nan.value:
            vx_add_rn(s0_1_1,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+24u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+24u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+24u+group]?nan.value:
            vx_add_rn(s0_1_2,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+24u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+24u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+24u+group]?nan.value:
            vx_add_rn(s0_1_3,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+32u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+32u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+32u+group]?nan.value:
            vx_add_rn(s0_2_0,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+32u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+32u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+32u+group]?nan.value:
            vx_add_rn(s0_2_1,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+40u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+40u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+40u+group]?nan.value:
            vx_add_rn(s0_2_2,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+40u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+40u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+40u+group]?nan.value:
            vx_add_rn(s0_2_3,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+48u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+48u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+48u+group]?nan.value:
            vx_add_rn(s0_3_0,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+48u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+48u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+48u+group]?nan.value:
            vx_add_rn(s0_3_1,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
    if(rb+56u+group<rows && cb+0u+pair<columns)
        a.y[0][(vx_u64)(rb+56u+group)*columns+cb+0u+pair]=((const vx_u32*)a.x[5])[rb+56u+group]?nan.value:
            vx_add_rn(s0_3_2,a.x[3]?a.x[3][cb+0u+pair]:0.0f);
    if(rb+56u+group<rows && cb+1u+pair<columns)
        a.y[0][(vx_u64)(rb+56u+group)*columns+cb+1u+pair]=((const vx_u32*)a.x[5])[rb+56u+group]?nan.value:
            vx_add_rn(s0_3_3,a.x[3]?a.x[3][cb+1u+pair]:0.0f);
#endif
}

/* Scalar arguments let the replay row recipe refresh the input/output cursor
 * without rebuilding a by-value operator descriptor for every token. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_packed_linear_decode_f32(
        const float* input, const vx_u32* packed_weights, const float* weight_scales,
        const float* bias, float* output, vx_u32 input_width, vx_u32 output_width,
        vx_u32 weight_bits, vx_u32 activation_bits, vx_u32 window,
        const vx_u32* workspace, vx_u32 phase, const unsigned char* argument_storage) {
    if(argument_storage)vx_cuda_restore_arguments(argument_storage,input,packed_weights,
        weight_scales,bias,output,input_width,output_width,weight_bits,activation_bits,
        window,workspace,phase);
    /* Inline consumers see only these fields. Zeroing the entire descriptor
     * before late inlining materializes a per-thread 336-byte local object. */
    VxLLMOpArgs a;
    a.first_unit=0u;a.unit_count=0u;
    a.kind=VX_OP_PACKED_LINEAR;a.batch=1;a.sequence=1;
    a.key_dim=(int)input_width;a.value_dim=(int)output_width;a.rate=128;
    a.groups=(int)weight_bits;a.normalize=(int)activation_bits;a.window=(int)window;
    a.x[0]=input;a.x[1]=(const float*)packed_weights;a.x[2]=weight_scales;a.x[3]=bias;
    a.x[4]=(const float*)workspace;
    a.x[5]=workspace?(const float*)(workspace+((input_width+127u)/128u)*32u):0;
    a.y[0]=output;
    if(phase==1u)vx_packed_cuda_prepare(&a);
    else if(activation_bits==8u && (window==1u || window==4u)) {
        if(workspace)vx_packed_cuda_a8_columns<true,true>(&a);
        else vx_packed_cuda_a8_columns<false,true>(&a);
    } else vx_packed_cuda_column(&a);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_llm_f32(VxLLMOpArgs a) {
    if(a.kind==VX_OP_PACKED_LINEAR) {
        if(a.window==5)vx_packed_cuda_prepare(&a);
        else if(a.normalize==16 && a.window==3) {
            if(a.x[4])vx_packed_cuda_a16_shared_tile<true>(&a);
            else vx_packed_cuda_a16_shared_tile<false>(&a);
        }
        else if(a.normalize==8 && a.window==2)vx_packed_cuda_a8_tile(&a);
        else if(a.normalize==8 && (a.window==1 || a.window==4)) {
            if(a.x[4])vx_packed_cuda_a8_columns<true>(&a);
            else vx_packed_cuda_a8_columns<false>(&a);
        }
        else vx_packed_cuda_column(&a);
        return;
    }
    vx_u64 id=(vx_u64)blockIdx.x*blockDim.x+threadIdx.x;
    vx_u64 count=(vx_u64)a.batch*a.sequence;
    if(a.kind==VX_OP_SIGNED_SQRT || a.kind==VX_OP_EMBEDDING_SHARD) count=a.y_elements[0];
    else if(a.kind==VX_OP_GATED_DELTA_NET) count=(vx_u64)a.batch*a.heads*a.value_dim;
    else if(a.kind==VX_OP_SPARSE_ATTENTION) count*=a.heads*a.value_dim;
    else if(a.kind==VX_OP_HYPER_RESIDUAL) count*=a.streams*a.key_dim;
    else if(a.kind==VX_OP_ROTARY_EMBEDDING) count*=a.heads*a.key_dim;
    vx_u64 first=a.first_unit,units=a.unit_count?a.unit_count:count;
    if(id>=units || first>count || units>count-first) return;
    id+=first;
    if(a.kind==VX_OP_GATED_DELTA_NET) vx_llm_delta_column(&a,id);
    else if(a.kind==VX_OP_SPARSE_ATTENTION) vx_llm_attention_column(&a,id);
    else vx_llm_run_unit(&a,id,0);
}

#define VX_NGRAM_INLINE static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE
#include "../../../kernels/ngram_ids.h"
#undef VX_NGRAM_INLINE
extern "C" VX_CUDA_GLOBAL void vx_cuda_ngram_ids_i32(const vx_i32* tokens,const vx_i32* multipliers,
        const vx_i32* sizes,const vx_i32* offsets,vx_i32* output,vx_u32 batch,vx_u32 sequence,
        vx_u32 n,vx_u32 p,vx_i32 eos,vx_i32 pad,vx_i32 vocabulary,vx_u32 row_start,vx_u32 row_count) {
    vx_u64 id=(vx_u64)blockIdx.x*blockDim.x+threadIdx.x;
    vx_u32 heads=(n-1)*p;
    if(id>=(vx_u64)row_count*heads || (vx_u64)row_start+row_count>(vx_u64)batch*sequence) return;
    id+=(vx_u64)row_start*heads;
    vx_u64 row=id/heads;
    vx_u32 t=(vx_u32)(row%sequence),b=(vx_u32)(row/sequence),h=(vx_u32)(id%heads);
    for(vx_u32 shift=0;shift<n&&shift<=t;shift++) {
        vx_i32 value=tokens[(vx_u64)b*sequence+t-shift];
        if(value<0||value>=vocabulary) {output[id]=-1;return;}
    }
    output[id]=vx_ngram_id_at(tokens,multipliers,sizes,offsets,sequence,p,eos,pad,b,t,h);
}
