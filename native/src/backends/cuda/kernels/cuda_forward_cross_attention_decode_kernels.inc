/* Decode with the same FP32 accumulation order as the cached path. The four stages preserve the baseline's
 * ascending channel dot, 48-key denominator/PV groups, and ascending group merge.
 * Domain: B1, Hq32, Hkv4, D64, 512<=K<=2048, causal=0, mask0/1.
 * Scratch: 32*2048 score floats + 32*43*64 PV partial floats = 614400 bytes. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_cross_sdpa_gqa_decode_f32(
        const float* query_values, const float* key_values,
        const float* value_values, const vx_i32* mask, float* output,
        vx_u32 query_sequence, vx_u32 key_sequence, vx_u32 query_width,
        vx_u32 kv_width, vx_u32 heads, vx_u32 kv_heads, vx_u32 head_dim,
        vx_u32 batch, float scale, vx_u32 causal, vx_u32 mask_mode,
        float* scratch, vx_u32 phase, const unsigned char* argument_storage) {
    if(argument_storage)vx_cuda_restore_arguments(argument_storage,
        query_values,key_values,value_values,mask,output,query_sequence,key_sequence,
        query_width,kv_width,heads,kv_heads,head_dim,batch,scale,causal,mask_mode,scratch,phase);
    query_values=vx_cuda_global_pointer(query_values);
    key_values=vx_cuda_global_pointer(key_values);
    value_values=vx_cuda_global_pointer(value_values);
    output=vx_cuda_global_pointer(output);
    scratch=vx_cuda_global_pointer(scratch);
    mask=vx_cuda_global_pointer(mask);
    const vx_u32 head=blockIdx.x,lane=threadIdx.x,kvhead=head/8u;
    float* scores=scratch+head*2048u;
    float* partials=scratch+32u*2048u+head*43u*64u;
    if(phase==0u) {
        const vx_u32 key=blockIdx.y*128u+lane;
        if(key<key_sequence) {
            float score=0.0f;
            if(mask_mode==0u || mask[key]!=0) {
                for(vx_u32 channel=0;channel<64u;++channel)
                    score+=query_values[head*64u+channel]*
                        key_values[key*256u+kvhead*64u+channel];
            }
            scores[key]=score*scale;
        }
        return;
    }
    if(phase==1u) {
        VX_CUDA_SHARED float statistics[3];
        if(lane<32u) {
            float maximum=-3.402823466e38f,any=0.0f;
            for(vx_u32 key=lane;key<key_sequence;key+=32u) {
                if(mask_mode && mask[key]==0)continue;
                maximum=vx_max(maximum,scores[key]);any=1.0f;
            }
            for(vx_u32 delta=16u;delta;delta/=2u) {
                maximum=vx_max(maximum,vx_cuda_attention_shuffle_down(maximum,delta));
                any=vx_max(any,vx_cuda_attention_shuffle_down(any,delta));
            }
            if(!lane){statistics[0]=maximum;statistics[2]=any;}
        }
        vx_sync_threads();
        if(statistics[2]==0.0f) {
            for(vx_u32 key=lane;key<key_sequence;key+=128u)scores[key]=0.0f;
            return;
        }
        for(vx_u32 key=lane;key<key_sequence;key+=128u)
            scores[key]=(!mask_mode || mask[key]!=0)?vx_exp(scores[key]-statistics[0]):0.0f;
        vx_sync_threads();
        if(lane<32u) {
            float denominator=0.0f;
            for(vx_u32 first=0;first<key_sequence;first+=48u*32u) {
                vx_u32 start=first+lane*48u;
                vx_u32 count=start<key_sequence?(key_sequence-start<48u?key_sequence-start:48u):0u;
                float partial=0.0f;
                for(vx_u32 offset=0;offset<count;++offset) {
                    vx_u32 key=start+offset;
                    if(mask_mode && mask[key]==0)continue;
                    partial+=scores[key];
                }
                vx_u32 parts=(key_sequence-first+47u)/48u;
                if(parts>32u)parts=32u;
                for(vx_u32 part=0;part<parts;++part) {
                    float value=vx_cuda_attention_shuffle_lane(partial,part);
                    if(!lane)denominator+=value;
                }
            }
            if(!lane)statistics[1]=denominator;
        }
        vx_sync_threads();
        for(vx_u32 key=lane;key<key_sequence;key+=128u)scores[key]/=statistics[1];
        return;
    }
    if(phase==2u) {
        vx_u32 part=blockIdx.y,start=part*48u;
        vx_u32 count=start<key_sequence?(key_sequence-start<48u?key_sequence-start:48u):0u;
        float partial=0.0f;
        for(vx_u32 offset=0;offset<count;++offset) {
            vx_u32 key=start+offset;
            if(mask_mode && mask[key]==0)continue;
            partial+=scores[key]*value_values[key*256u+kvhead*64u+lane];
        }
        partials[part*64u+lane]=partial;
        return;
    }
    if(phase==3u) {
        float result=0.0f;
        vx_u32 parts=(key_sequence+47u)/48u;
        for(vx_u32 part=0;part<parts;++part)result+=partials[part*64u+lane];
        output[head*64u+lane]=result;
    }
}

/* Aligned decode uses one 1024-thread block for each head/channel half. This
 * retains the cached path's scalar QK and K48 partial/merge order, while the
 * device mask guard returns before Q/K/V reads when no key is permitted.
 * Domain: B1,Hq32,Hkv4,D64,K512..2048,causal0,mask0/1; Q/K aligned to16bytes.
 * Shared storage: 8192 score +136 statistics +4096 partial bytes, rounded12432.
 * The host retains the four-stage route when alignment/device limits fail. */
template <vx_u32 Columns, vx_u32 Threads>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_attention_decode_vector(
        const float* q,const float* k,const float* v,const vx_i32* mask,float* out,
        vx_u32 keys,float scale,vx_u32 masking) {
    VX_CUDA_SHARED float scores[2048];
    VX_CUDA_SHARED float stats[Threads/32u+2u];
    VX_CUDA_SHARED float partials[Threads];
    const vx_u32 tid=threadIdx.x,head=blockIdx.x,kvhead=head/8u;
    const vx_u32 channel=blockIdx.y*Columns+tid%Columns,group=tid/Columns;
    if(masking) {
        float any=0.0f;
        for(vx_u32 key=tid;key<keys;key+=Threads)if(mask[key])any=1.0f;
        for(vx_u32 d=16u;d;d/=2u)any=vx_max(any,vx_cuda_attention_shuffle_down(any,d));
        if(!(tid%32u))stats[tid/32u]=any;
        vx_sync_threads();
        if(!tid){float any=0.0f;for(vx_u32 w=0;w<Threads/32u;++w)any=vx_max(any,stats[w]);stats[Threads/32u]=any;}
        vx_sync_threads();
        if(stats[Threads/32u]==0.0f) {
            if(tid<Columns)out[head*64u+channel]=0.0f;
            return;
        }
    }
    for(vx_u32 key=tid;key<keys;key+=Threads) {
        float dot=0.0f;
        if(!masking || mask[key]) {
#if defined(__CUDA_ARCH__)
            #pragma unroll 2
#endif
            for(vx_u32 ch=0;ch<64u;ch+=4u) {
                VxCudaFloat4 a=*(const VxCudaFloat4*)(q+head*64u+ch);
                VxCudaFloat4 b=*(const VxCudaFloat4*)(k+key*256u+kvhead*64u+ch);
                dot+=a.x*b.x;dot+=a.y*b.y;dot+=a.z*b.z;dot+=a.w*b.w;
            }
        }
        scores[key]=dot*scale;
    }
    vx_sync_threads();
    if(tid<32u) {
        float maximum=-3.402823466e38f;
        for(vx_u32 key=tid;key<keys;key+=32u)
            if(!masking || mask[key])maximum=vx_max(maximum,scores[key]);
        for(vx_u32 d=16u;d;d/=2u)maximum=vx_max(maximum,vx_cuda_attention_shuffle_down(maximum,d));
        if(!tid)stats[0]=maximum;
    }
    vx_sync_threads();
    for(vx_u32 key=tid;key<keys;key+=Threads)
        scores[key]=(!masking || mask[key])?vx_exp(scores[key]-stats[0]):0.0f;
    vx_sync_threads();
    if(tid<32u) {
        float denominator=0.0f;
        for(vx_u32 first=0;first<keys;first+=1536u) {
            vx_u32 start=first+tid*48u,count=start<keys?(keys-start<48u?keys-start:48u):0u;
            float partial=0.0f;
            for(vx_u32 off=0;off<count;++off)
                if(!masking || mask[start+off])partial+=scores[start+off];
            vx_u32 parts=(keys-first+47u)/48u;if(parts>32u)parts=32u;
            for(vx_u32 part=0;part<parts;++part) {
                float x=vx_cuda_attention_shuffle_lane(partial,part);
                if(!tid)denominator+=x;
            }
        }
        if(!tid)stats[1]=denominator;
    }
    vx_sync_threads();
    for(vx_u32 key=tid;key<keys;key+=Threads)scores[key]/=stats[1];
    vx_sync_threads();
    float result=0.0f;
    for(vx_u32 first=0;first<keys;first+=(Threads/Columns)*48u) {
        vx_u32 start=first+group*48u;
        float partial=0.0f;
        for(vx_u32 off=0;off<48u && start+off<keys;++off)
            if(!masking || mask[start+off])
                partial+=scores[start+off]*v[(start+off)*256u+kvhead*64u+channel];
        partials[tid]=partial;
        vx_sync_threads();
        if(tid<Columns)
            for(vx_u32 part=0;part<Threads/Columns && first+part*48u<keys;++part)
                result+=partials[part*Columns+tid];
        vx_sync_threads();
    }
    if(tid<Columns)out[head*64u+channel]=result;
}


extern "C" VX_CUDA_GLOBAL void vx_cuda_cross_sdpa_gqa_decode_vector_f32(
        const float* query_values, const float* key_values,
        const float* value_values, const vx_i32* mask, float* output,
        vx_u32 query_sequence, vx_u32 key_sequence, vx_u32 query_width,
        vx_u32 kv_width, vx_u32 heads, vx_u32 kv_heads, vx_u32 head_dim,
        vx_u32 batch, float scale, vx_u32 causal, vx_u32 mask_mode,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        query_values, key_values, value_values, mask, output, query_sequence,
        key_sequence, query_width, kv_width, heads, kv_heads, head_dim, batch,
        scale, causal, mask_mode);
    vx_attention_decode_vector<32u, 1024u>(vx_cuda_global_pointer(query_values),
        vx_cuda_global_pointer(key_values), vx_cuda_global_pointer(value_values),
        vx_cuda_global_pointer(mask), vx_cuda_global_pointer(output),
        key_sequence, scale, mask_mode);
}
