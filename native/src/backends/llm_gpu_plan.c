#include "llm_gpu_plan.h"
#include "../kernels/packed_linear.h"
#include <math.h>
#include <string.h>

static const char* const shader_names[VX_LLM_GPU_SHADER_COUNT] = {
    "llmDeltaGates", "llmDeltaNet", "llmSparseSelect", "llmSparseAttention",
    "llmKVCompress", "llmHyperConnection", "llmHyperHead", "llmHyperResidual",
    "llmRotaryEmbedding", "llmNGramIds", "llmQSAScores", "llmQSAMask",
    "llmSignedSqrt", "llmEmbeddingShard", "llmSparseIndexerScores", "llmSparseIndexerMask", "packedLinear", "moeRouterExtended", "moeGatedUp", "moeGatedDown", "rope", "copyTyped"
};
static const uint8_t shader_buffers[VX_LLM_GPU_SHADER_COUNT] = {
    7,7,5,8,7,7,5,5,4,5,7,3,2,3,5,3,5,8,6,5,4,2
};
/* Legacy fixed-shape callers supply a total only. A bounded owner supplies
 * both independently proved maxima; never assign their sum to slot zero. */
int vx_llm_gpu_workspace_slots(size_t total,const size_t raw[2],size_t slots[2]) {
    if(!raw || !slots || (total && total<16u))return 0;
    if(raw[0] || raw[1]) {
        if(raw[0]>SIZE_MAX-16u || raw[1]>SIZE_MAX-16u-raw[0] ||
           raw[0]+raw[1]+16u!=total)return 0;
        slots[0]=raw[0];slots[1]=raw[1];
    } else {slots[0]=total?total-16u:0;slots[1]=0;}
    return 1;
}
const char* vx_llm_gpu_shader_name(VxLlmGpuShader shader) {
    return (unsigned)shader<VX_LLM_GPU_SHADER_COUNT?shader_names[shader]:NULL;
}
uint32_t vx_llm_gpu_shader_bindings(VxLlmGpuShader shader) {
    return (unsigned)shader<VX_LLM_GPU_SHADER_COUNT?shader_buffers[shader]+1u:0;
}
static int product(uint64_t a,uint64_t b,uint64_t* result) {
    if(!a || !b || a>UINT32_MAX/b) return 0;
    *result=a*b;return 1;
}
static int tensor(VxLlmGpuBuffer* buffer,const void* host,uint64_t elements,int output) {
    if(!buffer || (host && (!elements || elements>UINT32_MAX/4u))) return 0;
    buffer->host=host;buffer->bytes=host?elements*4u:4u;
    buffer->workspace=host?0u:3u;buffer->output=output!=0;return 1;
}
static void scratch(VxLlmGpuBuffer* buffer,uint32_t index,uint64_t bytes,int output) {
    buffer->host=NULL;buffer->bytes=bytes;buffer->workspace=index;buffer->output=output!=0;
}
static void f32(uint32_t* word,float value) {memcpy(word,&value,sizeof(value));}
static int dispatch(VxLlmGpuDispatch* d,VxLlmGpuShader shader,uint64_t work,int groups,
                    VxLlmGpuEmit emit,void* user) {
    if(!work || work>UINT32_MAX) return 0;
    d->shader=shader;d->buffer_count=shader_buffers[shader];
    if(shader<VX_LLM_GPU_MOE_ROUTER)d->uniform[22]=(uint32_t)work;
    d->groups[0]=groups?(uint32_t)work:(uint32_t)((work+63u)/64u);
    d->groups[1]=d->groups[2]=1;
    return !emit || emit(user,d);
}
static void modern_uniform(VxLlmGpuDispatch* d,const VxLLMOpArgs* a) {
    memset(d,0,sizeof(*d));
    const int32_t values[21]={a->batch,a->sequence,a->keys,a->heads,a->kv_heads,a->key_dim,a->value_dim,a->streams,
        a->groups,a->rate,a->window,a->top_k,a->iterations,a->index_heads,a->index_dim,a->rotary_dim,
        a->rotary_offset,a->normalize,a->causal,a->interleaved,a->mask_mode};
    for(unsigned i=0;i<21;i++) d->uniform[i]=(uint32_t)values[i];
    d->uniform[23]=(uint32_t)a->first_unit;
    d->uniform[28]=(uint32_t)a->query_offset;
    f32(&d->uniform[24],a->eps);f32(&d->uniform[25],a->hc_eps);f32(&d->uniform[26],a->scale);
}
static int x(VxLlmGpuDispatch* d,unsigned binding,const VxLLMOpArgs* a,unsigned index) {
    return tensor(&d->buffers[binding],a->x[index],a->x_elements[index],0);
}
static int y(VxLlmGpuDispatch* d,unsigned binding,const VxLLMOpArgs* a,unsigned index) {
    return tensor(&d->buffers[binding],a->y[index],a->y_elements[index],1);
}
int vx_llm_gpu_plan(const VxLLMOpArgs* a,uint64_t workspace[2],VxLlmGpuEmit emit,void* user) {
    VxLlmGpuDispatch d;uint64_t rows,work,scratch_bytes=0;
    if(!a || !workspace || !a->x[0] || !a->y[0] || a->batch<=0 || a->sequence<=0 || a->key_dim<=0 ||
       a->query_offset<0 || a->query_offset>INT32_MAX-(a->sequence-1) ||
       !product((uint32_t)a->batch,(uint32_t)a->sequence,&rows)) return 0;
    workspace[0]=workspace[1]=0;modern_uniform(&d,a);
    switch(a->kind) {
        case VX_OP_PACKED_LINEAR: {
            uint64_t inputs,words,scales;
            uint32_t weight_bits=a->groups?(uint32_t)a->groups:4u;
            uint32_t per_word=weight_bits==8u?4u:8u;
            if(a->value_dim<=0 || !vx_packed_group_size_supported(a->rate) || (a->normalize!=8 && a->normalize!=16 && a->normalize!=32) ||
               (weight_bits!=4u && weight_bits!=8u) ||
               !a->x[1] || !a->x[2] || !product(rows,(uint32_t)a->key_dim,&inputs) ||
               !product((uint32_t)a->value_dim,((uint32_t)a->key_dim+per_word-1u)/per_word,&words) ||
               !product((uint32_t)a->value_dim,(uint32_t)a->key_dim/(uint32_t)a->rate+((uint32_t)a->key_dim%(uint32_t)a->rate!=0),&scales) ||
               !product(rows,(uint32_t)a->value_dim,&work) ||
               a->x_elements[0]<inputs || a->x_elements[1]!=words || a->x_elements[2]!=scales ||
               (a->x[3] && a->x_elements[3]!=(uint32_t)a->value_dim) || a->y_elements[0]<work ||
               a->first_unit>work || (a->first_unit && !a->unit_count) ||
               (a->unit_count && a->unit_count>work-a->first_unit))return 0;
            /* Cached prefill dispatches logical rows while retaining complete
             * slot owners, so later rows cannot invalidate the device prefix. */
            if(!x(&d,0,a,0)||!x(&d,1,a,1)||!x(&d,2,a,2)||!x(&d,3,a,3)||!y(&d,4,a,0))return 0;
            d.uniform[8]=weight_bits;
            d.uniform[21]=a->x[3]?1u:0u;
            return dispatch(&d,VX_LLM_GPU_PACKED_LINEAR,a->unit_count?a->unit_count:work,0,emit,user);
        }
        case VX_OP_GATED_DELTA_NET:
            if(a->heads<=0 || a->kv_heads<=0 || a->key_dim>512 || a->value_dim<=0 ||
               a->heads%a->kv_heads || !product(rows,(uint32_t)a->heads,&work) ||
               !product(work,16u,&scratch_bytes)) return 0;
            workspace[0]=scratch_bytes;
            if(a->y[1]) {
                uint64_t state_elements;
                if(!product((uint32_t)a->batch,(uint32_t)a->heads,&state_elements)||
                   !product(state_elements,(uint32_t)a->key_dim,&state_elements)||
                   !product(state_elements,(uint32_t)a->value_dim,&state_elements)||
                   state_elements>UINT32_MAX/4u || a->y_elements[1]!=state_elements)return 0;
                workspace[1]=state_elements*4u;
            }
            d.uniform[21]=(a->x[7]?1u:0u)|(a->y[1]?2u:0u);
            if(!x(&d,0,a,0)||!x(&d,1,a,1)||!x(&d,2,a,3)||!x(&d,3,a,4)||!x(&d,4,a,5)||!x(&d,5,a,6))return 0;
            scratch(&d.buffers[6],1,scratch_bytes,1);
            if(!dispatch(&d,VX_LLM_GPU_DELTA_GATES,work,0,emit,user))return 0;
            if(!x(&d,0,a,0)||!x(&d,1,a,1)||!x(&d,2,a,2)||!x(&d,4,a,7)||!y(&d,5,a,0))return 0;
            if(a->y[1])scratch(&d.buffers[6],2,workspace[1],1);
            else if(!y(&d,6,a,1))return 0;
            scratch(&d.buffers[3],1,scratch_bytes,0);
            if(!product((uint32_t)a->batch,(uint32_t)a->heads,&work)||!product(work,(uint32_t)a->value_dim,&work))return 0;
            if(!dispatch(&d,VX_LLM_GPU_DELTA_NET,work,0,emit,user))return 0;
            if(!a->y[1])return 1;
            /* Separate buffers satisfy WebGPU's read-only/storage-write
             * binding rules even when continuation reuses retained STATE. */
            memset(&d,0,sizeof(d));d.uniform[0]=(uint32_t)workspace[1];
            scratch(&d.buffers[0],2,workspace[1],0);
            if(!y(&d,1,a,1))return 0;
            return dispatch(&d,VX_LLM_GPU_COPY,workspace[1]/4u,0,emit,user);
        case VX_OP_SPARSE_ATTENTION:
            if(a->heads<=0||a->kv_heads<=0||a->heads%a->kv_heads||a->keys<=0||a->key_dim>512||
               a->value_dim<=0||a->value_dim>512||a->rate<=0||a->top_k<0) return 0;
            d.uniform[21]=(a->x[3]?4u:0u)|(a->x[4]?8u:0u)|(a->index_heads>0?16u:0u)|(a->x[5]&&!a->index_heads?128u:0u)|(a->positions?32u:0u);
            if(a->x[3]) {
                if(a->index_heads>0 && a->top_k>0 &&
                   (a->keys/a->rate>512||a->index_dim<=0||a->index_dim>512))return 0;
                if(a->x[5]&&!a->index_heads)d.uniform[11]=(uint32_t)a->keys;
                if(!product(rows,(uint64_t)d.uniform[11]+1u,&work)||!product(work,4u,&scratch_bytes))return 0;
                workspace[0]=scratch_bytes;
                if(!x(&d,0,a,5)||!x(&d,1,a,6)||!x(&d,2,a,7)||
                   !tensor(&d.buffers[3],a->positions,a->position_elements,0))return 0;
                scratch(&d.buffers[4],1,scratch_bytes,1);
                if(!dispatch(&d,VX_LLM_GPU_SPARSE_SELECT,rows,1,emit,user))return 0;
            }
            for(unsigned i=0;i<5;i++)if(!x(&d,i,a,i))return 0;
            if(!tensor(&d.buffers[5],a->mask,a->mask_elements,0)||!y(&d,7,a,0))return 0;
            if(a->x[3])scratch(&d.buffers[6],1,scratch_bytes,0);else if(!tensor(&d.buffers[6],NULL,0,0))return 0;
            if(!product(rows,(uint32_t)a->heads,&work))return 0;
            return dispatch(&d,VX_LLM_GPU_SPARSE_ATTENTION,work,0,emit,user);
        case VX_OP_KV_COMPRESS:
            if(a->key_dim>512||a->rate<=0||(a->groups!=1&&a->groups!=2))return 0;
            if(a->unit_count) {
                if(a->first_unit>=rows || a->unit_count>rows-a->first_unit)return 0;
                rows=a->unit_count;
            } else if(a->first_unit)return 0;
            d.uniform[21]=a->x[4]?64u:0u;
            for(unsigned i=0;i<6;i++)if(!x(&d,i,a,i))return 0;
            if(!y(&d,6,a,0))return 0;
            return dispatch(&d,VX_LLM_GPU_KV_COMPRESS,rows,0,emit,user);
        case VX_OP_HYPER_CONNECTION:
        case VX_OP_HYPER_HEAD:
            if(a->streams<=0||a->streams>16||a->iterations<=0)return 0;
            for(unsigned i=0;i<4;i++)if(!x(&d,i,a,i))return 0;
            if(!y(&d,4,a,0))return 0;
            if(a->kind==VX_OP_HYPER_CONNECTION) {
                if(!y(&d,5,a,1)||!y(&d,6,a,2))return 0;
                return dispatch(&d,VX_LLM_GPU_HYPER_CONNECTION,rows,0,emit,user);
            }
            return dispatch(&d,VX_LLM_GPU_HYPER_HEAD,rows,0,emit,user);
        case VX_OP_HYPER_RESIDUAL:
            if(a->streams<=0)return 0;
            for(unsigned i=0;i<4;i++)if(!x(&d,i,a,i))return 0;
            if(!y(&d,4,a,0))return 0;
            return dispatch(&d,VX_LLM_GPU_HYPER_RESIDUAL,a->y_elements[0],0,emit,user);
        case VX_OP_ROTARY_EMBEDDING:
            if(a->key_dim>512||a->rotary_dim<=0||a->rotary_dim>a->key_dim||a->rotary_offset<0||
               a->rotary_offset>a->key_dim-a->rotary_dim||(a->rotary_dim&1))return 0;
            for(unsigned i=0;i<3;i++)if(!x(&d,i,a,i))return 0;
            if(!y(&d,3,a,0))return 0;
            return dispatch(&d,VX_LLM_GPU_ROTARY,a->y_elements[0],0,emit,user);
        case VX_OP_SPARSE_INDEXER:
            if(a->heads<=0||a->keys!=a->sequence||a->rate<=0||a->top_k<=0||
               a->first_unit||a->unit_count||a->query_offset||
               (a->y[1]&&(a->window<=0||a->groups<=0||a->x[3]))||
               (!a->y[1]&&(a->window||a->groups))||
               !product(rows,(uint32_t)a->keys,&work)||!product(work,4u,&scratch_bytes))return 0;
            workspace[0]=scratch_bytes;
            d.uniform[21]=(a->x[3]?1u:0u)|(a->y[1]?2u:0u);
            for(unsigned i=0;i<4;i++)if(!x(&d,i,a,i))return 0;
            scratch(&d.buffers[4],1,scratch_bytes,1);
            if(!dispatch(&d,VX_LLM_GPU_SPARSE_INDEX_SCORES,work,0,emit,user))return 0;
            scratch(&d.buffers[0],1,scratch_bytes,0);
            if(!y(&d,1,a,0)||!y(&d,2,a,1))return 0;
            return dispatch(&d,VX_LLM_GPU_SPARSE_INDEX_MASK,work,0,emit,user);
        case VX_OP_QSA_INDEXER:
            if(a->key_dim>512||a->heads<=0||a->keys<=0||a->rate<=0||a->top_k<=0||
               a->rotary_dim<=0||a->rotary_dim>a->key_dim||(a->rotary_dim&1))return 0;
            if(!product(rows,((uint64_t)(uint32_t)a->keys+(uint32_t)a->rate-1u)/(uint32_t)a->rate,&work)||
               !product(work,4u,&scratch_bytes))return 0;
            workspace[0]=scratch_bytes;
            for(unsigned i=0;i<5;i++)if(!x(&d,i,a,i))return 0;
            if(!tensor(&d.buffers[5],a->mask,a->mask_elements,0))return 0;
            scratch(&d.buffers[6],1,scratch_bytes,1);
            if(!dispatch(&d,VX_LLM_GPU_QSA_SCORES,work,0,emit,user))return 0;
            scratch(&d.buffers[0],1,scratch_bytes,0);
            if(!tensor(&d.buffers[1],a->mask,a->mask_elements,0)||!y(&d,2,a,0))return 0;
            return dispatch(&d,VX_LLM_GPU_QSA_MASK,rows,0,emit,user);
        case VX_OP_SIGNED_SQRT:
            if(!isfinite(a->eps)||a->eps<=0||!x(&d,0,a,0)||!y(&d,1,a,0))return 0;
            return dispatch(&d,VX_LLM_GPU_SIGNED_SQRT,a->y_elements[0],0,emit,user);
        case VX_OP_EMBEDDING_SHARD:
            if(a->rotary_offset<0||a->value_dim<=0||a->keys<=0||
               (uint64_t)a->rotary_offset+a->value_dim>(uint32_t)a->keys||!x(&d,0,a,0)||!x(&d,1,a,1)||!y(&d,2,a,0))return 0;
            return dispatch(&d,VX_LLM_GPU_EMBEDDING_SHARD,a->y_elements[0],0,emit,user);
        default:return 0;
    }
}

static int moe_dispatch(VxLlmGpuDispatch* d,VxLlmGpuShader shader,uint64_t work,int groups,VxLlmGpuEmit emit,void* user) {
    if(!work || work>UINT32_MAX)return 0;
    d->uniform[17]=(uint32_t)work;
    return dispatch(d,shader,work,groups,emit,user);
}
int vx_llm_gpu_router_plan(const VxLlmGpuRouterArgs* a,VxLlmGpuEmit emit,void* user) {
    VxLlmGpuDispatch d;uint64_t weights,routes;
    if(!a||!a->input||!a->weight||!a->indices||!a->routes||!a->rows||!a->d_in||!a->experts||a->experts>1024||!a->top_k||a->top_k>a->experts||
       !a->groups||a->groups>128||a->experts%a->groups||!a->selected_groups||a->selected_groups>a->groups||
       !a->group_top_k||a->group_top_k>a->experts/a->groups||a->top_k>a->selected_groups*(a->experts/a->groups)||
       a->scoring>2||!isfinite(a->temperature)||a->temperature<=0||!isfinite(a->scale)||a->scale<=0||
       !isfinite(a->epsilon)||a->epsilon<0||(!!a->tokens!=!!a->hash)||
       (a->hash&&!a->vocabulary)||!product(a->d_in,a->experts,&weights)||!product(a->rows,a->top_k,&routes))return 0;
    memset(&d,0,sizeof(d));uint32_t v[16]={a->rows,a->d_in,0,0,a->experts,a->top_k,a->normalize,a->scoring,a->groups,
        a->selected_groups,a->group_top_k,a->bias!=NULL,a->correction!=NULL,a->hash!=NULL,a->vocabulary,0};
    memcpy(d.uniform,v,sizeof(v));f32(&d.uniform[20],a->temperature);f32(&d.uniform[21],a->scale);f32(&d.uniform[22],a->epsilon);
    uint64_t input,hash;if(!product(a->rows,a->d_in,&input))return 0;
    hash=0;if(a->hash&&!product(a->vocabulary,a->top_k,&hash))return 0;
    if(!tensor(&d.buffers[0],a->input,input,0)||!tensor(&d.buffers[1],a->weight,weights,0)||
       !tensor(&d.buffers[2],a->bias,a->experts,0)||!tensor(&d.buffers[3],a->correction,a->experts,0)||
       !tensor(&d.buffers[4],a->tokens,a->rows,0)||!tensor(&d.buffers[5],a->hash,hash,0)||
       !tensor(&d.buffers[6],a->indices,routes,1)||!tensor(&d.buffers[7],a->routes,routes,1))return 0;
    return moe_dispatch(&d,VX_LLM_GPU_MOE_ROUTER,a->rows,1,emit,user);
}
int vx_llm_gpu_gated_plan(const VxLlmGpuGatedArgs* a,uint64_t workspace[2],VxLlmGpuEmit emit,void* user) {
    VxLlmGpuDispatch d;uint64_t hidden,input,bank,down,routes,output;
    if(!a||!workspace||!a->input||!a->gate||!a->up||!a->down||!a->indices||!a->routes||!a->output||
       !a->rows||!a->d_in||!a->hidden||!a->d_out||!a->experts||!a->top_k||a->top_k>a->experts||
       !isfinite(a->limit)||a->limit<0||!product(a->rows,a->hidden,&hidden)||
       !product(a->rows,a->d_in,&input)||!product(a->experts,a->d_in,&bank)||!product(bank,a->hidden,&bank)||
       !product(a->experts,a->hidden,&down)||!product(down,a->d_out,&down)||!product(a->rows,a->top_k,&routes)||
       !product(a->rows,a->d_out,&output)||hidden>UINT32_MAX/4u)return 0;
    workspace[0]=hidden*4u;workspace[1]=0;memset(&d,0,sizeof(d));
    uint32_t v[16]={a->rows,a->d_in,a->hidden,a->d_out,a->experts,a->top_k,0,0,0,0,0,0,0,0,0,a->input_scale};
    memcpy(d.uniform,v,sizeof(v));f32(&d.uniform[23],a->limit);
    if(!emit)return 1;
    for(uint32_t slot=0;slot<a->top_k;slot++) {
        d.uniform[16]=slot;
        if(!tensor(&d.buffers[0],a->input,input,0)||!tensor(&d.buffers[1],a->gate,bank,0)||!tensor(&d.buffers[2],a->up,bank,0)||
           !tensor(&d.buffers[3],a->indices,routes,0)||!tensor(&d.buffers[4],a->routes,routes,0))return 0;
        scratch(&d.buffers[5],1,workspace[0],1);
        if(!moe_dispatch(&d,VX_LLM_GPU_MOE_UP,hidden,0,emit,user))return 0;
        scratch(&d.buffers[0],1,workspace[0],0);
        if(!tensor(&d.buffers[1],a->down,down,0)||!tensor(&d.buffers[2],a->indices,routes,0)||
           !tensor(&d.buffers[3],a->routes,routes,0)||!tensor(&d.buffers[4],a->output,output,1))return 0;
        if(!moe_dispatch(&d,VX_LLM_GPU_MOE_DOWN,output,0,emit,user))return 0;
    }
    return 1;
}
int vx_llm_gpu_ngram_plan(const VxLlmGpuNgramArgs* a,VxLlmGpuEmit emit,void* user) {
    VxLlmGpuDispatch d;uint64_t rows,heads,work;
    if(!a||!a->tokens||!a->multipliers||!a->sizes||!a->offsets||!a->output||!a->batch||!a->sequence||
       a->ngram<2||!a->heads_per_ngram||a->vocabulary<=0||a->eos< -1||a->eos>=a->vocabulary||a->pad<0||a->pad>=a->vocabulary||
       !product(a->batch,a->sequence,&rows)||!product(a->ngram-1u,a->heads_per_ngram,&heads)||!product(rows,heads,&work))return 0;
    memset(&d,0,sizeof(d));d.uniform[0]=a->batch;d.uniform[1]=a->sequence;d.uniform[3]=(uint32_t)heads;
    d.uniform[8]=a->heads_per_ngram;d.uniform[16]=(uint32_t)a->eos;d.uniform[17]=(uint32_t)a->pad;
    if(!tensor(&d.buffers[0],a->tokens,rows,0)||!tensor(&d.buffers[1],a->multipliers,(uint64_t)a->ngram*2u,0)||
       !tensor(&d.buffers[2],a->sizes,heads,0)||!tensor(&d.buffers[3],a->offsets,heads,0)||!tensor(&d.buffers[4],a->output,work,1))return 0;
    if(a->unit_count) {
        if(a->first_unit>=work || a->unit_count>work-a->first_unit)return 0;
        d.uniform[23]=(uint32_t)a->first_unit;work=a->unit_count;
    } else if(a->first_unit)return 0;
    return dispatch(&d,VX_LLM_GPU_NGRAM,work,0,emit,user);
}

int vx_llm_gpu_rope_plan(const VxLlmGpuRopeArgs* a,VxLlmGpuEmit emit,void* user) {
    VxLlmGpuDispatch d;uint64_t rows,elements,positions=0;
    if(!a||!a->input||!a->output||!a->batch||!a->sequence||!a->width||!a->heads||a->width%a->heads||
       a->width/a->heads>128||!a->rotary_width||a->rotary_width>a->width/a->heads||(a->rotary_width&1u)||
       a->position_mode>2||(!a->positions&&a->position_mode)||a->interleaved>1||!isfinite(a->theta)||a->theta<1.0f||
       a->sequence>INT32_MAX||a->position_offset>INT32_MAX-(a->sequence-1u)||!product(a->batch,a->sequence,&rows)||!product(rows,a->width,&elements))return 0;
    memset(&d,0,sizeof(d));size_t pairs=a->rotary_width/2u;
    for(size_t i=0;i<pairs;i++) {
        d.constants[i]=powf(a->theta,2.0f*(float)i/(float)a->rotary_width);
        if(!isfinite(d.constants[i])||d.constants[i]<1.0f)return 0;
    }
    const double tau_hi=6.2831853071795862320,tau_lo=2.4492935982947064143e-16;
    double high=1.0,low=0.0;
    for(size_t i=0;i<32;i++) {
        float first=(float)high;d.constants[pairs+2*i]=first;d.constants[pairs+2*i+1]=(float)((high-(double)first)+low);
        high*=2.0;low*=2.0;
        if(high>tau_hi||(high==tau_hi&&low>=tau_lo)) {
            double sum=high-tau_hi,virtual_value=sum-high;
            double residual=(high-(sum-virtual_value))+(-tau_hi-virtual_value)+low-tau_lo;
            high=sum+residual;low=residual-(high-sum);
        }
    }
    d.constants_bytes=(uint32_t)(pairs+64u)*sizeof(float);
    uint32_t params[8]={(uint32_t)elements,a->sequence,a->width,a->heads,a->rotary_width,a->position_offset,a->interleaved,a->position_mode};
    memcpy(d.uniform,params,sizeof(params));
    if(a->positions)positions=a->position_mode==1?a->sequence:rows;
    if(!tensor(&d.buffers[0],a->input,elements,0)||!tensor(&d.buffers[1],a->positions,positions,0)||
       !tensor(&d.buffers[3],a->output,elements,1))return 0;
    scratch(&d.buffers[2],4,d.constants_bytes,0);
    return dispatch(&d,VX_LLM_GPU_ROPE,elements,0,emit,user);
}
