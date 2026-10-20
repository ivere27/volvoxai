#include "llm_gpu_plan.h"

static const MetalKernel native_llm_kernels[VX_LLM_GPU_SHADER_COUNT] = {
    {"llmDeltaGates",METAL_SHADER_DIR "/llmDeltaGates.metal",8,7,64,1,1},
    {"llmDeltaNet",METAL_SHADER_DIR "/llmDeltaNet.metal",8,7,64,1,1},
    {"llmSparseSelect",METAL_SHADER_DIR "/llmSparseSelect.metal",6,5,64,1,1},
    {"llmSparseAttention",METAL_SHADER_DIR "/llmSparseAttention.metal",9,8,64,1,1},
    {"llmKVCompress",METAL_SHADER_DIR "/llmKVCompress.metal",8,7,64,1,1},
    {"llmHyperConnection",METAL_SHADER_DIR "/llmHyperConnection.metal",8,7,64,1,1},
    {"llmHyperHead",METAL_SHADER_DIR "/llmHyperHead.metal",6,5,64,1,1},
    {"llmHyperResidual",METAL_SHADER_DIR "/llmHyperResidual.metal",6,5,64,1,1},
    {"llmRotaryEmbedding",METAL_SHADER_DIR "/llmRotaryEmbedding.metal",5,4,64,1,1},
    {"llmNGramIds",METAL_SHADER_DIR "/llmNGramIds.metal",6,5,64,1,1},
    {"llmQSAScores",METAL_SHADER_DIR "/llmQSAScores.metal",8,7,64,1,1},
    {"llmQSAMask",METAL_SHADER_DIR "/llmQSAMask.metal",4,3,64,1,1},
    {"llmSignedSqrt",METAL_SHADER_DIR "/llmSignedSqrt.metal",3,2,64,1,1},
    {"llmEmbeddingShard",METAL_SHADER_DIR "/llmEmbeddingShard.metal",4,3,64,1,1},
    {"llmSparseIndexerScores",METAL_SHADER_DIR "/llmSparseIndexerScores.metal",6,5,64,1,1},
    {"llmSparseIndexerMask",METAL_SHADER_DIR "/llmSparseIndexerMask.metal",4,3,64,1,1},
    {"packedLinear",METAL_SHADER_DIR "/packedLinear.metal",6,5,64,1,1},
    {"moeRouterExtended",METAL_SHADER_DIR "/moeRouterExtended.metal",9,8,64,1,1},
    {"moeGatedUp",METAL_SHADER_DIR "/moeGatedUp.metal",7,6,64,1,1},
    {"moeGatedDown",METAL_SHADER_DIR "/moeGatedDown.metal",6,5,64,1,1},
    {"rope",METAL_SHADER_DIR "/rope.metal",5,4,64,1,1},
    {"copyTyped",METAL_SHADER_DIR "/copyTyped.metal",3,2,64,1,1},
};

static int native_llm_prepare(const uint64_t workspace[2]) {
    MetalContextState* state=metal_context_state_get(0);
    if(!state || !metal_ready() || workspace[0]>UINT32_MAX || workspace[1]>UINT32_MAX)return 0;
    if(!state->modern_zero_buffer) {
        if(state->domain_enforced)return 0;
        uint32_t zero[2]={0,0};
        id<MTLBuffer> input=create_buffer(sizeof(zero),zero),output=create_buffer(sizeof(zero),zero);
        if(!input||!output) {VX_METAL_RELEASE(input);VX_METAL_RELEASE(output);return 0;}
        state->modern_zero_buffer=input;state->modern_discard_buffer=output;
    }
    if(workspace[0]>state->modern_scratch_capacity) {
        if(state->domain_enforced)return 0;
        id<MTLBuffer> next=create_buffer((size_t)workspace[0],NULL);
        if(!next)return 0;
        VX_METAL_RELEASE(state->modern_scratch_buffer);state->modern_scratch_buffer=next;
        state->modern_scratch_capacity=(size_t)workspace[0];
        state->capacity_generation=metal_generation_next(state->capacity_generation);
    }
    if(workspace[1]>state->modern_state_capacity) {
        if(state->domain_enforced)return 0;
        id<MTLBuffer> next=create_buffer((size_t)workspace[1],NULL);
        if(!next)return 0;
        VX_METAL_RELEASE(state->modern_state_buffer);state->modern_state_buffer=next;
        state->modern_state_capacity=(size_t)workspace[1];
        state->capacity_generation=metal_generation_next(state->capacity_generation);
    }
    return 1;
}
static int native_llm_emit(void* user,const VxLlmGpuDispatch* d) {
    (void)user;
    MetalContextState* state=metal_context_state_get(0);
    MetalBinding bindings[9];id<MTLBuffer> constant_buffer=nil,params=nil;int ok=0;MetalTensorSlot* outputs[8]={0};
    if(!state || !d || (unsigned)d->shader>=VX_LLM_GPU_SHADER_COUNT)return 0;
    for(uint32_t i=0;i<d->buffer_count;i++) {
        const VxLlmGpuBuffer* ref=&d->buffers[i];
        if(ref->workspace==4) {
            if(ref->bytes!=d->constants_bytes)goto finish;
            constant_buffer=create_buffer(d->constants_bytes,d->constants);
            if(!constant_buffer)goto finish;
            bindings[i]=(MetalBinding){constant_buffer,(size_t)ref->bytes,0};
        } else if(ref->workspace) {
            if(ref->workspace>3)goto finish;
            id<MTLBuffer> buffer=ref->workspace==1?state->modern_scratch_buffer:ref->workspace==2?state->modern_state_buffer:
                (ref->output?state->modern_discard_buffer:state->modern_zero_buffer);
            if(!buffer || ref->bytes>(uint64_t)[buffer length])goto finish;
            bindings[i]=(MetalBinding){buffer,(size_t)ref->bytes,0};
        } else {
            MetalTensorSlot* slot=ref->output?graph_output_slot(ref->host,(size_t)ref->bytes):graph_ensure_device(ref->host,(size_t)ref->bytes,0);
            if(!slot)goto finish;
            bindings[i]=(MetalBinding){slot->buffer,(size_t)ref->bytes,0};if(ref->output)outputs[i]=slot;
        }
    }
    params=create_buffer(sizeof(d->uniform),d->uniform);
    if(!params)goto finish;
    bindings[d->buffer_count]=(MetalBinding){params,sizeof(d->uniform),0};
    ok=dispatch_kernel(&native_llm_kernels[d->shader],bindings,d->groups[0],d->groups[1],d->groups[2]);
finish:
    VX_METAL_RELEASE(constant_buffer);
    VX_METAL_RELEASE(params);if(!ok)return 0;
    for(uint32_t i=0;i<d->buffer_count;i++)if(outputs[i])graph_mark_device(outputs[i]);
    return 1;
}
#define VX_LLM_NATIVE_PREFIX metal
#include "llm_gpu_native.inc"

int metal_graph_rmsnorm_f32(const float* input,const float* weight,float* output,
                            int rows,int width,float epsilon) {
    if(!input||!weight||!output||rows<=0||width<=0||!isfinite(epsilon)||epsilon<=0||
       (uint64_t)(uint32_t)rows*(uint32_t)width>UINT32_MAX/4u)return 0;
    size_t bytes=(size_t)rows*(size_t)width*sizeof(float),wbytes=(size_t)width*sizeof(float);
    MetalTensorSlot *src=graph_ensure_device(input,bytes,0),*w=graph_ensure_device(weight,wbytes,1),*dst=graph_output_slot(output,bytes);
    if(!src||!w||!dst)return 0;
    struct {uint32_t rows,width;float epsilon;} values={(uint32_t)rows,(uint32_t)width,epsilon};
    id<MTLBuffer> params=create_buffer(sizeof(values),&values);if(!params)return 0;
    MetalBinding bindings[4]={{src->buffer,bytes,0},{w->buffer,wbytes,0},{dst->buffer,bytes,0},{params,sizeof(values),0}};
    int ok=dispatch_kernel(&k_rmsnorm,bindings,((uint32_t)rows+63u)/64u,1,1);
    VX_METAL_RELEASE(params);if(ok)graph_mark_device(dst);return ok;
}
