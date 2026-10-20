#include "llm_gpu_plan.h"

static const OglKernel native_llm_kernels[VX_LLM_GPU_SHADER_COUNT] = {
    {"llmDeltaGates",OGL_SHADER_DIR "/llmDeltaGates.comp",8,7},
    {"llmDeltaNet",OGL_SHADER_DIR "/llmDeltaNet.comp",8,7},
    {"llmSparseSelect",OGL_SHADER_DIR "/llmSparseSelect.comp",6,5},
    {"llmSparseAttention",OGL_SHADER_DIR "/llmSparseAttention.comp",9,8},
    {"llmKVCompress",OGL_SHADER_DIR "/llmKVCompress.comp",8,7},
    {"llmHyperConnection",OGL_SHADER_DIR "/llmHyperConnection.comp",8,7},
    {"llmHyperHead",OGL_SHADER_DIR "/llmHyperHead.comp",6,5},
    {"llmHyperResidual",OGL_SHADER_DIR "/llmHyperResidual.comp",6,5},
    {"llmRotaryEmbedding",OGL_SHADER_DIR "/llmRotaryEmbedding.comp",5,4},
    {"llmNGramIds",OGL_SHADER_DIR "/llmNGramIds.comp",6,5},
    {"llmQSAScores",OGL_SHADER_DIR "/llmQSAScores.comp",8,7},
    {"llmQSAMask",OGL_SHADER_DIR "/llmQSAMask.comp",4,3},
    {"llmSignedSqrt",OGL_SHADER_DIR "/llmSignedSqrt.comp",3,2},
    {"llmEmbeddingShard",OGL_SHADER_DIR "/llmEmbeddingShard.comp",4,3},
    {"llmSparseIndexerScores",OGL_SHADER_DIR "/llmSparseIndexerScores.comp",6,5},
    {"llmSparseIndexerMask",OGL_SHADER_DIR "/llmSparseIndexerMask.comp",4,3},
    {"packedLinear",OGL_SHADER_DIR "/packedLinear.comp",6,5},
    {"moeRouterExtended",OGL_SHADER_DIR "/moeRouterExtended.comp",9,8},
    {"moeGatedUp",OGL_SHADER_DIR "/moeGatedUp.comp",7,6},
    {"moeGatedDown",OGL_SHADER_DIR "/moeGatedDown.comp",6,5},
    {"rope",OGL_SHADER_DIR "/rope.comp",5,4},
    {"copyTyped",OGL_SHADER_DIR "/copyTyped.comp",3,2},
};

static int native_llm_prepare(const uint64_t workspace[2]) {
    OpenGLContextState* state=opengl_context_state_get(0);
    if(!state || !ogl_ready() || workspace[0]>UINT32_MAX || workspace[1]>UINT32_MAX)return 0;
    if(!state->modern_zero_buffer) {
        if(state->domain_enforced)return 0;
        uint32_t zero[2]={0,0};
        GLuint input=create_buffer(sizeof(zero),zero),output=create_buffer(sizeof(zero),zero);
        if(!input||!output) {if(input)p_glDeleteBuffers(1,&input);if(output)p_glDeleteBuffers(1,&output);return 0;}
        state->modern_zero_buffer=input;state->modern_discard_buffer=output;
    }
    if(workspace[0]>state->modern_scratch_capacity) {
        if(state->domain_enforced)return 0;
        GLuint next=create_buffer((size_t)workspace[0],NULL);
        if(!next)return 0;
        if(state->modern_scratch_buffer)p_glDeleteBuffers(1,&state->modern_scratch_buffer);
        state->modern_scratch_buffer=next;state->modern_scratch_capacity=(size_t)workspace[0];
        state->capacity_generation=opengl_generation_next(state->capacity_generation);
    }
    if(workspace[1]>state->modern_state_capacity) {
        if(state->domain_enforced)return 0;
        GLuint next=create_buffer((size_t)workspace[1],NULL);
        if(!next)return 0;
        if(state->modern_state_buffer)p_glDeleteBuffers(1,&state->modern_state_buffer);
        state->modern_state_buffer=next;state->modern_state_capacity=(size_t)workspace[1];
        state->capacity_generation=opengl_generation_next(state->capacity_generation);
    }
    return 1;
}
static int native_llm_emit(void* user,const VxLlmGpuDispatch* d) {
    (void)user;
    OpenGLContextState* state=opengl_context_state_get(0);
    GLuint buffers[9],constant_buffer=0,params=0;int ok=0;OglTensorSlot* outputs[8]={0};
    if(!state || !d || (unsigned)d->shader>=VX_LLM_GPU_SHADER_COUNT)return 0;
    for(uint32_t i=0;i<d->buffer_count;i++) {
        const VxLlmGpuBuffer* ref=&d->buffers[i];
        if(ref->workspace==4) {
            if(ref->bytes!=d->constants_bytes)goto finish;
            constant_buffer=create_buffer(d->constants_bytes,d->constants);
            if(!constant_buffer)goto finish;buffers[i]=constant_buffer;
        } else if(ref->workspace) {
            if(ref->workspace>3)goto finish;
            buffers[i]=ref->workspace==1?state->modern_scratch_buffer:ref->workspace==2?state->modern_state_buffer:
                (ref->output?state->modern_discard_buffer:state->modern_zero_buffer);
            if(!buffers[i] || (ref->workspace==1 && ref->bytes>state->modern_scratch_capacity) ||
               (ref->workspace==2 && ref->bytes>state->modern_state_capacity))goto finish;
        } else {
            OglTensorSlot* slot=ref->output?graph_output_slot(ref->host,(size_t)ref->bytes):graph_ensure_device(ref->host,(size_t)ref->bytes,0);
            if(!slot)goto finish;buffers[i]=slot->buffer;if(ref->output)outputs[i]=slot;
        }
    }
    params=params_buffer(d->uniform,sizeof(d->uniform));
    buffers[d->buffer_count]=params;
    if(!buffers[d->buffer_count])goto finish;
    ok=dispatch_kernel(&native_llm_kernels[d->shader],buffers,d->groups[0],d->groups[1],d->groups[2]);
finish:
    if(constant_buffer)p_glDeleteBuffers(1,&constant_buffer);
    if(params)p_glDeleteBuffers(1,&params);
    if(!ok)return 0;
    for(uint32_t i=0;i<d->buffer_count;i++)if(outputs[i])graph_mark_device(outputs[i]);
    return 1;
}
#define VX_LLM_NATIVE_PREFIX opengl
#include "llm_gpu_native.inc"
