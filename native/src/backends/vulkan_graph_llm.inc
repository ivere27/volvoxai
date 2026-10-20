#include "llm_gpu_plan.h"

static VkKernel native_llm_kernels[VX_LLM_GPU_SHADER_COUNT] = {
    {"llmDeltaGates","spv/llmDeltaGates.spv",8,7,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmDeltaNet","spv/llmDeltaNet.spv",8,7,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmSparseSelect","spv/llmSparseSelect.spv",6,5,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmSparseAttention","spv/llmSparseAttention.spv",9,8,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmKVCompress","spv/llmKVCompress.spv",8,7,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmHyperConnection","spv/llmHyperConnection.spv",8,7,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmHyperHead","spv/llmHyperHead.spv",6,5,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmHyperResidual","spv/llmHyperResidual.spv",6,5,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmRotaryEmbedding","spv/llmRotaryEmbedding.spv",5,4,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmNGramIds","spv/llmNGramIds.spv",6,5,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmQSAScores","spv/llmQSAScores.spv",8,7,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmQSAMask","spv/llmQSAMask.spv",4,3,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmSignedSqrt","spv/llmSignedSqrt.spv",3,2,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmEmbeddingShard","spv/llmEmbeddingShard.spv",4,3,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmSparseIndexerScores","spv/llmSparseIndexerScores.spv",6,5,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"llmSparseIndexerMask","spv/llmSparseIndexerMask.spv",4,3,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"packedLinear","spv/packedLinear.spv",6,5,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"moeRouterExtended","spv/moeRouterExtended.spv",9,8,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"moeGatedUp","spv/moeGatedUp.spv",7,6,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"moeGatedDown","spv/moeGatedDown.spv",6,5,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"rope","spv/rope.spv",5,4,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
    {"copyTyped","spv/copyTyped.spv",3,2,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,0},
};

static int native_llm_prepare(const uint64_t workspace[2]) {
    VulkanContextState* state=vk_context_current();
    if(!state || !vk_is_ready() || workspace[0]>UINT32_MAX || workspace[1]>UINT32_MAX)return 0;
    graph_scratch_begin();
    if(state->domain_enforced) {
        if(state->modern_workspace_bytes<16 || workspace[0]>state->modern_workspace_slots[0] ||
           workspace[1]>state->modern_workspace_slots[1])return 0;
    } else if(!state->modern_bootstrap_capacity || workspace[0]>state->modern_bootstrap_slots[0] ||
              workspace[1]>state->modern_bootstrap_slots[1]) {
        size_t aligned[2];
        if(!vk_graph_align_up((size_t)workspace[0],graph_alignment,&aligned[0]) ||
           !vk_graph_align_up((size_t)workspace[1],graph_alignment,&aligned[1]) ||
           aligned[0]>SIZE_MAX-2u*graph_alignment || aligned[1]>SIZE_MAX-2u*graph_alignment-aligned[0])return 0;
        size_t bytes=aligned[0]+aligned[1]+2u*graph_alignment;
        size_t offset=graph_scratch_alloc(bytes);
        if(offset==SIZE_MAX)return 0;
        state->modern_bootstrap_offset=offset;state->modern_bootstrap_capacity=bytes;
        state->modern_bootstrap_slots[0]=(size_t)workspace[0];state->modern_bootstrap_slots[1]=(size_t)workspace[1];
        state->modern_zero_ready=0;
    }
    if(!state->modern_zero_ready) {
        uint32_t zero[4]={0,0,0,0};
        size_t offset=state->domain_enforced?state->modern_workspace_offset:state->modern_bootstrap_offset;
        if(!vk_staging_upload(offset,zero,8) || !vk_staging_upload(offset+graph_alignment,zero,8))return 0;
        state->modern_zero_ready=1;
    }
    return 1;
}
static int native_llm_emit(void* user,const VxLlmGpuDispatch* d) {
    (void)user;
    VulkanContextState* state=vk_context_current();
    VkGraphBinding bindings[9];VkTensorSlot* outputs[8]={0};
    if(!state || !d || (unsigned)d->shader>=VX_LLM_GPU_SHADER_COUNT)return 0;
    for(uint32_t i=0;i<d->buffer_count;i++) {
        const VxLlmGpuBuffer* ref=&d->buffers[i];
        if(ref->workspace==4) {
            size_t offset=graph_scratch_upload(d->constants,d->constants_bytes);
            if(offset==SIZE_MAX || ref->bytes!=d->constants_bytes)return 0;
            bindings[i]=(VkGraphBinding){offset,(size_t)ref->bytes};
        } else if(ref->workspace) {
            if(ref->workspace>3)return 0;
            size_t base=state->domain_enforced?state->modern_workspace_offset:state->modern_bootstrap_offset;
            const size_t* slots=state->domain_enforced?state->modern_workspace_slots:state->modern_bootstrap_slots;
            size_t relative=ref->output?graph_alignment:0;
            if(ref->workspace<=2) {
                if(ref->bytes>slots[ref->workspace-1])return 0;
                relative=2u*graph_alignment;
                if(ref->workspace==2) {
                    size_t first;
                    if(!vk_graph_align_up(slots[0],graph_alignment,&first))return 0;
                    relative+=first;
                }
            }
            bindings[i]=(VkGraphBinding){base+relative,(size_t)ref->bytes};
        } else {
            VkTensorSlot* slot=ref->output?graph_output_slot(ref->host,(size_t)ref->bytes):graph_ensure_device(ref->host,(size_t)ref->bytes,0);
            if(!slot)return 0;
            bindings[i]=(VkGraphBinding){slot->offset,(size_t)ref->bytes};
            if(ref->output)outputs[i]=slot;
        }
    }
    size_t params=graph_scratch_upload(d->uniform,sizeof(d->uniform));
    if(params==SIZE_MAX)return 0;
    bindings[d->buffer_count]=(VkGraphBinding){params,sizeof(d->uniform)};
    if(!vk_dispatch_kernel(&native_llm_kernels[d->shader],bindings,d->groups[0],d->groups[1],d->groups[2]))return 0;
    for(uint32_t i=0;i<d->buffer_count;i++)if(outputs[i])graph_mark_device(outputs[i]);
    return 1;
}
#define VX_LLM_NATIVE_PREFIX vk
#include "llm_gpu_native.inc"
