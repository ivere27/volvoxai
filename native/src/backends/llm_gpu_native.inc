#include "../runtime/engine_internal.h"

/* Text-included thin native entrypoints. The physical planner is linked once;
 * each backend supplies only resource resolution and the ordinary dispatcher. */
#define VX_LLM_JOIN_(a,b) a##b
#define VX_LLM_JOIN(a,b) VX_LLM_JOIN_(a,b)
#define VX_LLM_ENTRY(name) VX_LLM_JOIN(VX_LLM_NATIVE_PREFIX,name)
int VX_LLM_ENTRY(_graph_llm_f32)(const VxLLMOpArgs* args) {
    uint64_t workspace[2];
    return vx_llm_gpu_plan(args,workspace,NULL,NULL) &&
        native_llm_prepare(workspace) &&
        vx_llm_gpu_plan(args,workspace,native_llm_emit,NULL);
}
static int native_llm_row_input(const void* full,uint64_t full_elements,
        const void** pointer,uint64_t elements,unsigned slot) {
    if(!*pointer || (*pointer==full && elements==full_elements))return 1;
    VxEngineState* state=vx_engine_state_current();
    uintptr_t begin=(uintptr_t)full,address=(uintptr_t)*pointer;
    if(!state->decode_row_stage_reserved || !full || !elements || elements>UINT32_MAX/4u ||
       full_elements>UINT32_MAX/4u || full_elements%elements || address<begin ||
       (address-begin)%((size_t)elements*4u) ||
       slot>=sizeof(state->decode_numeric_stage)/sizeof(state->decode_numeric_stage[0]) ||
       elements*4u>state->decode_numeric_stage_bytes[slot])return 0;
    uint64_t first=(address-begin)/((size_t)elements*4u);
    if(first>=full_elements/elements || first>INT32_MAX)return 0;
    int32_t index=(int32_t)first;
    void* stage=state->decode_numeric_stage[slot];
    if(!stage || !VX_LLM_ENTRY(_graph_row_index_transfer)(full,(size_t)full_elements*4u,
            &index,1,stage,(size_t)elements*4u,(uint32_t)elements,
            (uint32_t)(full_elements/elements),0))return 0;
    *pointer=stage;return 1;
}

int VX_LLM_ENTRY(_graph_llm_row_f32)(const VxLLMOpArgs* full,const VxLLMOpArgs* projected) {
    if(!full || !projected || full->kind!=projected->kind)return 0;
    VxEngineState* state=vx_engine_state_current();
    VxLLMOpArgs row=*projected;
    for(unsigned i=0;i<8;i++) {
        if(i==7 && row.kind==VX_OP_GATED_DELTA_NET)continue;
        const void* pointer=row.x[i];
        if(!native_llm_row_input(full->x[i],full->x_elements[i],&pointer,row.x_elements[i],i))return 0;
        row.x[i]=pointer;
    }
    /* Sparse/QSA leave K and index-K resident at full capacity, so their
     * otherwise unused staging slots hold the query mask and position ID. */
    const void* pointer=row.mask;
    if(!native_llm_row_input(full->mask,full->mask_elements,&pointer,row.mask_elements,1))return 0;
    row.mask=pointer;pointer=row.positions;
    if(!native_llm_row_input(full->positions,full->position_elements,&pointer,row.position_elements,6))return 0;
    row.positions=pointer;
    uint32_t first[3]={0};int staged[3]={0};
    for(unsigned i=0;i<3;i++) {
        if(!row.y[i] || (row.y[i]==full->y[i] && row.y_elements[i]==full->y_elements[i]))continue;
        uintptr_t begin=(uintptr_t)full->y[i],address=(uintptr_t)row.y[i];
        uint64_t width=row.y_elements[i];unsigned slot=8u+i;
        if(!full->y[i] || !state->decode_row_stage_reserved || !width || width>UINT32_MAX/4u ||
           full->y_elements[i]>UINT32_MAX/4u || full->y_elements[i]%width || address<begin ||
           (address-begin)%((size_t)width*4u) ||
           slot>=sizeof(state->decode_numeric_stage)/sizeof(state->decode_numeric_stage[0]) ||
           width*4u>state->decode_numeric_stage_bytes[slot])return 0;
        uint64_t index=(address-begin)/((size_t)width*4u);
        if(index>=full->y_elements[i]/width || index>INT32_MAX)return 0;
        first[i]=(uint32_t)index;staged[i]=1;row.y[i]=(float*)state->decode_numeric_stage[slot];
        if(!row.y[i])return 0;
    }
    if(!VX_LLM_ENTRY(_graph_llm_f32)(&row))return 0;
    for(unsigned i=0;i<3;i++)if(staged[i]) {
        int32_t index=(int32_t)first[i];
        if(!VX_LLM_ENTRY(_graph_row_index_transfer)(row.y[i],(size_t)row.y_elements[i]*4u,&index,1,
                full->y[i],(size_t)full->y_elements[i]*4u,(uint32_t)row.y_elements[i],
                (uint32_t)(full->y_elements[i]/row.y_elements[i]),1))return 0;
    }
    return 1;
}

int VX_LLM_ENTRY(_graph_llm_router)(const VxLlmGpuRouterArgs* args) {
    uint64_t workspace[2]={0,0};
    return vx_llm_gpu_router_plan(args,NULL,NULL) && native_llm_prepare(workspace) &&
        vx_llm_gpu_router_plan(args,native_llm_emit,NULL);
}
int VX_LLM_ENTRY(_graph_llm_gated)(const VxLlmGpuGatedArgs* args) {
    uint64_t workspace[2];
    return vx_llm_gpu_gated_plan(args,workspace,NULL,NULL) &&
        native_llm_prepare(workspace) &&
        vx_llm_gpu_gated_plan(args,workspace,native_llm_emit,NULL);
}
int VX_LLM_ENTRY(_graph_llm_ngram)(const VxLlmGpuNgramArgs* args) {
    uint64_t workspace[2]={0,0};
    return vx_llm_gpu_ngram_plan(args,NULL,NULL) && native_llm_prepare(workspace) &&
        vx_llm_gpu_ngram_plan(args,native_llm_emit,NULL);
}
int VX_LLM_ENTRY(_graph_llm_rope)(const VxLlmGpuRopeArgs* args) {
    uint64_t workspace[2]={0,0};
    return vx_llm_gpu_rope_plan(args,NULL,NULL) && native_llm_prepare(workspace) &&
        vx_llm_gpu_rope_plan(args,native_llm_emit,NULL);
}
#undef VX_LLM_ENTRY
#undef VX_LLM_JOIN
#undef VX_LLM_JOIN_
#undef VX_LLM_NATIVE_PREFIX
