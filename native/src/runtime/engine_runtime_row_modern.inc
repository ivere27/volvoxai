/* Batch-one projections for token-local multi-output nodes and finite causal
 * convolution. CUDA stages resident rows; it never reads activation bytes. */
#include "modern_row.h"
static int vx_native_row_selected(void) {
    if(g_vx_cpu_dispatch || !vx_engine_state_current()->decode_row_stage_reserved) return 0;
    return
#if VOLVOXAI_ENABLE_VULKAN
        g_use_vulkan ||
#endif
#if VOLVOXAI_ENABLE_OPENGL
        g_use_opengl ||
#endif
#if VOLVOXAI_ENABLE_METAL
        g_use_metal ||
#endif
        0;
}

static int vx_modern_row_span(T* tensor,int axis,int first,int rows,int stage,VxModernRowSpan* span) {
    if(!tensor || !tensor->data || tensor->elem_size!=4 || axis<0 || axis>=tensor->ndim ||
       first<0 || rows<1 || first>tensor->shape[axis]-rows) return 0;
    size_t before=1,width=1;
    for(int i=0;i<axis;i++) before*=(size_t)tensor->shape[i];
    for(int i=axis+1;i<tensor->ndim;i++) {
        if(tensor->shape[i]<=0 || width>SIZE_MAX/(uint32_t)tensor->shape[i]) return 0;
        width*=(uint32_t)tensor->shape[i];
    }
    if(before!=1 || !width || (uint64_t)tensor->shape[axis]*width!=(uint64_t)tensor->numel) return 0;
    *span=(VxModernRowSpan){.tensor=tensor,.saved=*tensor,.narrow=*tensor,.axis=axis,.first=first,.rows=rows,.stage=stage,.width=width};
    span->narrow.shape[axis]=rows;span->narrow.numel=(long)((size_t)rows*width);
    span->narrow.data=tensor->data+(size_t)first*width;span->narrow.owns=0;
    return 1;
}

int vx_engine_modern_row_projection(Node* node,int row,VxModernRowProjection* p) {
    if(!node || row<0 || g_decode_slots>1 || g_node_fusion[node-g_n].id) return 0;
    VxOperatorKind op=node->operator_kind;
    if(op!=VX_OP_MOE_ROUTER && op!=VX_OP_MOE_GATED_MLP && op!=VX_OP_MOE_LINEAR &&
       op!=VX_OP_GATHER && op!=VX_OP_REDUCE_MEAN && op!=VX_OP_REDUCE_SUM && op!=VX_OP_CONV_2D && op!=VX_OP_CONCAT && op!=VX_OP_SLICE && op!=VX_OP_TRANSPOSE && op!=VX_OP_EXPAND) return 0;
    memset(p,0,sizeof(*p));
    T* output=nout(node,VX_PORT_OUT);
    if(!output && node->nout) output=t_find(node->outs[0].name);
    if(!output || output->ndim<2 || output->shape[0]!=1) return 0;
    int axis=op==VX_OP_CONV_2D?2:1;
    if(axis>=output->ndim || output->shape[axis]<=row) return 0;
    int sequence=output->shape[axis];
    if(op==VX_OP_EXPAND) {
        T* input=node_input_default(node);
        if(!input || input->ndim!=output->ndim || input->ndim<3 || input->shape[0]!=1 ||
           input->shape[1]!=sequence || input->dtype!=output->dtype) return 0;
        for(int d=2;d<input->ndim;d++)
            if(input->shape[d]!=1 && input->shape[d]!=output->shape[d]) return 0;
    }
    if(op==VX_OP_TRANSPOSE) {
        T* input=node_input_default(node);
        if(!input || input->ndim!=output->ndim || input->ndim<3 || input->shape[0]!=1 ||
           input->shape[1]!=sequence || !node->parsed_params.has_perm ||
           node->parsed_params.perm[0]!=0 || node->parsed_params.perm[1]!=1) return 0;
        unsigned seen=0;
        for(int d=0;d<input->ndim;d++) {
            int source=node->parsed_params.perm[d];
            if(source<0 || source>=input->ndim || (seen&(1u<<source)) ||
               output->shape[d]!=input->shape[source]) return 0;
            seen|=1u<<source;
        }
    }
    if(op==VX_OP_SLICE) {
        T* input=node_input_default(node);
        if(!input || input->dtype!=T_I32 || output->dtype!=T_I32 || input->ndim!=2 || input->shape[0]!=3 ||
           input->shape[1]!=sequence || output->ndim!=2 || output->shape[0]!=1 ||
           !node->parsed_params.has_slice || node->parsed_params.slice_rank!=1 || node->parsed_params.axes[0]!=0 ||
           node->parsed_params.steps[0]!=1 ||
           node->parsed_params.starts[0]<0 || node->parsed_params.starts[0]>=3 ||
           node->parsed_params.ends[0]!=node->parsed_params.starts[0]+1) return 0;
        int first=node->parsed_params.starts[0]*sequence+row;
        p->inputs[0]=(VxModernRowSpan){.tensor=input,.saved=*input,.narrow=*output,.axis=0,.first=first,.rows=1,.stage=0,.width=1};
        p->inputs[0].narrow.shape[1]=1;p->inputs[0].narrow.numel=1;
        p->inputs[0].narrow.data=input->data+first;p->inputs[0].narrow.owns=0;
        p->inputs_count=1;p->copy=1;
    } else if(op==VX_OP_CONV_2D) {
        T* input=nin(node,VX_PORT_INPUT),*weight=nin(node,VX_PORT_WEIGHT);
        if(!input || !weight || input->ndim!=4 || output->ndim!=4 || input->shape[0]!=1 || input->shape[1]!=1 ||
           output->shape[1]!=1 || input->shape[2]!=sequence || weight->ndim!=4 || weight->shape[0]!=1 ||
           node->parsed_params.stride[0]!=1 || node->parsed_params.stride[1]!=1 ||
           node->parsed_params.dilation[0]!=1 || node->parsed_params.dilation[1]<1 ||
           node->parsed_params.pads[0]!=0 || node->parsed_params.pads[2]!=0 || node->parsed_params.pads[3]!=0) return 0;
        int64_t history=(int64_t)(weight->shape[1]-1)*node->parsed_params.dilation[1];
        if(history<0 || history>INT32_MAX || node->parsed_params.pads[1]!=history) return 0;
        int first=row>history?row-(int)history:0;
        if(!vx_modern_row_span(input,2,first,row-first+1,0,&p->inputs[0])) return 0;
        p->inputs_count=1;p->convolution=1;p->padding_left=(int)history-(row-first);
    } else {
        for(int i=0;i<node->nin;i++) {
            T* tensor=t_find(node->ins[i].name);
            if(!tensor) return 0;
            if(vx_decode_projection_invariant(node,tensor)) continue;
            int input_axis=tensor->ndim>=2 && tensor->shape[0]==1?1:
                tensor->ndim==1 && tensor->shape[0]==sequence?0:-1;
            if(input_axis<0 || tensor->shape[input_axis]!=sequence || p->inputs_count==3 ||
               !vx_modern_row_span(tensor,input_axis,row,1,p->inputs_count,&p->inputs[p->inputs_count])) return 0;
            p->inputs_count++;
        }
        if(!p->inputs_count) return 0;
        if(op==VX_OP_GATHER || op==VX_OP_REDUCE_MEAN || op==VX_OP_REDUCE_SUM || op==VX_OP_CONCAT) {
            T* input=node_input_default(node);
            int reduction=vx_node_param_i32(node,VX_NODE_PARAM_AXIS,-1);
            if(reduction<0 && input) reduction+=input->ndim;
            if(!input || ((op==VX_OP_REDUCE_MEAN || op==VX_OP_REDUCE_SUM) && reduction<=1)) return 0;
            if(op==VX_OP_GATHER && reduction==1) return 0;
            if(op==VX_OP_CONCAT && reduction<=1) return 0;
        }
    }
    for(int i=0;i<node->nout;i++) {
        T* tensor=t_find(node->outs[i].name);
        if(p->outputs_count==2 || !tensor || tensor->ndim<=axis || tensor->shape[axis]!=sequence ||
           !vx_modern_row_span(tensor,axis,row,1,3-p->outputs_count,&p->outputs[p->outputs_count])) return 0;
        p->outputs_count++;
    }
    return p->outputs_count>0 && p->inputs_count+p->outputs_count<=4;
}

static int vx_modern_row_compatible(Node* node,int row) {
    VxModernRowProjection p;
    return vx_engine_modern_row_projection(node,row,&p);
}

static int vx_native_plain_row_projection(Node* node,int row,VxModernRowProjection* p) {
    VxDecodeRowProjection common;
    if(!vx_decode_projection_describe(node,row,&common) || common.rows.slots!=1 ||
#if VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
       (!common.rows.unpaged && !vx_native_row_selected())) return 0;
#else
       !common.rows.unpaged) return 0;
#endif
    const GraphNodeFusion* fusion=&g_node_fusion[node-g_n];
    if(fusion->id) {
        if(fusion->id!=GRAPH_FUSION_CHAINED_ELEMENTWISE || node->operator_kind!=VX_OP_ADD ||
           !fusion->aux_tensor[0] || !vx_decode_projection_add_input(&common,t_find(fusion->aux_tensor),common.output.sequence,0)) return 0;
    }
    if(common.attention) {
        T* q=nin(node,VX_PORT_Q),*k=nin(node,VX_PORT_K),*v=nin(node,VX_PORT_V);
        int heads=node->parsed_params.heads,kv_heads=node->parsed_params.kv_heads;
        if(node->operator_kind!=VX_OP_CROSS_SDPA || !q || !k || !v ||
           q->dtype!=T_F32 || k->dtype!=T_F32 || v->dtype!=T_F32 || common.output.tensor->dtype!=T_F32 ||
           heads<1 || kv_heads<1 || heads%kv_heads || q->shape[q->ndim-1]%heads ||
           k->shape[k->ndim-1]%kv_heads || q->shape[q->ndim-1]/heads!=k->shape[k->ndim-1]/kv_heads ||
           q->shape[q->ndim-1]/heads>512 || v->shape[v->ndim-1]!=k->shape[k->ndim-1] ||
           (common.mask && (!common.mask->data || common.mask->dtype!=T_I32 ||
             !(common.mask->is_graph_input || volvoxai_engine_tensor_is_model_weight_locked(common.mask->name))))) return 0;
        float scale=node->parsed_params.has_scale?node->parsed_params.scale:1.f/sqrtf((float)(q->shape[q->ndim-1]/heads));
        if(!isfinite(scale) || scale<=0) return 0;
    }
    if(common.rope) {
        VxSequenceRoPE rope;
        if(!vx_sequence_rope_parameters(node,common.inputs[0].tensor,common.output.tensor,&rope) ||
           rope.batch!=1 || rope.theta<1.f || rope.width/rope.heads>128 ||
           (rope.positions && !(rope.positions->is_graph_input ||
               volvoxai_engine_tensor_is_model_weight_locked(rope.positions->name)))) return 0;
    }
    memset(p,0,sizeof(*p));p->copy=common.view;p->attention=common.attention;p->rope=common.rope;p->query=row;
    for(int i=0;i<common.input_count;i++) {
        VxDecodeRowOperand* input=&common.inputs[i];
        if(!vx_modern_row_span(input->tensor,input->axis,input->prefix?0:row,
               input->prefix?common.rows.key_capacity:1,i,&p->inputs[i])) return 0;
        p->inputs_count++;
    }
    if(!vx_modern_row_span(common.output.tensor,common.output.axis,row,1,3,&p->outputs[0])) return 0;
    p->outputs_count=1;
    return 1;
}

#if VOLVOXAI_ENABLE_CUDA || VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
int vx_runtime_backend_row_index_transfer(const void* source,size_t source_bytes,const int32_t* indices,uint32_t rows,
        void* destination,size_t destination_bytes,uint32_t width,uint32_t indexed_rows,uint32_t scatter) {
#if VOLVOXAI_ENABLE_CUDA
    if(g_use_cuda) return cuda_graph_row_index_transfer(source,source_bytes,indices,rows,destination,destination_bytes,width,indexed_rows,scatter);
#endif
#if VOLVOXAI_ENABLE_VULKAN
    if(g_use_vulkan) return vk_graph_row_index_transfer(source,source_bytes,indices,rows,destination,destination_bytes,width,indexed_rows,scatter);
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if(g_use_opengl) return opengl_graph_row_index_transfer(source,source_bytes,indices,rows,destination,destination_bytes,width,indexed_rows,scatter);
#endif
#if VOLVOXAI_ENABLE_METAL
    if(g_use_metal) return metal_graph_row_index_transfer(source,source_bytes,indices,rows,destination,destination_bytes,width,indexed_rows,scatter);
#endif
    return 0;
}
static int vx_row_device_copy(const void* source,void* destination,long elements) {
#if VOLVOXAI_ENABLE_CUDA
    if(g_use_cuda) return cuda_graph_copy_32(source,destination,elements);
#endif
#if VOLVOXAI_ENABLE_VULKAN
    if(g_use_vulkan) return vk_graph_copy_32(source,destination,elements);
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if(g_use_opengl) return opengl_graph_copy_32(source,destination,elements);
#endif
#if VOLVOXAI_ENABLE_METAL
    if(g_use_metal) return metal_graph_copy_32(source,destination,elements);
#endif
    return 0;
}
static int vx_modern_device_transfer(VxModernRowSpan* span,int scatter) {
    VxEngineState* state=vx_engine_state_current();
    size_t bytes=(size_t)span->narrow.numel*4u;
    if(!state->decode_row_stage_reserved || span->stage<0 || span->stage>=4 ||
       bytes>state->decode_numeric_stage_bytes[span->stage] || (size_t)span->rows*4u>state->decode_metadata_stage_bytes) return 0;
    span->narrow.data=(float*)state->decode_numeric_stage[span->stage];
    int32_t* indices=(int32_t*)state->decode_metadata_stage;
#if VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
    if(vx_native_row_selected() && vx_paged_tensor_locked(span->tensor)) {
        /* Only retained K/V banks use the page table. All other activation
         * rows keep their logical contiguous addresses, even in a paged
         * context. The current row set belongs to this engine-state scope. */
        const VxDecodeRowSet* rows=&g_decode_rows;
        if(!span->narrow.data || !indices || g_decode_slots!=1 || rows->slots!=1 || !rows->positions || !rows->empty ||
           !rows->kv_lengths || !rows->pages || rows->unpaged ||
           !rows->pages[0].page_table || rows->pages[0].page_tokens<1 ||
           rows->pages[0].pages_per_slot<1 || span->axis<0 || span->axis>=span->saved.ndim ||
           span->saved.shape[span->axis]<1 || !span->width || span->width>UINT32_MAX ||
           span->saved.numel<1 || (uint64_t)span->saved.numel%span->width ||
           (uint64_t)span->saved.numel>SIZE_MAX/sizeof(uint32_t) ||
           (uint64_t)span->saved.numel/span->width>UINT32_MAX ||
           span->saved.elem_size!=sizeof(uint32_t) || span->narrow.elem_size!=sizeof(uint32_t) ||
           span->narrow.numel<1 || (uint64_t)span->narrow.numel>SIZE_MAX/sizeof(uint32_t) ||
           (uint64_t)span->rows*span->width!=(uint64_t)span->narrow.numel ||
           (uint64_t)span->saved.shape[span->axis]*span->width!=(uint64_t)span->saved.numel) return 0;
        int status;
        if(!scatter && span->first==0 && span->rows==rows->key_capacity)
            status=vx_decode_row_set_prefix_rows(rows,span->saved.shape[span->axis],1,indices);
        else if(span->rows==1 && span->first==rows->positions[0])
            status=vx_decode_row_set_write_rows(rows,span->saved.shape[span->axis],1,indices);
        else return 0;
        if(status!=VX_DECODE_ROW_SET_OK) return 0;
        for(int i=0;i<span->rows;i++)
            if(indices[i]<0 || (uint64_t)indices[i]>=(uint64_t)span->saved.numel/span->width) return 0;
    } else
#endif
    for(int i=0;i<span->rows;i++) indices[i]=span->first+i;
    vx_runtime_backend_mark_host(indices,state->decode_metadata_stage_bytes,0);
    return vx_runtime_backend_row_index_transfer(scatter?span->narrow.data:span->saved.data,
        scatter?bytes:(size_t)span->saved.numel*4u,indices,(uint32_t)span->rows,
        scatter?span->saved.data:span->narrow.data,scatter?(size_t)span->saved.numel*4u:bytes,
        (uint32_t)span->width,(uint32_t)((size_t)span->saved.numel/span->width),(uint32_t)scatter);
}
#endif

#if VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
static const VxF32Ops* vx_native_row_ops(void) {
#if VOLVOXAI_ENABLE_VULKAN
    if(g_use_vulkan) return &g_vulkan_f32_ops;
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if(g_use_opengl) return &g_opengl_f32_ops;
#endif
#if VOLVOXAI_ENABLE_METAL
    if(g_use_metal) return &g_metal_f32_ops;
#endif
    return NULL;
}

/* Absolute query controls are public host inputs or immutable model weights.
 * Activation rows and retained K/V never pass through host memory. */
static int vx_native_special_row(Node* node,const VxModernRowProjection* p,
        const VxDecodeRowProjection* common,const VxSequenceRoPE* rope) {
    VxEngineState* state=vx_engine_state_current();
    const VxF32Ops* ops=vx_native_row_ops();
    if(!ops || !state->decode_metadata_stage) return -1;
    int32_t* control=(int32_t*)state->decode_metadata_stage;
    if(p->rope) {
        int32_t position;
        if(rope->positions) {
            size_t index=(size_t)p->query;
            memcpy(&position,(const unsigned char*)rope->positions->data+index*sizeof(int32_t),sizeof(position));
        } else {
            int64_t value=(int64_t)rope->position_offset+p->query;
            if(value<0 || value>INT32_MAX) return -1;
            position=(int32_t)value;
        }
        if(position<0 || state->decode_metadata_stage_bytes<sizeof(*control)) return -1;
        control[0]=position;
        vx_runtime_backend_mark_host(control,state->decode_metadata_stage_bytes,0);
        VxLlmGpuRopeArgs args={p->inputs[0].narrow.data,control,p->outputs[0].narrow.data,
            1,1,(uint32_t)rope->width,(uint32_t)rope->heads,(uint32_t)rope->rotary_width,
            0,(uint32_t)rope->interleaved,1,rope->theta};
        return ops->llm_rope && ops->llm_rope(&args)?0:-1;
    }
    if(p->attention) {
        if(common->keys<1 || (size_t)common->keys>state->decode_metadata_stage_bytes/sizeof(*control)) return -1;
        int source_keys=common->mask?common->mask->shape[common->mask->ndim-1]:common->keys;
        for(int key=0;key<common->keys;key++) control[key]=!common->rows.empty[0] &&
            (!common->causal || key<common->rows.kv_lengths[0]) &&
            attention_mask_allowed(common->mask,common->mask_layout,0,p->query,key,
                common->output.sequence,source_keys,0);
        vx_runtime_backend_mark_host(control,state->decode_metadata_stage_bytes,0);
        T* q=nin(node,VX_PORT_Q),*k=nin(node,VX_PORT_K),*v=nin(node,VX_PORT_V);
        VxLLMOpArgs args={0};
        args.kind=VX_OP_SPARSE_ATTENTION;args.batch=1;args.sequence=1;args.keys=common->keys;
        args.heads=node->parsed_params.heads;args.kv_heads=node->parsed_params.kv_heads;
        args.key_dim=q->shape[q->ndim-1]/args.heads;args.value_dim=args.key_dim;
        args.scale=node->parsed_params.has_scale?node->parsed_params.scale:1.f/sqrtf((float)args.key_dim);
        args.eps=args.hc_eps=1e-6f;args.rate=1;args.mask=control;args.mask_elements=(uint64_t)args.keys;args.mask_mode=1;
        args.x[0]=q->data;args.x[1]=k->data;args.x[2]=v->data;
        args.x_elements[0]=(uint64_t)q->numel;args.x_elements[1]=(uint64_t)k->numel;args.x_elements[2]=(uint64_t)v->numel;
        args.y[0]=p->outputs[0].narrow.data;args.y_elements[0]=(uint64_t)p->outputs[0].narrow.numel;
        return ops->llm && ops->llm(&args)?0:-1;
    }
    return -1;
}
#endif

static int vx_modern_row_run(Node* node,int index,int is_last,VxModernRowProjection* p,int device) {
    int status=-1;
    VxDecodeRowProjection common={0};VxSequenceRoPE rope={0};
    if((p->attention || p->rope) && (!device ||
       !vx_decode_projection_describe(node,p->query,&common) ||
       (p->rope && !vx_sequence_rope_parameters(node,common.inputs[0].tensor,common.output.tensor,&rope)))) return -1;
#if VOLVOXAI_ENABLE_CUDA || VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
    if(device) {
        for(int i=0;i<p->inputs_count;i++) if(!vx_modern_device_transfer(&p->inputs[i],0)) return -1;
        VxEngineState* state=vx_engine_state_current();
        for(int i=0;i<p->outputs_count;i++) {
            VxModernRowSpan* span=&p->outputs[i];
            if((size_t)span->narrow.numel*4u>state->decode_numeric_stage_bytes[span->stage]) return -1;
            span->narrow.data=(float*)state->decode_numeric_stage[span->stage];
        }
    }
#else
    if(device) return -1;
#endif
    for(int i=0;i<p->inputs_count;i++) *p->inputs[i].tensor=p->inputs[i].narrow;
    for(int i=0;i<p->outputs_count;i++) *p->outputs[i].tensor=p->outputs[i].narrow;
    int saved_pad=node->parsed_params.pads[1],active=g_active_row,execution=g_execution_row,slots=g_decode_slots;
    if(p->convolution) node->parsed_params.pads[1]=p->padding_left;
    g_active_row=-1;g_execution_row=-1;g_decode_slots=0;
    if(p->attention || p->rope) {
#if VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
        status=vx_native_special_row(node,p,&common,&rope);
#endif
    } else if(p->copy) {
#if VOLVOXAI_ENABLE_CUDA || VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
        if(device) status=vx_row_device_copy(p->inputs[0].narrow.data,p->outputs[0].narrow.data,p->outputs[0].narrow.numel)?0:-1;
        else
#endif
        {memcpy(p->outputs[0].narrow.data,p->inputs[0].narrow.data,(size_t)p->outputs[0].narrow.numel*4u);status=0;}
    } else status=run_node(node,index,is_last);
    if(device && status==0 && !p->copy && !p->attention && !p->rope) {
        const char* route=vx_engine_state_current()->runtime_route_backend[index];
        const char* backend=volvoxai_engine_backend_name();
        if(!route || !backend || strncmp(route,backend,strlen(backend))) status=-1;
    }
    g_active_row=active;g_execution_row=execution;g_decode_slots=slots;node->parsed_params.pads[1]=saved_pad;
    for(int i=0;i<p->inputs_count;i++) *p->inputs[i].tensor=p->inputs[i].saved;
    for(int i=0;i<p->outputs_count;i++) *p->outputs[i].tensor=p->outputs[i].saved;
#if VOLVOXAI_ENABLE_CUDA || VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
    if(device && status==0) for(int i=0;i<p->outputs_count;i++) if(!vx_modern_device_transfer(&p->outputs[i],1)) status=-1;
#endif
    return status;
}
