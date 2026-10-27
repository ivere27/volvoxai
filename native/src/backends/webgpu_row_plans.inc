#include "../runtime/decode_projection.h"
#include "../runtime/modern_row.h"

static int vx_webgpu_row_metadata(VxWebGpuNodePlan* plan, const int* values, size_t count) {
    if (!count || count > SIZE_MAX / sizeof(int)) return 0;
    size_t bytes = count * sizeof(int);
    int* slot = vx_webgpu_arena(bytes);
    if (!slot) return 0;
    memcpy(slot, values, bytes);
    if (!validating) vx_webgpu_mark_host(NULL, slot, bytes, 0);
    return vx_webgpu_bind(plan, slot, bytes);
}

static int vx_webgpu_row_transfer(VxWebGpuNodePlan* plan, const VxDecodeRowOperand* operand,
                                   const T* narrow, const VxDecodeRowSet* rows, int scatter) {
    size_t row_bytes = operand->width * operand->tensor->elem_size;
    int count = rows->slots * (operand->prefix ? rows->key_capacity : 1);
    if (count <= 0 || row_bytes > UINT32_MAX / (uint32_t)count) return 0;
    int* indices = vx_webgpu_arena((size_t)count * sizeof(int));
    if (!indices) return 0;
    int paged = vx_paged_tensor_locked(operand->tensor);
    int status = operand->prefix
        ? vx_decode_row_set_prefix_rows(rows, operand->sequence, paged, indices)
        : vx_decode_row_set_write_rows(rows, operand->sequence, paged, indices);
    if (status != VX_DECODE_ROW_SET_OK) return 0;
    // The common CPU scatter receives empty flags separately. The device
    // transfer represents the same skipped slot with a negative row index.
    if (!operand->prefix)
        for (int slot = 0; slot < rows->slots; slot++)
            if (rows->empty[slot]) indices[slot] = -1;
    size_t full_bytes = (size_t)operand->tensor->numel * operand->tensor->elem_size;
    size_t narrow_bytes = (size_t)count * row_bytes;
    for (int i = 0; i < count; i++) if (indices[i] >= 0 && (size_t)indices[i] >= full_bytes / row_bytes) return 0;
    if (!validating) vx_webgpu_mark_host(NULL, indices, (size_t)count * sizeof(int), 0);
    uint32_t params[4] = {(uint32_t)count, (uint32_t)row_bytes, (uint32_t)(full_bytes / row_bytes), (uint32_t)scatter};
    return vx_webgpu_bind(plan, scatter ? narrow->data : operand->tensor->data, scatter ? narrow_bytes : full_bytes) &&
        vx_webgpu_bind(plan, indices, (size_t)count * sizeof(int)) &&
        vx_webgpu_bind(plan, scatter ? operand->tensor->data : narrow->data, scatter ? full_bytes : narrow_bytes) &&
        vx_webgpu_params(plan, params, 4) &&
        vx_webgpu_grid(plan, VX_SHADER_INFERENCE_ROW_INDEX_TRANSFER, ((uint32_t)narrow_bytes + 255u) / 256u, 1, 1);
}

static int vx_webgpu_row_storage(const VxDecodeRowOperand* operand, const VxDecodeRowSet* rows, T* narrow) {
    *narrow = *operand->tensor;
    narrow->shape[operand->axis] = operand->prefix ? rows->key_capacity : 1;
    uint64_t elements = (uint64_t)rows->slots * narrow->shape[operand->axis] * operand->width;
    if (!elements || elements > INT32_MAX || elements > UINT32_MAX / narrow->elem_size) return 0;
    narrow->numel = (long)elements;
    narrow->data = vx_webgpu_arena((size_t)elements * narrow->elem_size);
    return narrow->data != NULL;
}

/* A modern row is a packed GPU span, never an interior host mirror. The
 * full-capacity banks remain bound directly; only projected query/control
 * inputs and token-local outputs are transferred by the resident row shader. */
static int vx_webgpu_modern_transfer(VxWebGpuNodePlan* plan, const void* full,
        uint64_t full_elements, void* packed, uint64_t width, uint32_t first,
        uint32_t rows, int scatter) {
    if(!full || !packed || !width || !rows || full_elements%width ||
       width>UINT32_MAX/4u || full_elements>UINT32_MAX/4u ||
       rows>UINT32_MAX/(width*4u) || first>full_elements/width ||
       rows>full_elements/width-first)return 0;
    int* indices=vx_webgpu_arena((size_t)rows*sizeof(int));
    if(!indices)return 0;
    for(uint32_t row=0;row<rows;row++)indices[row]=(int)(first+row);
    if(!validating)vx_webgpu_mark_host(NULL,indices,(size_t)rows*sizeof(int),0);
    uint32_t params[4]={rows,(uint32_t)width*4u,(uint32_t)(full_elements/width),(uint32_t)scatter};
    return vx_webgpu_bind(plan,scatter?packed:full,(size_t)(scatter?rows*width:full_elements)*4u) &&
        vx_webgpu_bind(plan,indices,(size_t)rows*sizeof(int)) &&
        vx_webgpu_bind(plan,scatter?full:packed,(size_t)(scatter?full_elements:rows*width)*4u) &&
        vx_webgpu_params(plan,params,4) &&
        vx_webgpu_grid(plan,VX_SHADER_INFERENCE_ROW_INDEX_TRANSFER,
            (uint32_t)((rows*width+63u)/64u),1,1);
}

static int vx_webgpu_modern_input(VxWebGpuNodePlan* plans,uint32_t* used,
        const void* full,uint64_t full_elements,const void** pointer,uint64_t elements) {
    if(!*pointer || (*pointer==full && elements==full_elements))return 1;
    uintptr_t begin=(uintptr_t)full,address=(uintptr_t)*pointer;
    if(!full || !elements || full_elements%elements || elements>UINT32_MAX/4u ||
       address<begin || (address-begin)%((size_t)elements*4u) ||
       (address-begin)/((size_t)elements*4u)>UINT32_MAX)return 0;
    uint32_t first=(uint32_t)((address-begin)/((size_t)elements*4u));
    void* stage=vx_webgpu_arena((size_t)elements*4u);
    if(!stage || !vx_webgpu_modern_transfer(&plans[(*used)++],full,full_elements,
            stage,elements,first,1,0))return 0;
    *pointer=stage;return 1;
}

static int vx_webgpu_plan_llm_row(const Node* node,VxWebGpuNodePlan* plans,uint32_t* count) {
    VxLLMOpArgs full,row;
    if(!vx_engine_llm_node_args((Node*)node,&full))return 0;
    int described=vx_engine_llm_row_args((Node*)node,g_active_row,&row);
    if(!described)return 0;
    if(described==2) {*count=0;return 1;}
    uint32_t used=0;
    for(unsigned i=0;i<8;i++) {
        if(i==7 && row.kind==VX_OP_GATED_DELTA_NET)continue;
        const void* pointer=row.x[i];
        if(!vx_webgpu_modern_input(plans,&used,full.x[i],full.x_elements[i],
                &pointer,row.x_elements[i]))return 0;
        row.x[i]=pointer;
    }
    const void* pointer=row.mask;
    if(!vx_webgpu_modern_input(plans,&used,full.mask,full.mask_elements,&pointer,row.mask_elements))return 0;
    row.mask=pointer;pointer=row.positions;
    if(!vx_webgpu_modern_input(plans,&used,full.positions,full.position_elements,&pointer,row.position_elements))return 0;
    row.positions=pointer;
    void* stage[3]={0};uint32_t first[3]={0};
    for(unsigned i=0;i<3;i++) {
        if(!row.y[i] || (row.y[i]==full.y[i] && row.y_elements[i]==full.y_elements[i]))continue;
        uintptr_t begin=(uintptr_t)full.y[i],address=(uintptr_t)row.y[i];
        uint64_t width=row.y_elements[i];
        if(!width || width>UINT32_MAX/4u || full.y_elements[i]%width || address<begin ||
           (address-begin)%((size_t)width*4u) || (address-begin)/((size_t)width*4u)>UINT32_MAX)return 0;
        first[i]=(uint32_t)((address-begin)/((size_t)width*4u));
        stage[i]=vx_webgpu_arena((size_t)width*4u);
        if(!stage[i])return 0;
        row.y[i]=stage[i];
    }
    uint64_t workspace[2];
    if(!vx_llm_gpu_plan(&row,workspace,NULL,NULL) ||
       (validating && !vx_webgpu_llm_reserve(workspace)))return 0;
    uint32_t capacity=vx_webgpu_plan_capacity(node);
    if(used>=capacity)return 0;
    VxWebGpuLlmEmission emission={plans+used,0,capacity-used};
    if(!vx_llm_gpu_plan(&row,workspace,vx_webgpu_llm_emit,&emission))return 0;
    used+=emission.count;
    for(unsigned i=0;i<3;i++)if(stage[i]) {
        if(used>=capacity || !vx_webgpu_modern_transfer(&plans[used++],full.y[i],
                full.y_elements[i],stage[i],row.y_elements[i],first[i],1,1))return 0;
    }
    *count=used;return 1;
}

static int vx_webgpu_plan_modern_projection(const Node* node,VxModernRowProjection* p,
        VxWebGpuNodePlan* plans,uint32_t* count) {
    uint32_t used=0;int accepted=0;
    for(int i=0;i<p->inputs_count;i++) {
        VxModernRowSpan* span=&p->inputs[i];
        span->narrow.data=vx_webgpu_arena((size_t)span->narrow.numel*4u);
        if(!span->narrow.data || !vx_webgpu_modern_transfer(&plans[used++],span->saved.data,
                span->saved.numel,span->narrow.data,span->width,span->first,span->rows,0))return 0;
    }
    for(int i=0;i<p->outputs_count;i++) {
        VxModernRowSpan* span=&p->outputs[i];
        span->narrow.data=vx_webgpu_arena((size_t)span->narrow.numel*4u);
        if(!span->narrow.data)return 0;
    }
    for(int i=0;i<p->inputs_count;i++)*p->inputs[i].tensor=p->inputs[i].narrow;
    for(int i=0;i<p->outputs_count;i++)*p->outputs[i].tensor=p->outputs[i].narrow;
    int saved_pad=node->parsed_params.pads[1];
    if(p->convolution)((Node*)node)->parsed_params.pads[1]=p->padding_left;
    if(p->copy) {
        p->outputs[0].narrow.data=p->inputs[0].narrow.data;
        accepted=1;
    } else if(node->operator_kind==VX_OP_CONCAT || node->operator_kind==VX_OP_CONCAT2) {
        uint32_t numerical=0;
        accepted=vx_webgpu_plan_concat(node,p->outputs[0].tensor,plans+used,&numerical);used+=numerical;
    } else if(vx_webgpu_llm_node(node)) {
        uint32_t numerical=0;
        accepted=vx_webgpu_plan_llm_sequence(node,plans+used,&numerical);used+=numerical;
    } else {
        T* primary=vx_webgpu_port(node,VX_PORT_INPUT);
        if(!primary)primary=vx_webgpu_port(node,VX_PORT_DATA);
        accepted=vx_webgpu_plan_node(node,primary,p->outputs[0].tensor,&plans[used++]);
    }
    ((Node*)node)->parsed_params.pads[1]=saved_pad;
    for(int i=0;i<p->inputs_count;i++)*p->inputs[i].tensor=p->inputs[i].saved;
    for(int i=0;i<p->outputs_count;i++)*p->outputs[i].tensor=p->outputs[i].saved;
    if(!accepted)return 0;
    for(int i=0;i<p->outputs_count;i++) {
        VxModernRowSpan* span=&p->outputs[i];
        if(used>=vx_webgpu_plan_capacity(node) || !vx_webgpu_modern_transfer(&plans[used++],
                span->saved.data,span->saved.numel,span->narrow.data,span->width,span->first,span->rows,1))return 0;
    }
    *count=used;return 1;
}

static int vx_webgpu_plan_row(const Node* node, T* input, T* output,
                              VxWebGpuNodePlan* plans, uint32_t* count) {
    (void)input;
    if(vx_webgpu_llm_node(node) && node->operator_kind!=VX_OP_MOE_ROUTER &&
            node->operator_kind!=VX_OP_MOE_GATED_MLP &&
            (node->operator_kind!=VX_OP_PACKED_LINEAR || g_decode_slots<=1)) {
        if(node->operator_kind==VX_OP_N_GRAM_IDS)
            return vx_webgpu_plan_llm_sequence(node,plans,count);
        return vx_webgpu_plan_llm_row(node,plans,count);
    }
    VxModernRowProjection modern;
    if(vx_engine_modern_row_projection((Node*)node,g_active_row,&modern))
        return vx_webgpu_plan_modern_projection(node,&modern,plans,count);
    VxDecodeRowProjection p;
    VxSequenceRoPE rope = {0};
    T saved[3], narrow[3], narrow_output, row_mask = {0};
    uint32_t used = 0;
    int accepted = 0;
    if (!vx_decode_projection_describe(node, g_active_row, &p) ||
        !vx_webgpu_row_storage(&p.output, &p.rows, &narrow_output)) return 0;
    if (p.rope && (!vx_sequence_rope_parameters((Node*)node, p.inputs[0].tensor,
            p.output.tensor, &rope) || rope.batch != p.rows.slots ||
            (size_t)rope.width != p.output.width)) return 0;
    for (int i = 0; i < p.input_count; i++) {
        saved[i] = *p.inputs[i].tensor;
        if (!vx_webgpu_row_storage(&p.inputs[i], &p.rows, &narrow[i]) ||
            !vx_webgpu_row_transfer(&plans[used++], &p.inputs[i], &narrow[i], &p.rows, 0)) return 0;
    }
    if (p.attention) {
        row_mask.dtype = T_I32; row_mask.elem_size = 4; row_mask.ndim = 2;
        row_mask.shape[0] = p.rows.slots; row_mask.shape[1] = p.keys;
        row_mask.numel = (long)p.rows.slots * p.keys;
        if (row_mask.numel <= 0 || (uint64_t)row_mask.numel > UINT32_MAX / 4u) return 0;
        row_mask.data = vx_webgpu_arena((size_t)row_mask.numel * 4u);
        VxWebGpuNodePlan* mask_plan = &plans[used++];
        uint32_t params[6] = {(uint32_t)p.rows.slots, (uint32_t)p.keys,
            (uint32_t)vx_webgpu_port(node, VX_PORT_K)->shape[p.output.tensor->ndim - 2],
            (uint32_t)p.output.sequence, (uint32_t)p.mask_layout, (uint32_t)p.causal};
        int* positions = vx_webgpu_arena((size_t)p.rows.slots * sizeof(int));
        int zero = 0;
        if (!positions) return 0;
        for (int slot = 0; slot < p.rows.slots; slot++) positions[slot] = p.rows.empty[slot] ? -1 : p.rows.positions[slot];
        if (!row_mask.data ||
            !(p.mask ? vx_webgpu_bind(mask_plan, p.mask->data, (size_t)p.mask->numel * 4u) : vx_webgpu_row_metadata(mask_plan, &zero, 1)) ||
            !vx_webgpu_row_metadata(mask_plan, positions, (size_t)p.rows.slots) ||
            !vx_webgpu_bind(mask_plan, row_mask.data, (size_t)row_mask.numel * 4u) ||
            !vx_webgpu_params(mask_plan, params, 6) ||
            !vx_webgpu_grid(mask_plan, VX_SHADER_INFERENCE_DECODE_KEEP_MASK, ((uint32_t)row_mask.numel + 63u) / 64u, 1, 1)) return 0;
    }
    for (int i = 0; i < p.input_count; i++) *p.inputs[i].tensor = narrow[i];
    T* primary = vx_webgpu_port(node, VX_PORT_INPUT);
    if (!primary) primary = vx_webgpu_port(node, VX_PORT_X);
    if (!primary) primary = vx_webgpu_port(node, VX_PORT_DATA);
    if (!primary) primary = vx_webgpu_port(node, VX_PORT_A);
    if (p.view) {
        // A proved row view has identical bytes. Its original shape operands
        // describe the whole tensor and are not evaluated a second time.
        narrow_output.data = primary->data;
        accepted = 1;
    } else if (p.rope) {
        int32_t* positions = vx_webgpu_arena((size_t)p.rows.slots * sizeof(int32_t));
        T packed_positions = {0};
        if (!positions) goto restore;
        for (int slot = 0; slot < p.rows.slots; ++slot) {
            if (p.rows.empty[slot]) { positions[slot] = 0; continue; }
            int position = p.rows.positions[slot];
            if (rope.positions && !validating) {
                size_t index = rope.position_mode == 1 ? (size_t)position :
                    (size_t)slot * (size_t)rope.sequence + (size_t)position;
                memcpy(&positions[slot], (const unsigned char*)rope.positions->data + index * sizeof(int32_t), sizeof(int32_t));
            } else positions[slot] = validating ? 0 : rope.position_offset + position;
        }
        packed_positions.dtype = T_I32; packed_positions.elem_size = 4;
        packed_positions.ndim = 2; packed_positions.shape[0] = p.rows.slots;
        packed_positions.shape[1] = 1; packed_positions.numel = p.rows.slots;
        packed_positions.data = (float*)positions;
        if (!validating) vx_webgpu_mark_host(NULL, positions, (size_t)p.rows.slots * sizeof(int32_t), 0);
        rope.batch = p.rows.slots; rope.sequence = 1; rope.positions = &packed_positions;
        rope.position_mode = 2; rope.position_offset = 0;
        accepted = vx_webgpu_plan_rope(node, primary, &narrow_output, &plans[used++], &rope);
    } else if (p.attention) {
        accepted = (node->operator_kind == VX_OP_Q_SDPA)
            ? vx_webgpu_plan_qsdpa_masked(node, &narrow_output, &plans[used++], &row_mask, 1)
            : vx_webgpu_plan_cross_sdpa_masked(node, &narrow_output, &plans[used++], &row_mask, 1);
    } else if (node->operator_kind == VX_OP_Q_LAYER_NORM) {
        uint32_t numerical = 0;
        accepted = vx_webgpu_plan_qnorm(node, primary, &narrow_output, &plans[used], &numerical, 0);
        used += numerical;
    } else if (node->operator_kind == VX_OP_PACKED_LINEAR) {
        T saved_output = *p.output.tensor;
        uint32_t numerical = 0;
        *p.output.tensor = narrow_output;
        accepted = vx_webgpu_plan_llm_sequence(node, &plans[used], &numerical);
        *p.output.tensor = saved_output;
        used += numerical;
    } else accepted = vx_webgpu_plan_node(node, primary, &narrow_output, &plans[used++]);
restore:
    for (int i = 0; i < p.input_count; i++) *p.inputs[i].tensor = saved[i];
    if (!accepted ||
        !vx_webgpu_row_transfer(&plans[used++], &p.output, &narrow_output, &p.rows, 1)) return 0;
    *count = used;
    (void)output;
    return 1;
}

int vx_webgpu_node_row_compatible(const char* output_name, int row) {
    VxEngineState* state = vx_engine_state_current();
    if (!state->webgpu_state || row < 0) return 0;
    const Node* node = NULL;
    for (int i = 0; i < state->node_count; i++)
        if (!strcmp(state->nodes[i].out, output_name)) { node = &state->nodes[i]; break; }
    if (!node) return 0;
    uint32_t capacity=vx_webgpu_plan_capacity(node);
    VxWebGpuNodePlan* plans=capacity?calloc(capacity,sizeof(*plans)):NULL;
    if(!plans)return 0;
    VxWebGpuState* original=state->webgpu_state;
    VxWebGpuState proof={.limits=original->limits};
    int previous=validating,active_row=g_active_row,value_domain=g_bounded_gpu_value_domain_proven;
    int node_index=g_vx_runtime_node.idx;
    state->webgpu_state=&proof;
    validating=1;g_active_row=row;g_vx_runtime_node.idx=(int)(node-g_n);
    g_bounded_gpu_value_domain_proven=1;
    uint32_t count=0;
    int accepted=vx_webgpu_plan_row(node,NULL,t_find(output_name),plans,&count) && count<=capacity;
    for(uint32_t i=0;accepted && i<count;i++)accepted=vx_webgpu_seal(&plans[i]);
    g_bounded_gpu_value_domain_proven=value_domain;g_active_row=active_row;validating=previous;
    g_vx_runtime_node.idx=node_index;state->webgpu_state=original;
    while(proof.arena) {VxWebGpuArenaBlock* next=proof.arena->next;free(proof.arena);proof.arena=next;}
    free(proof.spans);free(plans);
    return accepted;
}
