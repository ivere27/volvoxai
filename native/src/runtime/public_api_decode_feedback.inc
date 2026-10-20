/* The Generate driver enters the ordinary context transaction.
 * This descriptor proves that every feedback token is in the embedding's
 * vocabulary before any input bytes or device commands change. */
typedef struct {
    T* tokens;
    T* keep;
    T* output;
    int first;
    int end;
    VxTensorBinding roots[2];
} VxDecodeFeedback;

static int vx_incremental_has_recurrence(VxExecutionContext* context) {
    const VxEngineState* state=context->engine_state;
    for(int node=0;node<state->node_count;node++) if(state->nodes[node].operator_kind==VX_OP_GATED_DELTA_NET) return 1;
    return 0;
}

static int vx_incremental_modern_history(VxExecutionContext* context) {
    const cJSON* generation=cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model),"generation");
    if(cJSON_IsObject(cJSON_GetObjectItemCaseSensitive(generation,"incremental_controls"))) return 1;
    const VxEngineState* state=context->engine_state;
    for(int i=0;i<state->node_count;i++) {
        VxOperatorKind op=state->nodes[i].operator_kind;
        if((op>=VX_OP_GATED_DELTA_NET && op<=VX_OP_SPARSE_INDEXER) ||
           op==VX_OP_MOE_ROUTER || op==VX_OP_MOE_LINEAR || op==VX_OP_MOE_GATED_MLP) return 1;
    }
    return 0;
}

static int vx_incremental_previous_rows_immutable(VxExecutionContext* context,const VxTensorBinding* inputs,size_t count,int active) {
    const cJSON* generation=cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model),"generation");
    int explicit_controls=cJSON_IsObject(cJSON_GetObjectItemCaseSensitive(generation,"incremental_controls"));
    if(!explicit_controls && !vx_incremental_modern_history(context)) return 1;
    const cJSON* incremental=cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model),"incremental");
    const cJSON* updates=cJSON_GetObjectItemCaseSensitive(incremental,"update_inputs");
    VxEngineStateScope scope=vx_engine_state_scope_enter(context->engine_state);
    int valid=1;
    for(size_t i=0;i<count && valid;i++) {
        int kind=vx_incremental_control_kind(generation,inputs[i].name);
        const char* token=vx_execution_string(generation,"token_input"),*keep=vx_execution_string(generation,"keep_input");
        if(explicit_controls) {
            if(!kind && strcmp(inputs[i].name,token) && strcmp(inputs[i].name,keep)) continue;
        } else if(!vx_execution_names_contains(updates,inputs[i].name)) continue;
        T* previous=t_find(inputs[i].name);
        if(inputs[i].location!=VX_MEMORY_HOST || !inputs[i].data || !previous || !previous->data || previous->elem_size!=4 ||
           (explicit_controls && previous->dtype!=T_I32) ||
           inputs[i].byte_size!=(size_t)previous->numel*4u || previous->ndim<1) {valid=0;break;}
        if(!explicit_controls) {
            int axis=previous->ndim==1?0:1;
            int mask=0;
            for(int n=0;n<g_nn;n++) for(int port=0;port<g_n[n].nin;port++)
                if(g_n[n].ins[port].port==VX_PORT_MASK && !strcmp(g_n[n].ins[port].name,previous->name)) mask=1;
            if(mask && previous->dtype==T_I32 && previous->ndim==2 && previous->shape[0]==previous->shape[1]) axis=0;
            if(previous->ndim==4 && previous->shape[0]==1 && previous->shape[1]==1)
                for(int n=0;n<g_nn;n++) if(g_n[n].operator_kind==VX_OP_CONV_2D)
                    for(int port=0;port<g_n[n].nin;port++)
                        if(g_n[n].ins[port].port==VX_PORT_INPUT &&
                           !strcmp(g_n[n].ins[port].name,previous->name)) axis=2;
            size_t planes=1,width=1;
            for(int d=0;d<axis;d++) planes*=(size_t)previous->shape[d];
            int capacity=previous->shape[axis];
            if((planes!=1u && !(planes==3u && previous->ndim==2 && previous->dtype==T_I32)) ||
               active<1 || active>=capacity) {valid=0;break;}
            for(int d=axis+1;d<previous->ndim;d++) width*=(size_t)previous->shape[d];
            const unsigned char* before=(const unsigned char*)previous->data,*after=inputs[i].data;
            if(mask && previous->dtype==T_I32 && (previous->ndim==2 || previous->ndim==3) && width==(size_t)capacity) {
                for(int q=0;q<active && valid;q++) if(memcmp(before+(size_t)q*width*4u,after+(size_t)q*width*4u,(size_t)active*4u)) valid=0;
            } else for(size_t plane=0;plane<planes;plane++)
                if(memcmp(before+plane*(size_t)capacity*width*4u,after+plane*(size_t)capacity*width*4u,(size_t)active*width*4u)) valid=0;
            continue;
        }
        int capacity=previous->shape[previous->ndim-1];
        if(active<1 || active>=capacity) {valid=0;break;}
        const int32_t* before=(const int32_t*)previous->data,*after=(const int32_t*)inputs[i].data;
        if(kind==5) {
            for(int q=0;q<active && valid;q++) if(memcmp(before+(size_t)q*capacity,after+(size_t)q*capacity,(size_t)active*4u)) valid=0;
        } else {
            int axes=kind==6?3:1;
            for(int axis=0;axis<axes;axis++) if(memcmp(before+(size_t)axis*capacity,after+(size_t)axis*capacity,(size_t)active*4u)) valid=0;
        }
    }
    vx_engine_state_scope_leave(scope);
    return valid;
}

static VxStatus vx_incremental_controls_validate(VxExecutionContext* context,
        const VxTensorBinding* inputs,size_t count,int active,int end) {
    const cJSON* generation=cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model),"generation");
    const cJSON* controls=cJSON_GetObjectItemCaseSensitive(generation,"incremental_controls");
    VxEngineStateScope scope=vx_engine_state_scope_enter(context->engine_state);
    int logical_positions=0;
    for(int node=0;node<g_nn;node++) if(g_n[node].operator_kind==VX_OP_KV_COMPRESS) logical_positions=1;
    if(!controls && !logical_positions) {vx_engine_state_scope_leave(scope);return VX_STATUS_OK;}
    if(active<1 || end<active) {vx_engine_state_scope_leave(scope);return VX_STATUS_INVALID_ARGUMENT;}
    const cJSON* incremental=cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model),"incremental");
    const char* logical_position=vx_execution_string(incremental,"position_input");
    if(logical_positions && !logical_position) {vx_engine_state_scope_leave(scope);return VX_STATUS_INVALID_ARGUMENT;}
    const cJSON* chunk=cJSON_GetObjectItemCaseSensitive(controls,"chunk_size");
    int chunk_size=chunk?chunk->valueint:0;
    VxStatus status=VX_STATUS_OK;
    for(size_t i=0;i<context->compiled->model->input_count && status==VX_STATUS_OK;i++) {
        const char* name=context->compiled->model->inputs[i].name;
        int kind=vx_incremental_control_kind(generation,name);
        if(logical_positions && !strcmp(name,logical_position)) kind=3;
        if(kind<=0) continue;
        const VxTensorBinding* binding=NULL;
        for(size_t j=0;j<count;j++) if(!strcmp(inputs[j].name,name)) binding=&inputs[j];
        T* tensor=t_find(name);
        const int32_t* values=binding?binding->data:tensor?(const int32_t*)tensor->data:NULL;
        size_t bytes=binding?binding->byte_size:tensor?(size_t)tensor->numel*4u:0;
        int capacity=binding?(int)binding->shape[binding->rank-1]:tensor?tensor->shape[tensor->ndim-1]:0;
        if(!values || capacity<end || (binding && (binding->location!=VX_MEMORY_HOST || binding->dtype!=VX_DTYPE_I32))) {status=VX_STATUS_INVALID_ARGUMENT;break;}
        uint64_t required=(uint64_t)capacity*(kind==5?(uint32_t)capacity:kind==6?3u:1u)*4u;
        if(required!=bytes) {status=VX_STATUS_INVALID_ARGUMENT;break;}
        if(kind==3 || kind==4 || kind==6) {
            int axes=kind==6?3:1;
            for(int axis=0;axis<axes;axis++) for(int row=0;row<active;row++) {
                int32_t value=values[(size_t)axis*capacity+row];
                if(value<0 || (uint64_t)value+(uint32_t)(end-active)>INT32_MAX ||
                   (kind==4 && value!=row) || (kind==3 && logical_positions && value!=row)) status=VX_STATUS_INVALID_ARGUMENT;
            }
        } else if(kind==5) {
            if(chunk_size<=0) {status=VX_STATUS_INVALID_ARGUMENT;break;}
            for(int q=0;q<active;q++) for(int k=0;k<active;k++)
                if(values[(size_t)q*capacity+k]!=(q/chunk_size==k/chunk_size)) status=VX_STATUS_INVALID_ARGUMENT;
        }
    }
    vx_engine_state_scope_leave(scope);
    return status;
}

static int vx_incremental_controls_append(VxExecutionContext* context,int row) {
    const cJSON* generation=cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model),"generation");
    const cJSON* controls=cJSON_GetObjectItemCaseSensitive(generation,"incremental_controls");
    if(!controls) return 0;
    const cJSON* chunk=cJSON_GetObjectItemCaseSensitive(controls,"chunk_size");
    int chunk_size=chunk?chunk->valueint:0;
    for(size_t i=0;i<context->compiled->model->input_count;i++) {
        const char* name=context->compiled->model->inputs[i].name;
        int kind=vx_incremental_control_kind(generation,name);
        if(kind<=0) continue;
        T* tensor=t_find(name);
        if(!tensor || tensor->dtype!=T_I32 || !tensor->data || tensor->ndim<1) return -1;
        int capacity=tensor->shape[tensor->ndim-1];
        if(row<1 || row>=capacity) return -1;
        int32_t* values=(int32_t*)tensor->data;
        if(kind==5) {
            if(chunk_size<=0) return -1;
            for(int k=0;k<=row;k++) {
                values[(size_t)row*capacity+k]=row/chunk_size==k/chunk_size;
                values[(size_t)k*capacity+row]=k/chunk_size==row/chunk_size;
            }
        } else if(kind==6) {
            int32_t maximum=0;
            for(int axis=0;axis<3;axis++) for(int k=0;k<row;k++)
                if(values[(size_t)axis*capacity+k]>maximum) maximum=values[(size_t)axis*capacity+k];
            if(maximum==INT32_MAX) return -1;
            for(int axis=0;axis<3;axis++) values[(size_t)axis*capacity+row]=maximum+1;
        } else {
            if(kind==3 && values[row-1]==INT32_MAX) return -1;
            values[row]=kind==3?values[row-1]+1:kind==4?row:0;
        }
        vx_incremental_mark_tensor_locked(tensor);
        vx_runtime_backend_mark_host(tensor->data,(size_t)tensor->numel*4u,0);
    }
    return 0;
}

static T* vx_feedback_port(const Node* node, VxPortKind name) {
    for (int i = 0; i < node->nin; i++)
        if (node->ins[i].port == name) return t_find(node->ins[i].name);
    return NULL;
}

/* Validate the immutable graph proof without requiring bound input data or
 * a live prompt. Context admission and every feedback run share this check. */
static VxStatus vx_context_feedback_topology(VxExecutionContext* context,
    T* tokens, T* keep, T* output) {
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    int full_prefix = 0;
    if (!tokens || !keep || !output || tokens == keep || tokens == output || keep == output ||
        !tokens->is_graph_input || !keep->is_graph_input) goto done;
    int declared_output = 0;
    for (size_t i = 0; i < context->compiled->model->output_count; i++)
        if (!strcmp(context->compiled->model->outputs[i].name, output->name)) declared_output = 1;
    if (!declared_output) goto done;
    const Node* producer = NULL;
    for (int i = 0; i < g_nn; i++) {
        const Node* node = &g_n[i];
        if (!strcmp(node->out, output->name)) {
            if (producer) goto done;
            producer = node;
        }
        for (int port = 0; port < node->nin; port++)
            if (!strcmp(node->ins[port].name, output->name)) goto done;
    }
    if (!producer || ((producer->operator_kind != VX_OP_Q_ARG_MAX) && (producer->operator_kind != VX_OP_ARG_MAX))) goto done;
    T* scores = vx_feedback_port(producer, VX_PORT_INPUT);
    if (!scores || scores->ndim < 1) goto done;
    int axis = vx_node_param_i32(producer, VX_NODE_PARAM_AXIS, -1);
    if (axis < 0) axis += scores->ndim;
    if (axis != scores->ndim - 1) goto done;
    int vocabulary = scores->shape[axis], embeddings = 0, causal_mask = 0;
    if (context->execution_contract_enabled && context->execution_autoregressive) {
        const cJSON* contract = cJSON_GetObjectItemCaseSensitive(
            vx_execution_document(context->compiled->model), "generation");
        full_prefix = cJSON_IsObject(cJSON_GetObjectItemCaseSensitive(contract, "full_prefix"));
        const cJSON* initialize = vx_execution_entry(context->compiled->model,
            vx_execution_string(contract, "initialize_entrypoint"));
        const cJSON* transition = vx_execution_entry(context->compiled->model,
            vx_execution_string(contract, "transition_entrypoint"));
        const cJSON* initial_inputs = cJSON_GetObjectItemCaseSensitive(initialize, "inputs");
        const cJSON* transition_inputs = cJSON_GetObjectItemCaseSensitive(transition, "inputs");
        const cJSON* document = vx_execution_document(context->compiled->model);
        const cJSON* invariants = cJSON_GetObjectItemCaseSensitive(document, "invariant_inputs");
        const cJSON* updates = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetObjectItemCaseSensitive(document, "incremental"), "update_inputs");
        for (size_t i = 0; i < context->compiled->model->input_count; i++)
            if (!vx_execution_names_contains(initial_inputs, context->compiled->model->inputs[i].name)) goto done;
        const char* roots[] = {tokens->name, keep->name};
        for (size_t i = 0; i < 2; i++)
            if (vx_execution_names_contains(invariants, roots[i]) ||
                (!full_prefix && !vx_execution_names_contains(updates, roots[i])) ||
                !vx_execution_names_contains(transition_inputs, roots[i])) goto done;
        int initialize_full = 0, transition_rows = 0;
        const cJSON* ranges = cJSON_GetObjectItemCaseSensitive(initialize, "allowed_ranges");
        for (const cJSON* item = ranges ? ranges->child : NULL; item; item = item->next)
            initialize_full |= vx_execution_range(item) == VX_EXECUTION_RANGE_FULL;
        ranges = cJSON_GetObjectItemCaseSensitive(transition, "allowed_ranges");
        for (const cJSON* item = ranges ? ranges->child : NULL; item; item = item->next)
            transition_rows |= vx_execution_range(item) == (full_prefix ? VX_EXECUTION_RANGE_FULL : VX_EXECUTION_RANGE_SLOT_ROWS);
        if (!initialize_full || !transition_rows) goto done;
        const cJSON* domain = cJSON_GetObjectItemCaseSensitive(contract, "id_domain_size");
        const cJSON* reduction_axis = cJSON_GetObjectItemCaseSensitive(contract, "logits_axis");
        const char* logits = vx_execution_string(contract, "logits_output");
        if (!logits || strcmp(logits, scores->name) || !domain || domain->valueint != vocabulary ||
            !reduction_axis || reduction_axis->valueint != axis) goto done;
    }
    for (int i = 0; i < g_nn; i++) {
        const Node* node = &g_n[i];
        if (((node->operator_kind == VX_OP_Q_EMBEDDING) || (node->operator_kind == VX_OP_EMBEDDING) ||
             node->operator_kind == VX_OP_EMBEDDING_SHARD) &&
            vx_feedback_port(node, VX_PORT_INPUT) == tokens) {
            T* weight = vx_feedback_port(node, VX_PORT_WEIGHT);
            int shard = node->operator_kind == VX_OP_EMBEDDING_SHARD;
            if (!weight || weight->ndim != 2 || (shard
                ? vx_node_param_i32(node, VX_NODE_PARAM_VOCAB_SIZE, 0) != vocabulary
                : weight->shape[0] != vocabulary)) goto done;
            if (!shard && context->execution_contract_enabled && context->execution_autoregressive) {
                const VxDeclaredTensor* declared = vx_execution_tensor(context->logical_tensors,
                    context->logical_tensor_count, weight->name);
                if (declared && (declared->rank != 2 || declared->minimums[0] != vocabulary ||
                    declared->maximums[0] != vocabulary)) goto done;
            }
            embeddings++;
        }
        if (((node->operator_kind == VX_OP_Q_SDPA) || (node->operator_kind == VX_OP_CROSS_SDPA)) &&
            vx_feedback_port(node, VX_PORT_MASK) == keep && vx_node_param_bool(node, VX_NODE_PARAM_CAUSAL, 0)) causal_mask = 1;
        if (node->operator_kind == VX_OP_SDPA && vx_node_param_bool(node, VX_NODE_PARAM_CAUSAL, 0)) causal_mask = 1;
        if(node->operator_kind==VX_OP_SPARSE_ATTENTION && vx_node_param_bool(node,VX_NODE_PARAM_CAUSAL,1)) causal_mask=1;
        if(node->operator_kind==VX_OP_GATED_DELTA_NET && vx_execution_names_contains(
            cJSON_GetObjectItemCaseSensitive(cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model),"incremental"),"update_inputs"),keep->name)) causal_mask=1;
    }
    if (!embeddings || (!full_prefix && !causal_mask)) goto done;
    status = VX_STATUS_OK;
done:
    return status;
}

static VxStatus vx_context_feedback_describe(VxExecutionContext* context,
    const VxDecodeFeedbackOptions* options, int position, VxDecodeFeedback* out) {
    if (!options || !options->token_input || !options->keep_input || !options->token_output ||
        !options->token_input[0] || !options->keep_input[0] || !options->token_output[0] ||
        (!options->token_count && !options->validate_only) || position < 1 || context->decode_slots != 1 ||
        (options->has_cache_generation && options->cache_generation != context->decode_cache_generation))
        return VX_STATUS_INVALID_ARGUMENT;
    if (context->compiled->provider || context->decode_row_mode != VX_DECODE_ROW_REQUIRED ||
        (context->compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND &&
         context->compiled->builtin_backend != VX_BACKEND_KIND_WEBGPU &&
         (!vx_native_gpu_backend(context->compiled->builtin_backend) ||
          !context->engine_state->decode_row_stage_reserved))) return VX_STATUS_BACKEND_UNSUPPORTED;
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    T* tokens = t_find(options->token_input), *keep = t_find(options->keep_input), *output = t_find(options->token_output);
    if (!tokens || !keep || !output || tokens == keep || tokens == output || keep == output ||
        !tokens->is_graph_input || !keep->is_graph_input) goto done;
    T* tensors[3] = {tokens, keep, output};
    for (int i = 0; i < 3; i++) if (tensors[i]->dtype != T_I32 || tensors[i]->elem_size != 4 ||
        tensors[i]->ndim != 2 || tensors[i]->shape[0] != 1 || tensors[i]->shape[1] != tokens->shape[1] ||
        !tensors[i]->data) goto done;
    if ((uint64_t)position + options->token_count > (uint64_t)tokens->shape[1]) goto done;
    status = vx_context_feedback_topology(context, tokens, keep, output);
    if (status != VX_STATUS_OK) goto done;
    status=vx_incremental_controls_validate(context,NULL,0,position,position+(int)options->token_count);
    if(status!=VX_STATUS_OK) goto done;
    *out = (VxDecodeFeedback){.tokens = tokens, .keep = keep, .output = output,
        .first = position, .end = position + (int)options->token_count,
        .roots = {{.name = tokens->name}, {.name = keep->name}}};
    status = VX_STATUS_OK;
done:
    vx_engine_state_scope_leave(scope);
    return status;
}

#if VOLVOXAI_ENABLE_CUDA || VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
/* ArgMax and the embedding share a proved vocabulary domain. Keep the selected
 * ID device-authoritative while updating only host-authored keep metadata. */
static int vx_context_device_feedback(const VxDecodeFeedback* feedback, int position) {
    VxEngineState* state = vx_engine_state_current();
    if (!state->decode_row_stage_reserved || position < 1 ||
        position >= feedback->tokens->shape[1] ||
        !state->decode_numeric_stage[0] ||
        state->decode_numeric_stage_bytes[0] < sizeof(int32_t) ||
        !state->decode_metadata_stage ||
        state->decode_metadata_stage_bytes < sizeof(int32_t)) return -1;
    size_t bytes = (size_t)feedback->tokens->shape[1] * sizeof(int32_t);
    int32_t* index = (int32_t*)state->decode_metadata_stage;
    void* selected = state->decode_numeric_stage[0];
    *index = position - 1;
    vx_runtime_backend_mark_host(index, state->decode_metadata_stage_bytes, 0);
    if (!vx_runtime_backend_row_index_transfer(feedback->output->data, bytes, index, 1u,
            selected, sizeof(int32_t), 1u,
            (uint32_t)feedback->tokens->shape[1], 0u)) return -1;
    *index = position;
    vx_runtime_backend_mark_host(index, state->decode_metadata_stage_bytes, 0);
    if (!vx_runtime_backend_row_index_transfer(selected, sizeof(int32_t), index, 1u,
            feedback->tokens->data, bytes, 1u,
            (uint32_t)feedback->tokens->shape[1], 1u)) return -1;
    ((int32_t*)feedback->keep->data)[position] = 1;
    vx_runtime_backend_mark_host(feedback->keep->data, bytes, 0);
    return 0;
}
#endif

static int vx_context_feedback_run(VxExecutionContext* context,
    const VxDecodeFeedback* feedback, int32_t* positions) {
    positions[0] = feedback->first - 1;
    for (int position = feedback->first; position < feedback->end; position++) {
        if (context->generation_stop_reason) break;
        if (context->execution_contract_enabled &&
            atomic_load_explicit(&context->execution_cancel_requested, memory_order_acquire)) break;
        if (context->execution_contract_enabled && context->execution_deadline_ns &&
            vx_trace_now_ns() >= context->execution_deadline_ns) break;
#if VOLVOXAI_ENABLE_WEBGPU
        if (g_use_webgpu) {
            if (vx_webgpu_decode_feedback(feedback->tokens->name, feedback->keep->name,
                    feedback->output->name, position)) return -1;
        } else
#endif
#if VOLVOXAI_ENABLE_CUDA || VOLVOXAI_ENABLE_VULKAN || VOLVOXAI_ENABLE_OPENGL || VOLVOXAI_ENABLE_METAL
        if (context->engine_state->decode_row_stage_reserved && vx_native_gpu_backend(context->compiled->builtin_backend)) {
            if (vx_context_device_feedback(feedback, position)) return -1;
        } else
#endif
        {
            ((int32_t*)feedback->tokens->data)[position] = ((int32_t*)feedback->output->data)[position - 1];
            ((int32_t*)feedback->keep->data)[position] = 1;
        }
        vx_incremental_mark_tensor_locked(feedback->tokens);
        vx_incremental_mark_tensor_locked(feedback->keep);
        if(vx_incremental_controls_append(context,position)) return -1;
        positions[0] = position;
        if (volvoxai_engine_decode_session_step_rows(context->decode_session, positions, 1)) return -1;
        if (vx_generation_select_row(context, position)) return -1;
    }
    return 0;
}
