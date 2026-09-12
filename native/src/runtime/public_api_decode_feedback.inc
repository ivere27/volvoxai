/* Generated DecodeGenerate dispatch enters the ordinary context transaction.
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

static T* vx_feedback_port(const Node* node, VxPortKind name) {
    for (int i = 0; i < node->nin; i++)
        if (node->ins[i].port == name) return t_find(node->ins[i].name);
    return NULL;
}

static VxStatus vx_context_feedback_describe(VxExecutionContext* context,
    const VxDecodeFeedbackOptions* options, int position, VxDecodeFeedback* out) {
    if (!options || !options->token_input || !options->keep_input || !options->token_output ||
        !options->token_input[0] || !options->keep_input[0] || !options->token_output[0] ||
        !options->token_count || position < 1 || context->decode_lanes != 1 ||
        (options->has_cache_generation && options->cache_generation != context->decode_cache_generation))
        return VX_STATUS_INVALID_ARGUMENT;
    if (context->compiled->provider || context->decode_row_mode != VX_DECODE_ROW_REQUIRED ||
        (context->compiled->builtin_backend != VX_PORTABLE_BACKEND_KIND &&
         context->compiled->builtin_backend != VX_BACKEND_KIND_WEBGPU)) return VX_STATUS_BACKEND_UNSUPPORTED;
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
    for (int i = 0; i < g_nn; i++) {
        const Node* node = &g_n[i];
        if (((node->operator_kind == VX_OP_Q_EMBEDDING) || (node->operator_kind == VX_OP_EMBEDDING)) &&
            vx_feedback_port(node, VX_PORT_INPUT) == tokens) {
            T* weight = vx_feedback_port(node, VX_PORT_WEIGHT);
            if (!weight || weight->ndim != 2 || weight->shape[0] != vocabulary) goto done;
            embeddings++;
        }
        if (((node->operator_kind == VX_OP_Q_SDPA) || (node->operator_kind == VX_OP_CROSS_SDPA)) &&
            vx_feedback_port(node, VX_PORT_MASK) == keep && vx_node_param_bool(node, VX_NODE_PARAM_CAUSAL, 0)) causal_mask = 1;
    }
    if (!embeddings || !causal_mask) goto done;
    *out = (VxDecodeFeedback){.tokens = tokens, .keep = keep, .output = output,
        .first = position, .end = position + (int)options->token_count,
        .roots = {{.name = tokens->name}, {.name = keep->name}}};
    status = VX_STATUS_OK;
done:
    vx_engine_state_scope_leave(scope);
    return status;
}

static int vx_context_feedback_run(VxExecutionContext* context,
    const VxDecodeFeedback* feedback, int32_t* positions) {
    for (int position = feedback->first; position < feedback->end; position++) {
#if VOLVOXAI_ENABLE_WEBGPU
        if (g_use_webgpu) {
            if (vx_webgpu_decode_feedback(feedback->tokens->name, feedback->keep->name,
                    feedback->output->name, position)) return -1;
        } else
#endif
        {
            ((int32_t*)feedback->tokens->data)[position] = ((int32_t*)feedback->output->data)[position - 1];
            ((int32_t*)feedback->keep->data)[position] = 1;
        }
        vx_incremental_mark_tensor_locked(feedback->tokens);
        vx_incremental_mark_tensor_locked(feedback->keep);
        positions[0] = position;
        if (volvoxai_engine_decode_session_step_rows(context->decode_session, positions, 1)) return -1;
    }
    return 0;
}
