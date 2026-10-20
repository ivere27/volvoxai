#if VOLVOXAI_ENABLE_CUDA
/* CUDA keeps complete activation allocations resident while transformer
 * prefix/row calls pass interior host pointers as device-buffer views.  Limit
 * this windowing to the canonical [1, sequence, width] layout; every other
 * shape retains the complete-tensor kernel and its existing semantics. */
static int cuda_partial_sequence_rows(const T* tensor, int width,
                                      int* row_start, int* row_count) {
    long rows;
    if (g_active_row < 0 && g_prefix_rows <= 0) return 0;
    if (!tensor || !row_start || !row_count || tensor->ndim != 3 ||
        tensor->shape[0] != 1 || tensor->shape[1] <= 0 || width <= 0 ||
        tensor->shape[2] != width || tensor->numel % width) return -1;
    rows = tensor->numel / width;
    if (rows != tensor->shape[1] || rows > INT_MAX) return -1;
    seq_range(rows, row_start, row_count);
    return *row_start >= 0 && *row_count > 0 &&
           (long)*row_start + *row_count <= rows ? 1 : -1;
}

/* Ordered [1,S,H,D] normalization keeps all H groups of one token. */
static int cuda_partial_rmsnorm_rows(const T* tensor, int width,
                                     int* row_start, int* row_count) {
    if (g_active_row < 0 && g_prefix_rows <= 0) return 0;
    if (tensor && tensor->ndim == 4 && tensor->shape[0] == 1 &&
        tensor->shape[1] > 0 && tensor->shape[2] > 0 && width > 0 &&
        tensor->shape[3] == width) {
        int token_start, token_count;
        long groups = (long)tensor->shape[1] * tensor->shape[2];
        if (groups > INT_MAX || tensor->numel != groups * width) return -1;
        seq_range(tensor->shape[1], &token_start, &token_count);
        if (token_start < 0 || token_count <= 0 ||
            (long)token_start + token_count > tensor->shape[1]) return -1;
        *row_start = token_start * tensor->shape[2];
        *row_count = token_count * tensor->shape[2];
        return 1;
    }
    return cuda_partial_sequence_rows(tensor, width, row_start, row_count);
}

static int cuda_silu_mul_spans_disjoint(const T* a, const T* b) {
    uintptr_t x = (uintptr_t)a->data, y = (uintptr_t)b->data;
    size_t bytes = (size_t)a->numel * sizeof(float);
    return x <= y ? bytes <= y - x : bytes <= x - y;
}

/* Private incremental lowering of two adjacent semantic nodes. The caller
 * gates ordinary passes and keeps both nodes' selection/dirty bookkeeping.
 * No graph, source ID, skip flag or prepared reference is rewritten. Return
 * zero before any backend call when this tactic is ineligible; a backend
 * failure is terminal because commands may already have been submitted. */
int vx_runtime_cuda_silu_mul_pair_locked(int node_index, int row) {
    VxEngineState* state = vx_engine_state_current();
    if (!state || !g_use_cuda || !state->shape_policy.retain_activations ||
        state->preserve_node_boundaries || row < -1 || g_decode_slots > 1 ||
        node_index < 0 || node_index >= g_nn - 1) return 0;
    if (row >= 0 && g_decode_slots == 1 &&
        (g_decode_rows.slots != 1 || !g_decode_rows.positions ||
         !g_decode_rows.empty || g_decode_rows.empty[0] ||
         g_decode_rows.positions[0] != row)) return 0;
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return 0;
#endif
    Node* silu = &g_n[node_index];
    Node* mul = &g_n[node_index + 1];
    if (silu->operator_kind != VX_OP_SILU || mul->operator_kind != VX_OP_MUL ||
        silu->skip || silu->disabled || mul->skip || mul->disabled ||
        silu->fuse_relu6 || mul->fuse_relu6 || silu->nin != 1 || mul->nin != 2 ||
        silu->nout != 1 || mul->nout != 1 ||
        g_node_fusion[node_index].id || g_node_fusion[node_index + 1].id ||
        !vx_incremental_node_bindings_current(silu, state) ||
        !vx_incremental_node_bindings_current(mul, state) ||
        strcmp(silu->out, silu->outs[0].name) ||
        strcmp(mul->out, mul->outs[0].name)) return 0;
    T* input = vx_incremental_node_input_locked(silu, 0);
    T* activation = vx_incremental_node_output_locked(silu, -1);
    T* output = vx_incremental_node_output_locked(mul, -1);
    T* first = vx_incremental_node_input_locked(mul, 0);
    T* second = vx_incremental_node_input_locked(mul, 1);
    if (!activation || ((first == activation) + (second == activation)) != 1) return 0;
    T* other = first == activation ? second : first;
    T* tensors[4] = {input, other, activation, output};
    for (int i = 0; i < 4; ++i) {
        T* tensor = tensors[i];
        if (!tensor || !tensor->data || tensor->dtype != T_F32 ||
            tensor->elem_size != sizeof(float) || tensor->numel <= 0 ||
            (uint64_t)tensor->numel > UINT32_MAX ||
            (uint64_t)tensor->numel > SIZE_MAX / sizeof(float) ||
            !tensor_same_shape(tensor, activation) || vx_paged_tensor_locked(tensor)) return 0;
    }
    for (int i = 0; i < 2; ++i) {
        const VxIncrementalTensorBinding* prepared =
            vx_incremental_tensor_binding_locked(tensors[i], state);
        if (!prepared || prepared->producer >= node_index) return 0;
    }
    if (!cuda_silu_mul_spans_disjoint(activation, output) ||
        !cuda_silu_mul_spans_disjoint(activation, input) ||
        !cuda_silu_mul_spans_disjoint(activation, other) ||
        !cuda_silu_mul_spans_disjoint(output, input) ||
        !cuda_silu_mul_spans_disjoint(output, other)) return 0;
    long offset = 0, elements = activation->numel;
    if (row >= 0) {
        if (activation->ndim != 3 || activation->shape[0] != 1 ||
            activation->shape[1] <= row || activation->shape[2] <= 0 ||
            activation->numel != (long)activation->shape[1] * activation->shape[2]) return 0;
        elements = activation->shape[2];
        offset = (long)row * elements;
    } else {
        int start = 0, count = 0;
        int partial = cuda_partial_sequence_rows(activation,
            activation->ndim > 0 ? activation->shape[activation->ndim - 1] : 0,
            &start, &count);
        if (partial < 0) return 0;
        if (partial) {
            offset = (long)start * activation->shape[activation->ndim - 1];
            elements = (long)count * activation->shape[activation->ndim - 1];
        }
    }
    /* A first prefix has no retained row output to preserve. Ordinary SiLU
     * writes its active view and Mul uses its regular output route; the dual
     * route instead makes both complete owners coherent, uploading capacity
     * padding before its partial write. Keep that preserving tactic for row
     * updates and complete tensors, where the untouched outputs are live. */
    if (row < 0 && elements != activation->numel) return 0;
    if (!cuda_graph_silu_mul_dual_f32(input->data, other->data,
            activation->data, output->data, activation->numel, offset, elements)) return -1;
    if (state->runtime_route_backend && (size_t)(node_index + 1) < g_node_capacity) {
        const char* route = row >= 0 ? "cuda-graph-row+silu-mul" : "cuda-graph+silu-mul";
        state->runtime_route_backend[node_index] = route;
        state->runtime_route_backend[node_index + 1] = route;
    }
    return 1;
}

/* The checked byte-span predicate is shared with physical tensor dispatch. */
static int physical_tensor_ranges_overlap(const T* left, const T* right);

/* A row-only private lowering keeps Linear and Add, including both outputs,
 * intact. Zero is command-free fallback; submitted failures are terminal. */
int vx_runtime_cuda_gemv_add_pair_locked(int node_index, int row) {
    VxEngineState* state = vx_engine_state_current();
    if (!state || !g_use_cuda || !state->shape_policy.retain_activations ||
        state->preserve_node_boundaries || g_debug || row < 0 || row == INT32_MAX ||
        g_decode_slots > 1 || node_index < 0 || node_index >= g_nn - 1) return 0;
    if (g_decode_slots == 1 &&
        (g_decode_rows.slots != 1 || !g_decode_rows.positions ||
         !g_decode_rows.empty || g_decode_rows.empty[0] ||
         g_decode_rows.positions[0] != row)) return 0;
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return 0;
#endif
    Node* linear = &g_n[node_index];
    Node* add = &g_n[node_index + 1];
    if (linear->operator_kind != VX_OP_LINEAR || add->operator_kind != VX_OP_ADD ||
        linear->skip || linear->disabled || add->skip || add->disabled ||
        linear->fuse_relu6 || add->fuse_relu6 || linear->nin != 2 || add->nin != 2 ||
        linear->nout != 1 || add->nout != 1 ||
        g_node_fusion[node_index].id || g_node_fusion[node_index + 1].id ||
        linear->parsed_params.parsed != 1 || add->parsed_params.parsed != 1 ||
        vx_node_param_i32(add, VX_NODE_PARAM_RELU, 0) != 0 ||
        !vx_incremental_node_bindings_current(linear, state) ||
        !vx_incremental_node_bindings_current(add, state) ||
        strcmp(linear->out, linear->outs[0].name) ||
        strcmp(add->out, add->outs[0].name)) return 0;
    for (int node = node_index; node <= node_index + 1; node++)
        for (int input = 0; input < g_n[node].nin; input++)
            if (vx_adapter_run_has_target(g_n[node].ins[input].name)) return 0;
    T* input = nin(linear, VX_PORT_INPUT);
    T* weight = nin(linear, VX_PORT_WEIGHT);
    T* dense = vx_incremental_node_output_locked(linear, -1);
    T* output = vx_incremental_node_output_locked(add, -1);
    T* first = vx_incremental_node_input_locked(add, 0);
    T* second = vx_incremental_node_input_locked(add, 1);
    if (!dense || ((first == dense) + (second == dense)) != 1 ||
        nin(linear, VX_PORT_BIAS)) return 0;
    if (add->ins[0].port != VX_PORT_A || add->ins[1].port != VX_PORT_B) return 0;
    T* residual = first == dense ? second : first;
    T* tensors[5] = {input, weight, residual, dense, output};
    for (int i = 0; i < 5; i++) {
        const T* tensor = tensors[i];
        if (!tensor || !tensor->data || tensor->dtype != T_F32 ||
            tensor->elem_size != sizeof(float) || tensor->numel <= 0 ||
            (uint64_t)tensor->numel > UINT32_MAX ||
            (uint64_t)tensor->numel > SIZE_MAX / sizeof(float) ||
            vx_paged_tensor_locked(tensor)) return 0;
    }
    if (!vx_incremental_tensor_is_model_weight_locked(weight) ||
        input->ndim != 3 || dense->ndim != 3 || input->shape[0] != 1 ||
        dense->shape[0] != 1 || input->shape[1] <= row ||
        dense->shape[1] != input->shape[1] || input->shape[2] <= 0 ||
        dense->shape[2] <= 0 || (uint64_t)input->numel != (uint64_t)input->shape[1] * input->shape[2] ||
        (uint64_t)dense->numel != (uint64_t)dense->shape[1] * dense->shape[2] ||
        !tensor_same_shape(residual, dense) || !tensor_same_shape(output, dense) ||
        (uint64_t)weight->numel != (uint64_t)input->shape[2] * dense->shape[2]) return 0;
    int out_in = 0;
    if (linear_node_weight_layout(linear, input, dense, weight, &out_in) != 0 || !out_in)
        return 0;
    for (int i = 0; i <= 2; i += 2) {
        const VxIncrementalTensorBinding* prepared =
            vx_incremental_tensor_binding_locked(tensors[i], state);
        if (!prepared || prepared->producer >= node_index) return 0;
    }
    for (int destination = 3; destination < 5; destination++)
        for (int source = 0; source < destination; source++)
            if (physical_tensor_ranges_overlap(tensors[destination], tensors[source])) return 0;
    int result = cuda_graph_gemv_add_dual_f32(input->data, weight->data,
        residual->data, dense->data, output->data, input->shape[1], row,
        input->shape[2], dense->shape[2], first == residual);
    if (result > 0 && state->runtime_route_backend && (size_t)(node_index + 1) < g_node_capacity) {
        state->runtime_route_backend[node_index] = "cuda-graph-row+gemv-add";
        state->runtime_route_backend[node_index + 1] = "cuda-graph-row+gemv-add";
    }
    return result;
}

/* Same-input adjacent Linear nodes retain their two logical outputs. */
int vx_runtime_cuda_gemv_pair_locked(int node_index, int row) {
    VxEngineState* state = vx_engine_state_current();
    if (!state || !g_use_cuda || !state->shape_policy.retain_activations ||
        state->preserve_node_boundaries || g_debug || vx_trace_nodes(state->profiling) ||
        row < 0 || row == INT32_MAX || g_decode_slots > 1 ||
        node_index < 0 || node_index >= g_nn - 1) return 0;
    if (g_decode_slots == 1 &&
        (g_decode_rows.slots != 1 || !g_decode_rows.positions ||
         !g_decode_rows.empty || g_decode_rows.empty[0] ||
         g_decode_rows.positions[0] != row)) return 0;
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return 0;
#endif
    Node* first = &g_n[node_index];
    Node* second = &g_n[node_index + 1];
    Node* nodes[2] = {first, second};
    for (int i = 0; i < 2; ++i) {
        const Node* node = nodes[i];
        if (node->operator_kind != VX_OP_LINEAR || node->skip || node->disabled ||
            node->fuse_relu6 || node->nin != 2 || node->nout != 1 ||
            node->ins[0].port != VX_PORT_INPUT || node->ins[1].port != VX_PORT_WEIGHT ||
            node->parsed_params.parsed != 1 || g_node_fusion[node_index + i].id ||
            !vx_incremental_node_bindings_current(node, state) ||
            strcmp(node->out, node->outs[0].name)) return 0;
        for (int input = 0; input < node->nin; ++input)
            if (vx_adapter_run_has_target(node->ins[input].name)) return 0;
    }
    T* input = nin(first, VX_PORT_INPUT);
    T* weight0 = nin(first, VX_PORT_WEIGHT);
    T* weight1 = nin(second, VX_PORT_WEIGHT);
    T* output0 = vx_incremental_node_output_locked(first, -1);
    T* output1 = vx_incremental_node_output_locked(second, -1);
    if (input != nin(second, VX_PORT_INPUT) || nin(first, VX_PORT_BIAS) ||
        nin(second, VX_PORT_BIAS)) return 0;
    T* tensors[5] = {input, weight0, weight1, output0, output1};
    for (int i = 0; i < 5; ++i) {
        const T* tensor = tensors[i];
        if (!tensor || !tensor->data || tensor->dtype != T_F32 ||
            tensor->elem_size != sizeof(float) || tensor->numel <= 0 ||
            (uint64_t)tensor->numel > UINT32_MAX ||
            (uint64_t)tensor->numel > SIZE_MAX / sizeof(float) ||
            vx_paged_tensor_locked(tensor)) return 0;
    }
    if (!vx_incremental_tensor_is_model_weight_locked(weight0) ||
        !vx_incremental_tensor_is_model_weight_locked(weight1) ||
        input->ndim != 3 || output0->ndim != 3 || input->shape[0] != 1 ||
        output0->shape[0] != 1 || input->shape[1] <= row ||
        output0->shape[1] != input->shape[1] || input->shape[2] <= 0 || output0->shape[2] <= 0 ||
        (uint64_t)input->numel != (uint64_t)input->shape[1] * input->shape[2] ||
        (uint64_t)output0->numel != (uint64_t)output0->shape[1] * output0->shape[2] ||
        !tensor_same_shape(output0, output1) ||
        (uint64_t)weight0->numel != (uint64_t)input->shape[2] * output0->shape[2] ||
        (uint64_t)weight1->numel != (uint64_t)weight0->numel) return 0;
    for (int i = 0; i < 2; ++i) {
        int out_in = 0;
        if (linear_node_weight_layout(nodes[i], input, tensors[i + 3], tensors[i + 1], &out_in) != 0 || !out_in)
            return 0;
    }
    const VxIncrementalTensorBinding* binding = vx_incremental_tensor_binding_locked(input, state);
    if (!binding || binding->producer >= node_index) return 0;
    for (int destination = 3; destination < 5; ++destination)
        for (int source = 0; source < destination; ++source)
            if (physical_tensor_ranges_overlap(tensors[destination], tensors[source])) return 0;
    int result = cuda_graph_gemv_pair_out_in_f32(input->data, weight0->data,
        weight1->data, output0->data, output1->data, input->shape[1], row,
        input->shape[2], output0->shape[2]);
    if (result > 0 && state->runtime_route_backend && (size_t)(node_index + 1) < g_node_capacity) {
        state->runtime_route_backend[node_index] = "cuda-graph-row+gemv-pair";
        state->runtime_route_backend[node_index + 1] = "cuda-graph-row+gemv-pair";
    }
    return result;
}

static int try_cuda_graph_head_rope(Node* node, T* input, T* output) {
    VxSequenceRoPE parameters;
    if (!input) input = nin(node, VX_PORT_X);
    if (!vx_sequence_rope_parameters(node, input, output, &parameters) ||
        parameters.width / parameters.heads > 128 || parameters.theta < 1.0f)
        return 0;
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return 0;
#endif
    int start = 0, count = parameters.sequence;
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        if (parameters.batch != 1) return 0;
        seq_range(parameters.sequence, &start, &count);
        if (start < 0 || count <= 0 ||
            (long)start + count > parameters.sequence) return 0;
    }
    const int32_t* positions = parameters.positions
        ? (const int32_t*)parameters.positions->data + start : NULL;
    long position_elements = parameters.position_mode == 1 ? count :
        parameters.position_mode == 2 ? (long)parameters.batch * count : 0;
    int offset = parameters.position_offset;
    if (!parameters.position_mode) offset += start;
    materialize_tensor_f32(input);
    return cuda_graph_rope_heads_f32(input->data + (long)start * parameters.width,
        positions, position_elements, output->data + (long)start * parameters.width,
        parameters.batch, count, parameters.width, parameters.heads,
        parameters.rotary_width, parameters.theta, offset,
        parameters.interleaved, parameters.position_mode);
}

static int try_cuda_graph_compact_sdpa(Node* node, T* output) {
    T* query = nin(node, VX_PORT_Q);
    T* key = nin(node, VX_PORT_K);
    T* value = nin(node, VX_PORT_V);
    T* mask = nin(node, VX_PORT_MASK);
    if (!query || !key || !value || !output ||
        (query->ndim != 2 && query->ndim != 3) ||
        key->ndim != query->ndim || value->ndim != query->ndim ||
        output->ndim != query->ndim || query->dtype != T_F32 ||
        key->dtype != T_F32 || value->dtype != T_F32 || output->dtype != T_F32)
        return 0;
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return 0;
#endif
    int batch = query->ndim == 3 ? query->shape[0] : 1;
    int query_sequence = query->shape[query->ndim - 2];
    int key_sequence = key->shape[key->ndim - 2];
    int query_width = query->shape[query->ndim - 1];
    int kv_width = key->shape[key->ndim - 1];
    int heads = node->parsed_params.heads;
    int kv_heads = vx_node_param_i32(node, VX_NODE_PARAM_KV_HEADS, heads);
    if (batch <= 0 || query_sequence <= 0 || key_sequence <= 0 ||
        query_width <= 0 || kv_width <= 0 || heads <= 0 || kv_heads <= 0 ||
        heads % kv_heads || query_width % heads || kv_width % kv_heads ||
        query_width / heads != kv_width / kv_heads || query_width / heads > 128 ||
        (query->ndim == 3 && (key->shape[0] != batch || value->shape[0] != batch ||
                             output->shape[0] != batch)) ||
        value->shape[value->ndim - 2] != key_sequence ||
        value->shape[value->ndim - 1] != kv_width ||
        output->shape[output->ndim - 2] != query_sequence ||
        output->shape[output->ndim - 1] != query_width ||
        query->numel != (long)batch * query_sequence * query_width ||
        output->numel != query->numel ||
        key->numel != (long)batch * key_sequence * kv_width ||
        value->numel != key->numel) return 0;
    int mask_mode = attention_mask_mode(mask, batch, query_sequence, key_sequence);
    if (mask_mode < 0) return 0;
    uint64_t expected_mask = 0;
    if (mask_mode == 1) expected_mask = (uint32_t)key_sequence;
    else if (mask_mode == 2) expected_mask = (uint64_t)batch * key_sequence;
    else if (mask_mode == 3 || mask_mode == 4) {
        expected_mask = (uint64_t)query_sequence * key_sequence;
        if (mask_mode == 4) {
            if (expected_mask > (uint64_t)LONG_MAX / (uint32_t)batch) return 0;
            expected_mask *= (uint32_t)batch;
        }
    }
    if (expected_mask ? (!mask || mask->numel <= 0 ||
            (uint64_t)mask->numel != expected_mask) : mask != NULL)
        return 0;
    float scale = node->parsed_params.has_scale ? node->parsed_params.scale
        : 1.0f / sqrtf((float)(query_width / heads));
    if (!isfinite(scale)) return 0;
    materialize_tensor_f32(query);
    materialize_tensor_f32(key);
    materialize_tensor_f32(value);
    long mask_elements = mask ? mask->numel : 0;
    /* Batch-one prefix queries retain the full key/mask stride. Causality
     * excludes unwritten capacity rows before any K/V load. */
    if (g_active_row < 0 && g_prefix_rows > 0 && batch == 1 &&
        query_sequence == g_prefix_row_capacity &&
        g_prefix_rows < query_sequence && node->parsed_params.causal) {
        query_sequence = g_prefix_rows;
        if (mask_mode == 3 || mask_mode == 4)
            mask_elements = (long)query_sequence * key_sequence;
    }
    return cuda_graph_cross_sdpa_gqa_f32(query->data, key->data, value->data,
        mask ? (const int32_t*)mask->data : NULL, mask_elements,
        output->data, query_sequence, key_sequence, query_width, kv_width,
        heads, kv_heads, query_width / heads, batch, scale,
        node->parsed_params.causal ? 1 : 0, mask_mode);
}

static int try_cuda_graph_embedding_partial(Node* n, T* in, T* output) {
    T* tokens;
    T* weight;
    const int32_t* ids;
    int d_model;
    int vocab;
    int row_start;
    int row_count;
    if (g_active_row < 0 && g_prefix_rows <= 0) return 0;
    tokens = node_data_input(n, in);
    weight = nin(n, VX_PORT_WEIGHT);
    if (!tokens || !weight || !output || tokens->dtype != T_I32 ||
        weight->dtype != T_F32 || output->dtype != T_F32 ||
        tokens->ndim != 2 || tokens->shape[0] != 1 ||
        output->ndim != 3 || output->shape[0] != 1 ||
        output->shape[1] != tokens->shape[1] || weight->ndim <= 0)
        return -1;
    materialize_tensor_f32(weight);
    d_model = weight->shape[weight->ndim - 1];
    vocab = d_model > 0 && weight->numel % d_model == 0
        ? (int)(weight->numel / d_model) : 0;
    if (d_model <= 0 || vocab <= 0 || output->shape[2] != d_model ||
        output->numel != tokens->numel * (long)d_model ||
        tokens->numel <= 0 || tokens->numel > INT_MAX) return -1;
    seq_range(tokens->numel, &row_start, &row_count);
    if (row_start < 0 || row_count <= 0 ||
        (long)row_start + row_count > tokens->numel) return -1;
    ids = (const int32_t*)tokens->data;
    for (int row = row_start; row < row_start + row_count; row++)
        if (ids[row] < 0 || ids[row] >= vocab) return -1;
    return cuda_graph_embedding_f32(ids + row_start, weight->data,
            output->data + (long)row_start * d_model, row_count, d_model,
            vocab) ? 1 : -1;
}

static int try_cuda_graph_sdpa_partial(Node* n, T* in, T* output) {
    T* qkv;
    T* mask;
    int d_model;
    int batch;
    int sequence;
    int heads;
    int head_dim;
    int mask_mode;
    int query_start;
    int query_count;
    int causal;
    float scale;
    if (g_active_row < 0 && g_prefix_rows <= 0) return 0;
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return 0;
#endif
    qkv = nin(n, VX_PORT_QKV);
    if (!qkv) qkv = node_data_input(n, in);
    mask = nin(n, VX_PORT_MASK);
    if (!qkv || !output || (qkv->ndim != 2 && qkv->ndim != 3) ||
        output->ndim != qkv->ndim) return -1;
    materialize_tensor_f32(qkv);
    d_model = output->shape[output->ndim - 1];
    batch = qkv->ndim == 3 ? qkv->shape[0] : 1;
    sequence = qkv->shape[qkv->ndim - 2];
    heads = n->parsed_params.heads;
    if (qkv->dtype != T_F32 || d_model <= 0 || batch <= 0 ||
        sequence <= 0 || heads <= 0 || d_model % heads ||
        qkv->shape[qkv->ndim - 1] != 3 * d_model ||
        output->shape[output->ndim - 2] != sequence ||
        qkv->numel != (long)batch * sequence * 3 * d_model ||
        output->numel != (long)batch * sequence * d_model) return -1;
    head_dim = d_model / heads;
    mask_mode = attention_mask_mode(mask, batch, sequence, sequence);
    causal = n->parsed_params.causal ? 1 : 0;
    if (head_dim <= 0 || head_dim > 64 || mask_mode < 0) return -1;
    /* A non-causal prefix depends on rows that the prefix contract has not
     * produced yet, so retain the correct complete-tensor path for it. */
    if (!causal && g_prefix_rows > 0 && g_prefix_rows < sequence) return 0;
    seq_range(sequence, &query_start, &query_count);
    if (query_start < 0 || query_count <= 0 ||
        query_start + query_count > sequence) return -1;
    scale = n->parsed_params.has_scale
        ? n->parsed_params.scale : 1.0f / sqrtf((float)head_dim);
    return cuda_graph_sdpa_range_f32(qkv->data,
            mask ? (const int32_t*)mask->data : NULL,
            mask ? mask->numel : 0, output->data, sequence, d_model, heads,
            head_dim, batch, scale, causal, mask_mode, query_start,
            query_count) ? 1 : -1;
}

static int cuda_runtime_output_strides(const T* output,
                                       uint32_t strides[8]) {
    uint64_t stride = 1u;
    if (!output || !strides || output->ndim <= 0 || output->ndim > 8 ||
        output->numel <= 0 || output->numel > UINT32_MAX)
        return 0;
    memset(strides, 0, 8u * sizeof(uint32_t));
    for (int dimension = output->ndim - 1; dimension >= 0; dimension--) {
        if (output->shape[dimension] <= 0 || stride > UINT32_MAX)
            return 0;
        strides[dimension] = (uint32_t)stride;
        stride *= (uint32_t)output->shape[dimension];
        if (stride > UINT32_MAX) return 0;
    }
    return stride == (uint64_t)output->numel;
}

static int cuda_runtime_cast_dtype_supported(int dtype) {
    return dtype == T_F32 || dtype == T_I32 ||
        dtype == T_I8 || dtype == T_U8;
}

static int try_cuda_graph_compare_i32(Node* n, T* in, T* output,
                                      int operation) {
    T* a = nin(n, VX_PORT_A);
    T* b = nin(n, VX_PORT_B);
    uint32_t output_strides[8] = {0};
    uint32_t a_strides[8] = {0};
    uint32_t b_strides[8] = {0};
    if (!a) a = node_data_input(n, in);
    if (!a || !b || !output || a->dtype != T_I32 ||
        b->dtype != T_I32 || output->dtype != T_I32 ||
        gpu_broadcast_strides(a, output, a_strides) != 0 ||
        gpu_broadcast_strides(b, output, b_strides) != 0 ||
        !cuda_runtime_output_strides(output, output_strides))
        return 0;
    return cuda_graph_compare_i32(
        (const int32_t*)a->data, a->numel,
        (const int32_t*)b->data, b->numel,
        (int32_t*)output->data, output->numel,
        output_strides, a_strides, b_strides, output->ndim, operation);
}

static int try_cuda_graph_argmax_f32_i32(Node* n, T* in, T* output) {
    T* input = in ? in : node_data_input(n, NULL);
    uint64_t outer = 1u;
    uint64_t inner = 1u;
    uint64_t output_elements;
    uint64_t input_elements;
    int axis;
    int keepdims;
    if (!n || !input || !output || !input->data || !output->data ||
        input->dtype != T_F32 || output->dtype != T_I32 ||
        output->elem_size != sizeof(int32_t) ||
        input->ndim <= 0 || input->ndim > 8)
        return 0;
    axis = normalize_axis(n->parsed_params.axis, input->ndim);
    keepdims = n->parsed_params.keepdims;
    if (axis < 0 || axis >= input->ndim ||
        (keepdims != 0 && keepdims != 1) ||
        n->parsed_params.select_last_index != 0 ||
        input->shape[axis] <= 0)
        return 0;
    for (int dimension = 0; dimension < axis; dimension++) {
        uint32_t extent;
        if (input->shape[dimension] <= 0) return 0;
        extent = (uint32_t)input->shape[dimension];
        if (outer > UINT32_MAX / extent) return 0;
        outer *= extent;
    }
    for (int dimension = axis + 1;
         dimension < input->ndim; dimension++) {
        uint32_t extent;
        if (input->shape[dimension] <= 0) return 0;
        extent = (uint32_t)input->shape[dimension];
        if (inner > UINT32_MAX / extent) return 0;
        inner *= extent;
    }
    if (outer > UINT32_MAX / inner) return 0;
    output_elements = outer * inner;
    if ((uint64_t)input->shape[axis] >
        UINT32_MAX / output_elements)
        return 0;
    input_elements =
        output_elements * (uint32_t)input->shape[axis];
    if (input->numel <= 0 ||
        (uint64_t)input->numel != input_elements ||
        output->numel <= 0 ||
        (uint64_t)output->numel != output_elements)
        return 0;
    if (keepdims) {
        if (output->ndim != input->ndim) return 0;
        for (int dimension = 0;
             dimension < input->ndim; dimension++) {
            int expected =
                dimension == axis ? 1 : input->shape[dimension];
            if (output->shape[dimension] != expected) return 0;
        }
    } else {
        if (output->ndim != input->ndim - 1) return 0;
        for (int dimension = 0;
             dimension < input->ndim; dimension++) {
            int output_dimension;
            if (dimension == axis) continue;
            output_dimension =
                dimension < axis ? dimension : dimension - 1;
            if (output->shape[output_dimension] !=
                input->shape[dimension])
                return 0;
        }
    }
    materialize_tensor_f32(input);
    VxEngineState* state = vx_engine_state_current();
    if (state->generation_prefill_rows > 0 &&
        state->generation_prefill_tokens && state->generation_prefill_logits &&
        !strcmp(output->name, state->generation_prefill_tokens) &&
        !strcmp(input->name, state->generation_prefill_logits) &&
        inner == 1 && outer == (uint64_t)state->generation_prefill_capacity) {
        uint32_t row = (uint32_t)state->generation_prefill_rows - 1u;
        return cuda_graph_argmax_f32(
            input->data + (uint64_t)row * input->shape[axis],
            (int32_t*)output->data + row, 1u,
            (uint32_t)input->shape[axis], 1u);
    }
    return cuda_graph_argmax_f32(
        input->data, (int32_t*)output->data,
        (uint32_t)outer, (uint32_t)input->shape[axis],
        (uint32_t)inner);
}

static int try_cuda_graph_split_i32(Node* n, T* in) {
    T* input = node_data_input(n, in);
    int axis;
    int inner;
    int offset = 0;
    if (!input || input->dtype != T_I32 || input->ndim <= 0 ||
        input->ndim > 8 || n->nout <= 0)
        return 0;
    axis = normalize_axis(
        vx_node_param_i32(n, VX_NODE_PARAM_AXIS, 0), input->ndim);
    inner = tensor_inner_after_axis(input, axis);
    if (axis < 0 || axis >= input->ndim || input->shape[axis] <= 0 ||
        inner <= 0)
        return 0;
    /* n->outs is already in graph declaration order; do not sort keys such as
     * out10 lexically ahead of out2. */
    for (int output_index = 0; output_index < n->nout; output_index++) {
        T* destination = t_find(n->outs[output_index].name);
        if (!destination || destination->dtype != T_I32 ||
            destination->ndim != input->ndim ||
            destination->shape[axis] <= 0 ||
            offset > input->shape[axis] - destination->shape[axis])
            return 0;
        for (int dimension = 0; dimension < input->ndim; dimension++) {
            if (dimension != axis &&
                destination->shape[dimension] != input->shape[dimension])
                return 0;
        }
        offset += destination->shape[axis];
    }
    if (offset != input->shape[axis]) return 0;
    offset = 0;
    for (int output_index = 0; output_index < n->nout; output_index++) {
        T* destination = t_find(n->outs[output_index].name);
        int split_size = destination->shape[axis];
        if (!cuda_graph_split_32(
                input->data, destination->data,
                input->numel, destination->numel, inner,
                split_size, input->shape[axis], offset))
            return 0;
        offset += split_size;
    }
    return 1;
}

static int try_cuda_graph_node(Node* n, int idx, const char** node_backend) {
    T* o;
    T* in;
    VxOperatorKind op;
    (void)idx;
    if (!n || !node_backend) return 0;
    o = t_find(n->out);
    in = nin(n, VX_PORT_INPUT);
    op = n->operator_kind;
    if (!o) return 0;

    if (vx_llm_operator(op)) {
        int status=vx_llm_node_run(n,1);
        if (status==VX_LLM_NODE_SUBMITTED_ERROR) return -1;
        if (status!=1) return 0;
        *node_backend="cuda-graph-llm-prefix";
        return 1;
    }
    if(op==VX_OP_N_GRAM_IDS) {
        if(g_active_row>=0||g_prefix_rows>0||g_execution_row>=0) return 0;
        T* tokens=nin(n,VX_PORT_TOKEN_IDS),*multipliers=nin(n,VX_PORT_MULTIPLIERS),*sizes=nin(n,VX_PORT_VOCAB_SIZES),*offsets=nin(n,VX_PORT_OFFSETS);
        if(!tokens||!multipliers||!sizes||!offsets||tokens->ndim!=2||tokens->dtype!=T_I32||o->dtype!=T_I32) return 0;
        if(!cuda_graph_ngram_ids_i32((const int32_t*)tokens->data,(const int32_t*)multipliers->data,
            (const int32_t*)sizes->data,(const int32_t*)offsets->data,(int32_t*)o->data,tokens->shape[0],tokens->shape[1],
            vx_node_param_i32(n,VX_NODE_PARAM_NGRAM_SIZE,0),vx_node_param_i32(n,VX_NODE_PARAM_HEADS_PER_NGRAM,0),
            vx_node_param_i32(n,VX_NODE_PARAM_EOS_TOKEN_ID,-1),
            vx_node_param_i32(n,VX_NODE_PARAM_PAD_TOKEN_ID,vx_node_param_i32(n,VX_NODE_PARAM_EOS_TOKEN_ID,-1)),
            vx_node_param_i32(n,VX_NODE_PARAM_VOCAB_SIZE,0))) return 0;
        *node_backend="cuda-graph-ngram-ids";return 1;
    }

    if (op == VX_OP_ROPE) {
        if (try_cuda_graph_head_rope(n, in, o)) {
            *node_backend = "cuda-graph-rope";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_ARG_MAX) {
        if (try_cuda_graph_argmax_f32_i32(n, in, o)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_Q_BATCH_MATMUL) {
        QBatchMatMulPhysicalDescriptor descriptor;
        if (qbatch_matmul_physical_validate(n, o, &descriptor) != 0)
            return 0;
        if (cuda_graph_qbatch_matmul_i8u8(
                descriptor.a->data,
                descriptor.a->shape, descriptor.a->ndim,
                descriptor.a->quantization.scale,
                descriptor.a->quantization.zero_point,
                descriptor.a_dtype,
                descriptor.b->data,
                descriptor.b->shape, descriptor.b->ndim,
                descriptor.b->quantization.scale,
                descriptor.b->quantization.zero_point,
                descriptor.b_dtype,
                descriptor.output->data,
                descriptor.output->shape, descriptor.output->ndim,
                descriptor.output->quantization.scale,
                descriptor.output->quantization.zero_point,
                descriptor.output_dtype)) {
            *node_backend = "cuda-qbatch-matmul-w8a8";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_BATCH_MATMUL) {
        T* a = nin(n, VX_PORT_A);
        T* b = nin(n, VX_PORT_B);
        if (!a || !b || a->dtype != T_F32 || b->dtype != T_F32 ||
            o->dtype != T_F32)
            return 0;
        materialize_tensor_f32(a);
        materialize_tensor_f32(b);
        if (cuda_graph_batch_matmul_f32(
                a->data, a->shape, a->ndim,
                b->data, b->shape, b->ndim,
                o->data, o->shape, o->ndim)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_EQUAL) || (op == VX_OP_GREATER_OR_EQUAL)) {
        if (try_cuda_graph_compare_i32(
                n, in, o, (op == VX_OP_GREATER_OR_EQUAL) ? 1 : 0)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_NOT) {
        T* source = node_data_input(n, in);
        if (source && source->dtype == T_I32 && o->dtype == T_I32 &&
            source->numel == o->numel &&
            tensor_same_shape(source, o) &&
            cuda_graph_not_i32(
                (const int32_t*)source->data, (int32_t*)o->data,
                o->numel)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_CLIP) && o->dtype == T_I32) {
        T* source = node_data_input(n, in);
        if (source && source->dtype == T_I32 &&
            source->numel == o->numel && tensor_same_shape(source, o) &&
            cuda_graph_clip_i32(
                (const int32_t*)source->data, (int32_t*)o->data,
                o->numel, n->parsed_params.min_val_i32,
                n->parsed_params.max_val_i32)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_WHERE) {
        T* condition = nin(n, VX_PORT_CONDITION);
        T* a = nin(n, VX_PORT_X);
        T* b = nin(n, VX_PORT_Y);
        if (!condition) condition = nin(n, VX_PORT_COND);
        if (!a) a = nin(n, VX_PORT_A);
        if (!b) b = nin(n, VX_PORT_B);
        if (condition && a && b && condition->dtype == T_I32 &&
            (o->dtype == T_F32 || o->dtype == T_I32) &&
            a->dtype == o->dtype && b->dtype == o->dtype &&
            tensor_same_shape(condition, o) &&
            tensor_same_shape(a, o) && tensor_same_shape(b, o)) {
            if (o->dtype == T_F32) {
                materialize_tensor_f32(a);
                materialize_tensor_f32(b);
            }
            if (cuda_graph_where_32(
                    (const int32_t*)condition->data,
                    a->data, b->data, o->data, o->numel)) {
                *node_backend = "cuda-graph";
                return 1;
            }
        }
        return 0;
    }

    if (((op == VX_OP_EXPAND) || (op == VX_OP_BROADCAST)) &&
        (o->dtype == T_F32 || o->dtype == T_I32)) {
        T* source = in ? in : nin(n, VX_PORT_DATA);
        if (source && source->dtype == o->dtype) {
            if (source->dtype == T_F32) materialize_tensor_f32(source);
            if (cuda_graph_expand_32(
                    source->data, o->data, source->shape, source->ndim,
                    o->shape, o->ndim)) {
                *node_backend = "cuda-graph";
                return 1;
            }
        }
        return 0;
    }

    if ((op == VX_OP_TRANSPOSE) && o->dtype == T_I32) {
        T* source = in ? in : node_data_input(n, NULL);
        int permutation[8];
        if (!source || source->dtype != T_I32 ||
            source->ndim <= 0 || source->ndim > 8 ||
            source->numel != o->numel)
            return 0;
        for (int dimension = 0; dimension < source->ndim; dimension++)
            permutation[dimension] = source->ndim - 1 - dimension;
        if (n->parsed_params.has_perm) {
            if (n->parsed_params.perm_rank != source->ndim) return 0;
            memcpy(permutation, n->parsed_params.perm,
                   (size_t)source->ndim * sizeof(permutation[0]));
        }
        if (cuda_graph_transpose_32(
                source->data, o->data, source->shape,
                permutation, source->ndim)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_CAST) {
        T* source = in ? in : node_data_input(n, NULL);
        int ok = 0;
        int target_dtype = cast_target_dtype(n);
        if (!source || source->numel != o->numel || o->numel <= 0 ||
            target_dtype != o->dtype ||
            !cuda_runtime_cast_dtype_supported(source->dtype) ||
            !cuda_runtime_cast_dtype_supported(o->dtype))
            return 0;
        if (source->dtype == T_F32) materialize_tensor_f32(source);
        if (source->dtype == o->dtype && source->elem_size == sizeof(uint32_t))
            ok = cuda_graph_copy_32(source->data, o->data, o->numel);
        else
            ok = cuda_graph_cast_typed(
                source->data, source->dtype, o->data, o->dtype,
                o->numel);
        if (ok) {
            *node_backend = source->dtype == o->dtype
                ? "cuda-graph-copy" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (((op == VX_OP_RESHAPE) || (op == VX_OP_FLATTEN) ||
         (op == VX_OP_SQUEEZE) || (op == VX_OP_UNSQUEEZE) ||
         (op == VX_OP_IDENTITY)) && o->dtype == T_I32) {
        T* source = in ? in : node_data_input(n, NULL);
        if (source && source->dtype == T_I32 &&
            source->numel == o->numel && o->numel > 0 &&
            cuda_graph_copy_32(source->data, o->data, o->numel)) {
            *node_backend = "cuda-graph-copy";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_SPLIT) && o->dtype == T_I32) {
        if (try_cuda_graph_split_i32(n, in)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_CONCAT) && o->dtype == T_I32) {
        VxConcatInputs inputs;
        if (!concat_inputs_build(n, o, vx_node_param_i32(n, VX_NODE_PARAM_AXIS, 0),
                                 T_I32, &inputs)) return 0;
        int ok = cuda_graph_concat_32(inputs.raw, inputs.sizes, inputs.axes,
            inputs.count, o->data, o->shape[inputs.axis], inputs.inner);
        concat_inputs_clear(&inputs);
        if (ok) *node_backend = "cuda-graph";
        return ok;
    }

    if ((op == VX_OP_SLICE) &&
        (o->dtype == T_F32 || o->dtype == T_I32)) {
        T* source = in ? in : nin(n, VX_PORT_DATA);
        int starts[8] = {0};
        int steps[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        int axis_count;
        unsigned int seen_axes = 0u;
        if (!source || source->dtype != o->dtype ||
            source->ndim <= 0 || source->ndim > 8 ||
            o->ndim != source->ndim ||
            !n->parsed_params.has_slice)
            return 0;
        axis_count = n->parsed_params.slice_rank;
        if (axis_count <= 0 || axis_count > source->ndim)
            return 0;
        for (int index = 0; index < axis_count; index++) {
            int axis = n->parsed_params.axes[index];
            int step = n->parsed_params.steps[index];
            int raw_start = n->parsed_params.starts[index];
            int start;
            if (axis < 0) axis += source->ndim;
            if (axis < 0 || axis >= source->ndim ||
                step <= 0 ||
                (seen_axes & (1u << (unsigned int)axis)))
                return 0;
            seen_axes |= 1u << (unsigned int)axis;
            if (slice_canonical_start(
                    raw_start, source->shape[axis], o->shape[axis], step,
                    &start) != 0)
                return 0;
            starts[axis] = start;
            steps[axis] = step;
        }
        if (source->dtype == T_F32) materialize_tensor_f32(source);
        if (cuda_graph_slice_32(
                source->data, o->data, source->shape, source->ndim,
                o->shape, o->ndim, starts, steps)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (o->dtype != T_F32 && !(op == VX_OP_GATHER && o->dtype == T_I32)) return 0;

    if (op == VX_OP_CONV_2D) {
        T* weight = nin(n, VX_PORT_WEIGHT);
        T* bias = nin(n, VX_PORT_BIAS);
        const GraphNodeFusion* conv_add_fusion =
            graph_opt_node_fusion(idx, GRAPH_FUSION_CONV_ADD);
        T* residual = NULL;
        int stride_y = n->parsed_params.stride[0];
        int stride_x = n->parsed_params.stride[1];
        int dilation_y = n->parsed_params.dilation[0];
        int dilation_x = n->parsed_params.dilation[1];
        int pads[4] = {
            n->parsed_params.pads[0], n->parsed_params.pads[1],
            n->parsed_params.pads[2], n->parsed_params.pads[3]
        };
        int groups = n->parsed_params.groups;
        int relu = n->fuse_relu6 ? 2 : n->parsed_params.relu;
        T weight_f32;
        T bias_f32;
        T* prepared_weight;
        T* prepared_bias = bias;
        if (!in || !weight || in->dtype != T_F32 ||
            (weight->dtype != T_F32 && weight->dtype != T_F16) ||
            in->ndim != 4 || weight->ndim != 4 || o->ndim != 4 ||
            in->shape[0] <= 0 || o->shape[0] != in->shape[0])
            return 0;
        if (pads[0] < 0 || pads[1] < 0 || relu < 0 || relu > 2)
            return 0;
        if (conv_add_fusion) {
            int peer_idx = conv_add_fusion->peer_idx;
            Node* peer = peer_idx > idx && peer_idx < g_nn
                ? &g_n[peer_idx] : NULL;
            int residual_occurrences = 0;
            int conv_occurrences = 0;
            if (!peer || !peer->skip || (peer->operator_kind != VX_OP_ADD) ||
                peer->nin != 2 || strcmp(peer->out, n->out) ||
                peer->parsed_params.relu != 0 ||
                !conv_add_fusion->source_tensor[0] ||
                !conv_add_fusion->aux_tensor[0]) return 0;
            for (int input_index = 0; input_index < 2; input_index++) {
                if (!strcmp(peer->ins[input_index].name,
                            conv_add_fusion->aux_tensor))
                    residual_occurrences++;
                if (!strcmp(peer->ins[input_index].name,
                            conv_add_fusion->source_tensor))
                    conv_occurrences++;
            }
            residual = t_find(conv_add_fusion->aux_tensor);
            if (residual_occurrences != 1 || conv_occurrences != 1 ||
                !residual ||
                residual->dtype != T_F32 || residual->numel != o->numel ||
                !tensor_same_shape(residual, o)) return 0;
            int residual_producer =
                producer_index_any(conv_add_fusion->aux_tensor);
            if (residual_producer > idx) return 0;
        }
        prepared_weight = prepare_conv_weight_image(
            idx, weight, n->parsed_params.weight_layout,
            in->shape[3], o->shape[3], groups, &weight_f32);
        if (!prepared_weight) return 0;
        if (bias && bias->dtype == T_F16) {
            float* values = f16_tensor_f32_cache(g_f16bcache, idx, bias);
            if (!values) return 0;
            bias_f32 = *bias;
            bias_f32.data = values;
            bias_f32.dtype = T_F32;
            bias_f32.elem_size = sizeof(float);
            prepared_bias = &bias_f32;
        }
        if (prepared_bias && prepared_bias->dtype != T_F32)
            return 0;
        int launched = conv_add_fusion
            ? cuda_graph_conv2d_add_f32(in->data, o->data,
                prepared_weight->data,
                prepared_bias ? prepared_bias->data : NULL, residual->data,
                in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                o->shape[3], prepared_weight->shape[0],
                prepared_weight->shape[1], o->shape[1], o->shape[2],
                stride_y, stride_x, pads[0], pads[1], groups, relu,
                dilation_y, dilation_x)
            : cuda_graph_conv2d_f32(in->data, o->data,
                prepared_weight->data,
                prepared_bias ? prepared_bias->data : NULL,
                in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                o->shape[3], prepared_weight->shape[0],
                prepared_weight->shape[1], o->shape[1], o->shape[2],
                stride_y, stride_x, pads[0], pads[1], groups, relu,
                dilation_y, dilation_x);
        if (launched) {
            *node_backend = conv_add_fusion
                ? "cuda-graph+conv-add" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_ADD) || (op == VX_OP_MUL) ||
        (op == VX_OP_SUB) || (op == VX_OP_DIV)) {
        if (op == VX_OP_ADD) {
            T* a = nin(n, VX_PORT_A);
            T* b = nin(n, VX_PORT_B);
            const GraphNodeFusion* add3_fusion =
                graph_opt_node_fusion(idx, GRAPH_FUSION_CHAINED_ELEMENTWISE);
            int relu = vx_node_param_i32(n, VX_NODE_PARAM_RELU, 0);
            if (!a) a = in;
            if (add3_fusion) {
                int peer_idx = add3_fusion->peer_idx;
                Node* peer = peer_idx > idx && peer_idx < g_nn
                    ? &g_n[peer_idx] : NULL;
                T* c = add3_fusion->aux_tensor[0]
                    ? t_find(add3_fusion->aux_tensor) : NULL;
                int c_occurrences = 0;
                int add0_occurrences = 0;
                int peer_relu = peer
                    ? vx_node_param_i32(peer, VX_NODE_PARAM_RELU, 0) : -1;
                if (!peer || !peer->skip || (peer->operator_kind != VX_OP_ADD) ||
                    peer->nin != 2 || strcmp(peer->out, n->out) ||
                    n->nin != 2 || !a || !b || !c ||
                    !add3_fusion->source_tensor[0] ||
                    vx_node_param_i32(n, VX_NODE_PARAM_RELU, 0) != 0 ||
                    peer_relu < 0 || peer_relu > 2) return 0;
                for (int input_index = 0; input_index < 2; input_index++) {
                    if (!strcmp(peer->ins[input_index].name,
                                add3_fusion->aux_tensor))
                        c_occurrences++;
                    if (!strcmp(peer->ins[input_index].name,
                                add3_fusion->source_tensor))
                        add0_occurrences++;
                }
                if (c_occurrences != 1 || add0_occurrences != 1 ||
                    a->dtype != T_F32 ||
                    b->dtype != T_F32 || c->dtype != T_F32 ||
                    a->numel != b->numel || a->numel != c->numel ||
                    a->numel != o->numel || !tensor_same_shape(a, o) ||
                    !tensor_same_shape(b, o) || !tensor_same_shape(c, o) ||
                    producer_index_any(add3_fusion->aux_tensor) >= idx)
                    return 0;
                int row_start = 0;
                int row_count = 0;
                int partial = cuda_partial_sequence_rows(
                    o, o->ndim > 0 ? o->shape[o->ndim - 1] : 0,
                    &row_start, &row_count);
                if (partial < 0) return 0;
                long offset = partial
                    ? (long)row_start * o->shape[o->ndim - 1] : 0;
                long elements = partial
                    ? (long)row_count * o->shape[o->ndim - 1] : o->numel;
                if (!cuda_graph_add3_relu_f32(a->data + offset,
                        b->data + offset, c->data + offset, o->data + offset,
                        elements, peer_relu)) return 0;
                *node_backend = partial
                    ? "cuda-graph-row+add3" : "cuda-graph+add3";
                return 1;
            }
            if (a && b && a->dtype == T_F32 && b->dtype == T_F32 &&
                a->numel == b->numel && a->numel == o->numel &&
                relu >= 0 && relu <= 2) {
                int row_start = 0;
                int row_count = 0;
                int partial = cuda_partial_sequence_rows(
                    o, o->ndim > 0 ? o->shape[o->ndim - 1] : 0,
                    &row_start, &row_count);
                if (partial >= 0) {
                    long offset = partial
                        ? (long)row_start * o->shape[o->ndim - 1] : 0;
                    long elements = partial
                        ? (long)row_count * o->shape[o->ndim - 1] : o->numel;
                    if (cuda_graph_add_relu_f32(a->data + offset,
                            b->data + offset, o->data + offset, elements,
                            relu)) {
                        *node_backend = partial
                            ? "cuda-graph-row" : "cuda-graph";
                        return 1;
                    }
                }
            }
            if (relu) return 0;
        }
        if (try_gpu_graph_binary(n, in, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_SIGMOID) || (op == VX_OP_CLIP) ||
        (op == VX_OP_RELU) || (op == VX_OP_GELU) ||
        (op == VX_OP_SILU) ||
        (op == VX_OP_TANH) || (op == VX_OP_HARD_SWISH) ||
        (op == VX_OP_HARD_SIGMOID) || (op == VX_OP_LEAKY_RELU)) {
        int ok = 0;
        if (!in) in = node_data_input(n, NULL);
        if (!in) return 0;
        materialize_tensor_f32(in);
        if (in->dtype != T_F32 || in->numel != o->numel || o->numel <= 0) return 0;
        int row_start = 0;
        int row_count = 0;
        int partial = cuda_partial_sequence_rows(
            o, o->ndim > 0 ? o->shape[o->ndim - 1] : 0,
            &row_start, &row_count);
        if (partial < 0) return 0;
        long offset = partial
            ? (long)row_start * o->shape[o->ndim - 1] : 0;
        long elements = partial
            ? (long)row_count * o->shape[o->ndim - 1] : o->numel;
        const float* input_data = in->data + offset;
        float* output_data = o->data + offset;
        if (op == VX_OP_SIGMOID)
            ok = cuda_graph_sigmoid_f32(input_data, output_data, elements);
        else if (op == VX_OP_CLIP) {
            float minimum = n->parsed_params.min_val;
            float maximum = n->parsed_params.max_val;
            if (isnan(minimum) || isnan(maximum) || minimum > maximum) return 0;
            ok = cuda_graph_clip_f32(input_data, output_data, elements, minimum, maximum);
        } else if (op == VX_OP_RELU)
            ok = cuda_graph_relu_f32(input_data, output_data, elements);
        else if (op == VX_OP_GELU) {
            int approximate_tanh = 0;
            if (gpu_node_gelu_approximate_tanh(n, &approximate_tanh) != 0)
                return 0;
            ok = cuda_graph_gelu_f32(input_data, output_data, elements, approximate_tanh);
        } else if (op == VX_OP_SILU)
            ok = cuda_graph_silu_f32(input_data, output_data, elements);
        else if (op == VX_OP_TANH)
            ok = cuda_graph_tanh_f32(input_data, output_data, elements);
        else if (op == VX_OP_HARD_SWISH)
            ok = cuda_graph_hardswish_f32(input_data, output_data, elements);
        else if (op == VX_OP_HARD_SIGMOID)
            ok = cuda_graph_hardsigmoid_f32(input_data, output_data, elements);
        else
            ok = cuda_graph_leaky_relu_f32(
                input_data, output_data, elements,
                vx_node_param_f32(n, VX_NODE_PARAM_ALPHA, 0.01f));
        if (ok) {
            *node_backend = partial ? "cuda-graph-row" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_PRELU) {
        T* weight = nin(n, VX_PORT_WEIGHT);
        if (!weight) weight = nin(n, VX_PORT_SLOPE);
        int channels;
        if (!in || !weight || !o || in->ndim <= 0) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        channels = weight->numel == 1 ? 1 : in->shape[in->ndim - 1];
        if (in->dtype != T_F32 || weight->dtype != T_F32 ||
            o->dtype != T_F32 || channels <= 0 || in->numel != o->numel ||
            (weight->numel != 1 && weight->numel != channels)) return 0;
        if (cuda_graph_prelu_f32(in->data, weight->data, o->data, o->numel,
                                 channels)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_LAYER_NORM) {
        T* weight = nin(n, VX_PORT_WEIGHT);
        T* bias = nin(n, VX_PORT_BIAS);
        int width;
        long row_count;
        float epsilon;
        if (!in || !weight || in->ndim <= 0) return 0;
        width = vx_node_param_i32(
            n, VX_NODE_PARAM_D_MODEL, in->shape[in->ndim - 1]);
        epsilon = vx_node_param_f32(n, VX_NODE_PARAM_EPS, 1.0e-6f);
        if (normalization_dimensions(in, o, weight, bias, width,
                                     &row_count) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        if (bias) materialize_tensor_f32(bias);
        int row_start = 0;
        int selected_rows = 0;
        int partial = cuda_partial_sequence_rows(in, width, &row_start,
                                                  &selected_rows);
        if (partial < 0) return 0;
        if (!partial) selected_rows = (int)row_count;
        if (cuda_graph_layernorm_f32(in->data + (long)row_start * width,
                weight->data, bias ? bias->data : NULL,
                o->data + (long)row_start * width,
                selected_rows, width, epsilon)) {
            *node_backend = partial ? "cuda-graph-row" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_RMS_NORM) {
        T* weight = nin(n, VX_PORT_WEIGHT);
        int width;
        long row_count;
        float epsilon;
        if (!in || !weight || in->ndim <= 0) return 0;
        width = vx_node_param_i32(
            n, VX_NODE_PARAM_D_MODEL, in->shape[in->ndim - 1]);
        epsilon = vx_node_param_f32(n, VX_NODE_PARAM_EPS, 1.0e-6f);
        if (normalization_dimensions(in, o, weight, NULL, width,
                                     &row_count) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        int row_start = 0;
        int selected_rows = 0;
        int partial = cuda_partial_rmsnorm_rows(in, width, &row_start,
                                                &selected_rows);
        if (partial < 0) return 0;
        if (!partial) selected_rows = (int)row_count;
        if (cuda_graph_rmsnorm_f32(in->data + (long)row_start * width,
                weight->data, o->data + (long)row_start * width,
                selected_rows, width, epsilon)) {
            *node_backend = partial ? "cuda-graph-row" : "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_GROUP_NORM) {
        T* weight = nin(n, VX_PORT_WEIGHT);
        T* bias = nin(n, VX_PORT_BIAS);
        int groups = n->parsed_params.num_groups;
        int batch, height, width, channels;
        float epsilon = n->parsed_params.eps;
        if (!weight) weight = nin(n, VX_PORT_SCALE);
        if (groupnorm_dimensions(in, o, weight, bias, groups, &batch,
                                 &height, &width, &channels) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        if (bias) materialize_tensor_f32(bias);
        if (cuda_graph_groupnorm_f32(in->data, weight->data,
                                     bias ? bias->data : NULL,
                                     o->data, batch, height, width, channels,
                                     groups, epsilon)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_SOFTMAX) || (op == VX_OP_LOG_SOFTMAX)) {
        int axis;
        int width;
        long row_count;
        int ok;
        if (!in || in->ndim <= 0) return 0;
        axis = vx_node_param_i32(
            n, VX_NODE_PARAM_AXIS, in->ndim - 1);
        if (axis < 0) axis += in->ndim;
        if (axis != in->ndim - 1) return 0;
        materialize_tensor_f32(in);
        width = in->shape[in->ndim - 1];
        row_count = width > 0 ? in->numel / width : 0;
        if (in->dtype != T_F32 || width <= 0 || row_count <= 0 ||
            row_count > INT32_MAX || in->numel != row_count * width ||
            o->numel != in->numel) return 0;
        ok = (op == VX_OP_SOFTMAX)
            ? cuda_graph_softmax_f32(in->data, o->data, (int)row_count, width)
            : cuda_graph_logsoftmax_f32(in->data, o->data, (int)row_count, width);
        if (ok) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_REDUCE_SUM) || (op == VX_OP_REDUCE_MEAN)) {
        T* source = NULL;
        long rows = 0;
        int width = 0;
        float scale = 1.0f;
        if (last_axis_reduction_dimensions(
                n, o, &source, &rows, &width, &scale) != 0 ||
            rows > INT32_MAX) return 0;
        materialize_tensor_f32(source);
        if (cuda_graph_reduce_f32(source->data, o->data, (int)rows, width,
                                  scale)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_BATCH_NORM_2D) {
        T* weight = nin(n, VX_PORT_WEIGHT);
        T* bias = nin(n, VX_PORT_BIAS);
        T* mean = nin(n, VX_PORT_RUNNING_MEAN);
        T* variance = nin(n, VX_PORT_RUNNING_VAR);
        int channels;
        float epsilon = vx_node_param_f32(n, VX_NODE_PARAM_EPS, 1.0e-5f);
        if (!in || !weight || !bias || !mean || !variance ||
            in->ndim != 4 || o->ndim != 4 || !(epsilon > 0.0f) ||
            !isfinite(epsilon)) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        materialize_tensor_f32(bias);
        materialize_tensor_f32(mean);
        materialize_tensor_f32(variance);
        channels = in->shape[3];
        if (in->dtype != T_F32 || weight->dtype != T_F32 ||
            bias->dtype != T_F32 || mean->dtype != T_F32 ||
            variance->dtype != T_F32 || o->numel != in->numel ||
            channels <= 0 || weight->numel < channels ||
            bias->numel < channels || mean->numel < channels ||
            variance->numel < channels) return 0;
        if (cuda_graph_batchnorm2d_f32(in->data, weight->data, bias->data,
                mean->data, variance->data, o->data, in->shape[0],
                in->shape[1], in->shape[2], channels, epsilon)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_GLOBAL_AVERAGE_POOL) {
        if (!in || in->ndim != 4) return 0;
        materialize_tensor_f32(in);
        if (in->dtype != T_F32 ||
            o->numel != (long)in->shape[0] * in->shape[3]) return 0;
        if (cuda_graph_global_average_pool_f32(in->data, o->data,
                in->shape[0], in->shape[1], in->shape[2], in->shape[3])) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_AVERAGE_POOL_2D) {
        int kernel_y, kernel_x, stride_y, stride_x, padding_y, padding_x;
        if (!in || in->ndim != 4 || o->ndim != 4 ||
            in->dtype != T_F32) return 0;
        materialize_tensor_f32(in);
        if (gpu_node_param_pair(
                n, VX_NODE_PARAM_KERNEL, VX_NODE_ARRAY_KERNEL,
                1, &kernel_y, &kernel_x) != 0 ||
            gpu_node_param_pair(
                n, VX_NODE_PARAM_STRIDE, VX_NODE_ARRAY_STRIDE,
                1, &stride_y, &stride_x) != 0 ||
            gpu_node_spatial_pad_origin(n, &padding_y, &padding_x) != 0)
            return 0;
        if (cuda_graph_average_pool2d_f32(in->data, o->data, in->shape[0],
                in->shape[1], in->shape[2], in->shape[3], o->shape[1],
                o->shape[2], kernel_y, kernel_x, stride_y, stride_x,
                padding_y, padding_x)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_CONV_1D) {
        if (try_gpu_graph_conv1d(n, in, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_EMBEDDING) {
        int partial = try_cuda_graph_embedding_partial(n, in, o);
        if (partial != 0) {
            if (partial > 0) {
                *node_backend = "cuda-graph-row";
                return 1;
            }
            return 0;
        }
        if (try_gpu_graph_embedding(n, in, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_RESIZE) || (op == VX_OP_RESIZE_NEAREST_2D) ||
        (op == VX_OP_UPSAMPLE_NEAREST_2D)) {
        int mode;
        if (!in || in->dtype != T_F32 || in->ndim != 4 || o->ndim != 4 ||
            in->shape[0] <= 0 || o->shape[0] != in->shape[0] ||
            o->shape[3] != in->shape[3]) return 0;
        mode = 0;
        if ((op == VX_OP_RESIZE) &&
            gpu_node_resize_mode(n, &mode) != 0)
            return 0;
        if (cuda_graph_resize_f32(in->data, o->data, in->shape[0],
                in->shape[1], in->shape[2], in->shape[3], o->shape[1],
                o->shape[2], mode)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_MAX_POOL_2D) {
        int kernel_y, kernel_x, stride_y, stride_x, padding_y, padding_x;
        if (!in || in->dtype != T_F32 || in->ndim != 4 || o->ndim != 4 ||
            in->shape[0] <= 0 || o->shape[0] != in->shape[0] ||
            o->shape[3] != in->shape[3]) return 0;
        if (gpu_node_param_pair(
                n, VX_NODE_PARAM_KERNEL, VX_NODE_ARRAY_KERNEL,
                1, &kernel_y, &kernel_x) != 0 ||
            gpu_node_param_pair(
                n, VX_NODE_PARAM_STRIDE, VX_NODE_ARRAY_STRIDE,
                1, &stride_y, &stride_x) != 0 ||
            gpu_node_spatial_pad_origin(n, &padding_y, &padding_x) != 0)
            return 0;
        if (cuda_graph_maxpool2d_f32(in->data, o->data, in->shape[0],
                in->shape[1], in->shape[2], in->shape[3], o->shape[1],
                o->shape[2], kernel_y, kernel_x, stride_y, stride_x,
                padding_y, padding_x)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_TRANSPOSE) {
        int permutation[8];
        if (!in || in->ndim <= 0 || in->ndim > 8) return 0;
        materialize_tensor_f32(in);
        for (int dimension = 0; dimension < in->ndim; dimension++)
            permutation[dimension] = in->ndim - 1 - dimension;
        if (n->parsed_params.has_perm) {
            if (n->parsed_params.perm_rank != in->ndim) return 0;
            memcpy(permutation, n->parsed_params.perm,
                   (size_t)in->ndim * sizeof(permutation[0]));
        }
        if (in->dtype == T_F32 && o->numel == in->numel &&
            cuda_graph_transpose_f32(in->data, o->data, in->shape,
                                     permutation, in->ndim)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_WHERE) || (op == VX_OP_MASK)) {
        T* condition = nin(n, VX_PORT_COND);
        T* a = nin(n, VX_PORT_X);
        T* b = nin(n, VX_PORT_Y);
        if (!condition) condition = nin(n, VX_PORT_CONDITION);
        if (!a) a = nin(n, VX_PORT_A);
        if (!b) b = nin(n, VX_PORT_B);
        if (!condition || !a || !b) return 0;
        materialize_tensor_f32(condition);
        materialize_tensor_f32(a);
        materialize_tensor_f32(b);
        if (condition->dtype == T_F32 && a->dtype == T_F32 &&
            b->dtype == T_F32 && condition->numel == o->numel &&
            a->numel == o->numel && b->numel == o->numel &&
            cuda_graph_where_f32(condition->data, a->data, b->data,
                                 o->data, o->numel)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_EXPAND) || (op == VX_OP_BROADCAST)) {
        T* source = in ? in : nin(n, VX_PORT_DATA);
        if (!source) return 0;
        materialize_tensor_f32(source);
        if (source->dtype == T_F32 &&
            cuda_graph_expand_f32(source->data, o->data, source->shape,
                                  source->ndim, o->shape, o->ndim)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_GATHER) {
        T* indices = nin(n, VX_PORT_INDICES);
        int axis;
        int outer;
        int axis_size;
        int inner;
        int indices_elements;
        int output_elements;
        if (!in || !indices || in->ndim <= 0) return 0;
        axis = normalize_axis(
            vx_node_param_i32(n, VX_NODE_PARAM_AXIS, 0), in->ndim);
        materialize_tensor_f32(in);
        if (gather_i32_dimensions(in, indices, o, axis, &outer, &axis_size,
                                  &inner, &indices_elements,
                                  &output_elements) &&
            cuda_graph_gather_i32_f32(in->data,
                                      (const int32_t*)indices->data, o->data,
                                      outer, axis_size, inner,
                                      indices_elements, output_elements)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_PAD) {
        T* source = in ? in : nin(n, VX_PORT_DATA);
        const int32_t* pads;
        int pads_length;
        int padding_top;
        int padding_left;
        if (!source || source->ndim <= 0 || source->ndim > 4 ||
            o->ndim <= 0 || o->ndim > 4) return 0;
        materialize_tensor_f32(source);
        pads = vx_node_param_array(
            n, VX_NODE_PARAM_PADS, VX_NODE_ARRAY_PADS, &pads_length);
        if (!pads || (pads_length != 4 && pads_length != 8)) return 0;
        padding_top = pads_length == 8 ? pads[1] : pads[0];
        padding_left = pads_length == 8 ? pads[2] : pads[1];
        if (padding_top < 0 || padding_left < 0) return 0;
        if (source->dtype == T_F32 &&
            cuda_graph_pad4d_f32(source->data, o->data, source->shape,
                source->ndim, o->shape, o->ndim, padding_top, padding_left,
                vx_node_param_f32(n, VX_NODE_PARAM_VALUE, 0.0f))) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_SLICE) {
        T* source = in ? in : nin(n, VX_PORT_DATA);
        int starts4[4] = {0, 0, 0, 0};
        int steps4[4] = {1, 1, 1, 1};
        int axis_count;
        int base;
        unsigned int selected_axes = 0u;
        if (!source || source->ndim <= 0 || source->ndim > 4 ||
            o->ndim != source->ndim ||
            !n->parsed_params.has_slice) return 0;
        materialize_tensor_f32(source);
        axis_count = n->parsed_params.slice_rank;
        if (axis_count <= 0 || axis_count > source->ndim) return 0;
        base = 4 - source->ndim;
        for (int index = 0; index < axis_count; index++) {
            int axis = n->parsed_params.axes[index];
            int step = n->parsed_params.steps[index];
            int raw_start = n->parsed_params.starts[index];
            int start;
            if (axis < 0) axis += source->ndim;
            if (axis < 0 || axis >= source->ndim || step <= 0 ||
                (selected_axes & (1u << (unsigned int)axis)))
                return 0;
            if (slice_canonical_start(
                    raw_start, source->shape[axis], o->shape[axis], step,
                    &start) != 0)
                return 0;
            selected_axes |= 1u << (unsigned int)axis;
            starts4[base + axis] = start;
            steps4[base + axis] = step;
        }
        for (int dimension = 0; dimension < source->ndim; dimension++) {
            if (!(selected_axes & (1u << (unsigned int)dimension)) &&
                o->shape[dimension] != source->shape[dimension]) return 0;
        }
        if (source->dtype == T_F32 &&
            cuda_graph_slice4d_f32(source->data, o->data, source->shape,
                source->ndim, o->shape, o->ndim, starts4, steps4)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_CONV_TRANSPOSE_2D) {
        T* weight = nin(n, VX_PORT_WEIGHT);
        T* bias = nin(n, VX_PORT_BIAS);
        int kernel_y, kernel_x, stride_y, stride_x, padding_y, padding_x;
        if (!in || !weight || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        if (bias) materialize_tensor_f32(bias);
        if (gpu_node_param_pair(
                n, VX_NODE_PARAM_KERNEL, VX_NODE_ARRAY_KERNEL,
                1, &kernel_y, &kernel_x) != 0 ||
            gpu_node_param_pair(
                n, VX_NODE_PARAM_STRIDE, VX_NODE_ARRAY_STRIDE,
                1, &stride_y, &stride_x) != 0 ||
            gpu_node_param_pair(
                n, VX_NODE_PARAM_PADDING, VX_NODE_ARRAY_PADDING,
                0, &padding_y, &padding_x) != 0)
            return 0;
        if (in->dtype == T_F32 && weight->dtype == T_F32 &&
            (!bias || bias->dtype == T_F32) &&
            cuda_graph_conv_transpose2d_f32(in->data, weight->data,
                bias ? bias->data : NULL, o->data, in->shape[0],
                in->shape[1], in->shape[2], in->shape[3], o->shape[1],
                o->shape[2], o->shape[3], kernel_y, kernel_x, stride_y,
                stride_x, padding_y, padding_x)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_SDPA) {
        int partial = try_cuda_graph_sdpa_partial(n, in, o);
        if (partial != 0) {
            if (partial > 0) {
                *node_backend = "cuda-graph-row";
                return 1;
            }
            return 0;
        }
        if (try_gpu_graph_sdpa(n, in, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_CROSS_SDPA) {
        if (vx_node_param_has(n, VX_NODE_PARAM_KV_HEADS) ||
            (o->ndim > 0 && n->parsed_params.heads > 0 &&
             o->shape[o->ndim - 1] / n->parsed_params.heads > 64)) {
            int compact = try_cuda_graph_compact_sdpa(n, o);
            if (compact < 0) return -1;
            if (compact > 0) {
                *node_backend = "cuda-graph-gqa";
                return 1;
            }
            return 0;
        }
        if (try_gpu_graph_cross_sdpa(n, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_CROSS_ATTENTION) {
        if (try_gpu_graph_cross_attention(n, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_SPLIT) {
        if (try_gpu_graph_split(n, in, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_SPATIAL_SOFTARGMAX_Y) || (op == VX_OP_PROFILE_X) ||
        (op == VX_OP_PROFILE_Y) || (op == VX_OP_MEAN_HEIGHT)) {
        if (try_gpu_graph_profile_op(n, in, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_NON_MAX_SUPPRESSION) {
        if (try_gpu_graph_nms(n, o, &g_cuda_f32_ops)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_INTERPOLATE_1D) {
        int channels;
        int input_length;
        int output_length;
        if (!in || in->ndim != 2 || o->ndim != 2) return 0;
        materialize_tensor_f32(in);
        channels = in->shape[in->ndim - 2];
        input_length = in->shape[in->ndim - 1];
        output_length = o->shape[o->ndim - 1];
        if (in->dtype == T_F32 && o->shape[0] == channels &&
            cuda_graph_interp1d_f32(in->data, o->data, channels,
                                    input_length, output_length)) {
            *node_backend = "cuda-graph";
            return 1;
        }
        return 0;
    }

    if ((op == VX_OP_CAST) || (op == VX_OP_RESHAPE) ||
        (op == VX_OP_FLATTEN) || (op == VX_OP_SQUEEZE) ||
        (op == VX_OP_UNSQUEEZE) || (op == VX_OP_IDENTITY) ||
        (op == VX_OP_DROPOUT)) {
        T* source = in ? in : node_data_input(n, NULL);
#if VOLVOXAI_ENABLE_TRAINING
        if ((op == VX_OP_DROPOUT) && g_native_training_mode) {
            NativeDropoutParams dropout;
            if (!source || source->dtype != T_F32 ||
                source->numel != o->numel || o->numel <= 0 ||
                native_dropout_params(n, idx, &dropout) != 0) return 0;
            materialize_tensor_f32(source);
            if (cuda_graph_dropout_f32(source->data, o->data, o->numel,
                                       dropout.threshold, dropout.seed,
                                       dropout.counter, dropout.scale)) {
                *node_backend = "cuda-graph";
                return 1;
            }
            return 0;
        }
#endif
        if (!source) return 0;
        materialize_tensor_f32(source);
        if (source->dtype == T_F32 && source->numel == o->numel && o->numel > 0 &&
            cuda_graph_copy_f32(source->data, o->data, o->numel)) {
            *node_backend = "cuda-graph-copy";
            return 1;
        }
        return 0;
    }

    if (op == VX_OP_CONCAT) {
        VxConcatInputs inputs;
        if (!concat_inputs_build(n, o, vx_node_param_i32(n, VX_NODE_PARAM_AXIS, 0),
                                 T_F32, &inputs)) return 0;
        int sigmoid = vx_node_param_bool(n, VX_NODE_PARAM_SIGMOID, 0);
        int ok = cuda_graph_concat_f32(inputs.f32, inputs.sizes, inputs.axes,
            inputs.count, o->data, o->shape[inputs.axis], inputs.inner, sigmoid);
        concat_inputs_clear(&inputs);
        if (ok) *node_backend = sigmoid ? "cuda-graph+sigmoid" : "cuda-graph";
        return ok;
    }

    return 0;
}
#endif
