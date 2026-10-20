/* Previous production preflight, retained only as a behavioral test oracle. */
static const VxTensorBinding* reference_vx_staged_binding_find(
        const VxStagedBindingBatch* staged, const char* name) {
    if (!staged || !name) return NULL;
    for (size_t index = 0; index < staged->count; index++)
        if (!strcmp(staged->bindings[index].name, name))
            return &staged->bindings[index];
    return NULL;
}

static int reference_vx_native_gpu_public_f32_source(
        const VxEngineState* state,
        const char* tensor_name,
        const char** public_name,
        int before_node,
        int depth) {
    const T* tensor;
    const Node* producer;
    if (!state || !tensor_name || !public_name || before_node < 0 ||
        before_node > state->node_count || depth < 0 ||
        depth > state->node_count)
        return 0;
    tensor = vx_native_gpu_validation_tensor_find(state, tensor_name);
    if (!tensor || tensor->dtype != T_F32) return 0;
    if (tensor->is_graph_input) {
        *public_name = tensor->name;
        return 1;
    }
    producer = vx_native_gpu_validation_node_producer(state, tensor_name);
    if (!producer) return 2;
    {
        int producer_index = vx_native_gpu_validation_node_index(
            state, producer);
        if (producer_index < 0 || producer_index >= before_node) return 0;
        before_node = producer_index;
    }
    if (producer->operator_kind == VX_OP_MOE_ROUTER) return 2;
    if (vx_native_gpu_value_preserving_node(producer))
        return reference_vx_native_gpu_public_f32_source(
            state, vx_native_gpu_validation_node_data_name(producer),
            public_name, before_node, depth + 1);
    return 0;
}

static const VolvoxAIEngineResolvedTensor* reference_vx_resolved_tensor_find(
        const VolvoxAIEngineResolvedTensor* resolved,
        size_t resolved_count,
        const char* name) {
    if (!resolved || !name) return NULL;
    for (size_t index = 0; index < resolved_count; index++)
        if (!strcmp(resolved[index].name, name)) return &resolved[index];
    return NULL;
}

static int reference_vx_native_gpu_preflight_i32_binding(
        const VxStagedBindingBatch* staged,
        const char* public_name,
        int64_t minimum,
        int64_t maximum_exclusive,
        const Node* bank_node) {
    const VxTensorBinding* binding = reference_vx_staged_binding_find(
        staged, public_name);
    if (!binding) return 1; /* An unchanged decode-step input was preflighted. */
    if (binding->location != VX_MEMORY_HOST || binding->dtype != VX_DTYPE_I32 || !binding->data ||
        binding->byte_size % sizeof(int32_t) ||
        minimum >= maximum_exclusive)
        return 0;
    size_t count = binding->byte_size / sizeof(int32_t);
    for (size_t index = 0; index < count; index++) {
        int32_t value;
        memcpy(&value,
               (const unsigned char*)binding->data +
                   index * sizeof(value),
               sizeof(value));
        if ((int64_t)value < minimum ||
            (int64_t)value >= maximum_exclusive)
            return 0;
        if (bank_node && bank_node->resident_slot_rows) {
            int64_t slot = value < 0 ? (int64_t)value + bank_node->resident_slot_domain : value;
            if (slot < 0 || (uint64_t)slot >= bank_node->resident_slot_domain ||
                bank_node->resident_slot_rows[slot] == UINT32_MAX) return 0;
        }
    }
    return 1;
}

/* Inspect each original public leaf, never a device-produced intermediate.
 * Compilation has already checked immutable leaves and statically ranged
 * producers. Where needs both branches checked even if its condition would
 * select only one of them for the current request. */
static VxStatus reference_vx_native_gpu_preflight_i32_source(
        const VxEngineState* state,
        const VxStagedBindingBatch* staged,
        const char* tensor_name,
        int64_t minimum,
        int64_t maximum_exclusive,
        const Node* bank_node,
        int before_node,
        int depth) {
    const T* tensor;
    const Node* producer;
    if (!state || !staged || !tensor_name || minimum >= maximum_exclusive ||
        before_node < 0 || before_node > state->node_count ||
        depth < 0 || depth > state->node_count)
        return VX_STATUS_INVALID_GRAPH;
    tensor = vx_native_gpu_validation_tensor_find(state, tensor_name);
    if (!tensor || tensor->dtype != T_I32) return VX_STATUS_INVALID_GRAPH;
    if (tensor->is_graph_input)
        return reference_vx_native_gpu_preflight_i32_binding(staged, tensor->name,
            minimum, maximum_exclusive, bank_node)
            ? VX_STATUS_OK : VX_STATUS_INVALID_ARGUMENT;
    producer = vx_native_gpu_validation_node_producer(state, tensor_name);
    if (!producer) return VX_STATUS_OK;
    {
        int producer_index = vx_native_gpu_validation_node_index(state, producer);
        if (producer_index < 0 || producer_index >= before_node)
            return VX_STATUS_INVALID_GRAPH;
        before_node = producer_index;
    }
    if (producer->operator_kind == VX_OP_WHERE) {
        const char* left = vx_native_gpu_validation_node_input_name(producer, VX_PORT_X);
        const char* right = vx_native_gpu_validation_node_input_name(producer, VX_PORT_Y);
        if (!left) left = vx_native_gpu_validation_node_input_name(producer, VX_PORT_A);
        if (!right) right = vx_native_gpu_validation_node_input_name(producer, VX_PORT_B);
        VxStatus status = reference_vx_native_gpu_preflight_i32_source(state, staged, left,
            minimum, maximum_exclusive, bank_node, before_node, depth + 1);
        if (status != VX_STATUS_OK) return status;
        return reference_vx_native_gpu_preflight_i32_source(state, staged, right,
            minimum, maximum_exclusive, bank_node, before_node, depth + 1);
    }
    if ((producer->operator_kind == VX_OP_CLIP) ||
        (producer->operator_kind == VX_OP_N_GRAM_IDS) ||
        (producer->operator_kind == VX_OP_QSA_INDEXER) ||
        (producer->operator_kind == VX_OP_ARG_MAX) ||
        (producer->operator_kind == VX_OP_Q_ARG_MAX) ||
        (producer->operator_kind == VX_OP_EQUAL) ||
        (producer->operator_kind == VX_OP_GREATER_OR_EQUAL) ||
        (producer->operator_kind == VX_OP_NOT))
        return VX_STATUS_OK;
    if (producer->operator_kind == VX_OP_CAST) {
        const char* source_name = vx_native_gpu_validation_node_data_name(producer);
        const T* source = vx_native_gpu_validation_tensor_find(state, source_name);
        if (!source) return VX_STATUS_INVALID_GRAPH;
        if (source->dtype == T_U8 || source->dtype == T_I8) return VX_STATUS_OK;
        return reference_vx_native_gpu_preflight_i32_source(state, staged, source_name,
            minimum, maximum_exclusive, bank_node, before_node, depth + 1);
    }
    if (vx_native_gpu_value_preserving_node(producer))
        return reference_vx_native_gpu_preflight_i32_source(state, staged,
            vx_native_gpu_validation_node_data_name(producer),
            minimum, maximum_exclusive, bank_node, before_node, depth + 1);
    return VX_STATUS_INVALID_GRAPH;
}

static int reference_vx_native_gpu_preflight_finite_binding(
        const VxStagedBindingBatch* staged,
        const char* public_name) {
    const VxTensorBinding* binding = reference_vx_staged_binding_find(
        staged, public_name);
    if (!binding) return 1;
    if (binding->location != VX_MEMORY_HOST || binding->dtype != VX_DTYPE_F32 || !binding->data ||
        binding->byte_size % sizeof(float))
        return 0;
    size_t count = binding->byte_size / sizeof(float);
    for (size_t index = 0; index < count; index++) {
        float value;
        memcpy(&value,
               (const unsigned char*)binding->data +
                   index * sizeof(value),
               sizeof(value));
        if (!isfinite(value)) return 0;
    }
    return 1;
}

static int reference_vx_moe_preflight_route_binding(const VxStagedBindingBatch* staged,
    const char* name, uint32_t experts, const Node* consumer) {
    const VxTensorBinding* binding = reference_vx_staged_binding_find(staged, name);
    if (!binding) return 1;
    if (binding->location != VX_MEMORY_HOST || binding->dtype != VX_DTYPE_F32 || !binding->data || binding->byte_size % 4u) return 0;
    for (size_t i = 0; i < binding->byte_size / 4u; i++) {
        float value; memcpy(&value, (const unsigned char*)binding->data + i * 4u, 4u);
        if (!isfinite(value)) return 0;
        if (experts && (value < 0 || (double)value >= experts || value != (float)(uint32_t)value ||
            (consumer->resident_slot_rows && consumer->resident_slot_rows[(uint32_t)value] == UINT32_MAX))) return 0;
    }
    return 1;
}

/* Value predicates are graph-wide and run after canonical shape resolution
 * but before volvoxai_engine_commit_dynamic_shape copies an input, publishes a
 * plan, uploads a byte, or allows the first GPU dispatch. */
static VxStatus reference_vx_native_gpu_preflight_public_values(
        const VxExecutionContext* context,
        const VxStagedBindingBatch* staged,
        const VolvoxAIEngineResolvedTensor* resolved,
        size_t resolved_count) {
    const cJSON* nodes;
    const VxEngineState* state;
    if (!context || !staged ||
        !vx_builtin_gpu_backend(context->compiled->builtin_backend))
        return VX_STATUS_OK;
    state = context->engine_state;
    nodes = cJSON_GetObjectItemCaseSensitive(
        context->compiled->model->logical_graph, "nodes");
    if (!state || !cJSON_IsArray(nodes)) return VX_STATUS_INVALID_GRAPH;
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* op_item = cJSON_GetObjectItemCaseSensitive(
            node, "opType");
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        VxOperatorKind op = vx_operator_kind_from_name(cJSON_GetStringValue(op_item));
        const VxDeclaredTensor* logical_output = vx_native_gpu_node_output(
            node, context->logical_tensors, context->logical_tensor_count);
        const Node* validation_node = logical_output
            ? vx_native_gpu_validation_node_producer(
                  state, logical_output->name)
            : NULL;
        const int validation_node_index =
            vx_native_gpu_validation_node_index(state, validation_node);
        if (!op || !cJSON_IsObject(inputs) || validation_node_index < 0)
            return VX_STATUS_INVALID_GRAPH;
        if (op == VX_OP_N_GRAM_IDS || op == VX_OP_SPARSE_ATTENTION || op == VX_OP_EMBEDDING_SHARD) {
            const char* name = vx_native_gpu_validation_node_input_name(validation_node,
                op == VX_OP_N_GRAM_IDS ? VX_PORT_TOKEN_IDS : op == VX_OP_EMBEDDING_SHARD ? VX_PORT_INPUT : VX_PORT_POSITION_IDS);
            if (!name && op == VX_OP_SPARSE_ATTENTION) continue;
            int64_t maximum = op == VX_OP_N_GRAM_IDS || op == VX_OP_EMBEDDING_SHARD
                ? vx_node_param_i32(validation_node, VX_NODE_PARAM_VOCAB_SIZE, 0) : INT32_MAX;
            VxStatus status = reference_vx_native_gpu_preflight_i32_source(state, staged, name,
                0, maximum, NULL, validation_node_index, 0);
            if (status != VX_STATUS_OK) return status;
        } else if (op == VX_OP_ROPE) {
            const cJSON* positions_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "position_ids");
            if (!positions_ref) continue;
            if (!cJSON_IsString(positions_ref) || !positions_ref->valuestring ||
                vx_native_gpu_validation_node_producer(
                    state, positions_ref->valuestring))
                return VX_STATUS_INVALID_GRAPH;
            /* Every newly supplied position must pass before any upload. */
            VxStatus status = reference_vx_native_gpu_preflight_i32_source(state, staged,
                positions_ref->valuestring, 0, (int64_t)INT32_MAX + 1,
                NULL, validation_node_index, 0);
            if (status != VX_STATUS_OK) return status;
        } else if ((op == VX_OP_GATHER) || (op == VX_OP_GATHER_ELEMENTS)) {
            const cJSON* data_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "input");
            const cJSON* indices_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "indices");
            int64_t axis_size = 0;
            if (!cJSON_IsString(data_ref) || !data_ref->valuestring ||
                !cJSON_IsString(indices_ref) || !indices_ref->valuestring)
                return VX_STATUS_INVALID_GRAPH;
            {
                const VolvoxAIEngineResolvedTensor* data =
                    reference_vx_resolved_tensor_find(
                        resolved, resolved_count, data_ref->valuestring);
                const T* fixed = vx_native_gpu_validation_tensor_find(
                    state, data_ref->valuestring);
                int rank = data ? (int)data->rank : fixed ? fixed->ndim : 0;
                int64_t axis;
                if (!rank || !vx_native_gpu_integer_parameter(validation_node->params,
                        "axis", 0, -(int64_t)rank, (int64_t)rank - 1, &axis))
                    return VX_STATUS_INVALID_GRAPH;
                if (axis < 0) axis += rank;
                axis_size = data ? data->shape[axis] : fixed->shape[axis];
                if (validation_node->resident_slot_domain)
                    axis_size = validation_node->resident_slot_domain;
            }
            if (axis_size <= 0) return VX_STATUS_INVALID_ARGUMENT;
            VxStatus status = reference_vx_native_gpu_preflight_i32_source(state, staged,
                indices_ref->valuestring, -axis_size, axis_size, validation_node,
                validation_node_index, 0);
            if (status != VX_STATUS_OK) return status;
        } else if ((op == VX_OP_EMBEDDING) ||
                   (op == VX_OP_Q_EMBEDDING)) {
            const cJSON* ids_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "input");
            const cJSON* weight_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "weight");
            const T* weight;
            if (!cJSON_IsString(ids_ref) || !ids_ref->valuestring ||
                !cJSON_IsString(weight_ref) || !weight_ref->valuestring)
                return VX_STATUS_INVALID_GRAPH;
            weight = vx_native_gpu_validation_tensor_find(
                state, weight_ref->valuestring);
            if (!weight || weight->ndim <= 0 || weight->shape[0] <= 0)
                return VX_STATUS_INVALID_ARGUMENT;
            VxStatus status = reference_vx_native_gpu_preflight_i32_source(state, staged,
                ids_ref->valuestring, 0, weight->shape[0], NULL,
                validation_node_index, 0);
            if (status != VX_STATUS_OK) return status;
        } else if ((op == VX_OP_Q_GROUP_NORM) ||
                   (op == VX_OP_Q_LAYER_NORM)) {
            static const char* const roles[] = {"weight", "bias"};
            for (size_t role = 0; role < 2u; role++) {
                const cJSON* reference = cJSON_GetObjectItemCaseSensitive(
                    inputs, roles[role]);
                if (!reference && role == 0)
                    reference = cJSON_GetObjectItemCaseSensitive(inputs, "scale");
                const char* public_name = NULL;
                int source_kind;
                if (!cJSON_IsString(reference) || !reference->valuestring)
                    return VX_STATUS_INVALID_GRAPH;
                source_kind = reference_vx_native_gpu_public_f32_source(
                    state, reference->valuestring, &public_name,
                    validation_node_index, 0);
                if (!source_kind) return VX_STATUS_INVALID_GRAPH;
                if (source_kind == 1 &&
                    !reference_vx_native_gpu_preflight_finite_binding(
                        staged, public_name))
                    return VX_STATUS_INVALID_ARGUMENT;
            }
        } else if (op == VX_OP_MOE_ROUTER) {
            const char* token_name=vx_native_gpu_validation_node_input_name(validation_node,VX_PORT_TOKEN_IDS);
            const char* hash_name=vx_native_gpu_validation_node_input_name(validation_node,VX_PORT_HASH_INDICES);
            if (hash_name) {
                const T* hash=vx_native_gpu_validation_tensor_find(state,hash_name);
                if (!hash || hash->ndim != 2) return VX_STATUS_INVALID_GRAPH;
                VxStatus status = reference_vx_native_gpu_preflight_i32_source(state, staged,
                    token_name, 0, hash->shape[0], NULL, validation_node_index, 0);
                if (status != VX_STATUS_OK) return status;
            }
            const VxPortKind ports[]={VX_PORT_BIAS,VX_PORT_CORRECTION_BIAS};
            for (int i=0;i<2;++i) {
                const char* name=vx_native_gpu_validation_node_input_name(validation_node,ports[i]);
                if (!name) continue;
                const char* public_name=NULL;
                int kind=reference_vx_native_gpu_public_f32_source(state,name,&public_name,validation_node_index,0);
                if (!kind) return VX_STATUS_INVALID_GRAPH;
                if (kind==1 && !reference_vx_native_gpu_preflight_finite_binding(staged,public_name)) return VX_STATUS_INVALID_ARGUMENT;
            }
        } else if (op == VX_OP_MOE_LINEAR || op == VX_OP_MOE_GATED_MLP) {
            const T* weight = vx_native_gpu_validation_tensor_find(state,
                vx_native_gpu_validation_node_input_name(validation_node, op == VX_OP_MOE_GATED_MLP ? VX_PORT_GATE_WEIGHT : VX_PORT_EXPERT_WEIGHT));
            if (!weight || weight->ndim != 3) return VX_STATUS_INVALID_GRAPH;
            uint32_t experts = validation_node->resident_slot_domain
                ? validation_node->resident_slot_domain : (uint32_t)weight->shape[0];
            for (int role = 0; role < 2; role++) {
                const char* name = vx_native_gpu_validation_node_input_name(validation_node,
                    role ? VX_PORT_ROUTE_WEIGHTS : VX_PORT_ROUTE_INDICES);
                const char* public_name = NULL;
                int kind = reference_vx_native_gpu_public_f32_source(state, name, &public_name, validation_node_index, 0);
                if (!kind) return VX_STATUS_INVALID_GRAPH;
                if (kind == 1 && !reference_vx_moe_preflight_route_binding(staged,
                        public_name, role ? 0 : experts, validation_node)) return VX_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return VX_STATUS_OK;
}

