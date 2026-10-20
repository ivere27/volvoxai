/* Compile graph/control lineage once. Each bind checks only the current
 * public leaf values; no device-produced value is read through host storage. */
_Static_assert(VX_PUBLIC_VALUE_PREDICATE_LIMIT * sizeof(VxPublicValuePredicate) <= 256u * 1024u,
               "public value predicate metadata exceeds its private bound");

static VxStatus vx_public_value_add(VxPublicValuePlan* plan,
                                    VxPublicValuePredicate predicate) {
    for (size_t i = 0; i < plan->count; ++i)
        if (!memcmp(&plan->predicates[i], &predicate, sizeof(predicate)))
            return VX_STATUS_OK;
    if (plan->count == VX_PUBLIC_VALUE_PREDICATE_LIMIT)
        return VX_STATUS_OUT_OF_MEMORY;
    if (plan->count == plan->capacity) {
        size_t capacity = plan->capacity ? plan->capacity * 2u : 8u;
        VxPublicValuePredicate* next = realloc(plan->predicates, capacity * sizeof(*next));
        if (!next) return VX_STATUS_OUT_OF_MEMORY;
        plan->predicates = next;
        plan->capacity = capacity;
    }
    plan->predicates[plan->count++] = predicate;
    return VX_STATUS_OK;
}

static uint32_t vx_public_value_tensor_index(const VxEngineState* state,
                                             const char* name) {
    const T* tensor = vx_native_gpu_validation_tensor_find(state, name);
    return tensor ? (uint32_t)(tensor - state->tensors) : UINT32_MAX;
}

static VxPublicValuePredicate vx_public_value_predicate(uint16_t kind,
        int64_t minimum, int64_t maximum, uint32_t bank_node) {
    VxPublicValuePredicate predicate;
    memset(&predicate, 0, sizeof(predicate));
    predicate.kind = kind;
    predicate.minimum = minimum;
    predicate.maximum = maximum;
    predicate.tensor = predicate.bound_tensor = UINT32_MAX;
    predicate.bank_node = bank_node;
    return predicate;
}

static VxStatus vx_public_value_i32_visit(const VxEngineState* state,
        VxPublicValuePlan* plan, const char* name,
        VxPublicValuePredicate predicate, int before_node, int depth,
        unsigned char* visited) {
    if (!state || !name || before_node < 0 || before_node > state->node_count ||
        depth < 0 || depth > state->node_count)
        return VX_STATUS_INVALID_GRAPH;
    uint32_t index = vx_public_value_tensor_index(state, name);
    if (index == UINT32_MAX || state->tensors[index].dtype != T_I32)
        return VX_STATUS_INVALID_GRAPH;
    const T* tensor = &state->tensors[index];
    if (tensor->is_graph_input) {
        if (visited[index]) return VX_STATUS_OK;
        visited[index] = 1;
        predicate.tensor = index;
        return vx_public_value_add(plan, predicate);
    }
    const Node* producer = vx_native_gpu_validation_node_producer(state, name);
    if (!producer) return VX_STATUS_OK;
    int producer_index = vx_native_gpu_validation_node_index(state, producer);
    if (producer_index < 0 || producer_index >= before_node)
        return VX_STATUS_INVALID_GRAPH;
    if (visited[index]) return VX_STATUS_OK;
    visited[index] = 1;
    if (producer->operator_kind == VX_OP_WHERE) {
        const char* left = vx_native_gpu_validation_node_input_name(producer, VX_PORT_X);
        const char* right = vx_native_gpu_validation_node_input_name(producer, VX_PORT_Y);
        if (!left) left = vx_native_gpu_validation_node_input_name(producer, VX_PORT_A);
        if (!right) right = vx_native_gpu_validation_node_input_name(producer, VX_PORT_B);
        VxStatus status = vx_public_value_i32_visit(state, plan, left,
            predicate, producer_index, depth + 1, visited);
        return status == VX_STATUS_OK ? vx_public_value_i32_visit(state, plan,
            right, predicate, producer_index, depth + 1, visited) : status;
    }
    if (producer->operator_kind == VX_OP_CLIP || producer->operator_kind == VX_OP_N_GRAM_IDS ||
        producer->operator_kind == VX_OP_QSA_INDEXER || producer->operator_kind == VX_OP_SPARSE_INDEXER || producer->operator_kind == VX_OP_ARG_MAX ||
        producer->operator_kind == VX_OP_Q_ARG_MAX || producer->operator_kind == VX_OP_EQUAL ||
        producer->operator_kind == VX_OP_GREATER_OR_EQUAL || producer->operator_kind == VX_OP_NOT)
        return VX_STATUS_OK;
    if (producer->operator_kind == VX_OP_CAST) {
        const char* source_name = vx_native_gpu_validation_node_data_name(producer);
        const T* source = vx_native_gpu_validation_tensor_find(state, source_name);
        if (!source) return VX_STATUS_INVALID_GRAPH;
        if (source->dtype == T_U8 || source->dtype == T_I8) return VX_STATUS_OK;
        return vx_public_value_i32_visit(state, plan, source_name,
            predicate, producer_index, depth + 1, visited);
    }
    if (vx_native_gpu_value_preserving_node(producer))
        return vx_public_value_i32_visit(state, plan,
            vx_native_gpu_validation_node_data_name(producer), predicate,
            producer_index, depth + 1, visited);
    return VX_STATUS_INVALID_GRAPH;
}

static VxStatus vx_public_value_i32_compile(const VxEngineState* state,
        VxPublicValuePlan* plan, const char* name,
        VxPublicValuePredicate predicate, int before_node, int depth) {
    if (!state || state->tensor_count <= 0) return VX_STATUS_INVALID_GRAPH;
    /* A shared Where DAG visits each tensor once for this range, preventing
     * exponentially repeated proof walks. This cold scratch is graph-bounded. */
    unsigned char* visited = calloc((size_t)state->tensor_count, 1);
    if (!visited) return VX_STATUS_OUT_OF_MEMORY;
    VxStatus status = vx_public_value_i32_visit(state, plan, name, predicate,
        before_node, depth, visited);
    free(visited);
    return status;
}

static VxStatus vx_public_value_f32_compile(const VxEngineState* state,
        VxPublicValuePlan* plan, const char* name,
        VxPublicValuePredicate predicate, int before_node) {
    const char* public_name = NULL;
    int kind = vx_native_gpu_public_f32_source(state, name, &public_name, before_node, 0);
    if (!kind) return VX_STATUS_INVALID_GRAPH;
    if (kind != 1) return VX_STATUS_OK;
    predicate.tensor = vx_public_value_tensor_index(state, public_name);
    return predicate.tensor != UINT32_MAX ? vx_public_value_add(plan, predicate)
                                          : VX_STATUS_INVALID_GRAPH;
}

static VxStatus vx_public_value_shape_bound(VxPublicValuePlan* plan,
                                            VxPublicValuePredicate predicate) {
    predicate.kind = VX_PUBLIC_VALUE_SHAPE_BOUND;
    return vx_public_value_add(plan, predicate);
}

static VxStatus vx_public_value_plan_compile(VxExecutionContext* context,
                                             VxPublicValuePlan* plan) {
    const VxEngineState* state = context->engine_state;
    const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(
        context->compiled->model->logical_graph, "nodes");
    if (!state || !cJSON_IsArray(nodes)) return VX_STATUS_INVALID_GRAPH;
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* op_item = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
        VxOperatorKind op = vx_operator_kind_from_name(cJSON_GetStringValue(op_item));
        const VxDeclaredTensor* logical_output = vx_native_gpu_node_output(
            node, context->logical_tensors, context->logical_tensor_count);
        const Node* current = logical_output
            ? vx_native_gpu_validation_node_producer(state, logical_output->name) : NULL;
        int node_index = vx_native_gpu_validation_node_index(state, current);
        if (!op || !cJSON_IsObject(inputs) || node_index < 0)
            return VX_STATUS_INVALID_GRAPH;
        VxPublicValuePredicate predicate = vx_public_value_predicate(
            VX_PUBLIC_VALUE_I32_RANGE, 0, INT32_MAX, UINT32_MAX);
        VxStatus status = VX_STATUS_OK;
        if (op == VX_OP_N_GRAM_IDS || op == VX_OP_SPARSE_ATTENTION || op == VX_OP_EMBEDDING_SHARD) {
            const char* name = vx_native_gpu_validation_node_input_name(current,
                op == VX_OP_N_GRAM_IDS ? VX_PORT_TOKEN_IDS :
                op == VX_OP_EMBEDDING_SHARD ? VX_PORT_INPUT : VX_PORT_POSITION_IDS);
            if (!name && op == VX_OP_SPARSE_ATTENTION) continue;
            if (op != VX_OP_SPARSE_ATTENTION)
                predicate.maximum = vx_node_param_i32(current, VX_NODE_PARAM_VOCAB_SIZE, 0);
            if (predicate.maximum <= 0) return VX_STATUS_INVALID_GRAPH;
            status = vx_public_value_i32_compile(state, plan, name, predicate, node_index, 0);
        } else if (op == VX_OP_ROPE) {
            const cJSON* reference = cJSON_GetObjectItemCaseSensitive(inputs, "position_ids");
            if (!reference) continue;
            if (!cJSON_IsString(reference) || !reference->valuestring ||
                vx_native_gpu_validation_node_producer(state, reference->valuestring))
                return VX_STATUS_INVALID_GRAPH;
            predicate.maximum = (int64_t)INT32_MAX + 1;
            status = vx_public_value_i32_compile(state, plan, reference->valuestring,
                predicate, node_index, 0);
        } else if (op == VX_OP_GATHER || op == VX_OP_GATHER_ELEMENTS) {
            const cJSON* data = cJSON_GetObjectItemCaseSensitive(inputs, "input");
            const cJSON* ids = cJSON_GetObjectItemCaseSensitive(inputs, "indices");
            if (!cJSON_IsString(data) || !data->valuestring ||
                !cJSON_IsString(ids) || !ids->valuestring) return VX_STATUS_INVALID_GRAPH;
            predicate.bound_tensor = vx_public_value_tensor_index(state, data->valuestring);
            if (predicate.bound_tensor == UINT32_MAX) return VX_STATUS_INVALID_GRAPH;
            int rank = state->tensors[predicate.bound_tensor].ndim;
            int64_t axis;
            if (!rank || !vx_native_gpu_integer_parameter(current->params,
                    "axis", 0, -(int64_t)rank, (int64_t)rank - 1, &axis))
                return VX_STATUS_INVALID_GRAPH;
            predicate.axis = (int32_t)axis;
            predicate.bank_node = (uint32_t)node_index;
            predicate.bound_kind = VX_PUBLIC_BOUND_RESOLVED_AXIS;
            predicate.bound_failure = VX_STATUS_INVALID_ARGUMENT;
            predicate.negative_minimum = 1;
            status = vx_public_value_shape_bound(plan, predicate);
            if (status == VX_STATUS_OK)
                status = vx_public_value_i32_compile(state, plan, ids->valuestring,
                    predicate, node_index, 0);
        } else if (op == VX_OP_EMBEDDING || op == VX_OP_Q_EMBEDDING) {
            const cJSON* ids = cJSON_GetObjectItemCaseSensitive(inputs, "input");
            const cJSON* weight = cJSON_GetObjectItemCaseSensitive(inputs, "weight");
            if (!cJSON_IsString(ids) || !ids->valuestring || !cJSON_IsString(weight) ||
                !weight->valuestring) return VX_STATUS_INVALID_GRAPH;
            predicate.bound_tensor = vx_public_value_tensor_index(state, weight->valuestring);
            if (predicate.bound_tensor == UINT32_MAX) return VX_STATUS_INVALID_ARGUMENT;
            predicate.bound_kind = VX_PUBLIC_BOUND_CURRENT_AXIS;
            predicate.bound_failure = VX_STATUS_INVALID_ARGUMENT;
            status = vx_public_value_shape_bound(plan, predicate);
            if (status == VX_STATUS_OK)
                status = vx_public_value_i32_compile(state, plan, ids->valuestring,
                    predicate, node_index, 0);
        } else if (op == VX_OP_Q_GROUP_NORM || op == VX_OP_Q_LAYER_NORM) {
            const char* roles[] = {"weight", "bias"};
            predicate.kind = VX_PUBLIC_VALUE_F32_FINITE;
            for (size_t role = 0; role < 2 && status == VX_STATUS_OK; ++role) {
                const cJSON* reference = cJSON_GetObjectItemCaseSensitive(inputs, roles[role]);
                if (!reference && role == 0)
                    reference = cJSON_GetObjectItemCaseSensitive(inputs, "scale");
                if (!cJSON_IsString(reference) || !reference->valuestring)
                    return VX_STATUS_INVALID_GRAPH;
                status = vx_public_value_f32_compile(state, plan,
                    reference->valuestring, predicate, node_index);
            }
        } else if (op == VX_OP_MOE_ROUTER) {
            const char* hash_name = vx_native_gpu_validation_node_input_name(current, VX_PORT_HASH_INDICES);
            if (hash_name) {
                predicate.bound_tensor = vx_public_value_tensor_index(state, hash_name);
                if (predicate.bound_tensor == UINT32_MAX) return VX_STATUS_INVALID_GRAPH;
                predicate.bound_kind = VX_PUBLIC_BOUND_CURRENT_AXIS;
                predicate.bound_failure = VX_STATUS_INVALID_GRAPH;
                predicate.required_rank = 2;
                status = vx_public_value_shape_bound(plan, predicate);
                if (status == VX_STATUS_OK)
                    status = vx_public_value_i32_compile(state, plan,
                        vx_native_gpu_validation_node_input_name(current, VX_PORT_TOKEN_IDS),
                        predicate, node_index, 0);
                if (status != VX_STATUS_OK) return status;
            }
            const VxPortKind roles[] = {VX_PORT_BIAS, VX_PORT_CORRECTION_BIAS};
            predicate = vx_public_value_predicate(VX_PUBLIC_VALUE_F32_FINITE, 0, 0, UINT32_MAX);
            for (size_t role = 0; role < 2 && status == VX_STATUS_OK; ++role) {
                const char* name = vx_native_gpu_validation_node_input_name(current, roles[role]);
                if (name) status = vx_public_value_f32_compile(state, plan, name, predicate, node_index);
            }
        } else if (op == VX_OP_MOE_LINEAR || op == VX_OP_MOE_GATED_MLP) {
            const char* weight = vx_native_gpu_validation_node_input_name(current,
                op == VX_OP_MOE_GATED_MLP ? VX_PORT_GATE_WEIGHT : VX_PORT_EXPERT_WEIGHT);
            predicate.bound_tensor = vx_public_value_tensor_index(state, weight);
            if (predicate.bound_tensor == UINT32_MAX) return VX_STATUS_INVALID_GRAPH;
            predicate.bound_kind = VX_PUBLIC_BOUND_CURRENT_AXIS;
            predicate.bank_node = (uint32_t)node_index;
            predicate.required_rank = 3;
            predicate.bound_failure = VX_STATUS_INVALID_GRAPH;
            status = vx_public_value_shape_bound(plan, predicate);
            for (int role = 0; role < 2 && status == VX_STATUS_OK; ++role) {
                VxPublicValuePredicate route = predicate;
                route.kind = VX_PUBLIC_VALUE_F32_ROUTE;
                if (role) { route.bound_kind = VX_PUBLIC_BOUND_STATIC; route.maximum = 0; }
                status = vx_public_value_f32_compile(state, plan,
                    vx_native_gpu_validation_node_input_name(current,
                        role ? VX_PORT_ROUTE_WEIGHTS : VX_PORT_ROUTE_INDICES), route, node_index);
            }
        }
        if (status != VX_STATUS_OK) return status;
    }
    return VX_STATUS_OK;
}

static VxStatus vx_native_gpu_value_plan_prepare(VxExecutionContext* context) {
    if (!context || !vx_builtin_gpu_backend(context->compiled->builtin_backend))
        return VX_STATUS_OK;
    const VxEngineState* state = context->engine_state;
    if (!state) return VX_STATUS_INVALID_GRAPH;
    VxPublicValuePlan* old = &context->public_value_plan;
    const void* graph = context->compiled->model->logical_graph;
    if (old->valid && old->model_generation == state->model_generation &&
        old->graph == graph && old->nodes == state->nodes && old->tensors == state->tensors &&
        old->node_count == state->node_count && old->tensor_count == state->tensor_count)
        return VX_STATUS_OK;
    VxPublicValuePlan candidate = {0};
    VxStatus status = vx_public_value_plan_compile(context, &candidate);
    size_t bytes = candidate.capacity * sizeof(*candidate.predicates);
    if (status == VX_STATUS_OK && bytes > UINT64_MAX - context->allocated_bytes)
        status = VX_STATUS_OUT_OF_MEMORY;
    if (status != VX_STATUS_OK) { free(candidate.predicates); return status; }
    candidate.graph = graph; candidate.nodes = state->nodes; candidate.tensors = state->tensors;
    candidate.node_count = state->node_count; candidate.tensor_count = state->tensor_count;
    candidate.model_generation = state->model_generation; candidate.valid = 1;
    vx_public_value_plan_clear(context);
    context->public_value_plan = candidate;
    context->allocated_bytes += bytes;
    return VX_STATUS_OK;
}

static VxStatus vx_public_value_bounds(const VxExecutionContext* context,
        const VxPublicValuePredicate* predicate,
        const VolvoxAIEngineResolvedTensor* resolved, size_t resolved_count,
        int64_t* minimum, int64_t* maximum) {
    *minimum = predicate->minimum; *maximum = predicate->maximum;
    if (predicate->bound_kind == VX_PUBLIC_BOUND_STATIC) return VX_STATUS_OK;
    const T* tensor = &context->engine_state->tensors[predicate->bound_tensor];
    const VolvoxAIEngineResolvedTensor* shape = predicate->bound_kind == VX_PUBLIC_BOUND_RESOLVED_AXIS
        ? vx_resolved_tensor_find(resolved, resolved_count, tensor->name) : NULL;
    int rank = shape ? shape->rank : tensor->ndim;
    if (rank <= 0 || (predicate->required_rank && rank != predicate->required_rank))
        return predicate->bound_failure;
    int axis = predicate->axis < 0 ? predicate->axis + rank : predicate->axis;
    if (axis < 0 || axis >= rank) return VX_STATUS_INVALID_GRAPH;
    *maximum = shape ? shape->shape[axis] : tensor->shape[axis];
    if (predicate->bank_node != UINT32_MAX) {
        const Node* bank = &context->engine_state->nodes[predicate->bank_node];
        if (bank->resident_slot_domain) *maximum = bank->resident_slot_domain;
    }
    if (*maximum <= 0) return predicate->bound_failure;
    if (predicate->negative_minimum) *minimum = -*maximum;
    return VX_STATUS_OK;
}

static VxStatus vx_native_gpu_preflight_public_values(VxExecutionContext* context,
        const VxStagedBindingBatch* staged,
        const VolvoxAIEngineResolvedTensor* resolved, size_t resolved_count) {
    if (!context || !staged || !vx_builtin_gpu_backend(context->compiled->builtin_backend))
        return VX_STATUS_OK;
    VxStatus status = vx_native_gpu_value_plan_prepare(context);
    if (status != VX_STATUS_OK) return status;
    const VxEngineState* state = context->engine_state;
    for (size_t i = 0; i < context->public_value_plan.count; ++i) {
        const VxPublicValuePredicate* predicate = &context->public_value_plan.predicates[i];
        int64_t minimum, maximum;
        status = vx_public_value_bounds(context, predicate, resolved, resolved_count, &minimum, &maximum);
        if (status != VX_STATUS_OK) return status;
        if (predicate->kind == VX_PUBLIC_VALUE_SHAPE_BOUND) continue;
        const char* name = state->tensors[predicate->tensor].name;
        const Node* bank = predicate->bank_node == UINT32_MAX ? NULL : &state->nodes[predicate->bank_node];
        int valid = predicate->kind == VX_PUBLIC_VALUE_I32_RANGE
            ? vx_native_gpu_preflight_i32_binding(staged, name, minimum, maximum, bank)
            : predicate->kind == VX_PUBLIC_VALUE_F32_FINITE
                ? vx_native_gpu_preflight_finite_binding(staged, name)
                : vx_moe_preflight_route_binding(staged, name, (uint32_t)maximum, bank);
        if (!valid) return VX_STATUS_INVALID_ARGUMENT;
    }
    return VX_STATUS_OK;
}
