/* WebGPU physical admission follows the canonical, whole-domain shape proof.
 * Shape relations belong to that proof; this layer adds shader invariants,
 * host/static value preflight and maximum device/resource descriptors. */
#if VOLVOXAI_ENABLE_WEBGPU
/* Sparse banks need membership, not just an integer interval. Preserve the
 * source through view/copy nodes; arbitrary device producers have no proof of
 * membership in this particular resident subset. */
static int vx_webgpu_bank_index_source_proven(const VxEngineState* state,
    const char* name, const Node* consumer, int before, int depth) {
    if (!state || !name || !consumer || depth > state->node_count ||
        !consumer->resident_slot_rows || !consumer->resident_slot_domain) return 0;
    const T* tensor = vx_native_gpu_validation_tensor_find(state, name);
    if (!tensor || tensor->dtype != T_I32 || !tensor->data) return 0;
    if (tensor->is_graph_input) return 1;
    const Node* producer = vx_native_gpu_validation_node_producer(state, name);
    if (producer) {
        int index = vx_native_gpu_validation_node_index(state, producer);
        if (index < 0 || index >= before || !vx_native_gpu_value_preserving_node(producer)) return 0;
        return vx_webgpu_bank_index_source_proven(state,
            vx_native_gpu_validation_node_data_name(producer), consumer, index, depth + 1);
    }
    for (long i = 0; i < tensor->numel; i++) {
        int32_t raw; memcpy(&raw, (const unsigned char*)tensor->data + (size_t)i * 4u, 4u);
        int64_t slot = raw < 0 ? (int64_t)raw + consumer->resident_slot_domain : raw;
        if (slot < 0 || (uint64_t)slot >= consumer->resident_slot_domain ||
            consumer->resident_slot_rows[slot] == UINT32_MAX) return 0;
    }
    return 1;
}

static int vx_webgpu_fixed_axis(const VxDeclaredTensor* tensor, int axis) {
    int64_t extent;
    if (!tensor) return 0;
    if (axis < 0) axis += tensor->rank;
    return axis >= 0 && (uint32_t)axis < tensor->rank &&
        vx_declared_dimension_fixed_value(tensor, (uint32_t)axis, &extent) && extent > 0;
}

static int vx_webgpu_fixed_shape(const VxDeclaredTensor* tensor) {
    if (!tensor) return 0;
    for (uint32_t axis = 0; axis < tensor->rank; axis++)
        if (!vx_webgpu_fixed_axis(tensor, (int)axis)) return 0;
    return 1;
}

static int vx_webgpu_route_source_proven(const VxEngineState* state, const char* name,
    uint32_t experts, const Node* consumer, int before, int depth) {
    if (!name || depth > state->node_count) return 0;
    const T* tensor = vx_native_gpu_validation_tensor_find(state, name);
    if (!tensor || tensor->dtype != T_F32 || !tensor->data) return 0;
    if (tensor->is_graph_input) return 1;
    const Node* producer = vx_native_gpu_validation_node_producer(state, name);
    if (producer) {
        int index = vx_native_gpu_validation_node_index(state, producer);
        if (index < 0 || index >= before) return 0;
        if (vx_native_gpu_value_preserving_node(producer))
            return vx_webgpu_route_source_proven(state,
                vx_native_gpu_validation_node_data_name(producer), experts, consumer, index, depth + 1);
        if ((producer->operator_kind != VX_OP_MOE_ROUTER) || (experts && consumer->resident_slot_domain)) return 0;
        /* Router indices are generated from its finite expert loop. Router
         * weights are ordinary computed F32 values (training checks overflow);
         * their host bootstrap bytes are never used as execution evidence. */
        VxPortKind role = experts ? VX_PORT_INDICES : VX_PORT_WEIGHTS;
        int matches = 0;
        for (int port = 0; port < producer->nout; port++)
            if (!strcmp(producer->outs[port].name, name) && producer->outs[port].port == role) matches = 1;
        const T* weight = vx_native_gpu_validation_tensor_find(state,
            vx_native_gpu_validation_node_input_name(producer, VX_PORT_WEIGHT));
        return matches && weight && weight->ndim == 2 &&
            (!experts || (uint32_t)weight->shape[1] <= experts);
    }
    for (long i = 0; i < tensor->numel; i++) {
        float value; memcpy(&value, (const unsigned char*)tensor->data + (size_t)i * 4u, 4u);
        if (!isfinite(value)) return 0;
        if (experts && (value < 0 || (double)value >= experts || value != (float)(uint32_t)value ||
            (consumer->resident_slot_rows && consumer->resident_slot_rows[(uint32_t)value] == UINT32_MAX))) return 0;
    }
    return 1;
}

static int vx_webgpu_node_domain_proven(const cJSON* node,
    const VxDeclaredTensor* tensors, size_t tensor_count, VxEngineState* validation) {
    const cJSON* op_item = cJSON_GetObjectItemCaseSensitive(node, "opType");
    const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    VxOperatorKind op = vx_operator_kind_from_name(cJSON_GetStringValue(op_item));
    const VxDeclaredTensor* output = vx_native_gpu_node_output(node, tensors, tensor_count);
    VxDeclaredTensor input_fixed = {0}, weight_fixed = {0};
    const VxDeclaredTensor* input = vx_native_gpu_node_data_domain(
        node, tensors, tensor_count, validation, &input_fixed);
    const VxDeclaredTensor* weight = vx_native_gpu_node_input_domain(
        node, "weight", tensors, tensor_count, validation, &weight_fixed);
    const Node* runtime_node = output
        ? vx_native_gpu_validation_node_producer(validation, output->name) : NULL;
    int node_index = vx_native_gpu_validation_node_index(validation, runtime_node);
    if (!op || !output || node_index < 0) return 0;
    if (input && ((op == VX_OP_IDENTITY) || (op == VX_OP_RESHAPE) ||
        (op == VX_OP_FLATTEN) || (op == VX_OP_SQUEEZE) || (op == VX_OP_UNSQUEEZE))) {
        const cJSON* quantization = cJSON_GetObjectItemCaseSensitive(validation->graph_root, "quantization");
        const cJSON* descriptors = cJSON_GetObjectItemCaseSensitive(quantization, "tensors");
        const cJSON* descriptor = cJSON_GetObjectItemCaseSensitive(descriptors, input->name);
        const cJSON* scheme = cJSON_GetObjectItemCaseSensitive(descriptor, "scheme");
        if (cJSON_IsString(scheme) && (vx_generated_quantization_kind(scheme->valuestring) == VX_QUANTIZATION_PER_AXIS)) {
            /* This storage-copy route preserves the original axis layout;
             * it does not remap per-axis metadata across a shape change. */
            if (input->rank != output->rank) return 0;
            for (uint32_t axis = 0; axis < input->rank; axis++)
                if (!vx_declared_dimensions_provably_equal(input, axis, output, axis)) return 0;
        }
    }

    /* These widths select shader algorithms, affine tables or fixed banks.
     * Rows, spatial extents, reductions and typed copy domains remain bounded
     * dynamic. No F32-only operator list is imposed on the physical planner. */
    if ((op == VX_OP_MATMUL) || (op == VX_OP_LINEAR) ||
        (op == VX_OP_GEMM) || (op == VX_OP_Q_LINEAR)) {
        if (!vx_webgpu_fixed_axis(input, -1) || !vx_webgpu_fixed_axis(output, -1) ||
            !vx_webgpu_fixed_shape(weight)) return 0;
    } else if ((op == VX_OP_LAYER_NORM) || (op == VX_OP_RMS_NORM) ||
               (op == VX_OP_Q_LAYER_NORM)) {
        if (!vx_webgpu_fixed_axis(input, -1)) return 0;
    } else if ((op == VX_OP_GROUP_NORM) || (op == VX_OP_Q_GROUP_NORM)) {
        if (!vx_webgpu_fixed_axis(input, 3)) return 0;
    } else if ((op == VX_OP_CONV_1D) || (op == VX_OP_CONV_2D) ||
               (op == VX_OP_CONV_TRANSPOSE_2D) || (op == VX_OP_Q_CONV_2D)) {
        if (!vx_webgpu_fixed_axis(output, -1) ||
            ((op == VX_OP_Q_CONV_2D) && !vx_webgpu_fixed_shape(weight))) return 0;
    } else if ((op == VX_OP_SDPA) || (op == VX_OP_CROSS_SDPA) ||
               (op == VX_OP_Q_SDPA) || (op == VX_OP_CROSS_ATTENTION)) {
        VxDeclaredTensor query_fixed = {0};
        const VxDeclaredTensor* query = vx_native_gpu_node_input_domain(
            node, "q", tensors, tensor_count, validation, &query_fixed);
        if (!query) query = vx_native_gpu_node_input_domain(
            node, "qkv", tensors, tensor_count, validation, &query_fixed);
        if (!query) query = input;
        if (!vx_webgpu_fixed_axis(query, -1)) return 0;
    } else if (op == VX_OP_MOE_ROUTER) {
        if (!vx_webgpu_fixed_axis(weight, -1) ||
            !vx_webgpu_fixed_axis(output, -1)) return 0;
    } else if (op == VX_OP_MOE_LINEAR) {
        VxDeclaredTensor expert_fixed = {0};
        const VxDeclaredTensor* expert = vx_native_gpu_node_input_domain(
            node, "expert_weight", tensors, tensor_count, validation, &expert_fixed);
        if (!vx_webgpu_fixed_axis(expert, 0) || !vx_webgpu_fixed_axis(output, -1)) return 0;
        uint32_t experts = runtime_node->resident_slot_domain
            ? runtime_node->resident_slot_domain : (uint32_t)expert->minimums[0];
        if (!vx_webgpu_route_source_proven(validation,
                vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_ROUTE_INDICES), experts,
                runtime_node, node_index, 0) ||
            !vx_webgpu_route_source_proven(validation,
                vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_ROUTE_WEIGHTS), 0,
                runtime_node, node_index, 0)) return 0;
    }

    if ((op == VX_OP_GATHER) || (op == VX_OP_GATHER_ELEMENTS)) {
        const char* indices = vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_INDICES);
        int64_t axis;
        if (!input || !input->rank || !vx_native_gpu_integer_parameter(params, "axis", 0,
                -(int64_t)input->rank, (int64_t)input->rank - 1, &axis)) return 0;
        if (axis < 0) axis += input->rank;
        int64_t extent = runtime_node->resident_slot_domain
            ? runtime_node->resident_slot_domain : input->minimums[axis];
        if (runtime_node->resident_slot_domain &&
            ((op != VX_OP_GATHER) || axis != 0 ||
             !vx_webgpu_bank_index_source_proven(validation, indices, runtime_node, node_index, 0))) return 0;
        if (extent <= 0 || !vx_native_gpu_i32_value_source_proof(validation, tensors,
                tensor_count, indices, -extent, extent, node_index, 0)) return 0;
    } else if ((op == VX_OP_EMBEDDING) || (op == VX_OP_Q_EMBEDDING)) {
        if (runtime_node->resident_slot_domain || !weight || weight->rank != 2 || !vx_webgpu_fixed_shape(weight) ||
            !vx_native_gpu_i32_value_source_proof(validation, tensors, tensor_count,
                vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_INPUT),
                0, weight->minimums[0], node_index, 0)) return 0;
    } else if ((op == VX_OP_Q_LAYER_NORM) || (op == VX_OP_Q_GROUP_NORM)) {
        const char* gamma = vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_WEIGHT);
        if (!gamma) gamma = vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_SCALE);
        if (!vx_native_gpu_finite_f32_value_source_proof(validation, gamma, node_index, 0) ||
            !vx_native_gpu_finite_f32_value_source_proof(validation,
                vx_native_gpu_validation_node_input_name(runtime_node, VX_PORT_BIAS), node_index, 0)) return 0;
    }

    size_t capacity = (size_t)runtime_node->nin + (size_t)runtime_node->nout + 1u;
    if (capacity > SIZE_MAX / sizeof(VxWebGpuTensorBound)) return 0;
    VxWebGpuTensorBound* bounds = calloc(capacity, sizeof(*bounds));
    if (!bounds) return 0;
    size_t count = 0;
    int accepted = 0;
    for (size_t port = 0; port < capacity; port++) {
        const char* name = port < (size_t)runtime_node->nin ? runtime_node->ins[port].name :
            port < capacity - 1u ? runtime_node->outs[port - runtime_node->nin].name : runtime_node->out;
        const VxDeclaredTensor* tensor = vx_declared_tensor_find(tensors, tensor_count, name);
        if (!tensor) continue; /* Immutable weights retain their storage shape. */
        if (tensor->rank > 8u) goto done;
        bounds[count].name = tensor->name;
        bounds[count].rank = tensor->rank;
        memcpy(bounds[count++].maximum, tensor->maximums, sizeof(bounds[0].maximum));
    }
    accepted = vx_webgpu_prove_node_domain(validation, output->name, bounds, count);
done:
    free(bounds);
    return accepted;
}
#endif
