/* Concrete shape resolution: binding declared symbols to a ResolvedShapePlan.
 *
 * Fragment of public_api.c, included where these definitions used to sit.
 * Runs after the declared domain has already been proven, so these routines
 * substitute bound symbol values and assert the result rather than re-deciding
 * whether a graph is admissible. */

static void vx_shape_param_storage_clear(VxShapeParamStorage* storage) {
    if (!storage) return;
    for (size_t index = 0; index < storage->count; index++)
        free(storage->arrays ? storage->arrays[index] : NULL);
    free(storage->arrays);
    free(storage->values);
    memset(storage, 0, sizeof(*storage));
}

static const VxShapeTensorDescriptor* vx_shape_param_declared_output(
        const VxShapeNamedTensor* declared_outputs,
        size_t declared_output_count) {
    const VxShapeTensorDescriptor* output = NULL;
    for (size_t index = 0; index < declared_output_count; index++) {
        if (!declared_outputs[index].name ||
            strcmp(declared_outputs[index].name, "out")) continue;
        if (output) return NULL;
        output = &declared_outputs[index].descriptor;
    }
    return output;
}

static int vx_shape_params_materialize(
        const cJSON* object,
        VxOperatorKind operator_kind,
        const VxShapeNamedTensor* declared_outputs,
        size_t declared_output_count,
        VxShapeParamStorage* storage) {
    size_t count;
    size_t index = 0;
    if (!cJSON_IsObject(object) || !storage) return -1;
    memset(storage, 0, sizeof(*storage));
    count = (size_t)cJSON_GetArraySize(object);
    storage->values = count
        ? (VxShapeParam*)calloc(count, sizeof(*storage->values)) : NULL;
    storage->arrays = count
        ? (double**)calloc(count, sizeof(*storage->arrays)) : NULL;
    storage->count = count;
    if (count && (!storage->values || !storage->arrays)) return -2;
    for (const cJSON* item = object->child; item; item = item->next) {
        VxShapeParam* value;
        if (!item->string || !item->string[0]) return -1;
        if (operator_kind == VX_OP_Q_SDPA &&
            !strcmp(item->string, "scale") && cJSON_IsNull(item)) {
            continue;
        }
        value = &storage->values[index];
        value->name = item->string;
        if (cJSON_IsNumber(item)) {
            value->kind = VX_SHAPE_PARAM_NUMBER;
            value->value.number = item->valuedouble;
        } else if (cJSON_IsBool(item)) {
            value->kind = VX_SHAPE_PARAM_BOOLEAN;
            value->value.boolean = cJSON_IsTrue(item) ? 1 : 0;
        } else if (cJSON_IsString(item) && item->valuestring) {
            value->kind = VX_SHAPE_PARAM_STRING;
            value->value.string = item->valuestring;
        } else if (cJSON_IsArray(item)) {
            size_t array_count = (size_t)cJSON_GetArraySize(item);
            size_t array_index = 0;
            const VxShapeTensorDescriptor* declared =
                vx_shape_param_declared_output(
                    declared_outputs, declared_output_count);
            int symbolic_target = item->string &&
                !strcmp(item->string, "shape") &&
                (operator_kind == VX_OP_RESHAPE ||
                 operator_kind == VX_OP_EXPAND);
            double* numbers = array_count
                ? (double*)malloc(array_count * sizeof(*numbers)) : NULL;
            if (array_count && !numbers) return -2;
            storage->arrays[index] = numbers;
            for (const cJSON* element = item->child; element;
                 element = element->next, array_index++) {
                if (cJSON_IsNumber(element)) {
                    numbers[array_index] = element->valuedouble;
                } else if (cJSON_IsString(element) &&
                           element->valuestring && symbolic_target &&
                           declared && declared->shape &&
                           declared->rank == array_count &&
                           array_index < declared->rank &&
                           vx_shape_symbol_valid(element->valuestring) &&
                           declared->shape[array_index] > 0u &&
                           declared->shape[array_index] <=
                               VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER) {
                    uint64_t extent = declared->shape[array_index];
                    size_t prior_index = 0;
                    for (const cJSON* prior = item->child;
                         prior && prior != element;
                         prior = prior->next, prior_index++) {
                        if (cJSON_IsString(prior) && prior->valuestring &&
                            !strcmp(prior->valuestring,
                                    element->valuestring) &&
                            numbers[prior_index] != (double)extent)
                            return -1;
                    }
                    numbers[array_index] = (double)extent;
                } else {
                    return -1;
                }
            }
            value->kind = VX_SHAPE_PARAM_NUMBER_ARRAY;
            value->value.number_array.count = array_count;
            value->value.number_array.values = numbers;
        } else {
            return -1;
        }
        index++;
    }
    storage->count = index;
    return 0;
}


static uint64_t vx_logical_tensor_name_hash(const char* name) {
    uint64_t hash = UINT64_C(14695981039346656037);
    if (!name) return 0;
    while (*name) {
        hash ^= (unsigned char)*name++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static VxStatus vx_logical_tensor_name_index_build(
        VxExecutionContext* context) {
    size_t capacity = 1u;
    size_t* slots;
    if (!context || context->logical_tensor_name_slots ||
        context->logical_tensor_name_capacity) return VX_STATUS_INVALID_ARGUMENT;
    if (!context->logical_tensor_count) return VX_STATUS_OK;
    if (context->logical_tensor_count > SIZE_MAX / 2u)
        return VX_STATUS_OUT_OF_MEMORY;
    while (capacity < context->logical_tensor_count * 2u) {
        if (capacity > SIZE_MAX / 2u) return VX_STATUS_OUT_OF_MEMORY;
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(*slots)) return VX_STATUS_OUT_OF_MEMORY;
    slots = (size_t*)calloc(capacity, sizeof(*slots));
    if (!slots) return VX_STATUS_OUT_OF_MEMORY;
    for (size_t index = 0; index < context->logical_tensor_count; index++) {
        const char* name = context->logical_tensors[index].name;
        size_t slot;
        if (!name || !name[0]) {
            free(slots);
            return VX_STATUS_INVALID_GRAPH;
        }
        slot = (size_t)vx_logical_tensor_name_hash(name) & (capacity - 1u);
        while (slots[slot]) {
            if (!strcmp(context->logical_tensors[slots[slot] - 1u].name,
                        name)) {
                free(slots);
                return VX_STATUS_INVALID_GRAPH;
            }
            slot = (slot + 1u) & (capacity - 1u);
        }
        slots[slot] = index + 1u;
    }
    if (context->allocated_bytes > SIZE_MAX - capacity * sizeof(*slots)) {
        free(slots);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    context->logical_tensor_name_slots = slots;
    context->logical_tensor_name_capacity = capacity;
    context->allocated_bytes += capacity * sizeof(*slots);
    return VX_STATUS_OK;
}

static size_t vx_logical_tensor_index(const VxExecutionContext* context,
                                      const char* name) {
    if (!context || !name) return SIZE_MAX;
    if (context->logical_tensor_name_slots &&
        context->logical_tensor_name_capacity) {
        size_t slot = (size_t)vx_logical_tensor_name_hash(name) &
            (context->logical_tensor_name_capacity - 1u);
        for (size_t probe = 0;
             probe < context->logical_tensor_name_capacity; probe++) {
            size_t encoded = context->logical_tensor_name_slots[slot];
            if (!encoded) return SIZE_MAX;
            if (!strcmp(context->logical_tensors[encoded - 1u].name, name))
                return encoded - 1u;
            slot = (slot + 1u) &
                (context->logical_tensor_name_capacity - 1u);
        }
        return SIZE_MAX;
    }
    for (size_t index = 0; index < context->logical_tensor_count; index++)
        if (!strcmp(context->logical_tensors[index].name, name)) return index;
    return SIZE_MAX;
}

/* Logical tensors are input-first and then topological node-output order. The
 * first declaration of a symbol is therefore its canonical representative:
 * all inputs have already been checked for equality by input staging, and
 * every later inferred output is checked against this representative before
 * it is committed. Looking past the first declaration only repeats that same
 * equality proof and makes each dynamic bind quadratic in graph size. */
static int vx_resolved_symbol_from_partial(
        const VxExecutionContext* context,
        const VolvoxAIEngineResolvedTensor* resolved,
        const char* symbol,
        int64_t* out_extent) {
    if (!context || !resolved || !symbol || !out_extent) return -1;
    for (size_t index = 0; index < context->logical_tensor_count; index++) {
        const VxDeclaredTensor* logical = &context->logical_tensors[index];
        for (uint32_t axis = 0; axis < logical->rank; axis++) {
            if (!logical->symbols[axis] ||
                strcmp(logical->symbols[axis], symbol)) continue;
            if (resolved[index].shape[axis] <= 0) return 0;
            *out_extent = resolved[index].shape[axis];
            return 1;
        }
    }
    return 0;
}

static void vx_shape_quantization_from_private(
        const VxExecutionContext* context,
        size_t node_index,
        const char* input_port,
        const char* tensor_name,
        VxShapeTensorDescriptor* descriptor) {
    T* tensor = tensor_name ? t_find(tensor_name) : NULL;
    if (!tensor || !descriptor) return;
    if (tensor->quantization.valid) {
        descriptor->quantization.scheme = VX_SHAPE_QUANTIZATION_PER_TENSOR;
        descriptor->quantization.scale = tensor->quantization.scale;
        descriptor->quantization.zero_point = tensor->quantization.zero_point;
        return;
    }
    /* Per-axis weight metadata is normalized and aligned once by the private
     * engine. Reuse those central tables here instead of re-reading possibly
     * unaligned safetensors parameter bytes for every concrete shape plan. */
    if (!context || !context->engine_state ||
        node_index >= context->engine_state->node_capacity ||
        !input_port || strcmp(input_port, "weight")) return;
    if (context->engine_state->qlinear_metadata[node_index].valid) {
        const QLinearMetadata* metadata =
            &context->engine_state->qlinear_metadata[node_index];
        descriptor->quantization.scheme = VX_SHAPE_QUANTIZATION_PER_AXIS;
        descriptor->quantization.axis = 0u;
        descriptor->quantization.count = (size_t)metadata->d_out;
        descriptor->quantization.scales = metadata->weight_scales;
        descriptor->quantization.zero_points = metadata->weight_zero_points;
    } else if (context->engine_state->qconv_metadata[node_index].valid) {
        const QConv2DMetadata* metadata =
            &context->engine_state->qconv_metadata[node_index];
        descriptor->quantization.scheme = VX_SHAPE_QUANTIZATION_PER_AXIS;
        descriptor->quantization.axis = 0u;
        descriptor->quantization.count = metadata->output_channels;
        descriptor->quantization.scales = metadata->weight_scales;
        descriptor->quantization.zero_points = metadata->weight_zero_points;
    } else if (context->engine_state->qembedding_metadata[node_index].valid) {
        const QEmbeddingMetadata* metadata =
            &context->engine_state->qembedding_metadata[node_index];
        descriptor->quantization.scheme = VX_SHAPE_QUANTIZATION_PER_AXIS;
        descriptor->quantization.axis = 0u;
        descriptor->quantization.count = metadata->vocab;
        descriptor->quantization.scales = metadata->weight_scales;
        descriptor->quantization.zero_points = metadata->weight_zero_points;
    }
}

static int vx_shape_input_descriptor(
        const VxExecutionContext* context,
        const VolvoxAIEngineResolvedTensor* resolved,
        size_t node_index,
        const char* input_port,
        const char* tensor_name,
        VxShapeTensorDescriptor* descriptor,
        uint64_t shape[VX_MAX_TENSOR_RANK]) {
    size_t logical_index = vx_logical_tensor_index(context, tensor_name);
    memset(descriptor, 0, sizeof(*descriptor));
    memset(shape, 0, VX_MAX_TENSOR_RANK * sizeof(*shape));
    if (logical_index != SIZE_MAX) {
        const VolvoxAIEngineResolvedTensor* concrete =
            &resolved[logical_index];
        if (concrete->rank < 0 ||
            concrete->rank > (int)VX_MAX_TENSOR_RANK) return -1;
        descriptor->rank = (size_t)concrete->rank;
        descriptor->dtype = (VxDataType)concrete->dtype;
        for (int axis = 0; axis < concrete->rank; axis++) {
            if (concrete->shape[axis] <= 0) return -1;
            shape[axis] = (uint64_t)concrete->shape[axis];
        }
    } else {
        T* tensor = t_find(tensor_name);
        if (!tensor || tensor->ndim < 0 ||
            tensor->ndim > (int)VX_MAX_TENSOR_RANK || tensor->numel <= 0)
            return -1;
        descriptor->rank = (size_t)tensor->ndim;
        /* F16 is an immutable private storage optimization. Native v1 exposes
         * and computes it as F32, matching output snapshot conversion. */
        descriptor->dtype = tensor->dtype == T_F16
            ? VX_DTYPE_F32 : tensor->dtype;
        for (int axis = 0; axis < tensor->ndim; axis++) {
            if (tensor->shape[axis] <= 0) return -1;
            shape[axis] = (uint64_t)tensor->shape[axis];
        }
    }
    descriptor->shape = shape;
    vx_shape_quantization_from_private(
        context, node_index, input_port, tensor_name, descriptor);
    return 0;
}

static int vx_shape_declared_output_descriptor(
        const VxExecutionContext* context,
        const VolvoxAIEngineResolvedTensor* resolved,
        size_t logical_index,
        VxShapeTensorDescriptor* descriptor,
        uint64_t shape[VX_MAX_TENSOR_RANK]) {
    const VxDeclaredTensor* logical;
    int resolved_all = 1;
    if (!context || !resolved || logical_index >= context->logical_tensor_count)
        return -1;
    logical = &context->logical_tensors[logical_index];
    memset(descriptor, 0, sizeof(*descriptor));
    memset(shape, 0, VX_MAX_TENSOR_RANK * sizeof(*shape));
    descriptor->rank = logical->rank;
    descriptor->dtype = logical->dtype;
    /* The shape contract sizes its output slots from this descriptor, so it is
     * always returned structurally complete. An axis whose symbol is not bound
     * yet carries a valid placeholder and lowers the resolved flag; the caller
     * then declines to treat the declaration as semantic input. */
    for (uint32_t axis = 0; axis < logical->rank; axis++) {
        int64_t extent = logical->minimums[axis];
        if (logical->kinds[axis] == VX_DIMENSION_SYMBOLIC) {
            int found = vx_resolved_symbol_from_partial(
                context, resolved, logical->symbols[axis], &extent);
            if (found < 0) return found;
            if (!found) {
                shape[axis] = 1u;
                resolved_all = 0;
                continue;
            }
        }
        if (extent <= 0) return -1;
        shape[axis] = (uint64_t)extent;
    }
    descriptor->shape = shape;
    vx_shape_quantization_from_private(
        context, SIZE_MAX, NULL, logical->name, descriptor);
    return resolved_all;
}

static int vx_resolved_assert_output(
        const VxExecutionContext* context,
        VolvoxAIEngineResolvedTensor* resolved,
        size_t logical_index,
        const VxShapeTensorDescriptor* inferred) {
    const VxDeclaredTensor* logical;
    VolvoxAIEngineResolvedTensor* concrete;
    if (!context || !resolved || !inferred ||
        logical_index >= context->logical_tensor_count) return -1;
    logical = &context->logical_tensors[logical_index];
    concrete = &resolved[logical_index];
    if (inferred->dtype != logical->dtype || inferred->rank != logical->rank)
        return -1;
    for (uint32_t axis = 0; axis < logical->rank; axis++) {
        uint64_t raw_extent = inferred->shape[axis];
        int64_t bound_extent = 0;
        int bound;
        if (!raw_extent || raw_extent > INT_MAX ||
            raw_extent < (uint64_t)logical->minimums[axis] ||
            raw_extent > (uint64_t)logical->maximums[axis] ||
            raw_extent % (uint64_t)logical->multiples[axis]) return -1;
        if (logical->kinds[axis] == VX_DIMENSION_FIXED) {
            if (logical->symbols[axis] ||
                raw_extent != (uint64_t)logical->minimums[axis]) return -1;
        } else {
            if (!logical->symbols[axis] || !logical->symbols[axis][0]) return -1;
            bound = vx_resolved_symbol_from_partial(
                context, resolved, logical->symbols[axis], &bound_extent);
            if (bound < 0 || (bound && raw_extent != (uint64_t)bound_extent))
                return -1;
        }
        concrete->shape[axis] = (int)raw_extent;
    }
    return 0;
}

static VxStatus vx_resolve_node_shapes(
        const VxExecutionContext* context,
        const cJSON* node,
        size_t node_index,
        VolvoxAIEngineResolvedTensor* resolved) {
    const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
    const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
    const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
    const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    const char* shape_function;
    VxShapeNamedTensor* request_inputs = NULL;
    VxShapeNamedTensor* request_outputs = NULL;
    uint64_t* input_shapes = NULL;
    uint64_t* output_shapes = NULL;
    size_t* output_indices = NULL;
    VxShapeParamStorage param_storage = {0};
    VxConcreteShapeRequest request = {0};
    VxConcreteShapeResult result = VX_CONCRETE_SHAPE_RESULT_INITIALIZER;
    VxShapeContractError error = {0};
    size_t input_count;
    size_t output_count;
    size_t index;
    int declarations_bound = 1;
    VxStatus status = VX_STATUS_INVALID_GRAPH;
    if (!cJSON_IsString(op) || !op->valuestring ||
        !(shape_function = vx_shape_contract_function_id(op->valuestring)) ||
        !cJSON_IsObject(inputs) || !cJSON_IsObject(outputs) ||
        !cJSON_IsObject(params)) return VX_STATUS_BACKEND_UNSUPPORTED;
    input_count = (size_t)cJSON_GetArraySize(inputs);
    output_count = (size_t)cJSON_GetArraySize(outputs);
    if (!input_count || !output_count) return VX_STATUS_INVALID_GRAPH;
    request_inputs = calloc(input_count, sizeof(*request_inputs));
    request_outputs = calloc(output_count, sizeof(*request_outputs));
    input_shapes = calloc(input_count, VX_MAX_TENSOR_RANK * sizeof(*input_shapes));
    output_shapes = calloc(output_count, VX_MAX_TENSOR_RANK * sizeof(*output_shapes));
    output_indices = calloc(output_count, sizeof(*output_indices));
    if (!request_inputs || !request_outputs || !input_shapes || !output_shapes || !output_indices) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    index = 0;
    for (const cJSON* input = inputs->child; input;
         input = input->next, index++) {
        if (!input->string || !input->string[0] ||
            !cJSON_IsString(input) || !input->valuestring ||
            vx_shape_input_descriptor(
                context, resolved, node_index, input->string,
                input->valuestring,
                &request_inputs[index].descriptor,
                input_shapes + index * VX_MAX_TENSOR_RANK) != 0)
            goto done;
        request_inputs[index].name = input->string;
    }
    index = 0;
    for (const cJSON* output = outputs->child; output;
         output = output->next, index++) {
        const cJSON* tensor =
            cJSON_GetObjectItemCaseSensitive(output, "tensor");
        int descriptor_status;
        if (!output->string || !output->string[0] ||
            !cJSON_IsString(tensor) || !tensor->valuestring ||
            (output_indices[index] = vx_logical_tensor_index(
                 context, tensor->valuestring)) == SIZE_MAX) goto done;
        request_outputs[index].name = output->string;
        descriptor_status = vx_shape_declared_output_descriptor(
            context, resolved, output_indices[index],
            &request_outputs[index].descriptor,
            output_shapes + index * VX_MAX_TENSOR_RANK);
        if (descriptor_status < 0) goto done;
        if (!descriptor_status) declarations_bound = 0;
    }
    {
        int params_status = vx_shape_params_materialize(
            params, vx_operator_kind_from_name(op->valuestring),
            declarations_bound ? request_outputs : NULL,
            declarations_bound ? output_count : 0u,
            &param_storage);
        if (params_status != 0) {
            status = params_status == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                           VX_STATUS_INVALID_GRAPH;
            goto done;
        }
    }
    request.inputs = request_inputs;
    request.input_count = input_count;
    request.params = param_storage.values;
    request.param_count = param_storage.count;
    request.declared_outputs = request_outputs;
    request.declared_output_count = output_count;
    if (vx_shape_contract_infer(op->valuestring, &request, &result, &error) != 0) {
        if (context->engine_state && context->engine_state->debug)
            vx_engine_log(
                    "[debug] shape contract failed op=%s code=%d path=%s message=%s\n",
                    op->valuestring, (int)error.code,
                    error.path[0] ? error.path : "<none>",
                    error.detail[0] ? error.detail : "<none>");
        status = error.code == VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY
            ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_INVALID_GRAPH;
        goto done;
    }
    if (!result.shape_function_id ||
        strcmp(result.shape_function_id, shape_function) ||
        result.output_count != output_count) goto done;
    for (index = 0; index < output_count; index++) {
        const char* port = request_outputs[index].name;
        const VxShapeTensorDescriptor* inferred = NULL;
        for (size_t candidate = 0; candidate < result.output_count;
             candidate++) {
            if (result.outputs[candidate].name &&
                !strcmp(result.outputs[candidate].name, port)) {
                inferred = &result.outputs[candidate].descriptor;
                break;
            }
        }
        if (!inferred || vx_resolved_assert_output(
                context, resolved, output_indices[index], inferred) != 0) {
            if (context->engine_state && context->engine_state->debug)
                vx_engine_log(
                        "[debug] inferred output assertion failed op=%s port=%s tensor=%s\n",
                        op->valuestring, port,
                        context->logical_tensors[output_indices[index]].name);
            goto done;
        }
    }
    status = VX_STATUS_OK;
done:
    free(request_inputs); free(request_outputs);
    free(input_shapes); free(output_shapes); free(output_indices);
    vx_shape_contract_result_clear(&result);
    vx_shape_param_storage_clear(&param_storage);
    return status;
}

static VxStatus vx_resolve_graph_shapes(
        const VxExecutionContext* context,
        VolvoxAIEngineResolvedTensor* resolved) {
    const cJSON* nodes;
    if (!context || !context->compiled || !context->compiled->model ||
        !resolved || !context->compiled->model->logical_graph)
        return VX_STATUS_INVALID_ARGUMENT;
    nodes = cJSON_GetObjectItemCaseSensitive(
        context->compiled->model->logical_graph, "nodes");
    if (!cJSON_IsArray(nodes)) return VX_STATUS_INVALID_GRAPH;
    {
        size_t node_index = 0;
        for (const cJSON* node = nodes->child; node;
             node = node->next, node_index++) {
            VxStatus status = vx_resolve_node_shapes(
                context, node, node_index, resolved);
            if (status != VX_STATUS_OK) {
                if (context->engine_state && context->engine_state->debug) {
                    const cJSON* node_id = cJSON_GetObjectItemCaseSensitive(
                        node, "id");
                    const cJSON* op = cJSON_GetObjectItemCaseSensitive(
                        node, "opType");
                    vx_engine_log(
                            "[debug] graph shape resolution stopped node=%zu id=%s op=%s status=%d\n",
                            node_index,
                            cJSON_IsString(node_id) && node_id->valuestring
                                ? node_id->valuestring : "<unknown>",
                            cJSON_IsString(op) && op->valuestring
                                ? op->valuestring : "<unknown>",
                            (int)status);
                }
                return status;
            }
        }
    }
    for (size_t index = 0; index < context->logical_tensor_count; index++)
        for (int axis = 0; axis < resolved[index].rank; axis++)
            if (resolved[index].shape[axis] <= 0)
                return VX_STATUS_INVALID_GRAPH;
    return VX_STATUS_OK;
}

static VxStatus vx_resolved_plan_create(
        const VxExecutionContext* context,
        const int64_t* input_shapes,
        VolvoxAIEngineResolvedTensor** out_tensors,
        size_t* out_tensor_count,
        char** out_signature) {
    const VxModel* model;
    VolvoxAIEngineResolvedTensor* resolved = NULL;
    char* signature = NULL;
    size_t signature_capacity = 4u;
    size_t signature_offset = 0;
    if (!context || (!input_shapes && context->compiled->model->input_count) ||
        !out_tensors || !out_tensor_count ||
        !out_signature) return VX_STATUS_INVALID_ARGUMENT;
    *out_tensors = NULL;
    *out_tensor_count = 0;
    *out_signature = NULL;
    model = context->compiled->model;
    for (size_t input = 0; input < model->input_count; input++) {
        size_t name_length = strlen(model->inputs[input].name);
        if (name_length > SIZE_MAX - signature_capacity - 4u ||
            model->inputs[input].rank >
                (SIZE_MAX - signature_capacity - name_length - 4u) / 24u)
            return VX_STATUS_OUT_OF_MEMORY;
        signature_capacity += name_length + 4u +
            (size_t)model->inputs[input].rank * 24u;
    }
    signature = (char*)malloc(signature_capacity);
    resolved = (VolvoxAIEngineResolvedTensor*)calloc(
        context->logical_tensor_count, sizeof(*resolved));
    if (!signature || !resolved) {
        free(signature);
        free(resolved);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    signature_offset = (size_t)snprintf(signature, signature_capacity, "v1;");
    {
        const char* previous = NULL;
        for (size_t position = 0; position < model->input_count; position++) {
            size_t input = SIZE_MAX;
            int written;
            for (size_t candidate = 0; candidate < model->input_count;
                 candidate++) {
                const char* name = model->inputs[candidate].name;
                if ((previous && strcmp(name, previous) <= 0) ||
                    (input != SIZE_MAX &&
                     strcmp(name, model->inputs[input].name) >= 0)) continue;
                input = candidate;
            }
            if (input == SIZE_MAX) goto oom;
            written = snprintf(signature + signature_offset,
                               signature_capacity - signature_offset,
                               "%s[", model->inputs[input].name);
        if (written < 0 || (size_t)written >=
                             signature_capacity - signature_offset) goto oom;
        signature_offset += (size_t)written;
        for (uint32_t axis = 0; axis < model->inputs[input].rank; axis++) {
            written = snprintf(
                signature + signature_offset,
                signature_capacity - signature_offset,
                "%s%" PRId64,
                axis ? "," : "",
                input_shapes[input * VX_MAX_TENSOR_RANK + axis]);
            if (written < 0 || (size_t)written >=
                                 signature_capacity - signature_offset) goto oom;
            signature_offset += (size_t)written;
        }
        written = snprintf(signature + signature_offset,
                           signature_capacity - signature_offset, "];");
        if (written < 0 || (size_t)written >=
                             signature_capacity - signature_offset) goto oom;
        signature_offset += (size_t)written;
            previous = model->inputs[input].name;
        }
    }
    for (size_t index = 0; index < context->logical_tensor_count; index++) {
        const VxDeclaredTensor* logical = &context->logical_tensors[index];
        VolvoxAIEngineResolvedTensor* concrete = &resolved[index];
        concrete->name = logical->name;
        concrete->dtype = (int)logical->dtype;
        concrete->rank = (int)logical->rank;
    }
    for (size_t input = 0; input < model->input_count; input++) {
        size_t logical_index = vx_logical_tensor_index(
            context, model->inputs[input].name);
        if (logical_index == SIZE_MAX) goto invalid;
        for (uint32_t axis = 0; axis < model->inputs[input].rank; axis++) {
            int64_t extent =
                input_shapes[input * VX_MAX_TENSOR_RANK + axis];
            if (extent <= 0 || extent > INT_MAX) goto invalid;
            resolved[logical_index].shape[axis] = (int)extent;
        }
    }
    {
        VxStatus shape_status = vx_resolve_graph_shapes(context, resolved);
        if (shape_status != VX_STATUS_OK) {
            free(signature);
            free(resolved);
            return shape_status;
        }
    }
    *out_tensors = resolved;
    *out_tensor_count = context->logical_tensor_count;
    *out_signature = signature;
    return VX_STATUS_OK;
invalid:
    free(signature);
    free(resolved);
    return VX_STATUS_INVALID_GRAPH;
oom:
    free(signature);
    free(resolved);
    return VX_STATUS_OUT_OF_MEMORY;
}
