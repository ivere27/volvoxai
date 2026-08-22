/* Core-authored proof for scheduler coalescing over an explicit public B axis.
 *
 * This is deliberately stricter than ordinary bulk execution.  It walks the
 * exact typed logical graph and fails closed unless every public tensor carries
 * one leading symbolic request axis and every operator preserves lane
 * independence.  The resulting identity is bound to the canonical graph
 * fingerprint and must be echoed by an external provider.  Built-in GPU
 * routes consume the same proof directly.  Fixed-B=1 lifting is not performed
 * here; that requires a typed-IR transform before shape resolution. */

typedef struct VxIndependentBatchProof {
    int supported;
    uint32_t batch_axis;
    uint32_t min_batch;
    uint32_t max_batch;
    uint32_t multiple_of;
    size_t covered_nodes;
    char identity[128];
    char reason[192];
    char failed_node[VX_REPORT_NODE_CAPACITY];
} VxIndependentBatchProof;

static size_t vx_ib_object_count(const cJSON* object) {
    size_t count = 0;
    for (const cJSON* item = cJSON_IsObject(object) ? object->child : NULL;
         item; item = item->next) count++;
    return count;
}

static int vx_ib_object_exact(const cJSON* object,
                              const char* const* keys,
                              size_t key_count) {
    if (!cJSON_IsObject(object) || vx_ib_object_count(object) != key_count)
        return 0;
    for (size_t index = 0; index < key_count; index++)
        if (!cJSON_GetObjectItemCaseSensitive(object, keys[index])) return 0;
    return 1;
}

static int vx_ib_json_integer(const cJSON* value, int64_t* out) {
    double number;
    if (!cJSON_IsNumber(value) || !isfinite(value->valuedouble)) return 0;
    number = value->valuedouble;
    if (number < -9007199254740991.0 || number > 9007199254740991.0 ||
        floor(number) != number) return 0;
    if (out) *out = (int64_t)number;
    return 1;
}

static int vx_ib_normalized_axis(const cJSON* value, uint32_t rank,
                                 uint32_t* out) {
    int64_t raw;
    int64_t axis;
    if (!rank || !vx_ib_json_integer(value, &raw)) return 0;
    axis = raw < 0 ? raw + (int64_t)rank : raw;
    if (axis < 0 || axis >= (int64_t)rank) return 0;
    if (out) *out = (uint32_t)axis;
    return 1;
}

static int vx_ib_symbol_axis(const VxDeclaredTensor* tensor,
                             const char* symbol,
                             uint32_t* out_axis,
                             size_t* out_occurrences) {
    size_t occurrences = 0;
    uint32_t axis = 0;
    if (!tensor || !symbol) return 0;
    for (uint32_t index = 0; index < tensor->rank; index++) {
        if (!tensor->symbols[index] ||
            strcmp(tensor->symbols[index], symbol)) continue;
        axis = index;
        occurrences++;
    }
    if (out_axis && occurrences == 1u) *out_axis = axis;
    if (out_occurrences) *out_occurrences = occurrences;
    return occurrences == 1u;
}

static int vx_ib_leading_symbol(const VxDeclaredTensor* tensor,
                                const char* symbol) {
    uint32_t axis = UINT32_MAX;
    size_t occurrences = 0;
    return vx_ib_symbol_axis(tensor, symbol, &axis, &occurrences) &&
        occurrences == 1u && axis == 0u;
}

static int vx_ib_dimension_equal(const VxDeclaredTensor* left,
                                 uint32_t left_axis,
                                 const VxDeclaredTensor* right,
                                 uint32_t right_axis) {
    if (!left || !right || left_axis >= left->rank || right_axis >= right->rank ||
        left->kinds[left_axis] != right->kinds[right_axis]) return 0;
    if (left->kinds[left_axis] == VX_DIMENSION_FIXED)
        return left->minimums[left_axis] == right->minimums[right_axis];
    return left->symbols[left_axis] && right->symbols[right_axis] &&
        !strcmp(left->symbols[left_axis], right->symbols[right_axis]) &&
        left->minimums[left_axis] == right->minimums[right_axis] &&
        left->maximums[left_axis] == right->maximums[right_axis] &&
        left->multiples[left_axis] == right->multiples[right_axis];
}

static int vx_ib_shape_equal(const VxDeclaredTensor* left,
                             const VxDeclaredTensor* right) {
    if (!left || !right || left->rank != right->rank) return 0;
    for (uint32_t axis = 0; axis < left->rank; axis++)
        if (!vx_ib_dimension_equal(left, axis, right, axis)) return 0;
    return 1;
}

static const cJSON* vx_ib_node_inputs(const cJSON* node) {
    const cJSON* value = node
        ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    return cJSON_IsObject(value) ? value : NULL;
}

static const cJSON* vx_ib_node_params(const cJSON* node) {
    const cJSON* value = node
        ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
    return cJSON_IsObject(value) ? value : NULL;
}

static const VxDeclaredTensor* vx_ib_reference_tensor(
        const cJSON* node, const char* role,
        const VxDeclaredTensor* tensors, size_t tensor_count) {
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* reference = inputs
        ? cJSON_GetObjectItemCaseSensitive(inputs, role) : NULL;
    return cJSON_IsString(reference) && reference->valuestring
        ? vx_declared_tensor_find(tensors, tensor_count,
                                  reference->valuestring)
        : NULL;
}

static int vx_ib_reference_fixed(const cJSON* node, const char* role,
                                 const VxDeclaredTensor* tensors,
                                 size_t tensor_count) {
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* reference = inputs
        ? cJSON_GetObjectItemCaseSensitive(inputs, role) : NULL;
    return cJSON_IsString(reference) && reference->valuestring &&
        !vx_declared_tensor_find(tensors, tensor_count,
                                 reference->valuestring);
}

static const VxDeclaredTensor* vx_ib_single_output(
        const cJSON* node, const VxDeclaredTensor* tensors,
        size_t tensor_count) {
    const cJSON* outputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "outputs") : NULL;
    const cJSON* descriptor;
    const cJSON* reference;
    if (!cJSON_IsObject(outputs) || !(descriptor = outputs->child) ||
        descriptor->next) return NULL;
    reference = cJSON_GetObjectItemCaseSensitive(descriptor, "tensor");
    return cJSON_IsString(reference) && reference->valuestring
        ? vx_declared_tensor_find(tensors, tensor_count,
                                  reference->valuestring)
        : NULL;
}

static int vx_ib_preserves_shape(const cJSON* node,
                                 const VxDeclaredTensor* tensors,
                                 size_t tensor_count,
                                 const char* symbol) {
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    uint32_t input_axis;
    uint32_t output_axis;
    return input && output &&
        vx_ib_symbol_axis(input, symbol, &input_axis, NULL) &&
        vx_ib_symbol_axis(output, symbol, &output_axis, NULL) &&
        input_axis == output_axis && vx_ib_shape_equal(input, output);
}

static int vx_ib_logical_shape_matches_json(const VxDeclaredTensor* tensor,
                                            const cJSON* shape) {
    if (!tensor || !cJSON_IsArray(shape) ||
        (size_t)cJSON_GetArraySize(shape) != tensor->rank) return 0;
    for (uint32_t axis = 0; axis < tensor->rank; axis++) {
        const cJSON* item = cJSON_GetArrayItem(shape, (int)axis);
        int64_t fixed;
        if (tensor->kinds[axis] == VX_DIMENSION_FIXED) {
            if (!vx_ib_json_integer(item, &fixed) ||
                fixed != tensor->minimums[axis]) return 0;
        } else if (!cJSON_IsString(item) || !item->valuestring ||
                   !tensor->symbols[axis] ||
                   strcmp(item->valuestring, tensor->symbols[axis])) {
            return 0;
        }
    }
    return 1;
}

static int vx_ib_monomial_equal(const VxDeclaredTensor* left,
                                uint32_t left_begin, uint32_t left_end,
                                const VxDeclaredTensor* right,
                                uint32_t right_begin, uint32_t right_end) {
    uint64_t left_constant = 1u;
    uint64_t right_constant = 1u;
    if (!left || !right || left_begin > left_end || right_begin > right_end ||
        left_end > left->rank || right_end > right->rank) return 0;
    for (uint32_t axis = left_begin; axis < left_end; axis++) {
        if (left->kinds[axis] == VX_DIMENSION_FIXED) {
            uint64_t extent = (uint64_t)left->minimums[axis];
            if (!extent || left_constant > UINT64_MAX / extent) return 0;
            left_constant *= extent;
        }
    }
    for (uint32_t axis = right_begin; axis < right_end; axis++) {
        if (right->kinds[axis] == VX_DIMENSION_FIXED) {
            uint64_t extent = (uint64_t)right->minimums[axis];
            if (!extent || right_constant > UINT64_MAX / extent) return 0;
            right_constant *= extent;
        }
    }
    if (left_constant != right_constant) return 0;
    for (uint32_t axis = left_begin; axis < left_end; axis++) {
        size_t left_count = 0;
        size_t right_count = 0;
        const char* symbol = left->symbols[axis];
        if (!symbol) continue;
        for (uint32_t item = left_begin; item < left_end; item++)
            left_count += left->symbols[item] &&
                !strcmp(left->symbols[item], symbol);
        for (uint32_t item = right_begin; item < right_end; item++)
            right_count += right->symbols[item] &&
                !strcmp(right->symbols[item], symbol);
        if (left_count != right_count) return 0;
    }
    for (uint32_t axis = right_begin; axis < right_end; axis++) {
        const char* symbol = right->symbols[axis];
        int found = 0;
        if (!symbol) continue;
        for (uint32_t item = left_begin; item < left_end; item++)
            found |= left->symbols[item] &&
                !strcmp(left->symbols[item], symbol);
        if (!found) return 0;
    }
    return 1;
}

static const cJSON* vx_ib_quantization_descriptor(const VxModel* model,
                                                  const char* name) {
    const cJSON* quantization = model && model->logical_graph
        ? cJSON_GetObjectItemCaseSensitive(model->logical_graph,
                                           "quantization") : NULL;
    const cJSON* table = cJSON_IsObject(quantization)
        ? cJSON_GetObjectItemCaseSensitive(quantization, "tensors") : NULL;
    const cJSON* descriptor = cJSON_IsObject(table) && name
        ? cJSON_GetObjectItemCaseSensitive(table, name) : NULL;
    return cJSON_IsObject(descriptor) ? descriptor : NULL;
}

static int vx_ib_per_tensor_quantized(const VxModel* model,
                                      const cJSON* node,
                                      const char* role,
                                      const VxDeclaredTensor* output,
                                      int output_role) {
    const char* name = output_role && output ? output->name : NULL;
    const cJSON* inputs;
    const cJSON* reference;
    const cJSON* descriptor;
    const cJSON* scheme;
    if (!output_role) {
        inputs = vx_ib_node_inputs(node);
        reference = inputs
            ? cJSON_GetObjectItemCaseSensitive(inputs, role) : NULL;
        name = cJSON_IsString(reference) ? reference->valuestring : NULL;
    }
    descriptor = vx_ib_quantization_descriptor(model, name);
    scheme = descriptor
        ? cJSON_GetObjectItemCaseSensitive(descriptor, "scheme") : NULL;
    return cJSON_IsString(scheme) && scheme->valuestring &&
        !strcmp(scheme->valuestring, "per_tensor");
}

static int vx_ib_quantization_avoids_axis(const VxModel* model,
                                          const VxDeclaredTensor* tensor,
                                          const char* symbol) {
    const cJSON* descriptor;
    const cJSON* scheme;
    const cJSON* axis_item;
    uint32_t batch_axis;
    int64_t axis;
    if (!tensor || !vx_ib_symbol_axis(tensor, symbol, &batch_axis, NULL))
        return 1;
    descriptor = vx_ib_quantization_descriptor(model, tensor->name);
    if (!descriptor) return 1;
    scheme = cJSON_GetObjectItemCaseSensitive(descriptor, "scheme");
    if (!cJSON_IsString(scheme) || !scheme->valuestring) return 0;
    if (strcmp(scheme->valuestring, "per_axis"))
        return !strcmp(scheme->valuestring, "per_tensor");
    axis_item = cJSON_GetObjectItemCaseSensitive(descriptor, "axis");
    return vx_ib_json_integer(axis_item, &axis) && axis >= 0 &&
        axis < (int64_t)tensor->rank && (uint32_t)axis != batch_axis;
}

static int vx_ib_safe_broadcast(const cJSON* node,
                                const char* const* roles, size_t role_count,
                                const VxDeclaredTensor* tensors,
                                size_t tensor_count, const char* symbol) {
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    uint32_t output_axis;
    size_t batched = 0;
    if (!vx_ib_object_exact(inputs, roles, role_count) ||
        vx_ib_object_count(params) != 0u || !output ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL)) return 0;
    for (size_t index = 0; index < role_count; index++) {
        const VxDeclaredTensor* input = vx_ib_reference_tensor(
            node, roles[index], tensors, tensor_count);
        uint32_t input_axis;
        int64_t aligned;
        if (!input) {
            if (!vx_ib_reference_fixed(node, roles[index], tensors,
                                       tensor_count)) return 0;
            continue;
        }
        if (!vx_ib_symbol_axis(input, symbol, &input_axis, NULL)) return 0;
        aligned = (int64_t)input_axis + (int64_t)output->rank -
            (int64_t)input->rank;
        if (aligned < 0 || aligned != (int64_t)output_axis) return 0;
        batched++;
    }
    return batched > 0u;
}

static int vx_ib_safe_reshape(const cJSON* node,
                              const VxDeclaredTensor* tensors,
                              size_t tensor_count, const char* symbol) {
    static const char* const input_keys[] = {"input"};
    static const char* const param_keys[] = {"shape"};
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    uint32_t input_axis;
    uint32_t output_axis;
    if (!vx_ib_object_exact(inputs, input_keys, 1u) ||
        !vx_ib_object_exact(params, param_keys, 1u) || !input || !output ||
        !vx_ib_logical_shape_matches_json(
            output, cJSON_GetObjectItemCaseSensitive(params, "shape")) ||
        !vx_ib_symbol_axis(input, symbol, &input_axis, NULL) ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL)) return 0;
    return vx_ib_monomial_equal(input, 0u, input_axis,
                                output, 0u, output_axis) &&
        vx_ib_monomial_equal(input, input_axis + 1u, input->rank,
                             output, output_axis + 1u, output->rank);
}

static int vx_ib_safe_transpose(const cJSON* node,
                                const VxDeclaredTensor* tensors,
                                size_t tensor_count, const char* symbol) {
    static const char* const input_keys[] = {"input"};
    static const char* const param_keys[] = {"perm"};
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const cJSON* permutation = params
        ? cJSON_GetObjectItemCaseSensitive(params, "perm") : NULL;
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    uint32_t input_axis;
    uint32_t output_axis;
    unsigned char seen[VX_MAX_TENSOR_RANK] = {0};
    if (!vx_ib_object_exact(inputs, input_keys, 1u) ||
        !vx_ib_object_exact(params, param_keys, 1u) || !input || !output ||
        input->rank != output->rank || !cJSON_IsArray(permutation) ||
        (size_t)cJSON_GetArraySize(permutation) != input->rank ||
        !vx_ib_symbol_axis(input, symbol, &input_axis, NULL) ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL)) return 0;
    for (uint32_t axis = 0; axis < output->rank; axis++) {
        int64_t source;
        if (!vx_ib_json_integer(cJSON_GetArrayItem(permutation, (int)axis),
                                &source) || source < 0 ||
            source >= (int64_t)input->rank || seen[source] ||
            !vx_ib_dimension_equal(output, axis, input, (uint32_t)source))
            return 0;
        seen[source] = 1u;
        if (axis == output_axis && (uint32_t)source != input_axis) return 0;
    }
    return 1;
}

static int vx_ib_safe_squeeze(const cJSON* node,
                              const VxDeclaredTensor* tensors,
                              size_t tensor_count, const char* symbol,
                              int unsqueeze) {
    static const char* const input_keys[] = {"input"};
    static const char* const param_keys[] = {"axes"};
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const cJSON* axes = params
        ? cJSON_GetObjectItemCaseSensitive(params, "axes") : NULL;
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    unsigned char changed[VX_MAX_TENSOR_RANK] = {0};
    uint32_t input_axis;
    uint32_t output_axis;
    uint32_t source = 0;
    uint32_t destination = 0;
    uint32_t target_rank;
    if (!vx_ib_object_exact(inputs, input_keys, 1u) ||
        !vx_ib_object_exact(params, param_keys, 1u) || !input || !output ||
        !cJSON_IsArray(axes) ||
        !vx_ib_symbol_axis(input, symbol, &input_axis, NULL) ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL)) return 0;
    target_rank = unsqueeze ? output->rank : input->rank;
    if ((unsqueeze && output->rank != input->rank +
            (uint32_t)cJSON_GetArraySize(axes)) ||
        (!unsqueeze && input->rank != output->rank +
            (uint32_t)cJSON_GetArraySize(axes))) return 0;
    for (const cJSON* item = axes->child; item; item = item->next) {
        uint32_t axis;
        if (!vx_ib_normalized_axis(item, target_rank, &axis) || changed[axis])
            return 0;
        changed[axis] = 1u;
        if ((unsqueeze && axis == output_axis) ||
            (!unsqueeze && axis == input_axis)) return 0;
        if (unsqueeze) {
            if (output->kinds[axis] != VX_DIMENSION_FIXED ||
                output->minimums[axis] != 1) return 0;
        } else if (input->kinds[axis] != VX_DIMENSION_FIXED ||
                   input->minimums[axis] != 1) return 0;
    }
    while (source < input->rank && destination < output->rank) {
        if (unsqueeze && changed[destination]) {
            destination++;
            continue;
        }
        if (!unsqueeze && changed[source]) {
            source++;
            continue;
        }
        if (!vx_ib_dimension_equal(input, source, output, destination))
            return 0;
        source++;
        destination++;
    }
    while (unsqueeze && destination < output->rank && changed[destination])
        destination++;
    while (!unsqueeze && source < input->rank && changed[source]) source++;
    return source == input->rank && destination == output->rank;
}

static int vx_ib_safe_slice(const cJSON* node,
                            const VxDeclaredTensor* tensors,
                            size_t tensor_count, const char* symbol) {
    static const char* const input_keys[] = {"input"};
    static const char* const param_keys[] = {
        "starts", "ends", "axes", "steps"
    };
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const cJSON* starts = params
        ? cJSON_GetObjectItemCaseSensitive(params, "starts") : NULL;
    const cJSON* ends = params
        ? cJSON_GetObjectItemCaseSensitive(params, "ends") : NULL;
    const cJSON* axes = params
        ? cJSON_GetObjectItemCaseSensitive(params, "axes") : NULL;
    const cJSON* steps = params
        ? cJSON_GetObjectItemCaseSensitive(params, "steps") : NULL;
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    unsigned char seen[VX_MAX_TENSOR_RANK] = {0};
    uint32_t input_axis;
    uint32_t output_axis;
    int count;
    if (!vx_ib_object_exact(inputs, input_keys, 1u) ||
        !vx_ib_object_exact(params, param_keys, 4u) || !input || !output ||
        input->rank != output->rank || !cJSON_IsArray(starts) ||
        !cJSON_IsArray(ends) || !cJSON_IsArray(axes) ||
        !cJSON_IsArray(steps) ||
        (count = cJSON_GetArraySize(starts)) != cJSON_GetArraySize(ends) ||
        count != cJSON_GetArraySize(axes) ||
        count != cJSON_GetArraySize(steps) ||
        !vx_ib_symbol_axis(input, symbol, &input_axis, NULL) ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL) ||
        input_axis != output_axis) return 0;
    for (int index = 0; index < count; index++) {
        uint32_t axis;
        int64_t step;
        if (!vx_ib_normalized_axis(cJSON_GetArrayItem(axes, index),
                                   input->rank, &axis) || axis == input_axis ||
            seen[axis] ||
            !vx_ib_json_integer(cJSON_GetArrayItem(steps, index), &step) ||
            step != 1) return 0;
        seen[axis] = 1u;
    }
    return 1;
}

static int vx_ib_safe_concat(const cJSON* node,
                             const VxDeclaredTensor* tensors,
                             size_t tensor_count, const char* symbol) {
    static const char* const param_keys[] = {"axis"};
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    uint32_t output_axis;
    uint32_t concat_axis;
    size_t count = 0;
    if (!cJSON_IsObject(inputs) || !inputs->child || !inputs->child->next ||
        !vx_ib_object_exact(params, param_keys, 1u) || !output ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL) ||
        !vx_ib_normalized_axis(
            cJSON_GetObjectItemCaseSensitive(params, "axis"), output->rank,
            &concat_axis) || concat_axis == output_axis) return 0;
    for (const cJSON* reference = inputs->child; reference;
         reference = reference->next) {
        const VxDeclaredTensor* input =
            cJSON_IsString(reference) && reference->valuestring
            ? vx_declared_tensor_find(tensors, tensor_count,
                                      reference->valuestring) : NULL;
        uint32_t input_axis;
        if (!input || input->rank != output->rank ||
            !vx_ib_symbol_axis(input, symbol, &input_axis, NULL) ||
            input_axis != output_axis) return 0;
        count++;
    }
    return count >= 2u;
}

static int vx_ib_safe_expand(const cJSON* node,
                             const VxDeclaredTensor* tensors,
                             size_t tensor_count, const char* symbol) {
    static const char* const input_keys[] = {"input"};
    static const char* const param_keys[] = {"shape"};
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    uint32_t input_axis;
    uint32_t output_axis;
    int64_t aligned;
    if (!vx_ib_object_exact(inputs, input_keys, 1u) ||
        !vx_ib_object_exact(params, param_keys, 1u) || !output ||
        !vx_ib_logical_shape_matches_json(
            output, cJSON_GetObjectItemCaseSensitive(params, "shape")) ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL)) return 0;
    if (!input)
        return vx_ib_reference_fixed(node, "input", tensors, tensor_count);
    if (!vx_ib_symbol_axis(input, symbol, &input_axis, NULL)) return 0;
    aligned = (int64_t)input_axis + (int64_t)output->rank -
        (int64_t)input->rank;
    return aligned == (int64_t)output_axis;
}

static int vx_ib_safe_gather(const cJSON* node, const char* op,
                             const VxDeclaredTensor* tensors,
                             size_t tensor_count, const char* symbol) {
    static const char* const embedding_inputs[] = {"input", "weight"};
    static const char* const gather_inputs[] = {"indices", "input"};
    static const char* const gather_params[] = {"axis"};
    const int embedding = !strcmp(op, "Embedding");
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const char* table_role = embedding ? "weight" : "input";
    const char* indices_role = embedding ? "input" : "indices";
    const VxDeclaredTensor* indices = vx_ib_reference_tensor(
        node, indices_role, tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    uint32_t indices_axis;
    uint32_t output_axis;
    int64_t gather_axis = 0;
    if (!vx_ib_object_exact(inputs,
                            embedding ? embedding_inputs : gather_inputs, 2u) ||
        (embedding ? vx_ib_object_count(params) != 0u :
            !vx_ib_object_exact(params, gather_params, 1u)) ||
        !vx_ib_reference_fixed(node, table_role, tensors, tensor_count) ||
        !indices || !output ||
        (!embedding &&
         (!vx_ib_json_integer(
              cJSON_GetObjectItemCaseSensitive(params, "axis"),
              &gather_axis) || gather_axis != 0)) ||
        !vx_ib_symbol_axis(indices, symbol, &indices_axis, NULL) ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL) ||
        indices_axis != output_axis || output->rank <= indices->rank)
        return 0;
    for (uint32_t axis = 0; axis < indices->rank; axis++)
        if (!vx_ib_dimension_equal(indices, axis, output, axis)) return 0;
    return 1;
}

static int vx_ib_safe_reduction(const cJSON* node, const char* op,
                                const VxDeclaredTensor* tensors,
                                size_t tensor_count, const char* symbol) {
    static const char* const input_keys[] = {"input"};
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    const cJSON* axis_item = params
        ? cJSON_GetObjectItemCaseSensitive(params, "axis") : NULL;
    const cJSON* keepdims = params
        ? cJSON_GetObjectItemCaseSensitive(params, "keepdims") : NULL;
    const cJSON* select_last = params
        ? cJSON_GetObjectItemCaseSensitive(params, "select_last_index") : NULL;
    uint32_t input_axis;
    uint32_t output_axis;
    uint32_t reduce_axis;
    uint32_t output_cursor = 0;
    int keep;
    if (!vx_ib_object_exact(inputs, input_keys, 1u) || !input || !output ||
        (!cJSON_IsTrue(keepdims) && !cJSON_IsFalse(keepdims)) ||
        !vx_ib_normalized_axis(axis_item, input->rank, &reduce_axis) ||
        !vx_ib_symbol_axis(input, symbol, &input_axis, NULL) ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL) ||
        reduce_axis == input_axis) return 0;
    if (!strcmp(op, "ArgMax")) {
        if (vx_ib_object_count(params) != (select_last ? 3u : 2u) ||
            (select_last && (!cJSON_IsNumber(select_last) ||
                             select_last->valuedouble != 0.0))) return 0;
    } else if (vx_ib_object_count(params) != 2u) return 0;
    keep = cJSON_IsTrue(keepdims);
    if (output->rank != input->rank - (keep ? 0u : 1u)) return 0;
    for (uint32_t axis = 0; axis < input->rank; axis++) {
        if (axis == reduce_axis) {
            if (keep && (output->kinds[output_cursor] != VX_DIMENSION_FIXED ||
                         output->minimums[output_cursor] != 1)) return 0;
            if (keep) output_cursor++;
            continue;
        }
        if (!vx_ib_dimension_equal(input, axis, output, output_cursor++))
            return 0;
    }
    return output_axis == (keep || reduce_axis > input_axis
        ? input_axis : input_axis - 1u);
}

static int vx_ib_safe_softmax(const cJSON* node,
                              const VxDeclaredTensor* tensors,
                              size_t tensor_count, const char* symbol) {
    static const char* const input_keys[] = {"input"};
    static const char* const param_keys[] = {"axis"};
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    uint32_t batch_axis;
    uint32_t softmax_axis;
    return vx_ib_object_exact(inputs, input_keys, 1u) &&
        vx_ib_object_exact(params, param_keys, 1u) && input &&
        vx_ib_preserves_shape(node, tensors, tensor_count, symbol) &&
        vx_ib_symbol_axis(input, symbol, &batch_axis, NULL) &&
        vx_ib_normalized_axis(
            cJSON_GetObjectItemCaseSensitive(params, "axis"), input->rank,
            &softmax_axis) && softmax_axis != batch_axis;
}

static int vx_ib_safe_dense(const VxModel* model, const cJSON* node,
                            const char* op,
                            const VxDeclaredTensor* tensors,
                            size_t tensor_count, const char* symbol) {
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const int has_bias = inputs &&
        cJSON_GetObjectItemCaseSensitive(inputs, "bias") != NULL;
    const char* const with_bias[] = {"bias", "input", "weight"};
    const char* const without_bias[] = {"input", "weight"};
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    const int linear = !strcmp(op, "Linear");
    const int quantized = !strcmp(op, "QLinear") || !strcmp(op, "QGemm");
    uint32_t input_axis;
    uint32_t output_axis;
    if (!vx_ib_object_exact(inputs, has_bias ? with_bias : without_bias,
                            has_bias ? 3u : 2u) || !input || !output ||
        input->rank < 2u || input->rank != output->rank ||
        !vx_ib_reference_fixed(node, "weight", tensors, tensor_count) ||
        (has_bias && !vx_ib_reference_fixed(
            node, "bias", tensors, tensor_count)) ||
        !vx_ib_symbol_axis(input, symbol, &input_axis, NULL) ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL) ||
        input_axis >= input->rank - 1u || input_axis != output_axis)
        return 0;
    if (linear) {
        static const char* const linear_params[] = {"weight_layout"};
        const cJSON* layout;
        if (!vx_ib_object_exact(params, linear_params, 1u) ||
            !(layout = cJSON_GetObjectItemCaseSensitive(
                params, "weight_layout")) || !cJSON_IsString(layout) ||
            !layout->valuestring || strcmp(layout->valuestring, "din_dout"))
            return 0;
    } else if (vx_ib_object_count(params) != 0u) return 0;
    if (!quantized) return 1;
    return vx_ib_per_tensor_quantized(
               model, node, "input", NULL, 0) &&
        vx_ib_per_tensor_quantized(model, node, NULL, output, 1);
}

static int vx_ib_safe_matmul(const VxModel* model, const cJSON* node,
                             const char* op,
                             const VxDeclaredTensor* tensors,
                             size_t tensor_count, const char* symbol) {
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const int batch = !strcmp(op, "BatchMatMul") ||
        !strcmp(op, "QBatchMatMul");
    const int quantized = op[0] == 'Q';
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    uint32_t output_axis;
    size_t dynamic_operands = 0;
    if (batch) {
        static const char* const roles[] = {"a", "b"};
        if (!vx_ib_object_exact(inputs, roles, 2u) ||
            vx_ib_object_count(params) != 0u || !output ||
            output->rank < 3u ||
            !vx_ib_symbol_axis(output, symbol, &output_axis, NULL) ||
            output_axis >= output->rank - 2u) return 0;
        for (size_t index = 0; index < 2u; index++) {
            const VxDeclaredTensor* operand = vx_ib_reference_tensor(
                node, roles[index], tensors, tensor_count);
            uint32_t operand_axis;
            int64_t aligned;
            if (!operand) {
                if (!vx_ib_reference_fixed(node, roles[index], tensors,
                                           tensor_count)) return 0;
                continue;
            }
            if (operand->rank < 2u ||
                !vx_ib_symbol_axis(operand, symbol, &operand_axis, NULL) ||
                operand_axis >= operand->rank - 2u) return 0;
            aligned = (int64_t)operand_axis + (int64_t)output->rank -
                (int64_t)operand->rank;
            if (aligned != (int64_t)output_axis ||
                (quantized && !vx_ib_per_tensor_quantized(
                    model, node, roles[index], NULL, 0))) return 0;
            dynamic_operands++;
        }
        return dynamic_operands > 0u &&
            (!quantized || vx_ib_per_tensor_quantized(
                model, node, NULL, output, 1));
    }
    {
        const int has_bias = inputs &&
            cJSON_GetObjectItemCaseSensitive(inputs, "bias") != NULL;
        const char* const with_bias[] = {"bias", "input", "weight"};
        const char* const without_bias[] = {"input", "weight"};
        const VxDeclaredTensor* input = vx_ib_reference_tensor(
            node, "input", tensors, tensor_count);
        uint32_t input_axis;
        if (!vx_ib_object_exact(inputs, has_bias ? with_bias : without_bias,
                                has_bias ? 3u : 2u) ||
            vx_ib_object_count(params) != 0u || !input || !output ||
            input->rank < 2u || input->rank != output->rank ||
            !vx_ib_reference_fixed(node, "weight", tensors, tensor_count) ||
            (has_bias && !vx_ib_reference_fixed(
                node, "bias", tensors, tensor_count)) ||
            !vx_ib_symbol_axis(input, symbol, &input_axis, NULL) ||
            !vx_ib_symbol_axis(output, symbol, &output_axis, NULL) ||
            input_axis >= input->rank - 1u || input_axis != output_axis)
            return 0;
        return !quantized ||
            (vx_ib_per_tensor_quantized(
                 model, node, "input", NULL, 0) &&
             vx_ib_per_tensor_quantized(model, node, NULL, output, 1));
    }
}

static int vx_ib_safe_conv(const VxModel* model, const cJSON* node,
                           const char* op,
                           const VxDeclaredTensor* tensors,
                           size_t tensor_count, const char* symbol) {
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const int has_bias = inputs &&
        cJSON_GetObjectItemCaseSensitive(inputs, "bias") != NULL;
    const char* const with_bias[] = {"bias", "input", "weight"};
    const char* const without_bias[] = {"input", "weight"};
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    const cJSON* data_layout = params
        ? cJSON_GetObjectItemCaseSensitive(params, "data_layout") : NULL;
    const cJSON* weight_layout = params
        ? cJSON_GetObjectItemCaseSensitive(params, "weight_layout") : NULL;
    const int quantized = !strcmp(op, "QConv2D");
    if (!vx_ib_object_exact(inputs, has_bias ? with_bias : without_bias,
                            has_bias ? 3u : 2u) || !input || !output ||
        input->rank != 4u || output->rank != 4u ||
        !vx_ib_leading_symbol(input, symbol) ||
        !vx_ib_leading_symbol(output, symbol) ||
        !vx_ib_reference_fixed(node, "weight", tensors, tensor_count) ||
        (has_bias && !vx_ib_reference_fixed(
            node, "bias", tensors, tensor_count)) ||
        !cJSON_IsString(data_layout) || !data_layout->valuestring ||
        strcmp(data_layout->valuestring, "NHWC") ||
        !cJSON_IsString(weight_layout) || !weight_layout->valuestring ||
        strcmp(weight_layout->valuestring, quantized ? "OHWI" : "HWIO"))
        return 0;
    return !quantized ||
        (vx_ib_per_tensor_quantized(model, node, "input", NULL, 0) &&
         vx_ib_per_tensor_quantized(model, node, NULL, output, 1));
}

static int vx_ib_safe_normalization(const cJSON* node, const char* op,
                                    const VxDeclaredTensor* tensors,
                                    size_t tensor_count,
                                    const char* symbol) {
    static const char* const roles[] = {"bias", "input", "weight"};
    const cJSON* inputs = vx_ib_node_inputs(node);
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    uint32_t input_axis;
    uint32_t output_axis;
    const int layer = !strcmp(op, "LayerNorm");
    if (!vx_ib_object_exact(inputs, roles, 3u) || !input || !output ||
        !vx_ib_reference_fixed(node, "weight", tensors, tensor_count) ||
        !vx_ib_reference_fixed(node, "bias", tensors, tensor_count) ||
        !vx_ib_symbol_axis(input, symbol, &input_axis, NULL) ||
        !vx_ib_symbol_axis(output, symbol, &output_axis, NULL) ||
        input_axis != output_axis || !vx_ib_shape_equal(input, output))
        return 0;
    return layer ? input->rank >= 2u && input_axis < input->rank - 1u
                 : input->rank == 4u && input_axis == 0u;
}

static int vx_ib_safe_quantize(const VxModel* model, const cJSON* node,
                               const char* op,
                               const VxDeclaredTensor* tensors,
                               size_t tensor_count, const char* symbol) {
    static const char* const roles[] = {"input", "scale", "zero_point"};
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    const VxDeclaredTensor* input = vx_ib_reference_tensor(
        node, "input", tensors, tensor_count);
    const VxDeclaredTensor* output = vx_ib_single_output(
        node, tensors, tensor_count);
    const int quantize = !strcmp(op, "QuantizeLinear");
    return vx_ib_object_exact(inputs, roles, 3u) &&
        vx_ib_object_count(params) == 0u && input && output &&
        vx_ib_reference_fixed(node, "scale", tensors, tensor_count) &&
        vx_ib_reference_fixed(node, "zero_point", tensors, tensor_count) &&
        vx_ib_preserves_shape(node, tensors, tensor_count, symbol) &&
        vx_ib_per_tensor_quantized(model, node, quantize ? NULL : "input",
                                   quantize ? output : NULL, quantize);
}

static int vx_ib_safe_node(const VxModel* model, const cJSON* node,
                           const VxDeclaredTensor* tensors,
                           size_t tensor_count, const char* symbol) {
    const cJSON* op_item = node
        ? cJSON_GetObjectItemCaseSensitive(node, "opType") : NULL;
    const char* op = cJSON_IsString(op_item) ? op_item->valuestring : NULL;
    const cJSON* inputs = vx_ib_node_inputs(node);
    const cJSON* params = vx_ib_node_params(node);
    static const char* const unary[] = {"input"};
    static const char* const binary[] = {"a", "b"};
    static const char* const where_roles[] = {"a", "b", "condition"};
    if (!op) return 0;
    if (!strcmp(op, "Identity") || !strcmp(op, "Sigmoid") ||
        !strcmp(op, "SiLU") || !strcmp(op, "Not"))
        return vx_ib_object_exact(inputs, unary, 1u) &&
            vx_ib_object_count(params) == 0u &&
            vx_ib_preserves_shape(node, tensors, tensor_count, symbol);
    if (!strcmp(op, "GELU")) {
        static const char* const gelu_params[] = {"approximate"};
        const cJSON* approximate = params
            ? cJSON_GetObjectItemCaseSensitive(params, "approximate") : NULL;
        return vx_ib_object_exact(inputs, unary, 1u) &&
            vx_ib_object_exact(params, gelu_params, 1u) &&
            cJSON_IsString(approximate) && approximate->valuestring &&
            !strcmp(approximate->valuestring, "none") &&
            vx_ib_preserves_shape(node, tensors, tensor_count, symbol);
    }
    if (!strcmp(op, "Clip")) {
        static const char* const clip_params[] = {"min", "max"};
        const cJSON* minimum = params
            ? cJSON_GetObjectItemCaseSensitive(params, "min") : NULL;
        const cJSON* maximum = params
            ? cJSON_GetObjectItemCaseSensitive(params, "max") : NULL;
        return vx_ib_object_exact(inputs, unary, 1u) &&
            vx_ib_object_exact(params, clip_params, 2u) &&
            cJSON_IsNumber(minimum) && isfinite(minimum->valuedouble) &&
            cJSON_IsNumber(maximum) && isfinite(maximum->valuedouble) &&
            minimum->valuedouble <= maximum->valuedouble &&
            vx_ib_preserves_shape(node, tensors, tensor_count, symbol);
    }
    if (!strcmp(op, "Cast")) {
        static const char* const cast_params[] = {"to"};
        const cJSON* to = params
            ? cJSON_GetObjectItemCaseSensitive(params, "to") : NULL;
        return vx_ib_object_exact(inputs, unary, 1u) &&
            vx_ib_object_exact(params, cast_params, 1u) &&
            cJSON_IsString(to) && to->valuestring &&
            (!strcmp(to->valuestring, "float32") ||
             !strcmp(to->valuestring, "int32")) &&
            vx_ib_preserves_shape(node, tensors, tensor_count, symbol);
    }
    if (!strcmp(op, "Add") || !strcmp(op, "Sub") ||
        !strcmp(op, "Mul") || !strcmp(op, "Div") ||
        !strcmp(op, "Equal") || !strcmp(op, "GreaterOrEqual"))
        return vx_ib_safe_broadcast(node, binary, 2u, tensors, tensor_count,
                                    symbol);
    if (!strcmp(op, "Where"))
        return vx_ib_safe_broadcast(node, where_roles, 3u, tensors,
                                    tensor_count, symbol);
    if (!strcmp(op, "Reshape"))
        return vx_ib_safe_reshape(node, tensors, tensor_count, symbol);
    if (!strcmp(op, "Transpose"))
        return vx_ib_safe_transpose(node, tensors, tensor_count, symbol);
    if (!strcmp(op, "Squeeze") || !strcmp(op, "Unsqueeze"))
        return vx_ib_safe_squeeze(node, tensors, tensor_count, symbol,
                                  !strcmp(op, "Unsqueeze"));
    if (!strcmp(op, "Slice"))
        return vx_ib_safe_slice(node, tensors, tensor_count, symbol);
    if (!strcmp(op, "Concat"))
        return vx_ib_safe_concat(node, tensors, tensor_count, symbol);
    if (!strcmp(op, "Expand"))
        return vx_ib_safe_expand(node, tensors, tensor_count, symbol);
    if (!strcmp(op, "Gather") || !strcmp(op, "Embedding"))
        return vx_ib_safe_gather(node, op, tensors, tensor_count, symbol);
    if (!strcmp(op, "ReduceSum") || !strcmp(op, "ArgMax"))
        return vx_ib_safe_reduction(node, op, tensors, tensor_count, symbol);
    if (!strcmp(op, "Softmax"))
        return vx_ib_safe_softmax(node, tensors, tensor_count, symbol);
    if (!strcmp(op, "Linear") || !strcmp(op, "QLinear") ||
        !strcmp(op, "QGemm"))
        return vx_ib_safe_dense(model, node, op, tensors, tensor_count,
                                symbol);
    if (!strcmp(op, "MatMul") || !strcmp(op, "QMatMul") ||
        !strcmp(op, "BatchMatMul") || !strcmp(op, "QBatchMatMul"))
        return vx_ib_safe_matmul(model, node, op, tensors, tensor_count,
                                 symbol);
    if (!strcmp(op, "Conv2D") || !strcmp(op, "QConv2D"))
        return vx_ib_safe_conv(model, node, op, tensors, tensor_count,
                               symbol);
    if (!strcmp(op, "LayerNorm") || !strcmp(op, "GroupNorm"))
        return vx_ib_safe_normalization(node, op, tensors, tensor_count,
                                        symbol);
    if (!strcmp(op, "QuantizeLinear") || !strcmp(op, "DequantizeLinear"))
        return vx_ib_safe_quantize(model, node, op, tensors, tensor_count,
                                   symbol);
    return 0;
}

static void vx_ib_reject(VxIndependentBatchProof* proof,
                         const char* reason, const cJSON* node) {
    const cJSON* id = node
        ? cJSON_GetObjectItemCaseSensitive(node, "id") : NULL;
    proof->supported = 0;
    snprintf(proof->reason, sizeof(proof->reason), "%s",
             reason ? reason : "typed independent-batch proof failed");
    if (cJSON_IsString(id) && id->valuestring)
        snprintf(proof->failed_node, sizeof(proof->failed_node), "%s",
                 id->valuestring);
}

static void vx_independent_batch_prove(const VxModel* model,
                                       VxIndependentBatchProof* proof) {
    VxDeclaredTensor* tensors = NULL;
    size_t tensor_count = 0;
    const cJSON* nodes;
    const char* symbol;
    VxStatus status;
    memset(proof, 0, sizeof(*proof));
    proof->batch_axis = 0u;
    proof->min_batch = 1u;
    proof->multiple_of = 1u;
    if (!model || !model->input_count || !model->output_count ||
        !model->inputs[0].rank ||
        model->inputs[0].kinds[0] != VX_DIMENSION_SYMBOLIC ||
        !(symbol = model->inputs[0].symbols[0]) || !symbol[0]) {
        vx_ib_reject(proof,
                     "public inputs do not declare one leading symbolic batch axis",
                     NULL);
        return;
    }
    for (size_t index = 0; index < model->input_count; index++)
        if (!vx_ib_leading_symbol(&model->inputs[index], symbol)) {
            vx_ib_reject(proof,
                         "every public input must carry the same leading batch symbol exactly once",
                         NULL);
            return;
        }
    for (size_t index = 0; index < model->output_count; index++)
        if (!vx_ib_leading_symbol(&model->outputs[index], symbol)) {
            vx_ib_reject(proof,
                         "every public output must carry the same leading batch symbol exactly once",
                         NULL);
            return;
        }
    if (model->inputs[0].minimums[0] != 1 ||
        model->inputs[0].multiples[0] != 1 ||
        model->inputs[0].maximums[0] < 2 ||
        (uint64_t)model->inputs[0].maximums[0] > UINT32_MAX) {
        vx_ib_reject(proof,
                     "batch symbol requires min=1, multiple_of=1, and a finite max greater than one",
                     NULL);
        return;
    }
    proof->max_batch = (uint32_t)model->inputs[0].maximums[0];
    status = vx_logical_tensors_load(model, &tensors, &tensor_count);
    if (status != VX_STATUS_OK) {
        vx_ib_reject(proof,
                     "logical tensor table could not be built for batch proof",
                     NULL);
        return;
    }
    for (size_t index = 0; index < tensor_count; index++) {
        size_t occurrences = 0;
        (void)vx_ib_symbol_axis(&tensors[index], symbol, NULL, &occurrences);
        if (occurrences > 1u ||
            (occurrences == 1u && !vx_ib_quantization_avoids_axis(
                model, &tensors[index], symbol))) {
            vx_ib_reject(proof,
                         occurrences > 1u
                             ? "an execution tensor contains the batch symbol more than once"
                             : "per-axis quantization uses the public batch axis",
                         NULL);
            goto done;
        }
    }
    nodes = cJSON_GetObjectItemCaseSensitive(model->logical_graph, "nodes");
    if (!cJSON_IsArray(nodes) || !nodes->child) {
        vx_ib_reject(proof, "batch proof requires a non-empty typed graph",
                     NULL);
        goto done;
    }
    for (const cJSON* node = nodes->child; node; node = node->next) {
        if (!vx_ib_safe_node(model, node, tensors, tensor_count, symbol)) {
            vx_ib_reject(
                proof,
                "operator configuration is outside the typed independent-batch proof",
                node);
            goto done;
        }
        proof->covered_nodes++;
    }
    proof->supported = 1;
    proof->reason[0] = '\0';
    proof->failed_node[0] = '\0';
    snprintf(proof->identity, sizeof(proof->identity), "%s:%s",
             VX_BACKEND_INDEPENDENT_BATCH_PROOF_PROTOCOL,
             model->graph_fingerprint);
done:
    vx_declared_outputs_free(tensors, tensor_count);
}
