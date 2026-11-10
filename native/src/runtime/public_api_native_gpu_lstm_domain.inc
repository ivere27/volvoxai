static int vx_native_gpu_lstm_proven(const cJSON* node,
        const VxDeclaredTensor* tensors, size_t tensor_count,
        const VxEngineState* validation, const VxDeclaredTensor* input,
        const VxDeclaredTensor* output, const VxNativeGpuDomainLimits* limits) {
    static const char* const names[] = {"weight", "recurrent_weight", "bias", "initial_h", "initial_c"};
    VxDeclaredTensor fixed[5] = {{0}};
    const VxDeclaredTensor* values[5];
    for (unsigned i = 0; i < 5; ++i)
        values[i] = vx_native_gpu_node_input_domain(node, names[i], tensors, tensor_count, validation, &fixed[i]);
    const VxDeclaredTensor* w = values[0];
    const VxDeclaredTensor* r = values[1];
    int64_t direction;
    const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    if (!input || !output || !w || !r || input->rank != 3u || output->rank != 4u ||
        w->rank != 3u || r->rank != 3u ||
        !vx_native_gpu_integer_parameter(params, "direction", 0, 0, 2, &direction) ||
        !vx_native_gpu_cuda_typed_span_proven(input, VX_DTYPE_F32, limits) ||
        !vx_native_gpu_cuda_typed_span_proven(output, VX_DTYPE_F32, limits)) return 0;
    int64_t directions = direction == 2 ? 2 : 1;
    int64_t hidden = r->maximums[2];
    if (hidden <= 0 || hidden > 1024 || input->minimums[2] != input->maximums[2] ||
        output->minimums[1] != directions || output->maximums[1] != directions ||
        output->minimums[3] != hidden || output->maximums[3] != hidden ||
        !vx_declared_dimensions_provably_equal(input, 0u, output, 0u) ||
        !vx_declared_dimensions_provably_equal(input, 1u, output, 2u) ||
        input->maximums[1] > 65535 / directions ||
        (uint64_t)input->maximums[1] * (uint64_t)directions > limits->maximum_grid[0] ||
        limits->maximum_block[0] < 256u || limits->maximum_threads_per_block < 256u ||
        limits->maximum_shared_memory_bytes < 3u * 1024u * sizeof(float) ||
        limits->maximum_storage_bindings < 7u || limits->maximum_uniform_bindings < 1u ||
        limits->maximum_uniform_bytes < 32u) return 0;
    int64_t expected_w[3] = {directions, 4 * hidden, input->maximums[2]};
    int64_t expected_r[3] = {directions, 4 * hidden, hidden};
    for (unsigned axis = 0; axis < 3; ++axis)
        if (w->minimums[axis] != expected_w[axis] || w->maximums[axis] != expected_w[axis] ||
            r->minimums[axis] != expected_r[axis] || r->maximums[axis] != expected_r[axis]) return 0;
    for (unsigned i = 0; i < 5; ++i) {
        const VxDeclaredTensor* value = values[i];
        if (!value) continue;
        uint64_t elements;
        if (!vx_native_gpu_cuda_typed_span_proven(value, VX_DTYPE_F32, limits) ||
            !vx_native_gpu_tensor_maximum_elements(value, &elements) || elements > UINT32_MAX / 4u) return 0;
        if (i < 2) continue;
        if (value->rank != (i == 2 ? 2u : 3u) || value->minimums[0] != directions || value->maximums[0] != directions) return 0;
        if (i == 2) {
            if (value->minimums[1] != 8 * hidden || value->maximums[1] != 8 * hidden) return 0;
        } else if (!vx_declared_dimensions_provably_equal(input, 1u, value, 1u) ||
                value->minimums[2] != hidden || value->maximums[2] != hidden) return 0;
    }
    uint64_t input_elements, output_elements;
    return vx_native_gpu_tensor_maximum_elements(input, &input_elements) &&
        vx_native_gpu_tensor_maximum_elements(output, &output_elements) &&
        input_elements <= UINT32_MAX / 4u && output_elements <= UINT32_MAX / 4u;
}

static int vx_native_gpu_packed_lstm_proven(const cJSON* node,
        const VxDeclaredTensor* tensors, size_t tensor_count,
        const VxEngineState* validation, const VxDeclaredTensor* input,
        const VxDeclaredTensor* output, const VxNativeGpuDomainLimits* limits) {
    static const char* const names[] = {
        "weight", "scale", "recurrent_weight", "recurrent_scale",
        "bias", "initial_h", "initial_c"
    };
    VxDeclaredTensor fixed[7] = {{0}};
    const VxDeclaredTensor* values[7];
    for (unsigned i = 0; i < 7; ++i)
        values[i] = vx_native_gpu_node_input_domain(node, names[i], tensors, tensor_count, validation, &fixed[i]);
    if (!values[1])
        values[1] = vx_native_gpu_node_input_domain(node, "weight_scale", tensors, tensor_count, validation, &fixed[1]);
    const VxDeclaredTensor* w = values[0];
    const VxDeclaredTensor* ws = values[1];
    const VxDeclaredTensor* r = values[2];
    const VxDeclaredTensor* rs = values[3];
    int64_t direction, group_size, weight_bits;
    const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    if (!input || !output || !w || !ws || !r || !rs ||
        input->rank != 3u || output->rank != 4u ||
        w->rank != 3u || ws->rank != 3u || r->rank != 3u || rs->rank != 3u ||
        !vx_native_gpu_integer_parameter(params, "direction", 0, 0, 2, &direction) ||
        !vx_native_gpu_integer_parameter(params, "group_size", 0, 32, 256, &group_size) ||
        !vx_native_gpu_integer_parameter(params, "weight_bits", 4, 2, 8, &weight_bits) ||
        !vx_native_gpu_cuda_typed_span_proven(input, VX_DTYPE_F32, limits) ||
        !vx_native_gpu_cuda_typed_span_proven(output, VX_DTYPE_F32, limits)) return 0;
    if (group_size != 32 && group_size != 64 && group_size != 128 && group_size != 256) return 0;
    if (weight_bits != 2 && weight_bits != 3 && weight_bits != 4 && weight_bits != 8) return 0;
    int64_t directions = direction == 2 ? 2 : 1;
    int64_t gates_hidden = r->maximums[1];
    if (gates_hidden < 4 || (gates_hidden % 4) != 0) return 0;
    int64_t hidden = gates_hidden / 4;
    int64_t width = input->maximums[2];
    if (hidden <= 0 || hidden > 1024 || width <= 0 ||
        input->minimums[2] != input->maximums[2] ||
        output->minimums[1] != directions || output->maximums[1] != directions ||
        output->minimums[3] != hidden || output->maximums[3] != hidden ||
        !vx_declared_dimensions_provably_equal(input, 0u, output, 0u) ||
        !vx_declared_dimensions_provably_equal(input, 1u, output, 2u) ||
        input->maximums[1] > 65535 / directions ||
        (uint64_t)input->maximums[1] * (uint64_t)directions > limits->maximum_grid[0] ||
        limits->maximum_block[0] < 256u || limits->maximum_threads_per_block < 256u ||
        limits->maximum_shared_memory_bytes < 3u * 1024u * sizeof(float) ||
        limits->maximum_storage_bindings < 7u || limits->maximum_uniform_bindings < 1u ||
        limits->maximum_uniform_bytes < 32u) return 0;
    uint64_t input_elements, output_elements;
    return vx_native_gpu_tensor_maximum_elements(input, &input_elements) &&
        vx_native_gpu_tensor_maximum_elements(output, &output_elements) &&
        input_elements <= UINT32_MAX / 4u && output_elements <= UINT32_MAX / 4u;
}
