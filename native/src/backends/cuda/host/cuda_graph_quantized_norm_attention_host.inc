int cuda_graph_qadd_i8u8(const void* a, uint32_t a_elements,
                         const void* b, uint32_t b_elements,
                         void* output, uint32_t output_elements,
                         float a_scale, int32_t a_zero_point,
                         float b_scale, int32_t b_zero_point,
                         float output_scale, int32_t output_zero_point,
                         uint32_t a_dtype, uint32_t b_dtype,
                         uint32_t output_dtype, uint32_t relu) {
    CudaTensorView a_view, b_view, output_view;
    void* args[14];
    if (!a || !b || !output || !output_elements || a_elements != output_elements ||
        b_elements != output_elements || !byte_dtype_valid(a_dtype) ||
        !byte_dtype_valid(b_dtype) || !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(a_dtype, a_zero_point) ||
        !byte_zero_point_valid(b_dtype, b_zero_point) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) || relu > 2u ||
        !isfinite(a_scale) || a_scale <= 0.0f ||
        !isfinite(b_scale) || b_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        ranges_overlap(output, output_elements, a, a_elements) ||
        ranges_overlap(output, output_elements, b, b_elements)) return 0;
    if (!graph_ensure_input_view(a, a_elements, 0, &a_view) ||
        !graph_ensure_input_view(b, b_elements, 0, &b_view) ||
        !graph_output_view(output, output_elements, &output_view)) return 0;
    args[0] = &a_view.device;
    args[1] = &b_view.device;
    args[2] = &output_view.device;
    args[3] = &output_elements;
    args[4] = &a_scale;
    args[5] = &a_zero_point;
    args[6] = &b_scale;
    args[7] = &b_zero_point;
    args[8] = &output_scale;
    args[9] = &output_zero_point;
    args[10] = &a_dtype;
    args[11] = &b_dtype;
    args[12] = &output_dtype;
    args[13] = &relu;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qadd), output_elements, args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}

static int cuda_graph_qunary_i8u8(const void* input, void* output,
                                  uint32_t elements, float input_scale,
                                  int32_t input_zero_point, float output_scale,
                                  int32_t output_zero_point, uint32_t input_dtype,
                                  uint32_t output_dtype, int operation) {
    CudaTensorView input_view, output_view;
    CudaTensorSlot* table_slot;
    CudaQactLutEntry* table;
    int rebuilt;
    void* args[4];
    if (!input || !output || !elements || !byte_dtype_valid(input_dtype) ||
        !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(input_dtype, input_zero_point) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        ranges_overlap(output, elements, input, elements)) return 0;
    table = qact_lut_acquire(input_scale, input_zero_point, output_scale,
        output_zero_point, input_dtype, output_dtype, operation, &rebuilt);
    if (!table) return 0;
    table_slot = graph_get_slot(table->values, sizeof(table->values), 1);
    if (!table_slot) return 0;
    if (rebuilt) {
        table_slot->host_dirty = 1;
        table_slot->device_dirty = 0;
    }
    if (!graph_ensure_input_view(input, elements, 0, &input_view)) return 0;
    table_slot = graph_ensure_input(table->values, sizeof(table->values), 1);
    if (!table_slot || !graph_output_view(output, elements, &output_view))
        return 0;
    args[0] = &input_view.device;
    args[1] = &output_view.device;
    args[2] = &elements;
    args[3] = &table_slot->device;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qunary), elements, args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}

int cuda_graph_qsilu_i8u8(const void* input, void* output, uint32_t elements,
                          float input_scale, int32_t input_zero_point,
                          float output_scale, int32_t output_zero_point,
                          uint32_t input_dtype, uint32_t output_dtype) {
    return cuda_graph_qunary_i8u8(input, output, elements, input_scale,
        input_zero_point, output_scale, output_zero_point, input_dtype,
        output_dtype, 0);
}
int cuda_graph_qgelu_i8u8(const void* input, void* output, uint32_t elements,
                          float input_scale, int32_t input_zero_point,
                          float output_scale, int32_t output_zero_point,
                          uint32_t input_dtype, uint32_t output_dtype) {
    return cuda_graph_qunary_i8u8(input, output, elements, input_scale,
        input_zero_point, output_scale, output_zero_point, input_dtype,
        output_dtype, 1);
}

int cuda_graph_qgroupnorm_i8u8(const void* input, const float* weight,
                               const float* bias, void* output,
                               uint32_t batch, uint32_t height,
                               uint32_t width, uint32_t channels,
                               uint32_t groups, float input_scale,
                               int32_t input_zero_point, float output_scale,
                               int32_t output_zero_point, float epsilon,
                               uint32_t input_dtype, uint32_t output_dtype) {
    CudaTensorSlot *input_slot, *weight_slot, *bias_slot, *output_slot;
    uint32_t elements, tasks;
    size_t bytes, metadata_bytes, task_bytes;
    void* args[16];
    if (!input || !weight || !bias || !output || !batch || !height || !width ||
        !channels || !groups || channels % groups ||
        !byte_dtype_valid(input_dtype) || !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(input_dtype, input_zero_point) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !checked_product4_bytes(batch, height, width, channels, 1,
                                &elements, &bytes) ||
        !checked_elements_bytes((uint64_t)batch * groups, 1,
                                &tasks, &task_bytes) ||
        (uint64_t)channels * sizeof(float) > SIZE_MAX) return 0;
    metadata_bytes = (size_t)channels * sizeof(float);
    if (ranges_overlap(output, bytes, input, bytes) ||
        ranges_overlap(output, bytes, weight, metadata_bytes) ||
        ranges_overlap(output, bytes, bias, metadata_bytes)) return 0;
    for (uint32_t channel = 0; channel < channels; channel++)
        if (!isfinite(weight[channel]) || !isfinite(bias[channel])) return 0;
    input_slot = graph_ensure_input(input, bytes, 0);
    weight_slot = graph_ensure_input(weight, metadata_bytes, 1);
    bias_slot = graph_ensure_input(bias, metadata_bytes, 1);
    output_slot = graph_output(output, bytes);
    if (!input_slot || !weight_slot || !bias_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &weight_slot->device;
    args[2] = &bias_slot->device;
    args[3] = &output_slot->device;
    args[4] = &batch;
    args[5] = &height;
    args[6] = &width;
    args[7] = &channels;
    args[8] = &groups;
    args[9] = &input_scale;
    args[10] = &input_zero_point;
    args[11] = &output_scale;
    args[12] = &output_zero_point;
    args[13] = &epsilon;
    args[14] = &input_dtype;
    args[15] = &output_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qgroupnorm), tasks, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_qlayernorm_i8u8(const void* input, const float* weight,
                               const float* bias, void* output,
                               uint32_t rows, uint32_t d_model,
                               float input_scale, int32_t input_zero_point,
                               float output_scale, int32_t output_zero_point,
                               float epsilon, uint32_t input_dtype,
                               uint32_t output_dtype) {
    CudaTensorView input_view, output_view;
    CudaTensorSlot *weight_slot, *bias_slot;
    uint32_t elements;
    size_t bytes, metadata_bytes;
    void* args[13];
    if (!input || !weight || !bias || !output || !rows || !d_model ||
        !byte_dtype_valid(input_dtype) || !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(input_dtype, input_zero_point) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !checked_elements_bytes((uint64_t)rows * d_model, 1,
                                &elements, &bytes) ||
        (uint64_t)d_model * sizeof(float) > SIZE_MAX) return 0;
    metadata_bytes = (size_t)d_model * sizeof(float);
    if (ranges_overlap(output, bytes, input, bytes) ||
        ranges_overlap(output, bytes, weight, metadata_bytes) ||
        ranges_overlap(output, bytes, bias, metadata_bytes)) return 0;
    for (uint32_t channel = 0; channel < d_model; channel++)
        if (!isfinite(weight[channel]) || !isfinite(bias[channel])) return 0;
    if (!graph_ensure_input_view(input, bytes, 0, &input_view)) return 0;
    weight_slot = graph_ensure_input(weight, metadata_bytes, 1);
    bias_slot = graph_ensure_input(bias, metadata_bytes, 1);
    if (!graph_output_view(output, bytes, &output_view) || !weight_slot ||
        !bias_slot) return 0;
    args[0] = &input_view.device;
    args[1] = &weight_slot->device;
    args[2] = &bias_slot->device;
    args[3] = &output_view.device;
    args[4] = &rows;
    args[5] = &d_model;
    args[6] = &input_scale;
    args[7] = &input_zero_point;
    args[8] = &output_scale;
    args[9] = &output_zero_point;
    args[10] = &epsilon;
    args[11] = &input_dtype;
    args[12] = &output_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qlayernorm), rows, args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}

int cuda_graph_qsdpa_range_i8u8(const void* query, const void* key,
                                const void* value, const int32_t* mask,
                                void* output, uint32_t batch,
                                uint32_t query_sequence,
                                uint32_t key_sequence, uint32_t d_model,
                                uint32_t heads, float query_scale,
                                int32_t query_zero_point, float key_scale,
                                int32_t key_zero_point, float value_scale,
                                int32_t value_zero_point, float output_scale,
                                int32_t output_zero_point,
                                float attention_scale, uint32_t query_dtype,
                                uint32_t key_dtype, uint32_t value_dtype,
                                uint32_t output_dtype, uint32_t causal,
                                uint32_t mask_mode, uint32_t query_start,
                                uint32_t query_count) {
    CudaTensorSlot *query_slot, *key_slot, *value_slot, *mask_slot = NULL,
                   *output_slot;
    uint32_t query_elements, key_elements, tasks;
    uint32_t head_dim, mask_elements = 0;
    size_t query_bytes, key_bytes, output_bytes, task_bytes, mask_bytes = 0;
    CUdeviceptr no_mask = 0;
    CUdeviceptr mask_device;
    void* args[27];
    uint64_t query_magnitude, key_magnitude, maximum_dot;
    float query_key_scale, score_scale;
    if (!query || !key || !value || !output || !batch || !query_sequence ||
        !key_sequence || !d_model || !heads || !query_count ||
        query_start >= query_sequence ||
        query_count > query_sequence - query_start || d_model % heads ||
        d_model % 4u || !byte_dtype_valid(query_dtype) ||
        !byte_dtype_valid(key_dtype) || !byte_dtype_valid(value_dtype) ||
        !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(query_dtype, query_zero_point) ||
        !byte_zero_point_valid(key_dtype, key_zero_point) ||
        !byte_zero_point_valid(value_dtype, value_zero_point) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(query_scale) || query_scale <= 0.0f ||
        !isfinite(key_scale) || key_scale <= 0.0f ||
        !isfinite(value_scale) || value_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(attention_scale) || attention_scale <= 0.0f ||
        causal > 1u || mask_mode > 4u ||
        !checked_product4_bytes(batch, query_sequence, d_model, 1u, 1u,
                                &query_elements, &query_bytes) ||
        !checked_product4_bytes(batch, key_sequence, d_model, 1u, 1u,
                                &key_elements, &key_bytes) ||
        !checked_product4_bytes(batch, query_count, heads, 1u, 1u,
                                &tasks, &task_bytes)) return 0;
    output_bytes = query_bytes;
    head_dim = d_model / heads;
    if (!head_dim || head_dim % 4u || head_dim > 64u ||
        (mask_mode == 0u ? mask != NULL : mask == NULL)) return 0;
    if (mask_mode == 1u &&
        !checked_elements_bytes(key_sequence, sizeof(int32_t),
                                &mask_elements, &mask_bytes)) return 0;
    if (mask_mode == 2u &&
        !checked_elements_bytes((uint64_t)batch * key_sequence,
                                sizeof(int32_t), &mask_elements, &mask_bytes))
        return 0;
    if (mask_mode == 3u &&
        !checked_elements_bytes((uint64_t)query_sequence * key_sequence,
                                sizeof(int32_t), &mask_elements, &mask_bytes))
        return 0;
    if (mask_mode == 4u &&
        !checked_product4_bytes(batch, query_sequence, key_sequence, 1u,
                                sizeof(int32_t), &mask_elements, &mask_bytes))
        return 0;
    query_key_scale = query_scale * key_scale;
    score_scale = query_key_scale * attention_scale;
    query_magnitude = byte_centered_magnitude(query_dtype, query_zero_point);
    key_magnitude = byte_centered_magnitude(key_dtype, key_zero_point);
    maximum_dot = query_magnitude * key_magnitude * head_dim;
    if (!isfinite(query_key_scale) || query_key_scale <= 0.0f ||
        !isfinite(score_scale) || score_scale <= 0.0f ||
        !isfinite((float)maximum_dot * score_scale) ||
        ranges_overlap(output, output_bytes, query, query_bytes) ||
        ranges_overlap(output, output_bytes, key, key_bytes) ||
        ranges_overlap(output, output_bytes, value, key_bytes) ||
        (mask_bytes && ranges_overlap(output, output_bytes, mask, mask_bytes)))
        return 0;
    query_slot = graph_ensure_input(query, query_bytes, 0);
    key_slot = graph_ensure_input(key, key_bytes, 0);
    value_slot = graph_ensure_input(value, key_bytes, 0);
    if (mask_elements) mask_slot = graph_ensure_input(mask, mask_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!query_slot || !key_slot || !value_slot ||
        (mask_elements && !mask_slot) || !output_slot) return 0;
    mask_device = mask_slot ? mask_slot->device : no_mask;
    args[0] = &query_slot->device;
    args[1] = &key_slot->device;
    args[2] = &value_slot->device;
    args[3] = &mask_device;
    args[4] = &output_slot->device;
    args[5] = &batch;
    args[6] = &query_sequence;
    args[7] = &key_sequence;
    args[8] = &d_model;
    args[9] = &heads;
    args[10] = &query_scale;
    args[11] = &query_zero_point;
    args[12] = &key_scale;
    args[13] = &key_zero_point;
    args[14] = &value_scale;
    args[15] = &value_zero_point;
    args[16] = &output_scale;
    args[17] = &output_zero_point;
    args[18] = &attention_scale;
    args[19] = &query_dtype;
    args[20] = &key_dtype;
    args[21] = &value_dtype;
    args[22] = &output_dtype;
    args[23] = &causal;
    args[24] = &mask_mode;
    args[25] = &query_start;
    args[26] = &query_count;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qsdpa), tasks, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_qsdpa_i8u8(const void* query, const void* key,
                          const void* value, const int32_t* mask,
                          void* output, uint32_t batch, uint32_t query_sequence,
                          uint32_t key_sequence, uint32_t d_model,
                          uint32_t heads, float query_scale,
                          int32_t query_zero_point, float key_scale,
                          int32_t key_zero_point, float value_scale,
                          int32_t value_zero_point, float output_scale,
                          int32_t output_zero_point, float attention_scale,
                          uint32_t query_dtype, uint32_t key_dtype,
                          uint32_t value_dtype, uint32_t output_dtype,
                          uint32_t causal, uint32_t mask_mode) {
    return cuda_graph_qsdpa_range_i8u8(query, key, value, mask, output, batch,
        query_sequence, key_sequence, d_model, heads, query_scale,
        query_zero_point, key_scale, key_zero_point, value_scale,
        value_zero_point, output_scale, output_zero_point, attention_scale,
        query_dtype, key_dtype, value_dtype, output_dtype, causal, mask_mode,
        0u, query_sequence);
}
