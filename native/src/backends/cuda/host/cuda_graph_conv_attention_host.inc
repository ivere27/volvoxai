static int cuda_graph_conv2d_impl(const float* input, float* output,
                                  const float* weight, const float* bias,
                                  const float* residual, int batch,
                                  int height, int width, int channels,
                                  int output_channels, int kernel_height,
                                  int kernel_width, int output_height,
                                  int output_width, int stride_y, int stride_x,
                                  int padding_top, int padding_left, int groups,
                                  int relu, int dilation_y, int dilation_x) {
    CudaTensorSlot *input_slot, *weight_slot, *bias_slot = NULL;
    CudaTensorSlot *residual_slot = NULL, *output_slot;
    uint32_t input_count, weight_count, output_count;
    uint32_t params[17];
    size_t input_bytes, weight_bytes, output_bytes;
    size_t bias_bytes;
    uint64_t weight_elements;
    CUdeviceptr no_bias = 0;
    CUdeviceptr bias_device;
    void* args[21];
    void* fused_args[22];
    void* tiled_args[8];
    void* fused_tiled_args[9];
    void* depthwise_args[15];
    uint32_t rows;
    uint32_t depthwise_tasks = 0;
    unsigned int gx, gy;
    unsigned int block_x = 16u, block_y = 16u;
    uint32_t tile_rows = 16u, tile_columns = 16u;
    CUfunction tiled_function = residual
        ? CUDA_FORWARD_FUNCTION(k_conv2d_1x1_tiled_add) : CUDA_FORWARD_FUNCTION(k_conv2d_1x1_tiled);
    enum {
        CUDA_CONV1X1_TACTIC_BASELINE = 0,
        CUDA_CONV1X1_TACTIC_BM32_BN32_BK16 = 1,
        CUDA_CONV1X1_TACTIC_BM16_BN64_BK16 = 2,
    } tiled_tactic = CUDA_CONV1X1_TACTIC_BASELINE;
    enum {
        CUDA_DEPTHWISE_TACTIC_NONE = 0,
        CUDA_DEPTHWISE_TACTIC_3X3_C1 = 1,
        CUDA_DEPTHWISE_TACTIC_3X3_C4 = 2,
        CUDA_DEPTHWISE_TACTIC_5X5_C1 = 3,
        CUDA_DEPTHWISE_TACTIC_5X5_C4 = 4,
    } depthwise_tactic = CUDA_DEPTHWISE_TACTIC_NONE;
    (void)tiled_tactic;
    (void)depthwise_tactic;
    CUfunction depthwise_function = NULL;
    int tiled_1x1;
    int specialized_depthwise;
    int depthwise_c4;
    if (!input || !output || !weight || batch <= 0 || height <= 0 || width <= 0 ||
        channels <= 0 || output_channels <= 0 || kernel_height <= 0 ||
        kernel_width <= 0 || output_height <= 0 || output_width <= 0 ||
        stride_y <= 0 || stride_x <= 0 || padding_top < 0 || padding_left < 0 ||
        dilation_y <= 0 || dilation_x <= 0 || groups <= 0 ||
        channels % groups || output_channels % groups || relu < 0 || relu > 2 ||
        !(groups == 1 || groups == channels) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)height,
            (uint32_t)width, (uint32_t)channels, sizeof(float),
            &input_count, &input_bytes) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)output_height,
            (uint32_t)output_width, (uint32_t)output_channels, sizeof(float),
            &output_count, &output_bytes)) return 0;
    weight_elements = groups == channels
        ? (uint64_t)(uint32_t)kernel_height * (uint32_t)kernel_width *
          (uint32_t)channels * ((uint32_t)output_channels / (uint32_t)channels)
        : (uint64_t)(uint32_t)kernel_height * (uint32_t)kernel_width *
          ((uint32_t)channels / (uint32_t)groups) * (uint32_t)output_channels;
    if (!checked_elements_bytes(weight_elements, sizeof(float), &weight_count,
                                &weight_bytes) ||
        (uint64_t)(uint32_t)output_channels * sizeof(float) > SIZE_MAX) return 0;
    bias_bytes = (size_t)(uint32_t)output_channels * sizeof(float);
    if (residual &&
        (ranges_overlap(output, output_bytes, input, input_bytes) ||
         ranges_overlap(output, output_bytes, weight, weight_bytes) ||
         (bias && ranges_overlap(output, output_bytes, bias, bias_bytes)) ||
         ranges_overlap(output, output_bytes, residual, output_bytes))) return 0;
    params[0] = (uint32_t)batch;
    params[1] = (uint32_t)height;
    params[2] = (uint32_t)width;
    params[3] = (uint32_t)channels;
    params[4] = (uint32_t)output_channels;
    params[5] = (uint32_t)kernel_height;
    params[6] = (uint32_t)kernel_width;
    params[7] = (uint32_t)output_height;
    params[8] = (uint32_t)output_width;
    params[9] = (uint32_t)stride_y;
    params[10] = (uint32_t)stride_x;
    params[11] = (uint32_t)padding_top;
    params[12] = (uint32_t)padding_left;
    params[13] = (uint32_t)groups;
    params[14] = (uint32_t)relu;
    params[15] = (uint32_t)dilation_y;
    params[16] = (uint32_t)dilation_x;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    weight_slot = graph_ensure_input(weight, weight_bytes, 1);
    if (bias) bias_slot = graph_ensure_input(bias, bias_bytes, 1);
    if (residual)
        residual_slot = graph_ensure_input(residual, output_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !weight_slot || (bias && !bias_slot) ||
        (residual && !residual_slot) || !output_slot)
        return 0;
    bias_device = bias_slot ? bias_slot->device : no_bias;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &weight_slot->device;
    args[3] = &bias_device;
    for (int index = 0; index < 17; index++) args[index + 4] = &params[index];
    fused_args[0] = &input_slot->device;
    fused_args[1] = &output_slot->device;
    fused_args[2] = &weight_slot->device;
    fused_args[3] = &bias_device;
    fused_args[4] = residual_slot ? &residual_slot->device : NULL;
    for (int index = 0; index < 17; index++)
        fused_args[index + 5] = &params[index];
    depthwise_args[0] = &input_slot->device;
    depthwise_args[1] = &output_slot->device;
    depthwise_args[2] = &weight_slot->device;
    depthwise_args[3] = &bias_device;
    depthwise_args[4] = &params[0];
    depthwise_args[5] = &params[1];
    depthwise_args[6] = &params[2];
    depthwise_args[7] = &params[3];
    depthwise_args[8] = &params[7];
    depthwise_args[9] = &params[8];
    depthwise_args[10] = &params[9];
    depthwise_args[11] = &params[10];
    depthwise_args[12] = &params[11];
    depthwise_args[13] = &params[12];
    depthwise_args[14] = &params[14];
    rows = output_count / (uint32_t)output_channels;
    tiled_1x1 = groups == 1 && kernel_height == 1 && kernel_width == 1 &&
        stride_y == 1 && stride_x == 1 && dilation_y == 1 &&
        dilation_x == 1 && padding_top == 0 && padding_left == 0 &&
        output_height == height && output_width == width;
    if (residual && !tiled_1x1) return 0;
    specialized_depthwise = !residual && groups == channels &&
        output_channels == channels && kernel_height == kernel_width &&
        (kernel_height == 3 || kernel_height == 5) &&
        dilation_y == 1 && dilation_x == 1 &&
        stride_y <= 2 && stride_x <= 2;
    depthwise_c4 = specialized_depthwise && channels % 4 == 0 &&
        ((input_slot->device | output_slot->device | weight_slot->device |
          bias_device) & (CUdeviceptr)15u) == 0 &&
        !(kernel_height == 3 && stride_y == 2 && stride_x == 2 &&
          channels <= 128);
    if (specialized_depthwise) {
        depthwise_tasks = depthwise_c4 ? output_count / 4u : output_count;
        if (kernel_height == 3) {
            depthwise_function = depthwise_c4
                ? CUDA_FORWARD_FUNCTION(k_depthwise_conv2d_3x3_c4)
                : CUDA_FORWARD_FUNCTION(k_depthwise_conv2d_3x3_c1);
            depthwise_tactic = depthwise_c4
                ? CUDA_DEPTHWISE_TACTIC_3X3_C4
                : CUDA_DEPTHWISE_TACTIC_3X3_C1;
        } else {
            depthwise_function = depthwise_c4
                ? CUDA_FORWARD_FUNCTION(k_depthwise_conv2d_5x5_c4)
                : CUDA_FORWARD_FUNCTION(k_depthwise_conv2d_5x5_c1);
            depthwise_tactic = depthwise_c4
                ? CUDA_DEPTHWISE_TACTIC_5X5_C4
                : CUDA_DEPTHWISE_TACTIC_5X5_C1;
        }
    }
    if (tiled_1x1) {
        if (channels >= 16 && rows >= 17 && output_channels >= 48) {
            tiled_tactic = CUDA_CONV1X1_TACTIC_BM32_BN32_BK16;
            tiled_function = residual
                ? CUDA_FORWARD_FUNCTION(k_conv2d_1x1_bm32_bn32_bk16_add)
                : CUDA_FORWARD_FUNCTION(k_conv2d_1x1_bm32_bn32_bk16);
            tile_rows = 32u;
            tile_columns = 32u;
            block_x = 8u;
            block_y = 16u;
        } else if (channels >= 16 && rows >= 8 && output_channels >= 48) {
            tiled_tactic = CUDA_CONV1X1_TACTIC_BM16_BN64_BK16;
            tiled_function = residual
                ? CUDA_FORWARD_FUNCTION(k_conv2d_1x1_bm16_bn64_bk16_add)
                : CUDA_FORWARD_FUNCTION(k_conv2d_1x1_bm16_bn64_bk16);
            tile_rows = 16u;
            tile_columns = 64u;
            block_x = 16u;
            block_y = 8u;
        }
    }
    gx = (uint32_t)output_channels / tile_columns +
        ((uint32_t)output_channels % tile_columns != 0u);
    gy = rows / tile_rows + (rows % tile_rows != 0u);
    if (gy > 65535u) tiled_1x1 = 0;
    if (tiled_1x1) {
        tiled_args[0] = &input_slot->device;
        tiled_args[1] = &output_slot->device;
        tiled_args[2] = &weight_slot->device;
        tiled_args[3] = &bias_device;
        tiled_args[4] = &rows;
        tiled_args[5] = &params[3];
        tiled_args[6] = &params[4];
        tiled_args[7] = &params[14];
        fused_tiled_args[0] = &input_slot->device;
        fused_tiled_args[1] = &output_slot->device;
        fused_tiled_args[2] = &weight_slot->device;
        fused_tiled_args[3] = &bias_device;
        fused_tiled_args[4] = residual_slot ? &residual_slot->device : NULL;
        fused_tiled_args[5] = &rows;
        fused_tiled_args[6] = &params[3];
        fused_tiled_args[7] = &params[4];
        fused_tiled_args[8] = &params[14];
        if (!tiled_function || !gx || !gy ||
            !cuda_launch(tiled_function, gx, gy, 1, block_x, block_y, 1, 0,
                         residual ? fused_tiled_args : tiled_args)) return 0;
    } else if (specialized_depthwise) {
        if (!depthwise_function || !depthwise_tasks ||
            !cuda_launch_1d(depthwise_function, depthwise_tasks,
                            depthwise_args)) return 0;
    } else {
        if (!cuda_launch_1d(residual ? CUDA_FORWARD_FUNCTION(k_conv2d_add) : CUDA_FORWARD_FUNCTION(k_conv2d),
                            output_count,
                            residual ? fused_args : args)) return 0;
    }
    graph_mark_device(output_slot);
    return 1;
}


int cuda_graph_conv2d_f32(const float* input, float* output,
                          const float* weight, const float* bias,
                          int batch, int height, int width, int channels,
                          int output_channels, int kernel_height,
                          int kernel_width, int output_height,
                          int output_width, int stride_y, int stride_x,
                          int padding_top, int padding_left, int groups,
                          int relu, int dilation_y, int dilation_x) {
    return cuda_graph_conv2d_impl(input, output, weight, bias, NULL, batch,
        height, width, channels, output_channels, kernel_height, kernel_width,
        output_height, output_width, stride_y, stride_x, padding_top,
        padding_left, groups, relu, dilation_y, dilation_x);
}

int cuda_graph_conv2d_add_f32(const float* input, float* output,
                              const float* weight, const float* bias,
                              const float* residual, int batch, int height,
                              int width, int channels, int output_channels,
                              int kernel_height, int kernel_width,
                              int output_height, int output_width,
                              int stride_y, int stride_x, int padding_top,
                              int padding_left, int groups, int relu,
                              int dilation_y, int dilation_x) {
    if (!residual) return 0;
    return cuda_graph_conv2d_impl(input, output, weight, bias, residual, batch,
        height, width, channels, output_channels, kernel_height, kernel_width,
        output_height, output_width, stride_y, stride_x, padding_top,
        padding_left, groups, relu, dilation_y, dilation_x);
}

int cuda_graph_sdpa_range_f32(const float* qkv, const int32_t* mask,
                              long mask_elements, float* output,
                              int sequence, int d_model, int heads,
                              int head_dim, int batch, float attention_scale,
                              int causal, int mask_mode, int query_start,
                              int query_count) {
    CudaTensorSlot *qkv_slot, *mask_slot = NULL, *output_slot;
    uint32_t qkv_count, output_count, tasks;
    uint32_t params[9];
    size_t qkv_bytes, output_bytes, task_bytes, mask_bytes = 0;
    uint64_t expected_mask = 0;
    CUdeviceptr no_mask = 0;
    CUdeviceptr mask_device;
    void* args[13];
    if (!qkv || !output || sequence <= 0 || d_model <= 0 || heads <= 0 ||
        head_dim <= 0 || batch <= 0 || query_start < 0 || query_count <= 0 ||
        (uint64_t)(uint32_t)query_start + (uint32_t)query_count >
            (uint32_t)sequence ||
        (int64_t)d_model != (int64_t)heads * head_dim ||
        !isfinite(attention_scale) || (causal != 0 && causal != 1) ||
        mask_mode < 0 || mask_mode > 4 ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)sequence, 3u,
            (uint32_t)d_model, sizeof(float), &qkv_count, &qkv_bytes) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)sequence,
            (uint32_t)d_model, 1u, sizeof(float), &output_count,
            &output_bytes) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)query_count,
            (uint32_t)heads, 1u, 1u, &tasks, &task_bytes))
        return 0;
    if (mask_mode == 1) expected_mask = (uint32_t)sequence;
    else if (mask_mode == 2)
        expected_mask = (uint64_t)(uint32_t)batch * (uint32_t)sequence;
    else if (mask_mode == 3)
        expected_mask = (uint64_t)(uint32_t)sequence * (uint32_t)sequence;
    else if (mask_mode == 4)
        expected_mask = (uint64_t)(uint32_t)batch * (uint32_t)sequence *
            (uint32_t)sequence;
    if ((expected_mask == 0 && (mask || mask_elements != 0)) ||
        (expected_mask != 0 && (!mask || mask_elements <= 0 ||
         (uint64_t)mask_elements != expected_mask ||
         expected_mask > UINT32_MAX ||
         expected_mask > SIZE_MAX / sizeof(int32_t)))) return 0;
    if (expected_mask) mask_bytes = (size_t)expected_mask * sizeof(int32_t);
    if (ranges_overlap(output, output_bytes, qkv, qkv_bytes) ||
        (mask && ranges_overlap(output, output_bytes, mask, mask_bytes)))
        return 0;
    params[0] = (uint32_t)sequence;
    params[1] = (uint32_t)d_model;
    params[2] = (uint32_t)heads;
    params[3] = (uint32_t)head_dim;
    params[4] = (uint32_t)batch;
    params[5] = (uint32_t)causal;
    params[6] = (uint32_t)mask_mode;
    params[7] = (uint32_t)query_start;
    params[8] = (uint32_t)query_count;
    qkv_slot = graph_ensure_input(qkv, qkv_bytes, 0);
    if (mask_mode)
        mask_slot = graph_ensure_input(mask, mask_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!qkv_slot || (mask_mode && !mask_slot) || !output_slot) return 0;
    mask_device = mask_slot ? mask_slot->device : no_mask;
    args[0] = &qkv_slot->device;
    args[1] = &mask_device;
    args[2] = &output_slot->device;
    args[3] = &params[0];
    args[4] = &params[1];
    args[5] = &params[2];
    args[6] = &params[3];
    args[7] = &params[4];
    args[8] = &attention_scale;
    args[9] = &params[5];
    args[10] = &params[6];
    args[11] = &params[7];
    args[12] = &params[8];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_sdpa), tasks, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_sdpa_f32(const float* qkv, const int32_t* mask,
                        long mask_elements, float* output,
                        int sequence, int d_model, int heads, int head_dim,
                        int batch, float attention_scale, int causal,
                        int mask_mode) {
    return cuda_graph_sdpa_range_f32(qkv, mask, mask_elements, output,
        sequence, d_model, heads, head_dim, batch, attention_scale, causal,
        mask_mode, 0, sequence);
}

int cuda_graph_cross_sdpa_f32(const float* query, const float* key,
                              const float* value, const int32_t* mask,
                              long mask_elements, float* output,
                              int query_sequence, int key_sequence,
                              int d_model, int heads, int head_dim, int batch,
                              float attention_scale, int causal,
                              int mask_mode) {
    CudaTensorSlot *query_slot, *key_slot, *value_slot, *mask_slot = NULL,
                   *output_slot;
    uint32_t query_count, key_count, output_count, tasks;
    uint32_t params[8];
    size_t query_bytes, key_bytes, output_bytes, task_bytes;
    size_t mask_bytes = 0;
    uint64_t expected_mask = 0;
    CUdeviceptr no_mask = 0;
    CUdeviceptr mask_device;
    void* args[14];
    if (!query || !key || !value || !output || query_sequence <= 0 ||
        key_sequence <= 0 || d_model <= 0 || heads <= 0 || head_dim <= 0 ||
        batch <= 0 ||
        (int64_t)d_model != (int64_t)heads * head_dim || head_dim > 64 ||
        !isfinite(attention_scale) || (causal != 0 && causal != 1) ||
        mask_mode < 0 || mask_mode > 4 ||
        !checked_product4_bytes((uint32_t)batch,
            (uint32_t)query_sequence, (uint32_t)d_model, 1u,
            sizeof(float), &query_count, &query_bytes) ||
        !checked_product4_bytes((uint32_t)batch,
            (uint32_t)key_sequence, (uint32_t)d_model, 1u,
            sizeof(float), &key_count, &key_bytes) ||
        !checked_product4_bytes((uint32_t)batch,
            (uint32_t)query_sequence, (uint32_t)d_model, 1u,
            sizeof(float), &output_count, &output_bytes) ||
        !checked_product4_bytes((uint32_t)batch,
            (uint32_t)query_sequence, (uint32_t)heads, 1u, 1u,
            &tasks, &task_bytes)) return 0;
    if (mask_mode == 1) expected_mask = (uint32_t)key_sequence;
    else if (mask_mode == 2)
        expected_mask = (uint64_t)(uint32_t)batch * (uint32_t)key_sequence;
    else if (mask_mode == 3)
        expected_mask = (uint64_t)(uint32_t)query_sequence *
            (uint32_t)key_sequence;
    else if (mask_mode == 4)
        expected_mask = (uint64_t)(uint32_t)batch *
            (uint32_t)query_sequence * (uint32_t)key_sequence;
    if ((expected_mask == 0 && (mask || mask_elements != 0)) ||
        (expected_mask != 0 && (!mask || mask_elements <= 0 ||
         (uint64_t)mask_elements != expected_mask ||
         expected_mask > UINT32_MAX ||
         expected_mask > SIZE_MAX / sizeof(int32_t)))) return 0;
    if (expected_mask) mask_bytes = (size_t)expected_mask * sizeof(int32_t);
    if (ranges_overlap(output, output_bytes, query, query_bytes) ||
        ranges_overlap(output, output_bytes, key, key_bytes) ||
        ranges_overlap(output, output_bytes, value, key_bytes) ||
        (mask && ranges_overlap(output, output_bytes, mask, mask_bytes)))
        return 0;
    params[0] = (uint32_t)query_sequence;
    params[1] = (uint32_t)key_sequence;
    params[2] = (uint32_t)d_model;
    params[3] = (uint32_t)heads;
    params[4] = (uint32_t)head_dim;
    params[5] = (uint32_t)batch;
    params[6] = (uint32_t)causal;
    params[7] = (uint32_t)mask_mode;
    query_slot = graph_ensure_input(query, query_bytes, 0);
    key_slot = graph_ensure_input(key, key_bytes, 0);
    value_slot = graph_ensure_input(value, key_bytes, 0);
    if (mask) mask_slot = graph_ensure_input(mask, mask_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!query_slot || !key_slot || !value_slot || (mask && !mask_slot) ||
        !output_slot) return 0;
    mask_device = mask_slot ? mask_slot->device : no_mask;
    args[0] = &query_slot->device;
    args[1] = &key_slot->device;
    args[2] = &value_slot->device;
    args[3] = &mask_device;
    args[4] = &output_slot->device;
    for (int index = 0; index < 6; index++) args[index + 5] = &params[index];
    args[11] = &attention_scale;
    args[12] = &params[6];
    args[13] = &params[7];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_cross_sdpa), tasks, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_cross_attention_f32(const float* query,
                                    const float* key_value,
                                    const float* weight, const float* scale,
                                    const float* bias, float* output,
                                    int query_sequence, int key_sequence,
                                    int d_model, int heads, int head_dim,
                                    int batch, int has_scale, int has_bias) {
    CudaTensorSlot *query_slot, *key_value_slot, *weight_slot;
    CudaTensorSlot *scale_slot = NULL, *bias_slot = NULL, *output_slot;
    uint32_t query_count, key_value_count, weight_count, output_count, tasks;
    uint32_t params[8];
    size_t query_bytes, key_value_bytes, weight_bytes, output_bytes, task_bytes;
    size_t affine_bytes;
    CUdeviceptr no_scale = 0, no_bias = 0;
    CUdeviceptr scale_device, bias_device;
    void* args[14];
    if (!query || !key_value || !weight || !output || query_sequence <= 0 ||
        key_sequence <= 0 || d_model <= 0 || heads <= 0 || head_dim <= 0 ||
        batch <= 0 ||
        (int64_t)d_model != (int64_t)heads * head_dim || head_dim > 64 ||
        (has_scale != 0 && has_scale != 1) ||
        (has_bias != 0 && has_bias != 1) ||
        (has_scale && !scale) || (has_bias && !bias) ||
        !checked_product4_bytes((uint32_t)batch,
            (uint32_t)query_sequence, (uint32_t)d_model, 1u,
            sizeof(float), &query_count, &query_bytes) ||
        !checked_product4_bytes((uint32_t)batch,
            (uint32_t)key_sequence, (uint32_t)d_model, 1u,
            sizeof(float), &key_value_count, &key_value_bytes) ||
        !checked_product4_bytes(3u, (uint32_t)d_model,
            (uint32_t)d_model, 1u, sizeof(float), &weight_count,
            &weight_bytes) ||
        !checked_product4_bytes((uint32_t)batch,
            (uint32_t)query_sequence, (uint32_t)d_model, 1u,
            sizeof(float), &output_count, &output_bytes) ||
        !checked_product4_bytes((uint32_t)batch,
            (uint32_t)query_sequence, (uint32_t)heads, 1u, 1u,
            &tasks, &task_bytes) ||
        (uint64_t)3u * (uint32_t)d_model * sizeof(float) > SIZE_MAX)
        return 0;
    affine_bytes = (size_t)3u * (uint32_t)d_model * sizeof(float);
    if (ranges_overlap(output, output_bytes, query, query_bytes) ||
        ranges_overlap(output, output_bytes, key_value, key_value_bytes) ||
        ranges_overlap(output, output_bytes, weight, weight_bytes) ||
        (scale && ranges_overlap(output, output_bytes, scale, affine_bytes)) ||
        (bias && ranges_overlap(output, output_bytes, bias, affine_bytes)))
        return 0;
    params[0] = (uint32_t)query_sequence;
    params[1] = (uint32_t)key_sequence;
    params[2] = (uint32_t)d_model;
    params[3] = (uint32_t)heads;
    params[4] = (uint32_t)head_dim;
    params[5] = (uint32_t)batch;
    params[6] = (uint32_t)has_scale;
    params[7] = (uint32_t)has_bias;
    query_slot = graph_ensure_input(query, query_bytes, 0);
    key_value_slot = graph_ensure_input(key_value, key_value_bytes, 0);
    weight_slot = graph_ensure_input(weight, weight_bytes, 1);
    if (scale) scale_slot = graph_ensure_input(scale, affine_bytes, 1);
    if (bias) bias_slot = graph_ensure_input(bias, affine_bytes, 1);
    output_slot = graph_output(output, output_bytes);
    if (!query_slot || !key_value_slot || !weight_slot ||
        (scale && !scale_slot) || (bias && !bias_slot) || !output_slot)
        return 0;
    scale_device = scale_slot ? scale_slot->device : no_scale;
    bias_device = bias_slot ? bias_slot->device : no_bias;
    args[0] = &query_slot->device;
    args[1] = &key_value_slot->device;
    args[2] = &weight_slot->device;
    args[3] = &scale_device;
    args[4] = &bias_device;
    args[5] = &output_slot->device;
    for (int index = 0; index < 8; index++) args[index + 6] = &params[index];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_cross_attention), tasks, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

static int cuda_graph_profile_f32(CUfunction kernel, const float* input,
                                  float* output, int batch, int height,
                                  int width, int channels, int kind) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t input_count, output_count, tasks;
    uint32_t params[4];
    size_t input_bytes, output_bytes, task_bytes;
    uint64_t outputs_per_batch;
    uint64_t task_count;
    void* args[6];
    if (!kernel || !input || !output || batch <= 0 || height <= 0 ||
        width <= 0 || channels <= 0 || kind < 0 || kind > 3 ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)height,
            (uint32_t)width, (uint32_t)channels, sizeof(float),
            &input_count, &input_bytes)) return 0;
    if (kind == 2) {
        outputs_per_batch = (uint64_t)2u * (uint32_t)channels *
            (uint32_t)height;
        task_count = (uint64_t)(uint32_t)batch * (uint32_t)channels *
            (uint32_t)height;
    } else {
        outputs_per_batch = (uint64_t)(kind == 1 ? 2u : 1u) *
            (uint32_t)channels * (uint32_t)width;
        task_count = (uint64_t)(uint32_t)batch * (uint32_t)channels *
            (uint32_t)width;
    }
    if (!checked_elements_bytes((uint64_t)(uint32_t)batch *
            outputs_per_batch, sizeof(float), &output_count, &output_bytes) ||
        !checked_elements_bytes(task_count, 1, &tasks, &task_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes)) return 0;
    params[0] = (uint32_t)batch;
    params[1] = (uint32_t)height;
    params[2] = (uint32_t)width;
    params[3] = (uint32_t)channels;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    for (int index = 0; index < 4; index++) args[index + 2] = &params[index];
    if (!cuda_launch_1d(kernel, tasks, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_spatial_softargmax_y_f32(const float* input, float* output,
                                        int batch, int height, int width,
                                        int channels) {
    return cuda_graph_profile_f32(CUDA_FORWARD_FUNCTION(k_spatial_softargmax_y), input, output,
                                  batch, height, width, channels, 0);
}

int cuda_graph_profile_x_f32(const float* input, float* output,
                             int batch, int height, int width, int channels) {
    return cuda_graph_profile_f32(CUDA_FORWARD_FUNCTION(k_profile_x), input, output, batch, height,
                                  width, channels, 1);
}

int cuda_graph_profile_y_f32(const float* input, float* output,
                             int batch, int height, int width, int channels) {
    return cuda_graph_profile_f32(CUDA_FORWARD_FUNCTION(k_profile_y), input, output, batch, height,
                                  width, channels, 2);
}

int cuda_graph_mean_height_f32(const float* input, float* output,
                               int batch, int height, int width,
                               int channels) {
    return cuda_graph_profile_f32(CUDA_FORWARD_FUNCTION(k_mean_height), input, output, batch, height,
                                  width, channels, 3);
}

int cuda_graph_nms_f32(const float* boxes, const float* scores,
                       float* output, int batches, int spatial, int classes,
                       int max_output, int output_rows, float iou_threshold,
                       float score_threshold) {
    CudaTensorSlot *boxes_slot, *scores_slot, *output_slot;
    uint32_t boxes_count, scores_count, output_count;
    uint32_t params[5];
    size_t boxes_bytes, scores_bytes, output_bytes;
    void* args[10];
    if (!boxes || !scores || !output || batches <= 0 || spatial <= 0 ||
        classes <= 0 || max_output <= 0 || output_rows <= 0 ||
        !isfinite(iou_threshold) || iou_threshold < 0.0f ||
        iou_threshold >= 1.0f || !isfinite(score_threshold) ||
        !checked_elements_bytes((uint64_t)(uint32_t)batches *
            (uint32_t)spatial * 4u, sizeof(float), &boxes_count,
            &boxes_bytes) ||
        !checked_elements_bytes((uint64_t)(uint32_t)batches *
            (uint32_t)classes * (uint32_t)spatial, sizeof(float),
            &scores_count, &scores_bytes) ||
        !checked_elements_bytes((uint64_t)(uint32_t)output_rows * 3u,
            sizeof(float), &output_count, &output_bytes) ||
        ranges_overlap(output, output_bytes, boxes, boxes_bytes) ||
        ranges_overlap(output, output_bytes, scores, scores_bytes)) return 0;
    params[0] = (uint32_t)batches;
    params[1] = (uint32_t)spatial;
    params[2] = (uint32_t)classes;
    params[3] = (uint32_t)max_output;
    params[4] = (uint32_t)output_rows;
    boxes_slot = graph_ensure_input(boxes, boxes_bytes, 0);
    scores_slot = graph_ensure_input(scores, scores_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!boxes_slot || !scores_slot || !output_slot) return 0;
    args[0] = &boxes_slot->device;
    args[1] = &scores_slot->device;
    args[2] = &output_slot->device;
    for (int index = 0; index < 5; index++) args[index + 3] = &params[index];
    args[8] = &iou_threshold;
    args[9] = &score_threshold;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_nms), 1u, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}
