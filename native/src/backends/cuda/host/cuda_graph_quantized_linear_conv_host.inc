int cuda_graph_qlinear_i8u8(const void* input, const void* weight,
                            const float* weight_scales,
                            const int32_t* weight_zero_points,
                            const int32_t* bias, void* output,
                            uint32_t rows, uint32_t d_in, uint32_t d_out,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t weight_dtype,
                            uint32_t output_dtype) {
    CudaTensorView input_view, output_view;
    CudaTensorSlot *weight_slot, *scale_slot, *zero_slot, *bias_slot;
    size_t input_bytes, weight_bytes, output_bytes, metadata_bytes;
    uint32_t elements;
    void* args[16];
    if (!input || !weight || !weight_scales || !weight_zero_points || !bias || !output ||
        !byte_dtype_valid(input_dtype) || !byte_dtype_valid(weight_dtype) ||
        !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(input_dtype, input_zero_point) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !size_product_u32(rows, d_in, 1, &input_bytes) ||
        !size_product_u32(d_out, d_in, 1, &weight_bytes) ||
        !size_product_u32(rows, d_out, 1, &output_bytes) ||
        (uint64_t)d_out * sizeof(float) > SIZE_MAX) return 0;
    metadata_bytes = (size_t)d_out * sizeof(float);
    if (ranges_overlap(output, output_bytes, input, input_bytes) ||
        ranges_overlap(output, output_bytes, weight, weight_bytes) ||
        ranges_overlap(output, output_bytes, weight_scales, metadata_bytes) ||
        ranges_overlap(output, output_bytes, weight_zero_points, metadata_bytes) ||
        ranges_overlap(output, output_bytes, bias, metadata_bytes)) return 0;
    int64_t input_distance = input_dtype == VX_DTYPE_I8
        ? (input_zero_point < 0 ? 127 - (int64_t)input_zero_point
                                : (int64_t)input_zero_point + 128)
        : (input_zero_point < 128 ? 255 - (int64_t)input_zero_point
                                  : (int64_t)input_zero_point);
    for (uint32_t channel = 0; channel < d_out; channel++) {
        int32_t zp = weight_zero_points[channel];
        int64_t weight_distance;
        int64_t bias_distance = bias[channel] < 0 ? -(int64_t)bias[channel] : bias[channel];
        float multiplier;
        if (!byte_zero_point_valid(weight_dtype, zp) ||
            !vx_qlinear_compute_multiplier(input_scale, weight_scales[channel],
                                            output_scale, &multiplier))
            return 0;
        weight_distance = weight_dtype == VX_DTYPE_I8
            ? (zp < 0 ? 127 - (int64_t)zp : (int64_t)zp + 128)
            : (zp < 128 ? 255 - (int64_t)zp : (int64_t)zp);
        if (bias_distance + (int64_t)d_in * input_distance * weight_distance > INT32_MAX)
            return 0;
    }
    elements = (uint32_t)output_bytes;
    if (!graph_ensure_input_view(input, input_bytes, 0, &input_view)) return 0;
    weight_slot = graph_ensure_input(weight, weight_bytes, 1);
    scale_slot = graph_ensure_input(weight_scales, (size_t)d_out * sizeof(float), 1);
    zero_slot = graph_ensure_input(weight_zero_points, (size_t)d_out * sizeof(int32_t), 1);
    bias_slot = graph_ensure_input(bias, (size_t)d_out * sizeof(int32_t), 1);
    if (!graph_output_view(output, output_bytes, &output_view) || !weight_slot ||
        !scale_slot || !zero_slot || !bias_slot) return 0;
    args[0] = &input_view.device;
    args[1] = &weight_slot->device;
    args[2] = &scale_slot->device;
    args[3] = &zero_slot->device;
    args[4] = &bias_slot->device;
    args[5] = &output_view.device;
    args[6] = &elements;
    args[7] = &d_in;
    args[8] = &d_out;
    args[9] = &input_scale;
    args[10] = &input_zero_point;
    args[11] = &output_scale;
    args[12] = &output_zero_point;
    args[13] = &input_dtype;
    args[14] = &weight_dtype;
    args[15] = &output_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qlinear), elements, args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}

int cuda_graph_qembedding_i8u8(const int32_t* tokens, const void* weight,
                               const float* weight_scales,
                               const int32_t* weight_zero_points, void* output,
                               uint32_t token_count, uint32_t vocab,
                               uint32_t width, float output_scale,
                               int32_t output_zero_point,
                               uint32_t weight_dtype, uint32_t output_dtype) {
    CudaTensorView token_view, output_view;
    CudaTensorSlot *weight_slot, *scale_slot, *zero_slot;
    size_t token_bytes, weight_bytes, output_bytes, metadata_bytes;
    uint32_t elements;
    void* args[11];
    if (!tokens || !weight || !weight_scales || !weight_zero_points || !output ||
        !byte_dtype_valid(weight_dtype) || !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !size_product_u32(vocab, width, 1, &weight_bytes) ||
        !size_product_u32(token_count, width, 1, &output_bytes) ||
        (uint64_t)token_count * sizeof(int32_t) > SIZE_MAX ||
        (uint64_t)vocab * sizeof(float) > SIZE_MAX) return 0;
    token_bytes = (size_t)token_count * sizeof(int32_t);
    metadata_bytes = (size_t)vocab * sizeof(float);
    if (ranges_overlap(output, output_bytes, tokens, token_bytes) ||
        ranges_overlap(output, output_bytes, weight, weight_bytes) ||
        ranges_overlap(output, output_bytes, weight_scales, metadata_bytes) ||
        ranges_overlap(output, output_bytes, weight_zero_points, metadata_bytes))
        return 0;
    for (uint32_t i = 0; i < token_count; i++)
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= vocab) return 0;
    for (uint32_t i = 0; i < vocab; i++)
        if (!byte_zero_point_valid(weight_dtype, weight_zero_points[i]) ||
            !isfinite(weight_scales[i]) || weight_scales[i] <= 0.0f) return 0;
    elements = (uint32_t)output_bytes;
    if (!graph_ensure_input_view(tokens,
            (size_t)token_count * sizeof(int32_t), 0, &token_view)) return 0;
    weight_slot = graph_ensure_input(weight, weight_bytes, 1);
    scale_slot = graph_ensure_input(weight_scales, (size_t)vocab * sizeof(float), 1);
    zero_slot = graph_ensure_input(weight_zero_points, (size_t)vocab * sizeof(int32_t), 1);
    if (!graph_output_view(output, output_bytes, &output_view) || !weight_slot ||
        !scale_slot || !zero_slot) return 0;
    args[0] = &token_view.device;
    args[1] = &weight_slot->device;
    args[2] = &scale_slot->device;
    args[3] = &zero_slot->device;
    args[4] = &output_view.device;
    args[5] = &elements;
    args[6] = &width;
    args[7] = &output_scale;
    args[8] = &output_zero_point;
    args[9] = &weight_dtype;
    args[10] = &output_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qembedding), elements, args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}

int cuda_graph_qconv2d_i8u8(const void* input, const void* weight,
                            const float* weight_scales,
                            const int32_t* weight_zero_points,
                            const int32_t* bias, void* output,
                            uint32_t batch, uint32_t input_height,
                            uint32_t input_width, uint32_t input_channels,
                            uint32_t output_height, uint32_t output_width,
                            uint32_t output_channels,
                            uint32_t kernel_height, uint32_t kernel_width,
                            uint32_t input_per_group,
                            uint32_t stride_y, uint32_t stride_x,
                            uint32_t dilation_y, uint32_t dilation_x,
                            uint32_t padding_top, uint32_t padding_left,
                            uint32_t padding_bottom, uint32_t padding_right,
                            uint32_t groups, uint32_t relu,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t weight_dtype,
                            uint32_t output_dtype) {
    CudaTensorSlot *input_slot, *weight_slot, *scale_slot, *zero_slot,
                   *bias_slot = NULL, *output_slot;
    uint32_t input_elements, weight_elements, output_elements;
    size_t input_bytes, weight_bytes, output_bytes, metadata_bytes;
    uint64_t padded_height, padded_width, effective_height, effective_width;
    uint64_t expected_height, expected_width, terms, input_magnitude;
    CUdeviceptr no_bias = 0;
    CUdeviceptr bias_device;
    void* args[33];
    if (!input || !weight || !weight_scales || !weight_zero_points || !output ||
        !batch || !input_height || !input_width || !input_channels ||
        !output_height || !output_width || !output_channels ||
        !kernel_height || !kernel_width || !input_per_group || !stride_y ||
        !stride_x || !dilation_y || !dilation_x || !groups || relu > 2u ||
        (uint64_t)input_channels != (uint64_t)groups * input_per_group ||
        output_channels % groups || !byte_dtype_valid(input_dtype) ||
        !byte_dtype_valid(weight_dtype) || !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(input_dtype, input_zero_point) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !checked_product4_bytes(batch, input_height, input_width,
            input_channels, 1, &input_elements, &input_bytes) ||
        !checked_product4_bytes(output_channels, kernel_height, kernel_width,
            input_per_group, 1, &weight_elements, &weight_bytes) ||
        !checked_product4_bytes(batch, output_height, output_width,
            output_channels, 1, &output_elements, &output_bytes) ||
        (uint64_t)output_channels * sizeof(float) > SIZE_MAX) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    effective_height = (uint64_t)(kernel_height - 1u) * dilation_y + 1u;
    effective_width = (uint64_t)(kernel_width - 1u) * dilation_x + 1u;
    if (padded_height < effective_height || padded_width < effective_width)
        return 0;
    expected_height = (padded_height - effective_height) / stride_y + 1u;
    expected_width = (padded_width - effective_width) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width)
        return 0;
    terms = (uint64_t)kernel_height * kernel_width * input_per_group;
    input_magnitude = byte_centered_magnitude(input_dtype, input_zero_point);
    metadata_bytes = (size_t)output_channels * sizeof(float);
    if (ranges_overlap(output, output_bytes, input, input_bytes) ||
        ranges_overlap(output, output_bytes, weight, weight_bytes) ||
        ranges_overlap(output, output_bytes, weight_scales, metadata_bytes) ||
        ranges_overlap(output, output_bytes, weight_zero_points, metadata_bytes) ||
        (bias && ranges_overlap(output, output_bytes, bias, metadata_bytes)))
        return 0;
    for (uint32_t channel = 0; channel < output_channels; channel++) {
        uint64_t weight_magnitude, accumulator_bound, bias_magnitude = 0u;
        if (!isfinite(weight_scales[channel]) || weight_scales[channel] <= 0.0f ||
            !byte_zero_point_valid(weight_dtype, weight_zero_points[channel]))
            return 0;
        weight_magnitude = byte_centered_magnitude(
            weight_dtype, weight_zero_points[channel]);
        if (input_magnitude && weight_magnitude &&
            terms > (uint64_t)INT32_MAX / input_magnitude / weight_magnitude)
            return 0;
        accumulator_bound = input_magnitude * weight_magnitude * terms;
        if (bias) {
            int64_t bias_value = bias[channel];
            bias_magnitude = (uint64_t)(bias_value < 0 ? -bias_value : bias_value);
        }
        if (accumulator_bound > (uint64_t)INT32_MAX ||
            bias_magnitude > (uint64_t)INT32_MAX - accumulator_bound)
            return 0;
    }
    input_slot = graph_ensure_input(input, input_bytes, 0);
    weight_slot = graph_ensure_input(weight, weight_bytes, 1);
    scale_slot = graph_ensure_input(weight_scales, metadata_bytes, 1);
    zero_slot = graph_ensure_input(weight_zero_points,
        (size_t)output_channels * sizeof(int32_t), 1);
    if (bias) bias_slot = graph_ensure_input(bias,
        (size_t)output_channels * sizeof(int32_t), 1);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !weight_slot || !scale_slot || !zero_slot ||
        (bias && !bias_slot) || !output_slot) return 0;
    bias_device = bias_slot ? bias_slot->device : no_bias;
    args[0] = &input_slot->device;
    args[1] = &weight_slot->device;
    args[2] = &scale_slot->device;
    args[3] = &zero_slot->device;
    args[4] = &bias_device;
    args[5] = &output_slot->device;
    args[6] = &batch;
    args[7] = &input_height;
    args[8] = &input_width;
    args[9] = &input_channels;
    args[10] = &output_height;
    args[11] = &output_width;
    args[12] = &output_channels;
    args[13] = &kernel_height;
    args[14] = &kernel_width;
    args[15] = &input_per_group;
    args[16] = &stride_y;
    args[17] = &stride_x;
    args[18] = &dilation_y;
    args[19] = &dilation_x;
    args[20] = &padding_top;
    args[21] = &padding_left;
    args[22] = &padding_bottom;
    args[23] = &padding_right;
    args[24] = &groups;
    args[25] = &relu;
    args[26] = &input_scale;
    args[27] = &input_zero_point;
    args[28] = &output_scale;
    args[29] = &output_zero_point;
    args[30] = &input_dtype;
    args[31] = &weight_dtype;
    args[32] = &output_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qconv2d), output_elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}
