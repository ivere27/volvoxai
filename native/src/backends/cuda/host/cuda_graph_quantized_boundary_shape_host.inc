int cuda_graph_qargmax_i8u8(const void* input, int32_t* output,
                            uint32_t outer, uint32_t axis_size,
                            uint32_t inner, uint32_t input_dtype) {
    CudaTensorView input_view;
    CudaTensorView output_view;
    uint32_t input_elements, output_elements;
    size_t input_bytes, output_bytes;
    void* args[6];
    if (!input || !output || !outer || !axis_size || !inner ||
        axis_size > (uint32_t)INT32_MAX || !byte_dtype_valid(input_dtype) ||
        !checked_product4_bytes(outer, axis_size, inner, 1u, 1u,
                                &input_elements, &input_bytes) ||
        !checked_elements_bytes((uint64_t)outer * inner, sizeof(int32_t),
                                &output_elements, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes)) return 0;
    if (!graph_ensure_input_view(input, input_bytes, 0, &input_view) ||
        !graph_output_view(output, output_bytes, &output_view)) return 0;
    args[0] = &input_view.device;
    args[1] = &output_view.device;
    args[2] = &outer;
    args[3] = &axis_size;
    args[4] = &inner;
    args[5] = &input_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qargmax), output_elements, args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}

int cuda_graph_qmaskedmean_i8u8(const void* input, const int32_t* mask,
                                void* output, uint32_t batch,
                                uint32_t sequence, uint32_t width,
                                float input_scale, int32_t input_zero_point,
                                float output_scale, int32_t output_zero_point,
                                uint32_t input_dtype, uint32_t output_dtype) {
    CudaTensorSlot *input_slot, *mask_slot, *output_slot;
    uint32_t input_elements, mask_elements, output_elements;
    size_t input_bytes, output_bytes, mask_bytes;
    uint64_t input_magnitude;
    float multiplier;
    void* args[12];
    if (!input || !mask || !output || !batch || !sequence || !width ||
        !byte_dtype_valid(input_dtype) || !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(input_dtype, input_zero_point) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !checked_product4_bytes(batch, sequence, width, 1u, 1u,
                                &input_elements, &input_bytes) ||
        !checked_elements_bytes((uint64_t)batch * width, 1,
                                &output_elements, &output_bytes) ||
        !checked_elements_bytes((uint64_t)batch * sequence, sizeof(int32_t),
                                &mask_elements, &mask_bytes)) return 0;
    input_magnitude = byte_centered_magnitude(input_dtype, input_zero_point);
    multiplier = input_scale / output_scale;
    if ((input_magnitude &&
         (uint64_t)sequence > (uint64_t)INT32_MAX / input_magnitude) ||
        !isfinite(multiplier) || multiplier <= 0.0f ||
        ranges_overlap(output, output_bytes, input, input_bytes) ||
        ranges_overlap(output, output_bytes, mask, mask_bytes)) return 0;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    mask_slot = graph_ensure_input(mask, mask_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !mask_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &mask_slot->device;
    args[2] = &output_slot->device;
    args[3] = &batch;
    args[4] = &sequence;
    args[5] = &width;
    args[6] = &input_scale;
    args[7] = &input_zero_point;
    args[8] = &output_scale;
    args[9] = &output_zero_point;
    args[10] = &input_dtype;
    args[11] = &output_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qmaskedmean), output_elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_quantize_typed_f32_i8u8(const float* input, uint32_t elements,
                                       void* output, float output_scale,
                                       int32_t output_zero_point,
                                       uint32_t output_dtype) {
    CudaTensorSlot *input_slot, *output_slot;
    void* args[6];
    if (!input || !output || !elements || !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        (uint64_t)elements * sizeof(float) > SIZE_MAX ||
        ranges_overlap(output, elements, input,
                       (size_t)elements * sizeof(float))) return 0;
    input_slot = graph_ensure_input(input, (size_t)elements * sizeof(float), 0);
    output_slot = graph_output(output, elements);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &elements;
    args[3] = &output_scale;
    args[4] = &output_zero_point;
    args[5] = &output_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_quantize), elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_dequantize_typed_i8u8_f32(const void* input, uint32_t elements,
                                         float input_scale,
                                         int32_t input_zero_point,
                                         uint32_t input_dtype, float* output) {
    CudaTensorSlot *input_slot, *output_slot;
    void* args[6];
    if (!input || !output || !elements || !byte_dtype_valid(input_dtype) ||
        !byte_zero_point_valid(input_dtype, input_zero_point) ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        (uint64_t)elements * sizeof(float) > SIZE_MAX ||
        ranges_overlap(output, (size_t)elements * sizeof(float),
                       input, elements)) return 0;
    input_slot = graph_ensure_input(input, elements, 0);
    output_slot = graph_output(output, (size_t)elements * sizeof(float));
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &elements;
    args[3] = &input_scale;
    args[4] = &input_zero_point;
    args[5] = &input_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_dequantize), elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_requantize_linear_i8u8(const void* input,
                                      uint32_t input_elements,
                                      void* output, uint32_t output_elements,
                                      float input_scale,
                                      int32_t input_zero_point,
                                      float output_scale,
                                      int32_t output_zero_point,
                                      uint32_t input_dtype,
                                      uint32_t output_dtype) {
    CudaTensorSlot *input_slot, *output_slot;
    float multiplier;
    void* args[8];
    if (!input || !output || !input_elements ||
        input_elements != output_elements ||
        !byte_dtype_valid(input_dtype) || !byte_dtype_valid(output_dtype) ||
        !byte_zero_point_valid(input_dtype, input_zero_point) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        ranges_overlap(output, output_elements, input, input_elements)) return 0;
    multiplier = input_scale / output_scale;
    if (!isfinite(multiplier) || multiplier <= 0.0f) return 0;
    input_slot = graph_ensure_input(input, input_elements, 0);
    output_slot = graph_output(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &output_elements;
    args[3] = &multiplier;
    args[4] = &input_zero_point;
    args[5] = &output_zero_point;
    args[6] = &input_dtype;
    args[7] = &output_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_requantize), output_elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_copy_i8u8(const void* input, uint32_t input_elements,
                         void* output, uint32_t output_elements,
                         float input_scale, int32_t input_zero_point,
                         float output_scale, int32_t output_zero_point,
                         uint32_t input_dtype, uint32_t output_dtype) {
    CudaTensorSlot *input_slot, *output_slot;
    void* args[3];
    if (!input || !output || !input_elements ||
        input_elements != output_elements ||
        !byte_dtype_valid(input_dtype) || input_dtype != output_dtype ||
        !byte_zero_point_valid(input_dtype, input_zero_point) ||
        !byte_zero_point_valid(output_dtype, output_zero_point) ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        input_scale != output_scale || input_zero_point != output_zero_point ||
        ranges_overlap(output, output_elements, input, input_elements))
        return 0;
    input_slot = graph_ensure_input(input, input_elements, 0);
    output_slot = graph_output(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &output_elements;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_copy_byte), output_elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_transpose_i8u8(
        const void* input, void* output, const uint32_t* input_shape,
        const uint32_t* permutation, uint32_t rank, uint32_t elements,
        float input_scale, int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype) {
    CudaTensorSlot *input_slot, *output_slot;
    VxCudaTransposeParams params;
    uint32_t input_strides[8] = {0};
    uint32_t output_shape[8] = {0};
    uint32_t seen = 0;
    uint64_t stride = 1;
    uint64_t product = 1;
    void* args[4];
    if (!input || !output || !input_shape || !permutation ||
        rank == 0 || rank > 8 || elements == 0 ||
        !byte_qdesc_same(input_scale, input_zero_point, input_dtype,
                         output_scale, output_zero_point, output_dtype) ||
        ranges_overlap(output, elements, input, elements)) return 0;
    memset(&params, 0, sizeof(params));
    for (uint32_t reverse = rank; reverse-- > 0;) {
        if (!input_shape[reverse] || stride > UINT32_MAX) return 0;
        input_strides[reverse] = (uint32_t)stride;
        stride *= input_shape[reverse];
        if (stride > UINT32_MAX) return 0;
    }
    if (stride != elements) return 0;
    for (uint32_t dimension = 0; dimension < rank; dimension++) {
        uint32_t source = permutation[dimension];
        if (source >= rank || (seen & (1u << source))) return 0;
        seen |= 1u << source;
        output_shape[dimension] = input_shape[source];
        if (product > UINT32_MAX / output_shape[dimension]) return 0;
        product *= output_shape[dimension];
    }
    if (product != elements) return 0;
    stride = 1;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        params.output_strides[reverse] = (uint32_t)stride;
        stride *= output_shape[reverse];
    }
    for (uint32_t dimension = 0; dimension < rank; dimension++)
        params.input_strides[dimension] =
            input_strides[permutation[dimension]];
    params.rank = rank;
    input_slot = graph_ensure_input(input, elements, 0);
    output_slot = graph_output(output, elements);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &elements;
    args[3] = &params;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_transpose_byte),
                        elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_concat_i8u8(const void* const* inputs,
                           const uint32_t* input_elements,
                           const uint32_t* input_axes,
                           const float* input_scales,
                           const int32_t* input_zero_points,
                           const uint32_t* input_dtypes,
                           uint32_t input_count, void* output,
                           uint32_t output_elements, uint32_t output_axis,
                           uint32_t inner, float output_scale,
                           int32_t output_zero_point, uint32_t output_dtype) {
    CudaTensorSlot* output_slot;
    uint64_t output_width;
    uint64_t outer;
    uint64_t axis_sum = 0;
    uint32_t axis_offset = 0;
    if (!inputs || !input_elements || !input_axes || !input_scales ||
        !input_zero_points || !input_dtypes || !output || !input_count ||
        !output_elements || !output_axis || !inner ||
        !byte_qdesc_valid(output_scale, output_zero_point, output_dtype))
        return 0;
    output_width = (uint64_t)output_axis * inner;
    if (!output_width || output_width > output_elements ||
        output_elements % output_width) return 0;
    outer = output_elements / output_width;
    for (uint32_t index = 0; index < input_count; index++) {
        uint64_t expected;
        if (!inputs[index] || !input_axes[index] || axis_sum > output_axis ||
            (uint64_t)input_axes[index] > (uint64_t)output_axis - axis_sum)
            return 0;
        expected = outer * input_axes[index] * inner;
        if (expected > UINT32_MAX ||
            expected != input_elements[index] ||
            !byte_qdesc_same(input_scales[index], input_zero_points[index],
                             input_dtypes[index], output_scale,
                             output_zero_point, output_dtype) ||
            ranges_overlap(output, output_elements, inputs[index],
                           input_elements[index])) return 0;
        axis_sum += input_axes[index];
    }
    if (axis_sum != output_axis) return 0;
    output_slot = graph_output(output, output_elements);
    if (!output_slot) return 0;
    for (uint32_t index = 0; index < input_count; index++) {
        uint32_t source_elements = input_elements[index];
        uint32_t source_axis = input_axes[index];
        CudaTensorSlot* input_slot = graph_ensure_input(
            inputs[index], source_elements, 0);
        void* args[7];
        if (!input_slot) return 0;
        args[0] = &input_slot->device;
        args[1] = &output_slot->device;
        args[2] = &source_elements;
        args[3] = &axis_offset;
        args[4] = &source_axis;
        args[5] = &output_axis;
        args[6] = &inner;
        if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_concat_byte), source_elements, args))
            return 0;
        axis_offset += source_axis;
    }
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_maxpool2d_i8u8(const void* input, void* output,
                              uint32_t batch, uint32_t input_height,
                              uint32_t input_width, uint32_t channels,
                              uint32_t output_height, uint32_t output_width,
                              uint32_t kernel_y, uint32_t kernel_x,
                              uint32_t stride_y, uint32_t stride_x,
                              uint32_t padding_top, uint32_t padding_left,
                              uint32_t padding_bottom,
                              uint32_t padding_right, float input_scale,
                              int32_t input_zero_point, float output_scale,
                              int32_t output_zero_point, uint32_t input_dtype,
                              uint32_t output_dtype) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t input_elements, output_elements;
    size_t input_bytes, output_bytes;
    uint64_t padded_height, padded_width, expected_height, expected_width;
    uint32_t params[13];
    void* args[15];
    if (!input || !output || !kernel_y || !kernel_x || !stride_y ||
        !stride_x ||
        !byte_qdesc_same(input_scale, input_zero_point, input_dtype,
                         output_scale, output_zero_point, output_dtype) ||
        !checked_product4_bytes(batch, input_height, input_width, channels, 1,
                                &input_elements, &input_bytes) ||
        !checked_product4_bytes(batch, output_height, output_width, channels, 1,
                                &output_elements, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes)) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    if (padded_height < kernel_y || padded_width < kernel_x) return 0;
    expected_height = (padded_height - kernel_y) / stride_y + 1u;
    expected_width = (padded_width - kernel_x) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width)
        return 0;
    params[0] = batch;
    params[1] = input_height;
    params[2] = input_width;
    params[3] = channels;
    params[4] = output_height;
    params[5] = output_width;
    params[6] = kernel_y;
    params[7] = kernel_x;
    params[8] = stride_y;
    params[9] = stride_x;
    params[10] = padding_top;
    params[11] = padding_left;
    params[12] = input_dtype;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    for (int index = 0; index < 13; index++) args[index + 2] = &params[index];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_maxpool_byte), output_elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_resize_nearest_i8u8(const void* input, void* output,
                                   uint32_t batch, uint32_t input_height,
                                   uint32_t input_width, uint32_t channels,
                                   uint32_t output_height,
                                   uint32_t output_width, float input_scale,
                                   int32_t input_zero_point,
                                   float output_scale,
                                   int32_t output_zero_point,
                                   uint32_t input_dtype,
                                   uint32_t output_dtype) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t input_elements, output_elements;
    size_t input_bytes, output_bytes;
    uint32_t params[6];
    void* args[8];
    if (!input || !output ||
        !byte_qdesc_same(input_scale, input_zero_point, input_dtype,
                         output_scale, output_zero_point, output_dtype) ||
        !checked_product4_bytes(batch, input_height, input_width, channels, 1,
                                &input_elements, &input_bytes) ||
        !checked_product4_bytes(batch, output_height, output_width, channels, 1,
                                &output_elements, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes)) return 0;
    params[0] = batch;
    params[1] = input_height;
    params[2] = input_width;
    params[3] = channels;
    params[4] = output_height;
    params[5] = output_width;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    for (int index = 0; index < 6; index++) args[index + 2] = &params[index];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_resize_byte), output_elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}
