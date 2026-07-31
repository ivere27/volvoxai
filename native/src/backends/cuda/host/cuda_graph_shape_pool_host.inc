int cuda_graph_transpose_f32(const float* input, float* output,
                             const int* input_shape,
                             const int* permutation, int rank) {
    CudaTensorSlot *input_slot, *output_slot;
    VxCudaTransposeParams params;
    uint32_t original_input_strides[8] = {0};
    uint32_t elements;
    size_t bytes;
    uint64_t total = 1;
    uint64_t input_stride = 1;
    uint64_t output_stride = 1;
    unsigned int seen = 0;
    void* args[4];
    if (!input || !output || !input_shape || !permutation ||
        rank <= 0 || rank > 8) return 0;
    memset(&params, 0, sizeof(params));
    for (int dimension = 0; dimension < rank; dimension++) {
        int source = permutation[dimension];
        if (input_shape[dimension] <= 0 || source < 0 || source >= rank ||
            (seen & (1u << (unsigned int)source))) return 0;
        seen |= 1u << (unsigned int)source;
        total *= (uint32_t)input_shape[dimension];
        if (total > UINT32_MAX) return 0;
    }
    for (int dimension = rank - 1; dimension >= 0; dimension--) {
        if (input_stride > UINT32_MAX || output_stride > UINT32_MAX) return 0;
        original_input_strides[dimension] = (uint32_t)input_stride;
        params.output_strides[dimension] = (uint32_t)output_stride;
        input_stride *= (uint32_t)input_shape[dimension];
        output_stride *= (uint32_t)input_shape[permutation[dimension]];
    }
    for (int dimension = 0; dimension < rank; dimension++)
        params.input_strides[dimension] =
            original_input_strides[permutation[dimension]];
    params.rank = (uint32_t)rank;
    if (!checked_elements_bytes(total, sizeof(float), &elements, &bytes) ||
        ranges_overlap(output, bytes, input, bytes)) return 0;
    input_slot = graph_ensure_input(input, bytes, 0);
    output_slot = graph_output(output, bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &elements;
    args[3] = &params;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_transpose), elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}


int cuda_graph_where_f32(const float* condition, const float* a,
                         const float* b, float* output, long elements) {
    CudaTensorSlot *condition_slot, *a_slot, *b_slot, *output_slot;
    uint32_t count;
    size_t bytes;
    void* args[5];
    if (!condition || !a || !b || !output || elements <= 0 ||
        !checked_elements_bytes((uint64_t)elements, sizeof(float),
                                &count, &bytes) ||
        ranges_overlap(output, bytes, condition, bytes) ||
        ranges_overlap(output, bytes, a, bytes) ||
        ranges_overlap(output, bytes, b, bytes)) return 0;
    condition_slot = graph_ensure_input(condition, bytes, 0);
    a_slot = graph_ensure_input(a, bytes, 0);
    b_slot = graph_ensure_input(b, bytes, 0);
    output_slot = graph_output(output, bytes);
    if (!condition_slot || !a_slot || !b_slot || !output_slot) return 0;
    args[0] = &condition_slot->device;
    args[1] = &a_slot->device;
    args[2] = &b_slot->device;
    args[3] = &output_slot->device;
    args[4] = &count;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_where), count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_cast_copy_f32(const float* input, float* output,
                             long elements) {
    return cuda_graph_copy_f32(input, output, elements);
}

static int cuda_pad_shape4(const int* shape, int rank, uint32_t padded[4]) {
    int base;
    if (!shape || !padded || rank <= 0 || rank > 4) return 0;
    for (int index = 0; index < 4; index++) padded[index] = 1u;
    base = 4 - rank;
    for (int index = 0; index < rank; index++) {
        if (shape[index] <= 0) return 0;
        padded[base + index] = (uint32_t)shape[index];
    }
    return 1;
}

int cuda_graph_expand_f32(const float* input, float* output,
                          const int* input_shape, int input_rank,
                          const int* output_shape, int output_rank) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t input_dims[4], output_dims[4];
    uint32_t input_count, output_count;
    size_t input_bytes, output_bytes;
    void* args[10];
    if (!input || !output ||
        !cuda_pad_shape4(input_shape, input_rank, input_dims) ||
        !cuda_pad_shape4(output_shape, output_rank, output_dims)) return 0;
    for (int dimension = 0; dimension < 4; dimension++)
        if (input_dims[dimension] != 1u &&
            input_dims[dimension] != output_dims[dimension]) return 0;
    if (!checked_product4_bytes(input_dims[0], input_dims[1], input_dims[2],
            input_dims[3], sizeof(float), &input_count, &input_bytes) ||
        !checked_product4_bytes(output_dims[0], output_dims[1], output_dims[2],
            output_dims[3], sizeof(float), &output_count, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes)) return 0;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    for (int index = 0; index < 4; index++) args[index + 2] = &input_dims[index];
    for (int index = 0; index < 4; index++) args[index + 6] = &output_dims[index];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_expand), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_gather_i32_f32(const float* input, const int32_t* indices,
                              float* output, int outer, int axis_size,
                              int inner, int indices_elements,
                              int output_elements) {
    CudaTensorSlot *input_slot, *index_slot, *output_slot;
    uint32_t outer_count, axis_count, inner_count, indices_count, output_count;
    size_t input_bytes, index_bytes, output_bytes;
    void* args[7];
    if (!input || !indices || !output || outer <= 0 || axis_size <= 0 ||
        inner <= 0 || indices_elements <= 0 || output_elements <= 0 ||
        !checked_elements_bytes((uint64_t)(uint32_t)outer *
            (uint32_t)axis_size * (uint32_t)inner, sizeof(float),
            &output_count, &input_bytes) ||
        !checked_elements_bytes((uint32_t)indices_elements, sizeof(int32_t),
                                &indices_count, &index_bytes) ||
        !checked_elements_bytes((uint64_t)(uint32_t)outer *
            (uint32_t)indices_elements * (uint32_t)inner, sizeof(float),
            &output_count, &output_bytes) ||
        output_count != (uint32_t)output_elements ||
        ranges_overlap(output, output_bytes, input, input_bytes) ||
        ranges_overlap(output, output_bytes, indices, index_bytes)) return 0;
    outer_count = (uint32_t)outer;
    axis_count = (uint32_t)axis_size;
    inner_count = (uint32_t)inner;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    index_slot = graph_ensure_input(indices, index_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !index_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &index_slot->device;
    args[2] = &output_slot->device;
    args[3] = &outer_count;
    args[4] = &axis_count;
    args[5] = &inner_count;
    args[6] = &indices_count;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_gather), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_pad4d_f32(const float* input, float* output,
                         const int* input_shape, int input_rank,
                         const int* output_shape, int output_rank,
                         int padding_top, int padding_left, float value) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t input_dims[4], output_dims[4];
    uint32_t input_count, output_count;
    uint32_t padding_top_u32, padding_left_u32;
    size_t input_bytes, output_bytes;
    void* args[11];
    if (!input || !output || padding_top < 0 || padding_left < 0 ||
        !isfinite(value) ||
        !cuda_pad_shape4(input_shape, input_rank, input_dims) ||
        !cuda_pad_shape4(output_shape, output_rank, output_dims) ||
        input_dims[0] != output_dims[0] ||
        input_dims[3] != output_dims[3] ||
        (uint64_t)input_dims[1] + (uint32_t)padding_top > output_dims[1] ||
        (uint64_t)input_dims[2] + (uint32_t)padding_left > output_dims[2] ||
        !checked_product4_bytes(input_dims[0], input_dims[1], input_dims[2],
            input_dims[3], sizeof(float), &input_count, &input_bytes) ||
        !checked_product4_bytes(output_dims[0], output_dims[1], output_dims[2],
            output_dims[3], sizeof(float), &output_count, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes)) return 0;
    padding_top_u32 = (uint32_t)padding_top;
    padding_left_u32 = (uint32_t)padding_left;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &input_dims[0];
    args[3] = &input_dims[1];
    args[4] = &input_dims[2];
    args[5] = &input_dims[3];
    args[6] = &output_dims[1];
    args[7] = &output_dims[2];
    args[8] = &padding_top_u32;
    args[9] = &padding_left_u32;
    args[10] = &value;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_pad4d), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_slice4d_f32(const float* input, float* output,
                           const int* input_shape, int input_rank,
                           const int* output_shape, int output_rank,
                           const int* starts, const int* steps) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t input_dims[4], output_dims[4], starts_u32[4], steps_u32[4];
    uint32_t input_count, output_count;
    size_t input_bytes, output_bytes;
    void* args[17];
    if (!input || !output || !starts || !steps ||
        !cuda_pad_shape4(input_shape, input_rank, input_dims) ||
        !cuda_pad_shape4(output_shape, output_rank, output_dims)) return 0;
    for (int dimension = 0; dimension < 4; dimension++) {
        uint64_t last;
        if (starts[dimension] < 0 || steps[dimension] <= 0) return 0;
        starts_u32[dimension] = (uint32_t)starts[dimension];
        steps_u32[dimension] = (uint32_t)steps[dimension];
        last = starts_u32[dimension] +
            (uint64_t)(output_dims[dimension] - 1u) * steps_u32[dimension];
        if (last >= input_dims[dimension]) return 0;
    }
    if (!checked_product4_bytes(input_dims[0], input_dims[1], input_dims[2],
            input_dims[3], sizeof(float), &input_count, &input_bytes) ||
        !checked_product4_bytes(output_dims[0], output_dims[1], output_dims[2],
            output_dims[3], sizeof(float), &output_count, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes)) return 0;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    for (int index = 0; index < 4; index++) args[index + 2] = &output_dims[index];
    args[6] = &input_dims[1];
    args[7] = &input_dims[2];
    args[8] = &input_dims[3];
    for (int index = 0; index < 4; index++) args[index + 9] = &starts_u32[index];
    for (int index = 0; index < 4; index++) args[index + 13] = &steps_u32[index];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_slice4d), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_split_f32(const float* input, float* output,
                         long input_elements, long output_elements,
                         int inner, int split_size, int input_axis,
                         int offset) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t input_count, output_count;
    uint32_t inner_u32, split_u32, axis_u32, offset_u32;
    size_t input_bytes, output_bytes;
    void* args[7];
    if (!input || !output || input_elements <= 0 || output_elements <= 0 ||
        inner <= 0 || split_size <= 0 || input_axis <= 0 || offset < 0 ||
        (uint64_t)(uint32_t)offset + (uint32_t)split_size >
            (uint32_t)input_axis ||
        output_elements % ((long)split_size * inner) ||
        input_elements % ((long)input_axis * inner) ||
        output_elements / ((long)split_size * inner) !=
            input_elements / ((long)input_axis * inner) ||
        !checked_elements_bytes((uint64_t)input_elements, sizeof(float),
                                &input_count, &input_bytes) ||
        !checked_elements_bytes((uint64_t)output_elements, sizeof(float),
                                &output_count, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes)) return 0;
    inner_u32 = (uint32_t)inner;
    split_u32 = (uint32_t)split_size;
    axis_u32 = (uint32_t)input_axis;
    offset_u32 = (uint32_t)offset;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &output_count;
    args[3] = &inner_u32;
    args[4] = &split_u32;
    args[5] = &axis_u32;
    args[6] = &offset_u32;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_split), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_interp1d_f32(const float* input, float* output,
                            int channels, int input_length,
                            int output_length) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t input_count, output_count;
    uint32_t channels_u32, input_length_u32, output_length_u32;
    size_t input_bytes, output_bytes;
    void* args[5];
    if (!input || !output || channels <= 0 || input_length <= 0 ||
        output_length <= 0 ||
        !checked_elements_bytes((uint64_t)(uint32_t)channels *
            (uint32_t)input_length, sizeof(float), &input_count, &input_bytes) ||
        !checked_elements_bytes((uint64_t)(uint32_t)channels *
            (uint32_t)output_length, sizeof(float), &output_count,
            &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes)) return 0;
    channels_u32 = (uint32_t)channels;
    input_length_u32 = (uint32_t)input_length;
    output_length_u32 = (uint32_t)output_length;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &channels_u32;
    args[3] = &input_length_u32;
    args[4] = &output_length_u32;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_interp1d), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_conv1d_f32(const float* input, const float* weight,
                          const float* bias, float* output,
                          int batch, int input_channels, int input_length,
                          int output_channels, int output_length,
                          int kernel, int stride, int padding, int relu) {
    CudaTensorSlot *input_slot, *weight_slot, *bias_slot = NULL, *output_slot;
    uint32_t input_count, weight_count, output_count;
    uint32_t params[9];
    size_t input_bytes, weight_bytes, output_bytes, bias_bytes;
    CUdeviceptr no_bias = 0;
    CUdeviceptr bias_device;
    void* args[13];
    int64_t padded_length;
    if (!input || !weight || !output || batch <= 0 || input_channels <= 0 ||
        input_length <= 0 || output_channels <= 0 || output_length <= 0 ||
        kernel <= 0 || stride <= 0 || padding < 0 || relu < 0 || relu > 1)
        return 0;
    padded_length = (int64_t)input_length + 2LL * padding;
    if (padded_length < kernel ||
        (padded_length - kernel) / stride + 1 != output_length ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)input_channels,
            (uint32_t)input_length, 1u, sizeof(float), &input_count,
            &input_bytes) ||
        !checked_product4_bytes((uint32_t)output_channels,
            (uint32_t)input_channels, (uint32_t)kernel, 1u, sizeof(float),
            &weight_count, &weight_bytes) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)output_channels,
            (uint32_t)output_length, 1u, sizeof(float), &output_count,
            &output_bytes) ||
        (uint64_t)(uint32_t)output_channels * sizeof(float) > SIZE_MAX)
        return 0;
    bias_bytes = (size_t)(uint32_t)output_channels * sizeof(float);
    if (ranges_overlap(output, output_bytes, input, input_bytes) ||
        ranges_overlap(output, output_bytes, weight, weight_bytes) ||
        (bias && ranges_overlap(output, output_bytes, bias, bias_bytes)))
        return 0;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    weight_slot = graph_ensure_input(weight, weight_bytes, 1);
    if (bias) bias_slot = graph_ensure_input(bias, bias_bytes, 1);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !weight_slot || (bias && !bias_slot) || !output_slot)
        return 0;
    params[0] = (uint32_t)batch;
    params[1] = (uint32_t)input_channels;
    params[2] = (uint32_t)input_length;
    params[3] = (uint32_t)output_channels;
    params[4] = (uint32_t)output_length;
    params[5] = (uint32_t)kernel;
    params[6] = (uint32_t)stride;
    params[7] = (uint32_t)padding;
    params[8] = (uint32_t)relu;
    bias_device = bias_slot ? bias_slot->device : no_bias;
    args[0] = &input_slot->device;
    args[1] = &weight_slot->device;
    args[2] = &bias_device;
    args[3] = &output_slot->device;
    for (int index = 0; index < 9; index++) args[index + 4] = &params[index];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_conv1d), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_conv_transpose2d_f32(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    int batch, int input_height,
                                    int input_width, int input_channels,
                                    int output_height, int output_width,
                                    int output_channels, int kernel_height,
                                    int kernel_width, int stride_y,
                                    int stride_x, int padding_y,
                                    int padding_x) {
    CudaTensorSlot *input_slot, *weight_slot, *bias_slot = NULL, *output_slot;
    uint32_t input_count, weight_count, output_count;
    uint32_t params[13];
    size_t input_bytes, weight_bytes, output_bytes, bias_bytes;
    CUdeviceptr no_bias = 0;
    CUdeviceptr bias_device;
    void* args[17];
    if (!input || !weight || !output || batch <= 0 || input_height <= 0 ||
        input_width <= 0 || input_channels <= 0 || output_height <= 0 ||
        output_width <= 0 || output_channels <= 0 || kernel_height <= 0 ||
        kernel_width <= 0 || stride_y <= 0 || stride_x <= 0 ||
        padding_y < 0 || padding_x < 0 ||
        (int64_t)(input_height - 1) * stride_y - 2LL * padding_y +
            kernel_height != output_height ||
        (int64_t)(input_width - 1) * stride_x - 2LL * padding_x +
            kernel_width != output_width ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)input_height,
            (uint32_t)input_width, (uint32_t)input_channels, sizeof(float),
            &input_count, &input_bytes) ||
        !checked_product4_bytes((uint32_t)input_channels,
            (uint32_t)output_channels, (uint32_t)kernel_height,
            (uint32_t)kernel_width, sizeof(float), &weight_count,
            &weight_bytes) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)output_height,
            (uint32_t)output_width, (uint32_t)output_channels, sizeof(float),
            &output_count, &output_bytes) ||
        (uint64_t)(uint32_t)output_channels * sizeof(float) > SIZE_MAX)
        return 0;
    bias_bytes = (size_t)(uint32_t)output_channels * sizeof(float);
    if (ranges_overlap(output, output_bytes, input, input_bytes) ||
        ranges_overlap(output, output_bytes, weight, weight_bytes) ||
        (bias && ranges_overlap(output, output_bytes, bias, bias_bytes)))
        return 0;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    weight_slot = graph_ensure_input(weight, weight_bytes, 1);
    if (bias) bias_slot = graph_ensure_input(bias, bias_bytes, 1);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !weight_slot || (bias && !bias_slot) || !output_slot)
        return 0;
    params[0] = (uint32_t)batch;
    params[1] = (uint32_t)input_height;
    params[2] = (uint32_t)input_width;
    params[3] = (uint32_t)input_channels;
    params[4] = (uint32_t)output_height;
    params[5] = (uint32_t)output_width;
    params[6] = (uint32_t)output_channels;
    params[7] = (uint32_t)kernel_height;
    params[8] = (uint32_t)kernel_width;
    params[9] = (uint32_t)stride_y;
    params[10] = (uint32_t)stride_x;
    params[11] = (uint32_t)padding_y;
    params[12] = (uint32_t)padding_x;
    bias_device = bias_slot ? bias_slot->device : no_bias;
    args[0] = &input_slot->device;
    args[1] = &weight_slot->device;
    args[2] = &bias_device;
    args[3] = &output_slot->device;
    for (int index = 0; index < 13; index++) args[index + 4] = &params[index];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_convtranspose2d), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_quantize_linear_i8(const float* input, signed char* output,
                                  long elements, float input_scale,
                                  int input_zero_point, float output_scale,
                                  int output_zero_point) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t count;
    int32_t input_zero = input_zero_point;
    int32_t output_zero = output_zero_point;
    size_t input_bytes;
    void* args[7];
    if (!input || !output || elements <= 0 ||
        !isfinite(input_scale) || !isfinite(output_scale) ||
        output_scale <= 0.0f || output_zero_point < -128 ||
        output_zero_point > 127 ||
        !checked_elements_bytes((uint64_t)elements, sizeof(float),
                                &count, &input_bytes) ||
        ranges_overlap(output, count, input, input_bytes)) return 0;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, count);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &count;
    args[3] = &input_scale;
    args[4] = &input_zero;
    args[5] = &output_scale;
    args[6] = &output_zero;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_quantize_linear), count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_dequantize_linear_f32(const float* input, const float* scale,
                                     const float* zero_point, float* output,
                                     long elements, int has_zero_point) {
    CudaTensorSlot *input_slot, *scale_slot, *zero_slot = NULL, *output_slot;
    uint32_t count;
    uint32_t has_zero;
    size_t bytes;
    CUdeviceptr no_zero = 0;
    CUdeviceptr zero_device;
    void* args[6];
    if (!input || !scale || !output || elements <= 0 ||
        (has_zero_point != 0 && has_zero_point != 1) ||
        (has_zero_point && !zero_point) || !isfinite(scale[0]) ||
        (has_zero_point && !isfinite(zero_point[0])) ||
        !checked_elements_bytes((uint64_t)elements, sizeof(float),
                                &count, &bytes) ||
        ranges_overlap(output, bytes, input, bytes)) return 0;
    has_zero = (uint32_t)has_zero_point;
    input_slot = graph_ensure_input(input, bytes, 0);
    scale_slot = graph_ensure_input(scale, sizeof(float), 1);
    if (has_zero_point)
        zero_slot = graph_ensure_input(zero_point, sizeof(float), 1);
    output_slot = graph_output(output, bytes);
    if (!input_slot || !scale_slot || (has_zero_point && !zero_slot) ||
        !output_slot) return 0;
    zero_device = zero_slot ? zero_slot->device : no_zero;
    args[0] = &input_slot->device;
    args[1] = &scale_slot->device;
    args[2] = &zero_device;
    args[3] = &output_slot->device;
    args[4] = &count;
    args[5] = &has_zero;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_dequantize_linear), count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_resize_f32(const float* input, float* output,
                          int batch, int height, int width, int channels,
                          int output_height, int output_width, int mode) {
    CudaTensorSlot* input_slot;
    CudaTensorSlot* output_slot;
    uint32_t input_count, output_count;
    uint32_t params[7];
    size_t input_bytes, output_bytes;
    void* args[9];
    if (!input || !output || batch <= 0 || height <= 0 || width <= 0 ||
        channels <= 0 || output_height <= 0 || output_width <= 0 ||
        (mode != 0 && mode != 1) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)height,
            (uint32_t)width, (uint32_t)channels, sizeof(float),
            &input_count, &input_bytes) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)output_height,
            (uint32_t)output_width, (uint32_t)channels, sizeof(float),
            &output_count, &output_bytes)) return 0;
    params[0] = (uint32_t)batch;
    params[1] = (uint32_t)height;
    params[2] = (uint32_t)width;
    params[3] = (uint32_t)channels;
    params[4] = (uint32_t)output_height;
    params[5] = (uint32_t)output_width;
    params[6] = (uint32_t)mode;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    for (int index = 0; index < 7; index++) args[index + 2] = &params[index];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_resize), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_resize_nearest_f32(const float* input, float* output,
                                  int batch, int height, int width,
                                  int channels, int output_height,
                                  int output_width) {
    return cuda_graph_resize_f32(input, output, batch, height, width, channels,
                                 output_height, output_width, 0);
}

int cuda_graph_upsample2x_f32(const float* input, float* output,
                              int batch, int height, int width, int channels) {
    if (height > INT_MAX / 2 || width > INT_MAX / 2) return 0;
    return cuda_graph_resize_f32(input, output, batch, height, width, channels,
                                 height * 2, width * 2, 0);
}

int cuda_graph_concat_f32(const float** inputs, const long* sizes,
                          const int* input_axes, int count, float* output,
                          int output_axis, int inner, int sigmoid) {
    CudaTensorSlot* output_slot;
    uint32_t output_count;
    uint32_t axis_offset = 0;
    size_t output_bytes;
    uint64_t total = 0;
    uint64_t summed_axis = 0;
    if (!inputs || !sizes || !input_axes || !output || count <= 0 ||
        output_axis <= 0 || inner <= 0 || (sigmoid != 0 && sigmoid != 1))
        return 0;
    for (int index = 0; index < count; index++) {
        if (!inputs[index] || sizes[index] <= 0 || input_axes[index] <= 0)
            return 0;
        total += (uint64_t)sizes[index];
        summed_axis += (uint64_t)(uint32_t)input_axes[index];
    }
    if (summed_axis != (uint32_t)output_axis ||
        total % ((uint64_t)(uint32_t)output_axis * (uint32_t)inner) != 0 ||
        !checked_elements_bytes(total, sizeof(float), &output_count,
                                &output_bytes)) return 0;
    output_slot = graph_output(output, output_bytes);
    if (!output_slot) return 0;
    for (int index = 0; index < count; index++) {
        CudaTensorSlot* input_slot;
        uint32_t input_count;
        uint32_t input_axis = (uint32_t)input_axes[index];
        uint32_t output_axis_u32 = (uint32_t)output_axis;
        uint32_t inner_u32 = (uint32_t)inner;
        uint32_t apply_sigmoid = (uint32_t)sigmoid;
        size_t input_bytes;
        void* args[8];
        if (!checked_elements_bytes((uint64_t)sizes[index], sizeof(float),
                                    &input_count, &input_bytes)) return 0;
        input_slot = graph_ensure_input(inputs[index], input_bytes, 0);
        if (!input_slot) return 0;
        args[0] = &input_slot->device;
        args[1] = &output_slot->device;
        args[2] = &input_count;
        args[3] = &axis_offset;
        args[4] = &input_axis;
        args[5] = &output_axis_u32;
        args[6] = &inner_u32;
        args[7] = &apply_sigmoid;
        if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_concat), input_count, args)) return 0;
        axis_offset += input_axis;
    }
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_concat_flat_f32(const float** inputs, const long* sizes,
                               int count, float* output) {
    int input_axes_storage[256];
    long total = 0;
    if (!sizes || count <= 0 || count > 256) return 0;
    for (int index = 0; index < count; index++) {
        if (sizes[index] <= 0 || sizes[index] > INT_MAX - total) return 0;
        input_axes_storage[index] = (int)sizes[index];
        total += sizes[index];
    }
    if (total > INT_MAX) return 0;
    return cuda_graph_concat_f32(inputs, sizes, input_axes_storage, count,
                                 output, (int)total, 1, 0);
}

int cuda_graph_concat_sigmoid_flat_f32(const float** inputs,
                                       const long* sizes, int count,
                                       float* output) {
    int input_axes_storage[256];
    long total = 0;
    if (!sizes || count <= 0 || count > 256) return 0;
    for (int index = 0; index < count; index++) {
        if (sizes[index] <= 0 || sizes[index] > INT_MAX - total) return 0;
        input_axes_storage[index] = (int)sizes[index];
        total += sizes[index];
    }
    if (total > INT_MAX) return 0;
    return cuda_graph_concat_f32(inputs, sizes, input_axes_storage, count,
                                 output, (int)total, 1, 1);
}

int cuda_graph_maxpool2d_f32(const float* input, float* output,
                             int batch, int height, int width, int channels,
                             int output_height, int output_width,
                             int kernel_y, int kernel_x,
                             int stride_y, int stride_x,
                             int padding_y, int padding_x) {
    CudaTensorSlot* input_slot;
    CudaTensorSlot* output_slot;
    uint32_t input_count, output_count;
    uint32_t params[12];
    size_t input_bytes, output_bytes;
    void* args[14];
    if (!input || !output || batch <= 0 || height <= 0 || width <= 0 ||
        channels <= 0 || output_height <= 0 || output_width <= 0 ||
        kernel_y <= 0 || kernel_x <= 0 || stride_y <= 0 || stride_x <= 0 ||
        padding_y < 0 || padding_x < 0 ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)height,
            (uint32_t)width, (uint32_t)channels, sizeof(float),
            &input_count, &input_bytes) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)output_height,
            (uint32_t)output_width, (uint32_t)channels, sizeof(float),
            &output_count, &output_bytes)) return 0;
    params[0] = (uint32_t)batch;
    params[1] = (uint32_t)height;
    params[2] = (uint32_t)width;
    params[3] = (uint32_t)channels;
    params[4] = (uint32_t)output_height;
    params[5] = (uint32_t)output_width;
    params[6] = (uint32_t)kernel_y;
    params[7] = (uint32_t)kernel_x;
    params[8] = (uint32_t)stride_y;
    params[9] = (uint32_t)stride_x;
    params[10] = (uint32_t)padding_y;
    params[11] = (uint32_t)padding_x;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    for (int index = 0; index < 12; index++) args[index + 2] = &params[index];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_maxpool), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}
