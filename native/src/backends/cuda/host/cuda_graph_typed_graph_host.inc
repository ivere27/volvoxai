static int cuda_shape_strides(const int* shape, int rank,
                              uint32_t strides[8], uint32_t* elements) {
    uint64_t stride = 1u;
    if (!shape || !strides || !elements || rank <= 0 || rank > 8) return 0;
    memset(strides, 0, 8u * sizeof(uint32_t));
    for (int dimension = rank - 1; dimension >= 0; dimension--) {
        if (shape[dimension] <= 0 || stride > UINT32_MAX) return 0;
        strides[dimension] = (uint32_t)stride;
        stride *= (uint32_t)shape[dimension];
        if (stride > UINT32_MAX) return 0;
    }
    *elements = (uint32_t)stride;
    return 1;
}

int cuda_graph_batch_matmul_f32(
        const float* a, const int* a_shape, int a_rank,
        const float* b, const int* b_shape, int b_rank,
        float* output, const int* output_shape, int output_rank) {
    CudaTensorSlot *a_slot, *b_slot, *output_slot;
    VxCudaBatchMatMulParams params;
    uint32_t a_strides[8] = {0};
    uint32_t b_strides[8] = {0};
    uint32_t output_strides[8] = {0};
    uint32_t a_elements, b_elements, output_elements;
    size_t a_bytes, b_bytes, output_bytes;
    int batch_rank;
    int a_leading;
    int b_leading;
    uint64_t output_matrix_elements;
    void* args[5];
    if (!a || !a_shape || !b || !b_shape || !output || !output_shape ||
        a_rank < 2 || a_rank > 8 || b_rank < 2 || b_rank > 8 ||
        output_rank < 2 || output_rank > 8 ||
        output_rank != (a_rank > b_rank ? a_rank : b_rank) ||
        !cuda_shape_strides(a_shape, a_rank, a_strides, &a_elements) ||
        !cuda_shape_strides(b_shape, b_rank, b_strides, &b_elements) ||
        !cuda_shape_strides(output_shape, output_rank, output_strides,
                            &output_elements))
        return 0;
    batch_rank = output_rank - 2;
    a_leading = output_rank - a_rank;
    b_leading = output_rank - b_rank;
    output_matrix_elements =
        (uint64_t)(uint32_t)output_shape[output_rank - 2] *
        (uint32_t)output_shape[output_rank - 1];
    if (a_shape[a_rank - 1] != b_shape[b_rank - 2] ||
        output_shape[output_rank - 2] != a_shape[a_rank - 2] ||
        output_shape[output_rank - 1] != b_shape[b_rank - 1] ||
        output_matrix_elements == 0u ||
        output_matrix_elements > UINT32_MAX)
        return 0;
    memset(&params, 0, sizeof(params));
    for (int dimension = 0; dimension < batch_rank; dimension++) {
        int a_dimension = dimension - a_leading;
        int b_dimension = dimension - b_leading;
        int a_extent = a_dimension >= 0 ? a_shape[a_dimension] : 1;
        int b_extent = b_dimension >= 0 ? b_shape[b_dimension] : 1;
        int output_extent = output_shape[dimension];
        int expected_extent =
            a_extent > b_extent ? a_extent : b_extent;
        if ((a_extent != 1 && b_extent != 1 && a_extent != b_extent) ||
            output_extent != expected_extent)
            return 0;
        params.output_batch_strides[dimension] =
            output_strides[dimension] /
            (uint32_t)output_matrix_elements;
        if (a_dimension >= 0 && a_extent != 1)
            params.a_batch_strides[dimension] = a_strides[a_dimension];
        if (b_dimension >= 0 && b_extent != 1)
            params.b_batch_strides[dimension] = b_strides[b_dimension];
    }
    params.batch_rank = (uint32_t)batch_rank;
    params.m = (uint32_t)a_shape[a_rank - 2];
    params.k = (uint32_t)a_shape[a_rank - 1];
    params.n = (uint32_t)b_shape[b_rank - 1];
    if ((uint64_t)params.m * params.k > a_elements ||
        (uint64_t)params.k * params.n > b_elements ||
        (uint64_t)params.m * params.n > output_elements ||
        !checked_elements_bytes(a_elements, sizeof(float), &a_elements,
                                &a_bytes) ||
        !checked_elements_bytes(b_elements, sizeof(float), &b_elements,
                                &b_bytes) ||
        !checked_elements_bytes(output_elements, sizeof(float),
                                &output_elements, &output_bytes) ||
        ranges_overlap(output, output_bytes, a, a_bytes) ||
        ranges_overlap(output, output_bytes, b, b_bytes))
        return 0;
    a_slot = graph_ensure_input(a, a_bytes, 0);
    b_slot = graph_ensure_input(b, b_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    args[0] = &a_slot->device;
    args[1] = &b_slot->device;
    args[2] = &output_slot->device;
    args[3] = &output_elements;
    args[4] = &params;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_batch_matmul),
                        output_elements, args))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_qbatch_matmul_i8u8(
        const void* a, const int* a_shape, int a_rank,
        float a_scale, int32_t a_zero_point, uint32_t a_dtype,
        const void* b, const int* b_shape, int b_rank,
        float b_scale, int32_t b_zero_point, uint32_t b_dtype,
        void* output, const int* output_shape, int output_rank,
        float output_scale, int32_t output_zero_point,
        uint32_t output_dtype) {
    CudaTensorSlot *a_slot, *b_slot, *output_slot;
    VxCudaBatchMatMulParams params;
    uint32_t a_strides[8] = {0};
    uint32_t b_strides[8] = {0};
    uint32_t output_strides[8] = {0};
    uint32_t a_elements, b_elements, output_elements;
    uint64_t a_magnitude;
    uint64_t b_magnitude;
    uint64_t accumulator_bound;
    size_t a_bytes, b_bytes, output_bytes;
    float multiplier;
    int batch_rank;
    int a_leading;
    int b_leading;
    uint64_t output_matrix_elements;
    void* args[14];
    if (!a || !a_shape || !b || !b_shape || !output || !output_shape ||
        a_rank < 2 || a_rank > 8 || b_rank < 2 || b_rank > 8 ||
        output_rank < 2 || output_rank > 8 ||
        output_rank != (a_rank > b_rank ? a_rank : b_rank) ||
        !byte_qdesc_valid(a_scale, a_zero_point, a_dtype) ||
        !byte_qdesc_valid(b_scale, b_zero_point, b_dtype) ||
        !byte_qdesc_valid(output_scale, output_zero_point, output_dtype) ||
        !cuda_shape_strides(a_shape, a_rank, a_strides, &a_elements) ||
        !cuda_shape_strides(b_shape, b_rank, b_strides, &b_elements) ||
        !cuda_shape_strides(output_shape, output_rank, output_strides,
                            &output_elements))
        return 0;
    batch_rank = output_rank - 2;
    a_leading = output_rank - a_rank;
    b_leading = output_rank - b_rank;
    output_matrix_elements =
        (uint64_t)(uint32_t)output_shape[output_rank - 2] *
        (uint32_t)output_shape[output_rank - 1];
    if (a_shape[a_rank - 1] != b_shape[b_rank - 2] ||
        output_shape[output_rank - 2] != a_shape[a_rank - 2] ||
        output_shape[output_rank - 1] != b_shape[b_rank - 1] ||
        output_matrix_elements == 0u ||
        output_matrix_elements > UINT32_MAX)
        return 0;
    memset(&params, 0, sizeof(params));
    for (int dimension = 0; dimension < batch_rank; dimension++) {
        int a_dimension = dimension - a_leading;
        int b_dimension = dimension - b_leading;
        int a_extent = a_dimension >= 0 ? a_shape[a_dimension] : 1;
        int b_extent = b_dimension >= 0 ? b_shape[b_dimension] : 1;
        int output_extent = output_shape[dimension];
        int expected_extent =
            a_extent > b_extent ? a_extent : b_extent;
        if ((a_extent != 1 && b_extent != 1 && a_extent != b_extent) ||
            output_extent != expected_extent)
            return 0;
        params.output_batch_strides[dimension] =
            output_strides[dimension] /
            (uint32_t)output_matrix_elements;
        if (a_dimension >= 0 && a_extent != 1)
            params.a_batch_strides[dimension] = a_strides[a_dimension];
        if (b_dimension >= 0 && b_extent != 1)
            params.b_batch_strides[dimension] = b_strides[b_dimension];
    }
    params.batch_rank = (uint32_t)batch_rank;
    params.m = (uint32_t)a_shape[a_rank - 2];
    params.k = (uint32_t)a_shape[a_rank - 1];
    params.n = (uint32_t)b_shape[b_rank - 1];
    a_magnitude = byte_centered_magnitude(a_dtype, a_zero_point);
    b_magnitude = byte_centered_magnitude(b_dtype, b_zero_point);
    accumulator_bound = a_magnitude * b_magnitude * params.k;
    multiplier = (a_scale * b_scale) / output_scale;
    if ((uint64_t)params.m * params.k > a_elements ||
        (uint64_t)params.k * params.n > b_elements ||
        (uint64_t)params.m * params.n > output_elements ||
        accumulator_bound > INT32_MAX ||
        !isfinite(multiplier) || multiplier <= 0.0f ||
        !checked_elements_bytes(a_elements, 1u, &a_elements, &a_bytes) ||
        !checked_elements_bytes(b_elements, 1u, &b_elements, &b_bytes) ||
        !checked_elements_bytes(
            output_elements, 1u, &output_elements, &output_bytes) ||
        ranges_overlap(output, output_bytes, a, a_bytes) ||
        ranges_overlap(output, output_bytes, b, b_bytes))
        return 0;
    a_slot = graph_ensure_input(a, a_bytes, 0);
    b_slot = graph_ensure_input(b, b_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    args[0] = &a_slot->device;
    args[1] = &b_slot->device;
    args[2] = &output_slot->device;
    args[3] = &output_elements;
    args[4] = &params;
    args[5] = &a_scale;
    args[6] = &a_zero_point;
    args[7] = &b_scale;
    args[8] = &b_zero_point;
    args[9] = &output_scale;
    args[10] = &output_zero_point;
    args[11] = &a_dtype;
    args[12] = &b_dtype;
    args[13] = &output_dtype;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_qbatch_matmul),
                        output_elements, args))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_argmax_f32(const float* input, int32_t* output,
                          uint32_t outer, uint32_t axis_size,
                          uint32_t inner) {
    CudaTensorView input_view;
    CudaTensorView output_view;
    uint32_t input_elements;
    uint32_t output_elements;
    size_t input_bytes;
    size_t output_bytes;
    void* args[5];
    if (!input || !output || !outer || !axis_size || !inner ||
        axis_size > (uint32_t)INT32_MAX ||
        !checked_product4_bytes(
            outer, axis_size, inner, 1u, sizeof(float),
            &input_elements, &input_bytes) ||
        !checked_elements_bytes(
            (uint64_t)outer * inner, sizeof(int32_t),
            &output_elements, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes))
        return 0;
    if (!graph_ensure_input_view(input, input_bytes, 0, &input_view) ||
        !graph_output_view(output, output_bytes, &output_view))
        return 0;
    args[0] = &input_view.device;
    args[1] = &output_view.device;
    args[2] = &outer;
    args[3] = &axis_size;
    args[4] = &inner;
    if (!cuda_launch_1d(
            CUDA_FORWARD_FUNCTION(k_argmax_f32_i32),
            output_elements, args))
        return 0;
    graph_mark_device(output_view.slot);
    (void)input_elements;
    return 1;
}

int cuda_graph_compare_i32(
        const int32_t* a, long a_elements,
        const int32_t* b, long b_elements,
        int32_t* output, long output_elements,
        const uint32_t* output_strides,
        const uint32_t* a_strides,
        const uint32_t* b_strides,
        int rank, int operation) {
    CudaTensorSlot *a_slot, *b_slot, *output_slot;
    VxCudaBroadcastParams params;
    uint32_t output_count;
    int32_t op = operation;
    size_t a_bytes, b_bytes, output_bytes;
    void* args[6];
    if (!a || !b || !output || !output_strides || !a_strides || !b_strides ||
        a_elements <= 0 || b_elements <= 0 || output_elements <= 0 ||
        rank <= 0 || rank > 8 || operation < 0 || operation > 1 ||
        !checked_elements_bytes((uint64_t)a_elements, sizeof(int32_t),
                                &output_count, &a_bytes) ||
        !checked_elements_bytes((uint64_t)b_elements, sizeof(int32_t),
                                &output_count, &b_bytes) ||
        !checked_elements_bytes((uint64_t)output_elements, sizeof(int32_t),
                                &output_count, &output_bytes) ||
        ranges_overlap(output, output_bytes, a, a_bytes) ||
        ranges_overlap(output, output_bytes, b, b_bytes))
        return 0;
    memset(&params, 0, sizeof(params));
    memcpy(params.output_strides, output_strides,
           (size_t)rank * sizeof(uint32_t));
    memcpy(params.a_strides, a_strides, (size_t)rank * sizeof(uint32_t));
    memcpy(params.b_strides, b_strides, (size_t)rank * sizeof(uint32_t));
    params.rank = (uint32_t)rank;
    a_slot = graph_ensure_input(a, a_bytes, 0);
    b_slot = graph_ensure_input(b, b_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    args[0] = &a_slot->device;
    args[1] = &b_slot->device;
    args[2] = &output_slot->device;
    args[3] = &output_count;
    args[4] = &params;
    args[5] = &op;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_compare_i32),
                        output_count, args))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

static int cuda_graph_unary_i32(
        const int32_t* input, int32_t* output, long elements,
        int operation, int32_t minimum, int32_t maximum) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t count;
    size_t bytes;
    void* args[5];
    if (!input || !output || elements <= 0 ||
        (operation != 0 && operation != 1) ||
        (operation == 1 && minimum > maximum) ||
        !checked_elements_bytes((uint64_t)elements, sizeof(int32_t),
                                &count, &bytes) ||
        ranges_overlap(output, bytes, input, bytes))
        return 0;
    input_slot = graph_ensure_input(input, bytes, 0);
    output_slot = graph_output(output, bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &count;
    if (operation == 0) {
        if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_not_i32), count, args))
            return 0;
    } else {
        args[3] = &minimum;
        args[4] = &maximum;
        if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_clip_i32), count, args))
            return 0;
    }
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_not_i32(const int32_t* input, int32_t* output, long elements) {
    return cuda_graph_unary_i32(input, output, elements, 0, 0, 0);
}

int cuda_graph_clip_i32(const int32_t* input, int32_t* output, long elements,
                        int32_t minimum, int32_t maximum) {
    return cuda_graph_unary_i32(
        input, output, elements, 1, minimum, maximum);
}

int cuda_graph_where_32(const int32_t* condition, const void* a,
                        const void* b, void* output, long elements) {
    CudaTensorSlot *condition_slot, *a_slot, *b_slot, *output_slot;
    uint32_t count;
    size_t bytes;
    void* args[5];
    if (!condition || !a || !b || !output || elements <= 0 ||
        !checked_elements_bytes((uint64_t)elements, sizeof(uint32_t),
                                &count, &bytes) ||
        ranges_overlap(output, bytes, condition, bytes) ||
        ranges_overlap(output, bytes, a, bytes) ||
        ranges_overlap(output, bytes, b, bytes))
        return 0;
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
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_where_32), count, args))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_cast_typed(const void* input, int input_dtype,
                          void* output, int output_dtype, long elements) {
    CudaTensorSlot *input_slot, *output_slot;
    uint32_t count;
    uint32_t checked_count;
    int32_t input_type = input_dtype;
    int32_t output_type = output_dtype;
    size_t input_bytes;
    size_t output_bytes;
    size_t input_element_size;
    size_t output_element_size;
    void* args[5];
    if (!input || !output || elements <= 0 ||
        (input_dtype != VX_DTYPE_F32 && input_dtype != VX_DTYPE_I32 &&
         input_dtype != VX_DTYPE_I8 && input_dtype != VX_DTYPE_U8) ||
        (output_dtype != VX_DTYPE_F32 && output_dtype != VX_DTYPE_I32 &&
         output_dtype != VX_DTYPE_I8 && output_dtype != VX_DTYPE_U8))
        return 0;
    input_element_size =
        input_dtype == VX_DTYPE_I8 ||
        input_dtype == VX_DTYPE_U8 ? 1u : sizeof(uint32_t);
    output_element_size =
        output_dtype == VX_DTYPE_I8 ||
        output_dtype == VX_DTYPE_U8 ? 1u : sizeof(uint32_t);
    if (!checked_elements_bytes(
            (uint64_t)elements, input_element_size,
            &count, &input_bytes) ||
        !checked_elements_bytes(
            (uint64_t)elements, output_element_size,
            &checked_count, &output_bytes) ||
        checked_count != count ||
        ranges_overlap(output, output_bytes, input, input_bytes))
        return 0;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &count;
    args[3] = &input_type;
    args[4] = &output_type;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_cast_typed), count, args))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_cast_i32_f32(const void* input, void* output, long elements,
                            int input_is_i32) {
    if (input_is_i32 != 0 && input_is_i32 != 1) return 0;
    return cuda_graph_cast_typed(
        input, input_is_i32 ? VX_DTYPE_I32 : VX_DTYPE_F32,
        output, input_is_i32 ? VX_DTYPE_F32 : VX_DTYPE_I32,
        elements);
}

int cuda_graph_copy_32(const void* input, void* output, long elements) {
    CudaTensorView source;
    CudaTensorView destination;
    uint32_t count;
    size_t bytes;
    void* args[3];
    if (!input || !output || elements <= 0 ||
        !checked_elements_bytes((uint64_t)elements, sizeof(uint32_t),
                                &count, &bytes))
        return 0;
    if (!graph_ensure_input_view(input, bytes, 0, &source) ||
        !graph_output_view(output, bytes, &destination))
        return 0;
    args[0] = &source.device;
    args[1] = &destination.device;
    args[2] = &count;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_copy_32), count, args))
        return 0;
    graph_mark_device(destination.slot);
    return 1;
}

int cuda_graph_transpose_32(const void* input, void* output,
                            const int* input_shape,
                            const int* permutation, int rank) {
    CudaTensorSlot *input_slot, *output_slot;
    VxCudaTransposeParams params;
    uint32_t input_strides[8] = {0};
    uint32_t elements;
    size_t bytes;
    unsigned int seen = 0u;
    void* args[4];
    if (!input || !output || !input_shape || !permutation ||
        !cuda_shape_strides(input_shape, rank, input_strides, &elements))
        return 0;
    memset(&params, 0, sizeof(params));
    {
        uint64_t output_stride = 1u;
        for (int dimension = rank - 1; dimension >= 0; dimension--) {
            int source = permutation[dimension];
            if (source < 0 || source >= rank ||
                (seen & (1u << (unsigned int)source)) ||
                output_stride > UINT32_MAX)
                return 0;
            seen |= 1u << (unsigned int)source;
            params.output_strides[dimension] = (uint32_t)output_stride;
            params.input_strides[dimension] = input_strides[source];
            output_stride *= (uint32_t)input_shape[source];
        }
        if (output_stride != elements) return 0;
    }
    params.rank = (uint32_t)rank;
    if (!checked_elements_bytes(elements, sizeof(uint32_t), &elements,
                                &bytes) ||
        ranges_overlap(output, bytes, input, bytes))
        return 0;
    input_slot = graph_ensure_input(input, bytes, 0);
    output_slot = graph_output(output, bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &elements;
    args[3] = &params;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_transpose_32),
                        elements, args))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_expand_32(const void* input, void* output,
                         const int* input_shape, int input_rank,
                         const int* output_shape, int output_rank) {
    CudaTensorSlot *input_slot, *output_slot;
    VxCudaTransposeParams params;
    uint32_t input_strides[8] = {0};
    uint32_t output_strides[8] = {0};
    uint32_t input_elements, output_elements;
    size_t input_bytes, output_bytes;
    int leading;
    void* args[4];
    if (!input || !output || input_rank < 0 || input_rank > output_rank ||
        output_rank <= 0 ||
        output_rank > 8 ||
        (input_rank > 0 &&
         !cuda_shape_strides(input_shape, input_rank, input_strides,
                             &input_elements)) ||
        !cuda_shape_strides(output_shape, output_rank, output_strides,
                            &output_elements))
        return 0;
    if (input_rank == 0) input_elements = 1u;
    leading = output_rank - input_rank;
    memset(&params, 0, sizeof(params));
    for (int dimension = 0; dimension < output_rank; dimension++) {
        int input_dimension = dimension - leading;
        int input_extent =
            input_dimension >= 0 ? input_shape[input_dimension] : 1;
        if (input_extent != 1 && input_extent != output_shape[dimension])
            return 0;
        params.output_strides[dimension] = output_strides[dimension];
        if (input_dimension >= 0 && input_extent != 1)
            params.input_strides[dimension] =
                input_strides[input_dimension];
    }
    params.rank = (uint32_t)output_rank;
    if (!checked_elements_bytes(input_elements, sizeof(uint32_t),
                                &input_elements, &input_bytes) ||
        !checked_elements_bytes(output_elements, sizeof(uint32_t),
                                &output_elements, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes))
        return 0;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &output_elements;
    args[3] = &params;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_expand_32),
                        output_elements, args))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_slice_32(const void* input, void* output,
                        const int* input_shape, int input_rank,
                        const int* output_shape, int output_rank,
                        const int* starts, const int* steps) {
    CudaTensorSlot *input_slot, *output_slot;
    VxCudaSliceParams params;
    uint32_t input_elements, output_elements;
    size_t input_bytes, output_bytes;
    void* args[4];
    if (!input || !output || !starts || !steps ||
        input_rank != output_rank ||
        !cuda_shape_strides(input_shape, input_rank, params.input_strides,
                            &input_elements) ||
        !cuda_shape_strides(output_shape, output_rank, params.output_strides,
                            &output_elements))
        return 0;
    memset(params.starts, 0, sizeof(params.starts));
    memset(params.steps, 0, sizeof(params.steps));
    for (int dimension = 0; dimension < input_rank; dimension++) {
        uint64_t last;
        if (starts[dimension] < 0 || steps[dimension] <= 0)
            return 0;
        params.starts[dimension] = (uint32_t)starts[dimension];
        params.steps[dimension] = (uint32_t)steps[dimension];
        last = params.starts[dimension] +
            (uint64_t)(uint32_t)(output_shape[dimension] - 1) *
            params.steps[dimension];
        if (last >= (uint32_t)input_shape[dimension]) return 0;
    }
    params.rank = (uint32_t)input_rank;
    if (!checked_elements_bytes(input_elements, sizeof(uint32_t),
                                &input_elements, &input_bytes) ||
        !checked_elements_bytes(output_elements, sizeof(uint32_t),
                                &output_elements, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes))
        return 0;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &output_elements;
    args[3] = &params;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_slice_32),
                        output_elements, args))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_split_32(const void* input, void* output,
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
        !checked_elements_bytes((uint64_t)input_elements, sizeof(uint32_t),
                                &input_count, &input_bytes) ||
        !checked_elements_bytes((uint64_t)output_elements, sizeof(uint32_t),
                                &output_count, &output_bytes) ||
        ranges_overlap(output, output_bytes, input, input_bytes))
        return 0;
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
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_split_32),
                        output_count, args))
        return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_concat_32(const void* const* inputs, const long* sizes,
                         const int* input_axes, int count, void* output,
                         int output_axis, int inner) {
    CudaTensorSlot* output_slot;
    uint32_t output_count;
    uint32_t axis_offset = 0u;
    size_t output_bytes;
    uint64_t total = 0u;
    uint64_t summed_axis = 0u;
    if (!inputs || !sizes || !input_axes || !output || count <= 0 ||
        output_axis <= 0 || inner <= 0)
        return 0;
    for (int index = 0; index < count; index++) {
        if (!inputs[index] || sizes[index] <= 0 || input_axes[index] <= 0)
            return 0;
        total += (uint64_t)sizes[index];
        summed_axis += (uint64_t)(uint32_t)input_axes[index];
    }
    if (summed_axis != (uint32_t)output_axis ||
        total % ((uint64_t)(uint32_t)output_axis * (uint32_t)inner) != 0u ||
        !checked_elements_bytes(total, sizeof(uint32_t),
                                &output_count, &output_bytes))
        return 0;
    output_slot = graph_output(output, output_bytes);
    if (!output_slot) return 0;
    for (int index = 0; index < count; index++) {
        CudaTensorSlot* input_slot;
        uint32_t input_count;
        uint32_t input_axis = (uint32_t)input_axes[index];
        uint32_t output_axis_u32 = (uint32_t)output_axis;
        uint32_t inner_u32 = (uint32_t)inner;
        size_t input_bytes;
        void* args[7];
        if (!checked_elements_bytes((uint64_t)sizes[index],
                                    sizeof(uint32_t),
                                    &input_count, &input_bytes) ||
            ranges_overlap(output, output_bytes, inputs[index], input_bytes))
            return 0;
        input_slot = graph_ensure_input(inputs[index], input_bytes, 0);
        if (!input_slot) return 0;
        args[0] = &input_slot->device;
        args[1] = &output_slot->device;
        args[2] = &input_count;
        args[3] = &axis_offset;
        args[4] = &input_axis;
        args[5] = &output_axis_u32;
        args[6] = &inner_u32;
        if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_concat_32),
                            input_count, args))
            return 0;
        axis_offset += input_axis;
    }
    graph_mark_device(output_slot);
    return 1;
}
