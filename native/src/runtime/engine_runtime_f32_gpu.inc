/* Backend-neutral bindings for the F32 graph kernels shared by Vulkan,
 * OpenGL, and Metal.  The operation helpers below only validate graph data;
 * they never select a concrete GPU.  Selection is made once by the VxBackend
 * registry and this table supplies that backend's existing kernel entry
 * points. */
typedef int (*VxBinaryF32Fn)(const float*, long, const float*, long, float*, long,
                             const uint32_t*, const uint32_t*, const uint32_t*, int, int);
typedef int (*VxConv1DF32Fn)(const float*, const float*, const float*, float*,
                             int, int, int, int, int, int, int, int, int);
typedef int (*VxEmbeddingF32Fn)(const int32_t*, const float*, float*, int, int, int);
typedef int (*VxSdpaF32Fn)(const float*, const int32_t*, long, float*,
                           int, int, int, int, int, float, int, int);
typedef int (*VxCrossSdpaF32Fn)(const float*, const float*, const float*,
                                const int32_t*, long, float*, int, int, int, int,
                                int, int, float, int, int);
typedef int (*VxCrossAttentionF32Fn)(const float*, const float*, const float*,
                                     const float*, const float*, float*, int, int,
                                     int, int, int, int, int, int);
typedef int (*VxDequantF32Fn)(const float*, const float*, const float*, float*, long, int);
typedef int (*VxQuantI8Fn)(const float*, signed char*, long, float, int, float, int);
typedef int (*VxSplitF32Fn)(const float*, float*, long, long, int, int, int, int);
typedef int (*VxProfileF32Fn)(const float*, float*, int, int, int, int);
typedef int (*VxNmsF32Fn)(const float*, const float*, float*, int, int, int, int, int,
                          float, float);
typedef int (*VxCompareI32Fn)(const int32_t*, long, const int32_t*, long,
                              int32_t*, long, const uint32_t*, const uint32_t*,
                              const uint32_t*, int, int);
typedef int (*VxNotI32Fn)(const int32_t*, int32_t*, long);
typedef int (*VxClipI32Fn)(const int32_t*, int32_t*, long, int32_t, int32_t);
typedef int (*VxWhere32Fn)(const int32_t*, const void*, const void*, void*, long);
typedef int (*VxExpand32Fn)(const void*, void*, const int*, int,
                            const int*, int);
typedef int (*VxBatchMatMulF32Fn)(
    const float*, const int*, int, const float*, const int*, int,
    float*, const int*, int);
typedef int (*VxCastTypedFn)(const void*, int, void*, int, long);
typedef int (*VxCopy32Fn)(const void*, void*, long);
typedef int (*VxConcat32Fn)(const void* const*, const long*, const int*, int,
                            void*, int, int);
typedef int (*VxArgMaxF32I32Fn)(const float*, int32_t*, uint32_t, uint32_t,
                                uint32_t);
typedef int (*VxLinearF32Fn)(const float*, const float*, const float*, float*,
                             int, int, int, int);
#if VOLVOXAI_ENABLE_TRAINING
typedef int (*VxSdpaTrainingF32Fn)(const float*, const int32_t*, long, float*,
                                   int, int, int, int, int, float, int, int,
                                   uint32_t, uint32_t, uint32_t, float);
typedef int (*VxCrossSdpaTrainingF32Fn)(const float*, const float*, const float*,
                                        const int32_t*, long, float*, int, int,
                                        int, int, int, int, float, int, int,
                                        uint32_t, uint32_t, uint32_t, float);
#endif

typedef int (*VxMoERouterF32Fn)(const float*, const float*, const float*,
                                float*, float*, int, int, int, int, float, int);
typedef int (*VxMoELinearF32Fn)(const float*, const float*, const float*,
                                const float*, const float*, float*, int, int,
                                int, int, int, const uint32_t*, uint32_t);

typedef struct {
    VxBinaryF32Fn binary;
    VxConv1DF32Fn conv1d;
    VxEmbeddingF32Fn embedding;
    VxSdpaF32Fn sdpa;
#if VOLVOXAI_ENABLE_TRAINING
    VxSdpaTrainingF32Fn sdpa_training;
#endif
    VxCrossSdpaF32Fn cross_sdpa;
#if VOLVOXAI_ENABLE_TRAINING
    VxCrossSdpaTrainingF32Fn cross_sdpa_training;
#endif
    VxCrossAttentionF32Fn cross_attention;
    VxDequantF32Fn dequantize_linear;
    VxQuantI8Fn quantize_linear;
    VxSplitF32Fn split;
    VxProfileF32Fn spatial_softargmax_y;
    VxProfileF32Fn profile_x;
    VxProfileF32Fn profile_y;
    VxProfileF32Fn mean_height;
    VxNmsF32Fn nms;
    VxCompareI32Fn compare_i32;
    VxNotI32Fn not_i32;
    VxClipI32Fn clip_i32;
    VxWhere32Fn where_32;
    VxExpand32Fn expand_32;
    VxBatchMatMulF32Fn batch_matmul_f32;
    VxCastTypedFn cast_typed;
    VxCopy32Fn copy_32;
    VxConcat32Fn concat_32;
    VxArgMaxF32I32Fn argmax_f32_i32;
    VxLinearF32Fn linear_f32;
    VxMoELinearF32Fn moe_linear;
    VxMoERouterF32Fn moe_router;
} VxF32Ops;

#if VOLVOXAI_ENABLE_VULKAN
static const VxF32Ops g_vulkan_f32_ops = {
    vk_graph_binary_f32, vk_graph_conv1d_f32, vk_graph_embedding_f32,
    vk_graph_sdpa_f32,
#if VOLVOXAI_ENABLE_TRAINING
    vk_graph_sdpa_training_f32,
#endif
    vk_graph_cross_sdpa_f32,
#if VOLVOXAI_ENABLE_TRAINING
    vk_graph_cross_sdpa_training_f32,
#endif
    vk_graph_cross_attention_f32, vk_graph_dequantize_linear_f32,
    vk_graph_quantize_linear_i8, vk_graph_split_f32,
    vk_graph_spatial_softargmax_y_f32, vk_graph_profile_x_f32,
    vk_graph_profile_y_f32, vk_graph_mean_height_f32, vk_graph_nms_f32,
    vk_graph_compare_i32, vk_graph_not_i32, vk_graph_clip_i32,
    vk_graph_where_32, vk_graph_expand_32, vk_graph_batch_matmul_f32,
    vk_graph_cast_typed, vk_graph_copy_32,
    vk_graph_concat_32, vk_graph_argmax_f32,
    vk_graph_linear_f32,
    vk_graph_moe_linear_f32,
    vk_graph_moe_router_f32
};
#endif

#if VOLVOXAI_ENABLE_OPENGL
static const VxF32Ops g_opengl_f32_ops = {
    opengl_graph_binary_f32, opengl_graph_conv1d_f32, opengl_graph_embedding_f32,
    opengl_graph_sdpa_f32,
#if VOLVOXAI_ENABLE_TRAINING
    opengl_graph_sdpa_training_f32,
#endif
    opengl_graph_cross_sdpa_f32,
#if VOLVOXAI_ENABLE_TRAINING
    opengl_graph_cross_sdpa_training_f32,
#endif
    opengl_graph_cross_attention_f32, opengl_graph_dequantize_linear_f32,
    opengl_graph_quantize_linear_i8, opengl_graph_split_f32,
    opengl_graph_spatial_softargmax_y_f32, opengl_graph_profile_x_f32,
    opengl_graph_profile_y_f32, opengl_graph_mean_height_f32, opengl_graph_nms_f32,
    opengl_graph_compare_i32, opengl_graph_not_i32, opengl_graph_clip_i32,
    opengl_graph_where_32, opengl_graph_expand_32,
    opengl_graph_batch_matmul_f32,
    opengl_graph_cast_typed, opengl_graph_copy_32,
    opengl_graph_concat_32, opengl_graph_argmax_f32,
    opengl_graph_linear_f32,
    opengl_graph_moe_linear_f32,
    opengl_graph_moe_router_f32
};
#endif

#if VOLVOXAI_ENABLE_METAL
static const VxF32Ops g_metal_f32_ops = {
    metal_graph_binary_f32, metal_graph_conv1d_f32, metal_graph_embedding_f32,
    metal_graph_sdpa_f32,
#if VOLVOXAI_ENABLE_TRAINING
    metal_graph_sdpa_training_f32,
#endif
    metal_graph_cross_sdpa_f32,
#if VOLVOXAI_ENABLE_TRAINING
    metal_graph_cross_sdpa_training_f32,
#endif
    metal_graph_cross_attention_f32, metal_graph_dequantize_linear_f32,
    metal_graph_quantize_linear_i8, metal_graph_split_f32,
    metal_graph_spatial_softargmax_y_f32, metal_graph_profile_x_f32,
    metal_graph_profile_y_f32, metal_graph_mean_height_f32, metal_graph_nms_f32,
    metal_graph_compare_i32, metal_graph_not_i32, metal_graph_clip_i32,
    metal_graph_where_32, metal_graph_expand_32,
    metal_graph_batch_matmul_f32,
    metal_graph_cast_typed, metal_graph_copy_32,
    metal_graph_concat_32, metal_graph_argmax_f32,
    metal_graph_linear_f32,
    metal_graph_moe_linear_f32,
    metal_graph_moe_router_f32
};
#endif

#if VOLVOXAI_ENABLE_CUDA
static const VxF32Ops g_cuda_f32_ops = {
    .binary = cuda_graph_binary_f32,
    .conv1d = cuda_graph_conv1d_f32,
    .embedding = cuda_graph_embedding_f32,
    .sdpa = cuda_graph_sdpa_f32,
#if VOLVOXAI_ENABLE_TRAINING
    .sdpa_training = cuda_graph_sdpa_training_f32,
#endif
    .cross_sdpa = cuda_graph_cross_sdpa_f32,
#if VOLVOXAI_ENABLE_TRAINING
    .cross_sdpa_training = cuda_graph_cross_sdpa_training_f32,
#endif
    .cross_attention = cuda_graph_cross_attention_f32,
    .dequantize_linear = cuda_graph_dequantize_linear_f32,
    .quantize_linear = cuda_graph_quantize_linear_i8,
    .split = cuda_graph_split_f32,
    .spatial_softargmax_y = cuda_graph_spatial_softargmax_y_f32,
    .profile_x = cuda_graph_profile_x_f32,
    .profile_y = cuda_graph_profile_y_f32,
    .mean_height = cuda_graph_mean_height_f32,
    .nms = cuda_graph_nms_f32,
    .compare_i32 = cuda_graph_compare_i32,
    .not_i32 = cuda_graph_not_i32,
    .clip_i32 = cuda_graph_clip_i32,
    .where_32 = cuda_graph_where_32,
    .expand_32 = cuda_graph_expand_32,
    .batch_matmul_f32 = cuda_graph_batch_matmul_f32,
    .cast_typed = cuda_graph_cast_typed,
    .copy_32 = cuda_graph_copy_32,
    .concat_32 = cuda_graph_concat_32,
    .argmax_f32_i32 = NULL,
    .linear_f32 = NULL,
};
#endif

static T* node_data_input(Node* n, T* input) {
    if (input) return input;
    T* x = nin(n, "data");
    if (!x) x = nin(n, "x");
    if (!x && n->nin > 0) x = t_find(n->ins[0].name);
    return x;
}

static int normalize_axis(int axis, int rank) {
    if (axis < 0) axis += rank;
    return axis;
}

static int tensor_inner_after_axis(const T* t, int axis) {
    int inner = 1;
    if (!t || axis < 0 || axis >= t->ndim) return 0;
    for (int i = axis + 1; i < t->ndim; i++) inner *= t->shape[i];
    return inner;
}

static int gpu_output_strides_u32(const T* output, uint32_t strides[8]) {
    uint64_t stride = 1u;
    if (!output || !strides || output->ndim <= 0 || output->ndim > 8 ||
        output->numel <= 0 || output->numel > UINT32_MAX)
        return 0;
    memset(strides, 0, 8u * sizeof(uint32_t));
    for (int dimension = output->ndim - 1; dimension >= 0; dimension--) {
        if (output->shape[dimension] <= 0 || stride > UINT32_MAX)
            return 0;
        strides[dimension] = (uint32_t)stride;
        stride *= (uint32_t)output->shape[dimension];
        if (stride > UINT32_MAX) return 0;
    }
    return stride == (uint64_t)output->numel;
}

static int try_gpu_graph_compare_i32(Node* node, T* output,
                                     const VxF32Ops* ops, int operation) {
    T* a = nin(node, "a");
    T* b = nin(node, "b");
    uint32_t output_strides[8] = {0};
    uint32_t a_strides[8] = {0};
    uint32_t b_strides[8] = {0};
    if (!ops || !ops->compare_i32 || !a || !b || !output ||
        a->dtype != T_I32 || b->dtype != T_I32 ||
        output->dtype != T_I32 || (operation != 0 && operation != 1) ||
        gpu_broadcast_strides(a, output, a_strides) != 0 ||
        gpu_broadcast_strides(b, output, b_strides) != 0 ||
        !gpu_output_strides_u32(output, output_strides))
        return 0;
    return ops->compare_i32(
        (const int32_t*)a->data, a->numel,
        (const int32_t*)b->data, b->numel,
        (int32_t*)output->data, output->numel,
        output_strides, a_strides, b_strides, output->ndim, operation);
}

static int try_gpu_graph_argmax_f32_i32(Node* node, T* output,
                                        const VxF32Ops* ops) {
    T* input = node_input_default(node);
    uint64_t outer = 1u;
    uint64_t inner = 1u;
    int axis;
    int keepdims;
    if (!ops || !ops->argmax_f32_i32 || !node || !input || !output ||
        input->dtype != T_F32 || output->dtype != T_I32 ||
        input->ndim <= 0 || input->ndim > 8 || input->data == output->data)
        return 0;
    axis = normalize_axis(p_int(node->params, "axis", 0), input->ndim);
    keepdims = p_int(node->params, "keepdims", 1);
    if (axis < 0 || axis >= input->ndim ||
        (keepdims != 0 && keepdims != 1) ||
        p_int(node->params, "select_last_index", 0) != 0 ||
        input->shape[axis] <= 0)
        return 0;
    for (int dimension = 0; dimension < axis; dimension++) {
        if (input->shape[dimension] <= 0 ||
            outer > UINT32_MAX / (uint32_t)input->shape[dimension])
            return 0;
        outer *= (uint32_t)input->shape[dimension];
    }
    for (int dimension = axis + 1; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0 ||
            inner > UINT32_MAX / (uint32_t)input->shape[dimension])
            return 0;
        inner *= (uint32_t)input->shape[dimension];
    }
    if (outer == 0 || inner == 0 || outer * inner != (uint64_t)output->numel)
        return 0;
    if (keepdims) {
        if (output->ndim != input->ndim) return 0;
        for (int dimension = 0; dimension < input->ndim; dimension++) {
            int expected = dimension == axis ? 1 : input->shape[dimension];
            if (output->shape[dimension] != expected) return 0;
        }
    } else {
        if (output->ndim != input->ndim - 1) return 0;
        for (int dimension = 0; dimension < input->ndim; dimension++) {
            int output_dimension;
            if (dimension == axis) continue;
            output_dimension = dimension < axis ? dimension : dimension - 1;
            if (output->shape[output_dimension] != input->shape[dimension])
                return 0;
        }
    }
    return ops->argmax_f32_i32(
        input->data, (int32_t*)output->data, (uint32_t)outer,
        (uint32_t)input->shape[axis], (uint32_t)inner);
}

static int try_gpu_graph_concat_i32(Node* node, T* output,
                                    const VxF32Ops* ops) {
    const void* inputs[MAXIN];
    long sizes[MAXIN];
    int input_axes[MAXIN];
    long total = 0;
    long summed_axis = 0;
    int axis;
    int inner;
    if (!ops || !ops->concat_32 || !node || !output ||
        output->dtype != T_I32 || output->ndim <= 0 ||
        output->ndim > 8 || node->nin <= 0 || node->nin > MAXIN)
        return 0;
    axis = normalize_axis(p_int(node->params, "axis", 1), output->ndim);
    if (axis < 0 || axis >= output->ndim || output->shape[axis] <= 0)
        return 0;
    for (int input_index = 0; input_index < node->nin; input_index++) {
        T* input = t_find(node->ins[input_index].name);
        if (!input || input->dtype != T_I32 ||
            input->ndim != output->ndim || input->shape[axis] <= 0)
            return 0;
        for (int dimension = 0; dimension < output->ndim; dimension++) {
            if (dimension != axis &&
                input->shape[dimension] != output->shape[dimension])
                return 0;
        }
        inputs[input_index] = input->data;
        sizes[input_index] = input->numel;
        input_axes[input_index] = input->shape[axis];
        total += input->numel;
        summed_axis += input->shape[axis];
    }
    inner = tensor_inner_after_axis(output, axis);
    if (total != output->numel ||
        summed_axis != output->shape[axis] || inner <= 0)
        return 0;
    return ops->concat_32(
        inputs, sizes, input_axes, node->nin, output->data,
        output->shape[axis], inner);
}

/* Shared typed-control route for Vulkan, OpenGL, and Metal. These operators
 * preserve ordinary F32/I32 graph storage; none is a quantized byte-domain
 * shortcut. The backend entry points use the same validation contract as the
 * CUDA typed graph path so strict all-device compilation cannot silently
 * accept a narrower implementation on one native GPU. */
static int try_gpu_graph_typed_control(Node* node, T* input, T* output,
                                       const VxF32Ops* ops) {
    const char* op;
    if (!node || !output || !ops) return 0;
    op = node->op;
    if (!strcmp(op, "Equal") || !strcmp(op, "GreaterOrEqual"))
        return try_gpu_graph_compare_i32(
            node, output, ops, !strcmp(op, "GreaterOrEqual") ? 1 : 0);
    if (!strcmp(op, "Not")) {
        T* source = node_input_default(node);
        return ops->not_i32 && source && source->dtype == T_I32 &&
            output->dtype == T_I32 && tensor_same_shape(source, output) &&
            ops->not_i32((const int32_t*)source->data,
                         (int32_t*)output->data, output->numel);
    }
    if (!strcmp(op, "Clip") && output->dtype == T_I32) {
        T* source = node_input_default(node);
        int32_t minimum;
        int32_t maximum;
        return ops->clip_i32 && source && source->dtype == T_I32 &&
            tensor_same_shape(source, output) &&
            node_i32_parameter(node->params, "min", INT32_MIN, &minimum) == 0 &&
            node_i32_parameter(node->params, "max", INT32_MAX, &maximum) == 0 &&
            minimum <= maximum &&
            ops->clip_i32((const int32_t*)source->data,
                          (int32_t*)output->data, output->numel,
                          minimum, maximum);
    }
    if (!strcmp(op, "Where") || !strcmp(op, "Mask")) {
        T* condition = nin(node, "condition");
        T* a = nin(node, "x");
        T* b = nin(node, "y");
        if (!condition) condition = nin(node, "cond");
        if (!condition) condition = nin(node, "mask");
        if (!a) a = nin(node, "a");
        if (!b) b = nin(node, "b");
        return ops->where_32 && condition && a && b &&
            condition->dtype == T_I32 && tensor_plain_32(a) &&
            tensor_plain_32(b) && tensor_plain_32(output) &&
            a->dtype == b->dtype && output->dtype == a->dtype &&
            tensor_same_shape(condition, output) &&
            tensor_same_shape(a, output) && tensor_same_shape(b, output) &&
            ops->where_32((const int32_t*)condition->data, a->data, b->data,
                          output->data, output->numel);
    }
    if (!strcmp(op, "Expand")) {
        T* source = input ? input : node_input_default(node);
        return ops->expand_32 && source && source->dtype == T_I32 &&
            output->dtype == T_I32 && tensor_plain_32(source) &&
            tensor_plain_32(output) &&
            ops->expand_32(
                source->data, output->data, source->shape, source->ndim,
                output->shape, output->ndim);
    }
    if (!strcmp(op, "Cast")) {
        T* source = input ? input : node_input_default(node);
        cJSON* target = node->params
            ? cJSON_GetObjectItemCaseSensitive(node->params, "to") : NULL;
        int target_dtype;
        if (!ops->cast_typed || !source || source->numel != output->numel ||
            !tensor_plain_32(source) || !tensor_plain_32(output) || !target ||
            (target_dtype = graph_declared_dtype(target)) < 0 ||
            target_dtype != output->dtype)
            return 0;
        return ops->cast_typed(
            source->data, source->dtype, output->data, output->dtype,
            output->numel);
    }
    if ((!strcmp(op, "Reshape") || !strcmp(op, "Flatten") ||
         !strcmp(op, "Squeeze") || !strcmp(op, "Unsqueeze") ||
         !strcmp(op, "Identity")) && output->dtype == T_I32) {
        T* source = input ? input : node_input_default(node);
        return ops->copy_32 && source && source->dtype == T_I32 &&
            source->numel == output->numel &&
            ops->copy_32(source->data, output->data, output->numel);
    }
    if (!strcmp(op, "Concat") && output->dtype == T_I32)
        return try_gpu_graph_concat_i32(node, output, ops);
    if (!strcmp(op, "ArgMax"))
        return try_gpu_graph_argmax_f32_i32(node, output, ops);
    return 0;
}

static int try_gpu_graph_batch_matmul_f32(
        Node* node, T* output, const VxF32Ops* ops) {
    T* a;
    T* b;
    if (!node || !output || !ops || !ops->batch_matmul_f32 ||
        strcmp(node->op, "BatchMatMul"))
        return 0;
    a = nin(node, "a");
    b = nin(node, "b");
    if (!a || !b || a->dtype != T_F32 || b->dtype != T_F32 ||
        output->dtype != T_F32)
        return 0;
    materialize_tensor_f32(a);
    materialize_tensor_f32(b);
    return ops->batch_matmul_f32(
        a->data, a->shape, a->ndim,
        b->data, b->shape, b->ndim,
        output->data, output->shape, output->ndim);
}

static int try_gpu_graph_linear_f32(
        Node* node, T* input, T* output, const VxF32Ops* ops) {
    T* weight;
    T* bias;
    long rows;
    int d_in;
    int d_out;
    int output_major_weight = 0;
    if (!node || !input || !output || !ops || !ops->linear_f32 ||
        (strcmp(node->op, "MatMul") && strcmp(node->op, "Gemm") &&
         strcmp(node->op, "Linear")) || input->dtype != T_F32 ||
        output->dtype != T_F32 || input->ndim <= 0 || output->ndim <= 0)
        return 0;
    weight = nin(node, "weight");
    bias = nin(node, "bias");
    if (!weight || weight->dtype != T_F32 || weight->ndim != 2 ||
        (bias && bias->dtype != T_F32))
        return 0;
    d_in = input->shape[input->ndim - 1];
    d_out = output->shape[output->ndim - 1];
    if (d_in <= 0 || d_out <= 0 || input->numel <= 0 ||
        output->numel <= 0 || input->numel % d_in ||
        output->numel % d_out)
        return 0;
    rows = input->numel / d_in;
    if (rows <= 0 || rows > INT_MAX || output->numel / d_out != rows ||
        d_in > LONG_MAX / d_out ||
        weight->numel != (long)d_in * d_out ||
        (bias && bias->numel != d_out) ||
        linear_node_weight_layout(
            node, input, output, weight, &output_major_weight) != 0)
        return 0;
    return ops->linear_f32(
        input->data, weight->data, bias ? bias->data : NULL, output->data,
        (int)rows, d_in, d_out, output_major_weight);
}

static int gather_i32_dimensions(const T* data, const T* indices,
                                 const T* output, int axis,
                                 int* outer, int* axis_size, int* inner,
                                 int* indices_elements, int* output_elements) {
    uint64_t outer_count = 1;
    uint64_t inner_count = 1;
    uint64_t expected_input;
    uint64_t expected_output;
    if (!data || !indices || !output || !outer || !axis_size || !inner ||
        !indices_elements || !output_elements || data->dtype != T_F32 ||
        indices->dtype != T_I32 || output->dtype != T_F32 ||
        data->ndim <= 0 || data->ndim > 8 || axis < 0 || axis >= data->ndim ||
        data->shape[axis] <= 0 || indices->numel <= 0 || output->numel <= 0)
        return 0;
    for (int dimension = 0; dimension < axis; dimension++) {
        if (data->shape[dimension] <= 0 ||
            outer_count > INT_MAX / (uint32_t)data->shape[dimension]) return 0;
        outer_count *= (uint32_t)data->shape[dimension];
    }
    for (int dimension = axis + 1; dimension < data->ndim; dimension++) {
        if (data->shape[dimension] <= 0 ||
            inner_count > INT_MAX / (uint32_t)data->shape[dimension]) return 0;
        inner_count *= (uint32_t)data->shape[dimension];
    }
    if (indices->numel > INT_MAX || output->numel > INT_MAX) return 0;
    expected_input = outer_count * (uint32_t)data->shape[axis] * inner_count;
    expected_output = outer_count * (uint32_t)indices->numel * inner_count;
    if (expected_input != (uint64_t)data->numel ||
        expected_output != (uint64_t)output->numel) return 0;
    *outer = (int)outer_count;
    *axis_size = data->shape[axis];
    *inner = (int)inner_count;
    *indices_elements = (int)indices->numel;
    *output_elements = (int)output->numel;
    return 1;
}

static int try_gpu_graph_binary(Node* n, T* in, T* o, const VxF32Ops* ops) {
    T* a = nin(n, "a");
    T* b = nin(n, "b");
    if (!a) a = node_data_input(n, in);
    if (!a || !b || !o) return 0;
    materialize_tensor_f32(a);
    materialize_tensor_f32(b);
    if (a->dtype != T_F32 || b->dtype != T_F32 || o->dtype != T_F32 ||
        o->ndim <= 0 || o->ndim > 8 || o->numel <= 0 || o->numel > UINT32_MAX)
        return 0;
    uint32_t output_strides[8] = {0}, a_strides[8], b_strides[8];
    if (gpu_broadcast_strides(a, o, a_strides) != 0 ||
        gpu_broadcast_strides(b, o, b_strides) != 0) return 0;
    long stride = 1;
    for (int dimension = o->ndim - 1; dimension >= 0; dimension--) {
        if (o->shape[dimension] <= 0 || stride > UINT32_MAX) return 0;
        output_strides[dimension] = (uint32_t)stride;
        stride *= o->shape[dimension];
    }
    int op = !strcmp(n->op, "Sub") ? 1 : (!strcmp(n->op, "Div") ? 2 :
             (!strcmp(n->op, "Add") ? 3 : 0));
    return ops && ops->binary && ops->binary(a->data, a->numel, b->data, b->numel, o->data,
                                              o->numel, output_strides, a_strides, b_strides,
                                              o->ndim, op);
}

static int try_gpu_graph_conv1d(Node* n, T* in, T* o, const VxF32Ops* ops) {
    T* x = node_data_input(n, in);
    T* w = nin(n, "weight");
    T* b = nin(n, "bias");
    if (!x || !w || !o || w->ndim != 3) return 0;
    if ((x->ndim != 2 && x->ndim != 3) || o->ndim != x->ndim) return 0;
    materialize_tensor_f32(x);
    materialize_tensor_f32(w);
    if (b) materialize_tensor_f32(b);
    if (x->dtype != T_F32 || w->dtype != T_F32 || (b && b->dtype != T_F32)) return 0;
    /* NLC activations [batch, l, c]; WIO weights [k, in_c, out_c]. */
    int in_l = x->ndim == 3 ? x->shape[1] : x->shape[0];
    int in_c = x->ndim == 3 ? x->shape[2] : x->shape[1];
    int out_l = o->ndim == 3 ? o->shape[1] : o->shape[0];
    int out_c = o->ndim == 3 ? o->shape[2] : o->shape[1];
    int batch = x->ndim == 3 ? x->shape[0] : 1;
    int output_batch = o->ndim == 3 ? o->shape[0] : 1;
    int kernel = w->shape[0];
    if (batch <= 0 || batch != output_batch || in_c <= 0 || in_l <= 0 || out_c <= 0 || out_l <= 0 ||
        x->numel != (long)batch * in_c * in_l || o->numel != (long)batch * out_c * out_l ||
        w->shape[1] != in_c || w->shape[2] != out_c || (b && b->numel < out_c)) return 0;
    int sy, sx, py, px;
    p_pair(n->params, "stride", 1, &sy, &sx);
    p_pair(n->params, "padding", 0, &py, &px);
    int relu = p_int(n->params, "relu", 0) ? 1 : 0;
    long padded_l = (long)in_l + 2L * py;
    if (sy <= 0 || py < 0 || kernel <= 0 || padded_l < kernel ||
        (padded_l - kernel) / sy + 1 != out_l) return 0;
    return ops && ops->conv1d && ops->conv1d(x->data, w->data, b ? b->data : NULL, o->data,
                                              batch, in_c, in_l, out_c, out_l, kernel, sy, py, relu);
}

static int try_gpu_graph_embedding(Node* n, T* in, T* o, const VxF32Ops* ops) {
    T* tokens = node_data_input(n, in);
    T* weight = nin(n, "weight");
    if (!tokens || !weight || !o || tokens->dtype != T_I32 || weight->dtype != T_F32 ||
        o->dtype != T_F32 || weight->ndim <= 0 || tokens->numel <= 0 ||
        tokens->numel > INT32_MAX) return 0;
    materialize_tensor_f32(weight);
    int d_model = weight->shape[weight->ndim - 1];
    int vocab = d_model > 0 && weight->numel % d_model == 0
        ? (int)(weight->numel / d_model) : 0;
    if (d_model <= 0 || vocab <= 0 || o->numel != tokens->numel * (long)d_model) return 0;
    const int32_t* ids = (const int32_t*)tokens->data;
    if (!g_bounded_gpu_value_domain_proven)
        for (long token = 0; token < tokens->numel; token++) {
            int32_t id;
            memcpy(&id,
                   (const unsigned char*)tokens->data +
                       (size_t)token * sizeof(id),
                   sizeof(id));
            if (id < 0 || id >= vocab) return 0;
        }
    return ops && ops->embedding && ops->embedding(ids, weight->data, o->data,
                                                    (int)tokens->numel, d_model, vocab);
}

/* Shared MoERouter validation for the device routes. The router weight is
 * [d_model, experts], so no slot table is involved. */
static int try_gpu_graph_moe_router(Node* n, T* o, const VxF32Ops* ops) {
    T* in = nin(n, "input");
    T* weight = nin(n, "weight");
    T* bias = nin(n, "bias");
    T* indices = nout(n, "indices");
    T* routes = nout(n, "weights");
    if (!in) in = nin(n, "x");
    if (!weight) weight = nin(n, "router_weight");
    if (!indices) indices = nout(n, "route_indices");
    if (!routes) routes = nout(n, "route_weights");
    (void)o;
    if (!in || !weight || !indices || !routes) return 0;
    if (in->dtype != T_F32 || weight->dtype != T_F32 ||
        indices->dtype != T_F32 || routes->dtype != T_F32 ||
        (bias && bias->dtype != T_F32)) return 0;
    if (in->ndim <= 0 || weight->ndim != 2 || indices->ndim <= 0) return 0;
    materialize_tensor_f32(in);
    materialize_tensor_f32(weight);
    if (bias) materialize_tensor_f32(bias);
    int d_model = in->shape[in->ndim - 1];
    int experts = p_int(n->params, "num_experts", weight->shape[1]);
    if (d_model <= 0 || experts <= 0 || weight->shape[0] != d_model ||
        weight->shape[1] != experts || (bias && bias->numel != experts)) return 0;
    long rows = in->numel / d_model;
    if (rows <= 0 || rows > INT_MAX || indices->numel != routes->numel ||
        indices->numel % rows) return 0;
    long top_k = indices->numel / rows;
    if (top_k <= 0 || top_k > experts || top_k > 8 ||
        top_k != p_int(n->params, "top_k", (int)top_k)) return 0;
    float temperature = p_flt(n->params, "temperature", 1.0f);
    int normalize = p_int(n->params, "normalize", 1) ? 1 : 0;
    if (!(temperature > 0.0f)) return 0;
    return ops && ops->moe_router && ops->moe_router(
        in->data, weight->data, bias ? bias->data : NULL, indices->data,
        routes->data, (int)rows, d_model, experts, (int)top_k, temperature,
        normalize);
}

/* Shared MoELinear validation for the device routes. Mirrors the portable
 * kernel's contract, including the resident-slot table a partially resident
 * bank carries on the node. */
static int try_gpu_graph_moe_linear(Node* n, T* o, const VxF32Ops* ops) {
    T* in = nin(n, "input");
    T* ew = nin(n, "expert_weight");
    T* eb = nin(n, "expert_bias");
    T* indices = nin(n, "route_indices");
    T* weights = nin(n, "route_weights");
    if (!in) in = nin(n, "x");
    if (!ew) ew = nin(n, "weight");
    if (!eb) eb = nin(n, "bias");
    if (!indices) indices = nin(n, "indices");
    if (!weights) weights = nin(n, "weights");
    if (!in || !ew || !indices || !weights || !o) return 0;
    if (in->dtype != T_F32 || ew->dtype != T_F32 || indices->dtype != T_F32 ||
        weights->dtype != T_F32 || o->dtype != T_F32 ||
        (eb && eb->dtype != T_F32)) return 0;
    if (in->ndim <= 0 || ew->ndim != 3 || o->ndim <= 0) return 0;
    materialize_tensor_f32(in);
    materialize_tensor_f32(ew);
    materialize_tensor_f32(indices);
    materialize_tensor_f32(weights);
    if (eb) materialize_tensor_f32(eb);
    int experts = ew->shape[0];
    int d_in = ew->shape[1];
    int d_out = ew->shape[2];
    if (experts <= 0 || d_in <= 0 || d_out <= 0 ||
        in->shape[in->ndim - 1] != d_in || o->shape[o->ndim - 1] != d_out ||
        (eb && eb->numel != (long)experts * d_out)) return 0;
    long tokens = in->numel / d_in;
    if (tokens <= 0 || tokens > INT_MAX || o->numel != tokens * d_out ||
        indices->numel != weights->numel || indices->numel % tokens) return 0;
    long top_k = indices->numel / tokens;
    if (top_k <= 0 || top_k > INT_MAX) return 0;
    return ops && ops->moe_linear && ops->moe_linear(
        in->data, ew->data, eb ? eb->data : NULL, indices->data, weights->data,
        o->data, (int)tokens, d_in, d_out, experts, (int)top_k,
        n->resident_slot_rows, n->resident_slot_domain);
}

static int try_gpu_graph_sdpa(Node* n, T* in, T* o, const VxF32Ops* ops) {
    T* qkv = nin(n, "qkv");
    if (!qkv) qkv = node_data_input(n, in);
    T* mask = nin(n, "mask");
    if (!qkv || !o || (qkv->ndim != 2 && qkv->ndim != 3) || o->ndim != qkv->ndim) return 0;
    materialize_tensor_f32(qkv);
    if (qkv->dtype != T_F32 || qkv->shape[qkv->ndim - 1] != o->shape[o->ndim - 1] * 3) return 0;
    int d_model = o->shape[o->ndim - 1];
    int batch = qkv->ndim == 3 ? qkv->shape[0] : 1;
    int output_batch = o->ndim == 3 ? o->shape[0] : 1;
    int seq_len = qkv->shape[qkv->ndim - 2];
    int heads = p_int(n->params, "heads", 8);
    int mask_mode = attention_mask_mode(mask, batch, seq_len, seq_len);
    if (heads <= 0 || d_model % heads) return 0;
    int head_dim = d_model / heads;
    if (batch <= 0 || batch != output_batch || o->shape[o->ndim - 2] != seq_len ||
        qkv->numel != (long)batch * seq_len * 3 * d_model ||
        o->numel != (long)batch * seq_len * d_model || head_dim > 64 || mask_mode < 0) return 0;
    float scale = p_flt(n->params, "scale", 1.0f / sqrtf((float)head_dim));
    int causal = p_int(n->params, "causal", 1) ? 1 : 0;
#if VOLVOXAI_ENABLE_TRAINING
    NativeAttentionDropoutParams dropout;
    if (native_attention_dropout_params(n, (int)(n - g_n), &dropout) != 0) return 0;
    if (g_native_training_mode && dropout.enabled) {
        return ops && ops->sdpa_training && ops->sdpa_training(qkv->data,
            mask ? (const int32_t*)mask->data : NULL, mask ? mask->numel : 0,
            o->data, seq_len, d_model, heads, head_dim, batch, scale, causal,
            mask_mode, dropout.threshold, dropout.seed, dropout.counter, dropout.scale);
    }
#endif
    return ops && ops->sdpa && ops->sdpa(qkv->data, mask ? (const int32_t*)mask->data : NULL,
                                         mask ? mask->numel : 0, o->data, seq_len, d_model,
                                         heads, head_dim, batch, scale, causal, mask_mode);
}

static int try_gpu_graph_cross_sdpa(Node* n, T* o, const VxF32Ops* ops) {
    T* q = nin(n, "q");
    T* k = nin(n, "k");
    T* v = nin(n, "v");
    T* mask = nin(n, "mask");
    if (!q || !k || !v || !o || (q->ndim != 2 && q->ndim != 3) ||
        k->ndim != q->ndim || v->ndim != q->ndim || o->ndim != q->ndim) return 0;
    materialize_tensor_f32(q);
    materialize_tensor_f32(k);
    materialize_tensor_f32(v);
    int d_model = o->shape[o->ndim - 1];
    if (q->dtype != T_F32 || k->dtype != T_F32 || v->dtype != T_F32 ||
        q->shape[q->ndim - 1] != d_model || k->shape[k->ndim - 1] != d_model ||
        v->shape[v->ndim - 1] != d_model) return 0;
    int batch = q->ndim == 3 ? q->shape[0] : 1;
    int k_batch = k->ndim == 3 ? k->shape[0] : 1;
    int v_batch = v->ndim == 3 ? v->shape[0] : 1;
    int output_batch = o->ndim == 3 ? o->shape[0] : 1;
    int seq_q = q->shape[q->ndim - 2];
    int seq_kv = k->shape[k->ndim - 2];
    int mask_mode = attention_mask_mode(mask, batch, seq_q, seq_kv);
    if (batch <= 0 || batch != k_batch || batch != v_batch || batch != output_batch ||
        seq_q <= 0 || seq_kv <= 0 || v->shape[v->ndim - 2] != seq_kv ||
        o->shape[o->ndim - 2] != seq_q || v->numel != (long)batch * seq_kv * d_model ||
        k->numel != v->numel || q->numel != (long)batch * seq_q * d_model ||
        o->numel != q->numel || mask_mode < 0) return 0;
    int heads = p_int(n->params, "heads", 8);
    if (heads <= 0 || d_model % heads) return 0;
    int head_dim = d_model / heads;
    if (head_dim > 64) return 0;
    float scale = p_flt(n->params, "scale", 1.0f / sqrtf((float)head_dim));
    int causal = p_int(n->params, "causal", 0) ? 1 : 0;
#if VOLVOXAI_ENABLE_TRAINING
    NativeAttentionDropoutParams dropout;
    if (native_attention_dropout_params(n, (int)(n - g_n), &dropout) != 0) return 0;
    if (g_native_training_mode && dropout.enabled) {
        return ops && ops->cross_sdpa_training && ops->cross_sdpa_training(q->data, k->data, v->data,
            mask ? (const int32_t*)mask->data : NULL, mask ? mask->numel : 0,
            o->data, seq_q, seq_kv, d_model, heads, head_dim, batch, scale, causal,
            mask_mode, dropout.threshold, dropout.seed, dropout.counter, dropout.scale);
    }
#endif
    return ops && ops->cross_sdpa && ops->cross_sdpa(q->data, k->data, v->data,
                                                     mask ? (const int32_t*)mask->data : NULL,
                                                     mask ? mask->numel : 0, o->data, seq_q, seq_kv,
                                                     d_model, heads, head_dim, batch, scale, causal, mask_mode);
}

static int try_gpu_graph_cross_attention(Node* n, T* o, const VxF32Ops* ops) {
    T* q = nin(n, "q");
    T* kv = nin(n, "kv");
    T* w = nin(n, "weight");
    T* scale = nin(n, "scale");
    T* bias = nin(n, "bias");
    if (!q || !kv || !w || !o || (q->ndim != 2 && q->ndim != 3) ||
        kv->ndim != q->ndim || o->ndim != q->ndim) return 0;
    materialize_tensor_f32(q);
    materialize_tensor_f32(kv);
    materialize_tensor_f32(w);
    if (scale) materialize_tensor_f32(scale);
    if (bias) materialize_tensor_f32(bias);
    int d_model = o->shape[o->ndim - 1];
    if (q->dtype != T_F32 || kv->dtype != T_F32 || w->dtype != T_F32 ||
        (scale && scale->dtype != T_F32) || (bias && bias->dtype != T_F32) ||
        q->shape[q->ndim - 1] != d_model || kv->shape[kv->ndim - 1] != d_model ||
        d_model > 64 || w->numel < (long)3 * d_model * d_model) return 0;
    if (scale && scale->numel < 3 * d_model) return 0;
    if (bias && bias->numel < 3 * d_model) return 0;
    int batch = q->ndim == 3 ? q->shape[0] : 1;
    int kv_batch = kv->ndim == 3 ? kv->shape[0] : 1;
    int output_batch = o->ndim == 3 ? o->shape[0] : 1;
    int seq_q = q->shape[q->ndim - 2];
    int seq_kv = kv->shape[kv->ndim - 2];
    int heads = p_int(n->params, "heads", 8);
    if (batch <= 0 || batch != kv_batch || batch != output_batch || seq_q <= 0 || seq_kv <= 0 ||
        o->shape[o->ndim - 2] != seq_q || q->numel != (long)batch * seq_q * d_model ||
        kv->numel != (long)batch * seq_kv * d_model || o->numel != q->numel ||
        heads <= 0 || d_model % heads) return 0;
    int head_dim = d_model / heads;
    if (head_dim > 64) return 0;
    return ops && ops->cross_attention && ops->cross_attention(
        q->data, kv->data, w->data, scale ? scale->data : NULL,
        bias ? bias->data : NULL, o->data, seq_q, seq_kv,
        d_model, heads, head_dim, batch, scale != NULL, bias != NULL);
}

static int try_gpu_graph_split(Node* n, T* in, const VxF32Ops* ops) {
    T* x = node_data_input(n, in);
    if (!x || n->nout <= 0) return 0;
    int axis = normalize_axis(p_int(n->params, "axis", 0), x->ndim);
    if (axis < 0 || axis >= x->ndim) return 0;
    materialize_tensor_f32(x);
    if (x->dtype != T_F32) return 0;
    int inner = tensor_inner_after_axis(x, axis);
    /* Graph parsing retains declaration order in n->outs.  That order defines
     * Split slices; lexical key sorting would place out10 before out2. */
    int offset = 0;
    for (int oi = 0; oi < n->nout; oi++) {
        T* dst = t_find(n->outs[oi].name);
        if (!dst || dst->dtype != T_F32 || dst->ndim != x->ndim) return 0;
        for (int d = 0; d < x->ndim; d++) {
            if (d != axis && dst->shape[d] != x->shape[d]) return 0;
        }
        offset += dst->shape[axis];
    }
    if (offset != x->shape[axis]) return 0;
    offset = 0;
    for (int oi = 0; oi < n->nout; oi++) {
        T* dst = t_find(n->outs[oi].name);
        int split_size = dst->shape[axis];
        int ok = ops && ops->split && ops->split(
            x->data, dst->data, x->numel, dst->numel, inner, split_size, x->shape[axis], offset);
        if (!ok) return 0;
        offset += split_size;
    }
    return 1;
}

static int cpu_split(Node* n, T* in) {
    T* x = node_data_input(n, in);
    if (!tensor_plain_32(x) || x->ndim <= 0 || n->nout <= 0) return -1;
    int axis = normalize_axis(p_int(n->params, "axis", 0), x->ndim);
    if (axis < 0 || axis >= x->ndim || x->shape[axis] <= 0) return -1;
    materialize_tensor_f32(x);
    int inner = tensor_inner_after_axis(x, axis);
    long outer = inner > 0 ? x->numel / ((long)x->shape[axis] * inner) : 0;
    if (inner <= 0 || outer <= 0 || x->numel != outer * x->shape[axis] * inner) return -1;
    /* Match the declaration order used by every device Split path. */
    int axis_offset = 0;
    for (int oi = 0; oi < n->nout; oi++) {
        T* dst = t_find(n->outs[oi].name);
        if (!tensor_plain_32(dst) || dst->dtype != x->dtype ||
            dst->ndim != x->ndim || dst->shape[axis] <= 0) return -1;
        for (int d = 0; d < x->ndim; d++) {
            if (d != axis && dst->shape[d] != x->shape[d]) return -1;
        }
        if (axis_offset > x->shape[axis] - dst->shape[axis]) return -1;
        long block = (long)dst->shape[axis] * inner;
        if (dst->numel != outer * block) return -1;
        if (outer > (long)UINT32_MAX) return -1;
        int ok = x->dtype == T_F32
            ? split_slice_f32(
                x->data, dst->data, (uint32_t)outer,
                (uint32_t)x->shape[axis],
                (uint32_t)dst->shape[axis], (uint32_t)inner,
                (uint32_t)axis_offset)
            : split_slice_u32(
                (const uint32_t*)x->data, (uint32_t*)dst->data,
                (uint32_t)outer, (uint32_t)x->shape[axis],
                (uint32_t)dst->shape[axis], (uint32_t)inner,
                (uint32_t)axis_offset);
        if (!ok) return -1;
        vk_mark_host_tensor(dst);
        axis_offset += dst->shape[axis];
    }
    return axis_offset == x->shape[axis] ? 0 : -1;
}

static int try_gpu_graph_profile_op(Node* n, T* in, T* o, const VxF32Ops* ops) {
    T* x = node_data_input(n, in);
    if (!x || !o || x->ndim != 4 || x->shape[0] <= 0) return 0;
    materialize_tensor_f32(x);
    if (x->dtype != T_F32) return 0;
    int batch = x->shape[0], h = x->shape[1], w = x->shape[2], c = x->shape[3];
    int ok = 0;
    if (!strcmp(n->op, "SpatialSoftargmaxY")) {
        if (o->numel != (long)batch * c * w) return 0;
        if (ops && ops->spatial_softargmax_y)
            ok = ops->spatial_softargmax_y(x->data, o->data, batch, h, w, c);
    } else if (!strcmp(n->op, "ProfileX")) {
        if (o->numel != (long)batch * 2 * c * w) return 0;
        if (ops && ops->profile_x) ok = ops->profile_x(x->data, o->data, batch, h, w, c);
    } else if (!strcmp(n->op, "ProfileY")) {
        if (o->numel != (long)batch * 2 * c * h) return 0;
        if (ops && ops->profile_y) ok = ops->profile_y(x->data, o->data, batch, h, w, c);
    } else if (!strcmp(n->op, "MeanHeight")) {
        if (o->numel != (long)batch * c * w) return 0;
        if (ops && ops->mean_height) ok = ops->mean_height(x->data, o->data, batch, h, w, c);
    }
    return ok;
}

static int try_gpu_graph_nms(Node* n, T* o, const VxF32Ops* ops) {
    T* boxes = nin(n, "boxes");
    T* scores = nin(n, "scores");
    if (!boxes || !scores || !o || boxes->ndim != 3 || scores->ndim != 3 || boxes->shape[2] != 4 ||
        scores->shape[0] != boxes->shape[0] || scores->shape[2] != boxes->shape[1] || o->numel % 3) return 0;
    materialize_tensor_f32(boxes);
    materialize_tensor_f32(scores);
    if (boxes->dtype != T_F32 || scores->dtype != T_F32) return 0;
    T* max_t = nin(n, "max_output_boxes_per_class");
    T* iou_t = nin(n, "iou_threshold");
    T* score_t = nin(n, "score_threshold");
    int max_output = p_int(n->params, "max_output_boxes_per_class", 0);
    float iou = p_flt(n->params, "iou_threshold", 0.5f);
    float score = p_flt(n->params, "score_threshold", 0.0f);
    if ((max_t && vk_sync_host_tensor(max_t) != 0) ||
        (iou_t && vk_sync_host_tensor(iou_t) != 0) ||
        (score_t && vk_sync_host_tensor(score_t) != 0)) return 0;
    if (max_t && max_t->numel > 0) max_output = (int)tensor_scalar_any(max_t, 0);
    if (iou_t && iou_t->numel > 0) iou = tensor_scalar_any(iou_t, 0);
    if (score_t && score_t->numel > 0) score = tensor_scalar_any(score_t, 0);
    int output_rows = (int)(o->numel / 3);
    if (max_output <= 0) max_output = output_rows;
    return ops && ops->nms && ops->nms(boxes->data, scores->data, o->data,
                                       boxes->shape[0], boxes->shape[1], scores->shape[1],
                                       max_output, output_rows, iou, score);
}

/* Vulkan/OpenGL/Metal domain slots are keyed by complete host-span bases.
 * Incremental row and prefix execution deliberately crosses the established
 * hybrid host boundary instead of presenting an interior pointer as a new
 * graph allocation. In particular, Linear is full-span here; only CUDA has a
 * proved resident subspan route. */
#if VOLVOXAI_ENABLE_VULKAN
static int try_vulkan_graph_node(Node* n, int idx, const char** node_backend) {
    if (!n || g_active_row >= 0 || g_prefix_rows > 0) return 0;
    T* o = t_find(n->out);
    T* in = nin(n, "input");
    cJSON* p = n->params;
    const char* op = n->op;
    if (!o) return 0;
    if (try_gpu_graph_typed_control(n, in, o, &g_vulkan_f32_ops)) {
        *node_backend = "vulkan-graph";
        return 1;
    }
    if (try_gpu_graph_batch_matmul_f32(n, o, &g_vulkan_f32_ops)) {
        *node_backend = "vulkan-graph";
        return 1;
    }
    if (try_gpu_graph_linear_f32(n, in, o, &g_vulkan_f32_ops)) {
        *node_backend = "vulkan-graph";
        return 1;
    }
    if (o->dtype != T_F32) return 0;

    if (!strcmp(op, "QConv2D")) {
        return 0;
    }

    if (!strcmp(op, "Conv2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || in->dtype != T_F32 || (w->dtype != T_F32 && w->dtype != T_F16) ||
            o->ndim != 4 || in->ndim != 4 || w->ndim != 4) return 0;
        if (in->shape[0] <= 0 || o->shape[0] != in->shape[0]) return 0;
        int sy, sx, py, px, dy, dx; p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px); p_pair(p, "dilation", 1, &dy, &dx);
        int pads[4] = {py, px, py, px}; if (p && cJSON_GetObjectItem(p, "pads")) p_iarr(p, "pads", pads, 4, 0);
        int groups = p_int(p, "groups", 1), relu = n->fuse_relu6 ? 2 : p_int(p, "relu", 0);
        if (pads[0] < 0 || pads[1] < 0 || relu > 2) return 0;
        T wtmp, btmp;
        T* wp = prepare_conv_weight_image(idx, w, p, in->shape[3], o->shape[3], groups, &wtmp);
        if (!wp) return 0;
        T* bp = b;
        if (b && b->dtype == T_F16) {
            float* bf = f16_tensor_f32_cache(g_f16bcache, idx, b);
            if (!bf) return 0;
            btmp = *b; btmp.data = bf; btmp.dtype = T_F32; btmp.elem_size = sizeof(float);
            bp = &btmp;
        }
        if (b && bp->dtype != T_F32) return 0;
        if (vk_graph_conv2d_f32(in->data, o->data, wp->data, bp ? bp->data : NULL,
                                in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                o->shape[3], wp->shape[0], wp->shape[1], o->shape[1], o->shape[2],
                                sy, sx, pads[0], pads[1], groups, relu, dy, dx)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Add")) {
        T* a = nin(n, "a"); T* b = nin(n, "b");
        if (!a) a = in;
        if (a && b && a->dtype == T_F32 && b->dtype == T_F32 &&
            a->numel == b->numel && a->numel == o->numel &&
            vk_graph_add_relu_f32(a->data, b->data, o->data, o->numel, p_int(p, "relu", 0))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        if (p_int(p, "relu", 0) == 0 && try_gpu_graph_binary(n, in, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Mul") || !strcmp(op, "Sub") || !strcmp(op, "Div")) {
        if (try_gpu_graph_binary(n, in, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Sigmoid")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            vk_graph_sigmoid_f32(in->data, o->data, o->numel)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Clip")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            vk_graph_clip_f32(in->data, o->data, o->numel,
                              p_flt(p, "min", -1e30f), p_flt(p, "max", 1e30f))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ReLU") || !strcmp(op, "GELU") || !strcmp(op, "SiLU") ||
        !strcmp(op, "Tanh") || !strcmp(op, "HardSwish") || !strcmp(op, "HardSigmoid")) {
        if (!in) return 0;
        materialize_tensor_f32(in);
        if (in->dtype != T_F32 || in->numel != o->numel) return 0;
        int ok = 0;
        if (!strcmp(op, "ReLU")) ok = vk_graph_relu_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "GELU")) {
            int approximate_tanh = 0;
            if (gelu_approximate_tanh(p, &approximate_tanh) != 0) return 0;
            ok = vk_graph_gelu_f32(in->data, o->data, o->numel, approximate_tanh);
        }
        else if (!strcmp(op, "SiLU")) ok = vk_graph_silu_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "Tanh")) ok = vk_graph_tanh_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "HardSwish")) ok = vk_graph_hardswish_f32(in->data, o->data, o->numel);
        else ok = vk_graph_hardsigmoid_f32(in->data, o->data, o->numel);
        if (ok) { *node_backend = "vulkan-graph"; return 1; }
        return 0;
    }

    if (!strcmp(op, "LeakyReLU")) {
        if (!in) return 0;
        materialize_tensor_f32(in);
        if (in->dtype == T_F32 && in->numel == o->numel &&
            vk_graph_leaky_relu_f32(in->data, o->data, o->numel, p_flt(p, "alpha", 0.01f))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "PReLU")) {
        T* w = nin(n, "weight");
        if (!in || !w) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w);
        int c = in->ndim > 0 ? in->shape[in->ndim - 1] : 1;
        if (in->dtype == T_F32 && w->dtype == T_F32 && in->numel == o->numel && w->numel >= c &&
            vk_graph_prelu_f32(in->data, w->data, o->data, o->numel, c)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "LayerNorm")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || !b || in->ndim <= 0) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); materialize_tensor_f32(b);
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        long rows = d_model > 0 ? in->numel / d_model : 0;
        float epsilon = p_flt(p, "eps", 1.0e-6f);
        if (in->dtype == T_F32 && w->dtype == T_F32 && b->dtype == T_F32 && o->numel == in->numel &&
            d_model > 0 && in->numel % d_model == 0 && rows > 0 && rows <= INT32_MAX &&
            w->numel == d_model && b->numel == d_model &&
            epsilon > 0.0f && isfinite(epsilon) &&
            vk_graph_layernorm_f32(in->data, w->data, b->data, o->data,
                                   (int)rows, d_model, epsilon)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "GroupNorm")) {
        T* weight = nin(n, "weight");
        if (!weight) weight = nin(n, "scale");
        T* bias = nin(n, "bias");
        int groups = p_int(p, "num_groups", p_int(p, "groups", 1));
        float epsilon = p_flt(p, "eps", 1.0e-5f);
        int batch, height, width, channels;
        if (!bias || groupnorm_dimensions(in, o, weight, bias, groups, &batch, &height,
                                          &width, &channels) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        materialize_tensor_f32(bias);
        if (vk_graph_groupnorm_f32(in->data, weight->data, bias->data, o->data,
                                   batch, height, width, channels, groups, epsilon)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "RMSNorm")) {
        T* w = nin(n, "weight");
        if (!in || !w) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w);
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        int rows = d_model > 0 ? (int)(in->numel / d_model) : 0;
        if (in->dtype == T_F32 && w->dtype == T_F32 && o->numel == in->numel && w->numel >= d_model &&
            vk_graph_rmsnorm_f32(in->data, w->data, o->data, rows, d_model, p_flt(p, "eps", 1e-6f))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Softmax") || !strcmp(op, "LogSoftmax")) {
        if (!in || in->ndim <= 0) return 0;
        int axis = p_int(p, "axis", in->ndim - 1);
        if (axis < 0) axis += in->ndim;
        if (axis != in->ndim - 1) return 0;
        materialize_tensor_f32(in);
        int d = in->shape[in->ndim - 1];
        int rows = d > 0 ? (int)(in->numel / d) : 0;
        if (in->dtype != T_F32 || o->numel != in->numel) return 0;
        int ok = (!strcmp(op, "Softmax"))
            ? vk_graph_softmax_f32(in->data, o->data, rows, d)
            : vk_graph_logsoftmax_f32(in->data, o->data, rows, d);
        if (ok) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ReduceSum") || !strcmp(op, "ReduceMean")) {
        T* source = NULL;
        long rows = 0;
        int width = 0;
        float scale = 1.0f;
        if (last_axis_reduction_dimensions(n, o, &source, &rows, &width, &scale) != 0 ||
            rows > INT32_MAX) return 0;
        materialize_tensor_f32(source);
        if (vk_graph_reduce_f32(source->data, o->data, (int)rows, width, scale)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "BatchNorm2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias"); T* rm = nin(n, "running_mean"); T* rv = nin(n, "running_var");
        if (!in || !w || !b || !rm || !rv || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); materialize_tensor_f32(b); materialize_tensor_f32(rm); materialize_tensor_f32(rv);
        int c = in->shape[3];
        if (in->dtype == T_F32 && w->dtype == T_F32 && b->dtype == T_F32 && rm->dtype == T_F32 && rv->dtype == T_F32 &&
            w->numel >= c && b->numel >= c && rm->numel >= c && rv->numel >= c &&
            vk_graph_batchnorm2d_f32(in->data, w->data, b->data, rm->data, rv->data, o->data,
                                     in->shape[0], in->shape[1], in->shape[2], c, p_flt(p, "eps", 1e-5f))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "GlobalAveragePool")) {
        if (!in || in->ndim != 4) return 0;
        materialize_tensor_f32(in);
        if (in->dtype == T_F32 && o->numel == (long)in->shape[0] * in->shape[3] &&
            vk_graph_global_average_pool_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3])) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Conv1D")) {
        if (try_gpu_graph_conv1d(n, in, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "AveragePool2D")) {
        if (!in || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in);
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx); p_pair(p, "stride", 1, &sy, &sx);
        p_spatial_pad_origin(p, &py, &px);
        if (in->dtype == T_F32 &&
            vk_graph_average_pool2d_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                        o->shape[1], o->shape[2], ky, kx, sy, sx, py, px)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Embedding")) {
        if (try_gpu_graph_embedding(n, in, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Resize") || !strcmp(op, "ResizeNearest2D") || !strcmp(op, "UpsampleNearest2D")) {
        if (in && o && in->dtype == T_F32 && in->ndim == 4 && o->ndim == 4 &&
            in->shape[0] > 0 && o->shape[0] == in->shape[0]) {
            int mode = (!strcmp(op, "Resize") && strcmp(p_str(p, "mode", "bilinear"), "nearest")) ? 1 : 0;
            if (o->shape[3] == in->shape[3] && vk_graph_resize_f32(in->data, o->data,
                    in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                    o->shape[1], o->shape[2], mode)) {
                *node_backend = "vulkan-graph";
                return 1;
            }
        }
        return 0;
    }

    if (!strcmp(op, "MaxPool2D")) {
        if (!in || !o || in->dtype != T_F32 || in->ndim != 4 || o->ndim != 4 ||
            in->shape[0] <= 0 || o->shape[0] != in->shape[0] || o->shape[3] != in->shape[3]) return 0;
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx);
        p_pair(p, "stride", 1, &sy, &sx);
        p_spatial_pad_origin(p, &py, &px);
        if (vk_graph_maxpool2d_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                   o->shape[1], o->shape[2], ky, kx, sy, sx, py, px)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Transpose")) {
        if (!in || in->ndim <= 0 || in->ndim > 8) return 0;
        materialize_tensor_f32(in);
        int perm[8];
        for (int i = 0; i < in->ndim; i++) perm[i] = in->ndim - 1 - i;
        if (p && cJSON_GetObjectItem(p, "perm"))
            p_iarr(p, "perm", perm, in->ndim, 0);
        if (in->dtype == T_F32 && o->numel == in->numel &&
            vk_graph_transpose_f32(in->data, o->data, in->shape, perm, in->ndim)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Cast")) {
        T* x = in ? in : nin(n, "data");
        if (!x) return 0;
        materialize_tensor_f32(x);
        if (x->dtype == T_F32 && x->numel == o->numel && vk_graph_cast_copy_f32(x->data, o->data, o->numel)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Where") || !strcmp(op, "Mask")) {
        T* cond = nin(n, "cond"); if (!cond) cond = nin(n, "condition");
        T* a = nin(n, "x"); if (!a) a = nin(n, "a");
        T* b = nin(n, "y"); if (!b) b = nin(n, "b");
        if (!cond || !a || !b) return 0;
        materialize_tensor_f32(cond); materialize_tensor_f32(a); materialize_tensor_f32(b);
        if (cond->dtype == T_F32 && a->dtype == T_F32 && b->dtype == T_F32 &&
            cond->numel == o->numel && a->numel == o->numel && b->numel == o->numel &&
            vk_graph_where_f32(cond->data, a->data, b->data, o->data, o->numel)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Expand") || !strcmp(op, "Broadcast")) {
        T* x = in ? in : nin(n, "data");
        if (!x) return 0;
        materialize_tensor_f32(x);
        if (x->dtype == T_F32 && vk_graph_expand_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Gather")) {
        T* idx = nin(n, "indices");
        int outer, axis_size, inner, indices_elements, output_elements;
        int axis;
        if (!in || !idx || in->ndim <= 0) return 0;
        axis = normalize_axis(p_int(p, "axis", 0), in->ndim);
        materialize_tensor_f32(in);
        if (gather_i32_dimensions(in, idx, o, axis, &outer, &axis_size,
                                  &inner, &indices_elements,
                                  &output_elements) &&
            vk_graph_gather_i32_f32(in->data, (const int32_t*)idx->data,
                                    o->data, outer, axis_size, inner,
                                    indices_elements, output_elements)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Pad")) {
        T* x = in ? in : nin(n, "data");
        if (!x || x->ndim > 4 || o->ndim > 4) return 0;
        materialize_tensor_f32(x);
        int pads[8] = {0}; p_iarr(p, "pads", pads, 8, 0);
        cJSON* pads_node = p ? cJSON_GetObjectItem(p, "pads") : NULL;
        int pads_len = pads_node && cJSON_IsArray(pads_node) ? cJSON_GetArraySize(pads_node) : 0;
        int pt = pads_len == 8 ? pads[1] : pads[0];
        int pl = pads_len == 8 ? pads[2] : pads[1];
        if (pt < 0 || pl < 0) return 0;
        if (x->dtype == T_F32 && vk_graph_pad4d_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim,
                                                    pt, pl, p_flt(p, "value", 0.0f))) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Slice")) {
        T* x = in ? in : nin(n, "data");
        if (!x || x->ndim > 4 || o->ndim > 4) return 0;
        materialize_tensor_f32(x);
        int starts4[4] = {0, 0, 0, 0}, steps4[4] = {1, 1, 1, 1};
        int starts[8] = {0}, steps[8] = {1,1,1,1,1,1,1,1}, axes[8] = {0,1,2,3,4,5,6,7};
        p_iarr(p, "starts", starts, 8, 0); p_iarr(p, "steps", steps, 8, 1); p_iarr(p, "axes", axes, 8, 0);
        cJSON* axes_node = p ? cJSON_GetObjectItem(p, "axes") : NULL;
        int n_axes = axes_node && cJSON_IsArray(axes_node) ? cJSON_GetArraySize(axes_node) : x->ndim;
        int base = 4 - x->ndim;
        for (int i = 0; i < n_axes && i < 8; i++) {
            int ax = axes_node ? axes[i] : i;
            if (ax < 0) ax += x->ndim;
            if (ax < 0 || ax >= x->ndim || steps[i] <= 0) return 0;
            int padded = base + ax;
            int st = starts[i] < 0 ? starts[i] + x->shape[ax] : starts[i];
            if (st < 0) return 0;
            starts4[padded] = st;
            steps4[padded] = steps[i];
        }
        if (x->dtype == T_F32 && vk_graph_slice4d_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim, starts4, steps4)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ConvTranspose2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); if (b) materialize_tensor_f32(b);
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx); p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px);
        if (in->dtype == T_F32 && w->dtype == T_F32 && (!b || b->dtype == T_F32) &&
            vk_graph_conv_transpose2d_f32(in->data, w->data, b ? b->data : NULL, o->data,
                in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                o->shape[1], o->shape[2], o->shape[3], ky, kx, sy, sx, py, px)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "MoERouter")) {
        if (try_gpu_graph_moe_router(n, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "MoELinear")) {
        if (try_gpu_graph_moe_linear(n, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SDPA")) {
        if (try_gpu_graph_sdpa(n, in, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossSDPA")) {
        if (try_gpu_graph_cross_sdpa(n, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossAttention")) {
        if (try_gpu_graph_cross_attention(n, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Split")) {
        if (try_gpu_graph_split(n, in, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SpatialSoftargmaxY") || !strcmp(op, "ProfileX") ||
        !strcmp(op, "ProfileY") || !strcmp(op, "MeanHeight")) {
        if (try_gpu_graph_profile_op(n, in, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "NonMaxSuppression")) {
        if (try_gpu_graph_nms(n, o, &g_vulkan_f32_ops)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Interpolate1D")) {
        if (!in || in->ndim != 2 || o->ndim != 2) return 0;
        materialize_tensor_f32(in);
        int channels = in->shape[in->ndim - 2];
        int in_l = in->shape[in->ndim - 1];
        int out_l = o->shape[o->ndim - 1];
        if (in->dtype == T_F32 && vk_graph_interp1d_f32(in->data, o->data, channels, in_l, out_l)) {
            *node_backend = "vulkan-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Dropout")) {
        if (!in || in->dtype != T_F32 || in->numel != o->numel) return 0;
#if VOLVOXAI_ENABLE_TRAINING
        if (g_native_training_mode) {
            NativeDropoutParams dropout;
            if (native_dropout_params(n, idx, &dropout) != 0) return 0;
            if (vk_graph_dropout_f32(in->data, o->data, o->numel, dropout.threshold,
                                     dropout.seed, dropout.counter, dropout.scale)) {
                *node_backend = "vulkan-graph";
                return 1;
            }
        } else
#endif
        if (vk_graph_alias_f32(in->data, o->data, o->numel)) {
            *node_backend = "vulkan-graph-alias";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Reshape") || !strcmp(op, "Flatten") || !strcmp(op, "Squeeze")
            || !strcmp(op, "Unsqueeze") || !strcmp(op, "Identity")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            vk_graph_alias_f32(in->data, o->data, o->numel)) {
            *node_backend = "vulkan-graph-alias";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Concat")) {
        int axis = p_int(p, "axis", 1);
        if (axis < 0) axis += o->ndim;
        if (axis < 0 || axis >= o->ndim || o->shape[axis] <= 0) return 0;
        const float* ptrs[MAXIN];
        long sizes[MAXIN];
        int input_axes[MAXIN];
        long total = 0;
        long summed_axis = 0;
        for (int i = 0; i < n->nin; i++) {
            T* src = t_find(n->ins[i].name);
            if (!src || src->dtype != T_F32 || src->ndim != o->ndim || src->shape[axis] <= 0) return 0;
            for (int d = 0; d < o->ndim; d++) {
                if (d != axis && src->shape[d] != o->shape[d]) return 0;
            }
            materialize_tensor_f32(src);
            ptrs[i] = src->data;
            sizes[i] = src->numel;
            input_axes[i] = src->shape[axis];
            total += sizes[i];
            summed_axis += input_axes[i];
        }
        int sigmoid = p_int(p, "sigmoid", 0) ? 1 : 0;
        int inner = tensor_inner_after_axis(o, axis);
        int ok = total == o->numel && summed_axis == o->shape[axis] && inner > 0 &&
                 vk_graph_concat_f32(ptrs, sizes, input_axes, n->nin, o->data,
                                     o->shape[axis], inner, sigmoid);
        if (ok) {
            *node_backend = sigmoid ? "vulkan-graph+sigmoid" : "vulkan-graph";
            return 1;
        }
        return 0;
    }

    return 0;
}
#endif

#if VOLVOXAI_ENABLE_OPENGL
static int try_opengl_graph_node(Node* n, int idx, const char** node_backend) {
    if (!n || g_active_row >= 0 || g_prefix_rows > 0) return 0;
    T* o = t_find(n->out);
    T* in = nin(n, "input");
    cJSON* p = n->params;
    const char* op = n->op;
    if (!o) return 0;
    if (try_gpu_graph_typed_control(n, in, o, &g_opengl_f32_ops)) {
        *node_backend = "opengl-graph";
        return 1;
    }
    if (try_gpu_graph_batch_matmul_f32(n, o, &g_opengl_f32_ops)) {
        *node_backend = "opengl-graph";
        return 1;
    }
    if (try_gpu_graph_linear_f32(n, in, o, &g_opengl_f32_ops)) {
        *node_backend = "opengl-graph";
        return 1;
    }
    if (o->dtype != T_F32) return 0;

    if (!strcmp(op, "QConv2D")) {
        return 0;
    }

    if (!strcmp(op, "Conv2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || in->dtype != T_F32 || (w->dtype != T_F32 && w->dtype != T_F16) ||
            o->ndim != 4 || in->ndim != 4 || w->ndim != 4) return 0;
        int sy, sx, py, px, dy, dx; p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px); p_pair(p, "dilation", 1, &dy, &dx);
        int pads[4] = {py, px, py, px}; if (p && cJSON_GetObjectItem(p, "pads")) p_iarr(p, "pads", pads, 4, 0);
        int groups = p_int(p, "groups", 1), relu = n->fuse_relu6 ? 2 : p_int(p, "relu", 0);
        if (in->shape[0] <= 0 || o->shape[0] != in->shape[0]) return 0;
        if (pads[0] < 0 || pads[1] < 0 || relu > 2) return 0;
        T wtmp, btmp;
        T* wp = prepare_conv_weight_image(idx, w, p, in->shape[3], o->shape[3], groups, &wtmp);
        if (!wp) return 0;
        T* bp = b;
        if (b && b->dtype == T_F16) {
            float* bf = f16_tensor_f32_cache(g_f16bcache, idx, b);
            if (!bf) return 0;
            btmp = *b; btmp.data = bf; btmp.dtype = T_F32; btmp.elem_size = sizeof(float);
            bp = &btmp;
        }
        if (b && bp->dtype != T_F32) return 0;
        int kh = wp->shape[0], kw = wp->shape[1];
        if (opengl_graph_conv2d_f32(in->data, o->data, wp->data, bp ? bp->data : NULL,
                                    in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                    o->shape[3], kh, kw, o->shape[1], o->shape[2],
                                    sy, sx, pads[0], pads[1], groups, relu, dy, dx)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Add")) {
        T* a = nin(n, "a"); T* b = nin(n, "b");
        if (!a) a = in;
        const GraphNodeFusion* add3_fusion = graph_opt_node_fusion(idx, GRAPH_FUSION_CHAINED_ELEMENTWISE);
        if (add3_fusion) {
            Node* peer = (add3_fusion->peer_idx >= 0 && add3_fusion->peer_idx < g_nn) ? &g_n[add3_fusion->peer_idx] : NULL;
            T* c = t_find(add3_fusion->aux_tensor);
            if (peer && a && b && c &&
                a->dtype == T_F32 && b->dtype == T_F32 && c->dtype == T_F32 &&
                a->numel == b->numel && a->numel == c->numel && a->numel == o->numel &&
                opengl_graph_add3_relu_f32(a->data, b->data, c->data, o->data, o->numel,
                                           p_int(peer->params, "relu", 0))) {
                *node_backend = "opengl-graph+add3";
                return 1;
            }
            return 0;
        }
        if (a && b && a->dtype == T_F32 && b->dtype == T_F32 &&
            a->numel == b->numel && a->numel == o->numel &&
            opengl_graph_add_relu_f32(a->data, b->data, o->data, o->numel, p_int(p, "relu", 0))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        if (p_int(p, "relu", 0) == 0 && try_gpu_graph_binary(n, in, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Mul") || !strcmp(op, "Sub") || !strcmp(op, "Div")) {
        if (try_gpu_graph_binary(n, in, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Sigmoid")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            opengl_graph_sigmoid_f32(in->data, o->data, o->numel)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Clip")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            opengl_graph_clip_f32(in->data, o->data, o->numel,
                                  p_flt(p, "min", -1e30f), p_flt(p, "max", 1e30f))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ReLU") || !strcmp(op, "GELU") || !strcmp(op, "SiLU") ||
        !strcmp(op, "Tanh") || !strcmp(op, "HardSwish") || !strcmp(op, "HardSigmoid")) {
        if (!in) return 0;
        materialize_tensor_f32(in);
        if (in->dtype != T_F32 || in->numel != o->numel) return 0;
        int ok = 0;
        if (!strcmp(op, "ReLU")) ok = opengl_graph_relu_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "GELU")) {
            int approximate_tanh = 0;
            if (gelu_approximate_tanh(p, &approximate_tanh) != 0) return 0;
            ok = opengl_graph_gelu_f32(in->data, o->data, o->numel, approximate_tanh);
        }
        else if (!strcmp(op, "SiLU")) ok = opengl_graph_silu_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "Tanh")) ok = opengl_graph_tanh_f32(in->data, o->data, o->numel);
        else if (!strcmp(op, "HardSwish")) ok = opengl_graph_hardswish_f32(in->data, o->data, o->numel);
        else ok = opengl_graph_hardsigmoid_f32(in->data, o->data, o->numel);
        if (ok) { *node_backend = "opengl-graph"; return 1; }
        return 0;
    }

    if (!strcmp(op, "LeakyReLU")) {
        if (!in) return 0;
        materialize_tensor_f32(in);
        if (in->dtype == T_F32 && in->numel == o->numel &&
            opengl_graph_leaky_relu_f32(in->data, o->data, o->numel, p_flt(p, "alpha", 0.01f))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "PReLU")) {
        T* w = nin(n, "weight");
        if (!in || !w) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w);
        int c = in->ndim > 0 ? in->shape[in->ndim - 1] : 1;
        if (in->dtype == T_F32 && w->dtype == T_F32 && in->numel == o->numel && w->numel >= c &&
            opengl_graph_prelu_f32(in->data, w->data, o->data, o->numel, c)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "LayerNorm")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || !b || in->ndim <= 0) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); materialize_tensor_f32(b);
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        long rows = d_model > 0 ? in->numel / d_model : 0;
        float epsilon = p_flt(p, "eps", 1.0e-6f);
        if (in->dtype == T_F32 && w->dtype == T_F32 && b->dtype == T_F32 && o->numel == in->numel &&
            d_model > 0 && in->numel % d_model == 0 && rows > 0 && rows <= INT32_MAX &&
            w->numel == d_model && b->numel == d_model &&
            epsilon > 0.0f && isfinite(epsilon) &&
            opengl_graph_layernorm_f32(in->data, w->data, b->data, o->data,
                                       (int)rows, d_model, epsilon)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "GroupNorm")) {
        T* weight = nin(n, "weight");
        if (!weight) weight = nin(n, "scale");
        T* bias = nin(n, "bias");
        int groups = p_int(p, "num_groups", p_int(p, "groups", 1));
        float epsilon = p_flt(p, "eps", 1.0e-5f);
        int batch, height, width, channels;
        if (!bias || groupnorm_dimensions(in, o, weight, bias, groups, &batch, &height,
                                          &width, &channels) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
        materialize_tensor_f32(in);
        materialize_tensor_f32(weight);
        materialize_tensor_f32(bias);
        if (opengl_graph_groupnorm_f32(in->data, weight->data, bias->data, o->data,
                                       batch, height, width, channels, groups, epsilon)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "RMSNorm")) {
        T* w = nin(n, "weight");
        if (!in || !w) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w);
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        int rows = d_model > 0 ? (int)(in->numel / d_model) : 0;
        if (in->dtype == T_F32 && w->dtype == T_F32 && o->numel == in->numel && w->numel >= d_model &&
            opengl_graph_rmsnorm_f32(in->data, w->data, o->data, rows, d_model, p_flt(p, "eps", 1e-6f))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Softmax") || !strcmp(op, "LogSoftmax")) {
        if (!in || in->ndim <= 0) return 0;
        int axis = p_int(p, "axis", in->ndim - 1);
        if (axis < 0) axis += in->ndim;
        if (axis != in->ndim - 1) return 0;
        materialize_tensor_f32(in);
        int d = in->shape[in->ndim - 1];
        int rows = d > 0 ? (int)(in->numel / d) : 0;
        if (in->dtype != T_F32 || o->numel != in->numel) return 0;
        int ok = (!strcmp(op, "Softmax"))
            ? opengl_graph_softmax_f32(in->data, o->data, rows, d)
            : opengl_graph_logsoftmax_f32(in->data, o->data, rows, d);
        if (ok) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ReduceSum") || !strcmp(op, "ReduceMean")) {
        T* source = NULL;
        long rows = 0;
        int width = 0;
        float scale = 1.0f;
        if (last_axis_reduction_dimensions(n, o, &source, &rows, &width, &scale) != 0 ||
            rows > INT32_MAX) return 0;
        materialize_tensor_f32(source);
        if (opengl_graph_reduce_f32(source->data, o->data, (int)rows, width, scale)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "BatchNorm2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias"); T* rm = nin(n, "running_mean"); T* rv = nin(n, "running_var");
        if (!in || !w || !b || !rm || !rv || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); materialize_tensor_f32(b); materialize_tensor_f32(rm); materialize_tensor_f32(rv);
        int c = in->shape[3];
        if (in->dtype == T_F32 && w->dtype == T_F32 && b->dtype == T_F32 && rm->dtype == T_F32 && rv->dtype == T_F32 &&
            w->numel >= c && b->numel >= c && rm->numel >= c && rv->numel >= c &&
            opengl_graph_batchnorm2d_f32(in->data, w->data, b->data, rm->data, rv->data, o->data,
                                         in->shape[0], in->shape[1], in->shape[2], c, p_flt(p, "eps", 1e-5f))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "GlobalAveragePool")) {
        if (!in || in->ndim != 4) return 0;
        materialize_tensor_f32(in);
        if (in->dtype == T_F32 && o->numel == (long)in->shape[0] * in->shape[3] &&
            opengl_graph_global_average_pool_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3])) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Conv1D")) {
        if (try_gpu_graph_conv1d(n, in, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "AveragePool2D")) {
        if (!in || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in);
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx); p_pair(p, "stride", 1, &sy, &sx);
        p_spatial_pad_origin(p, &py, &px);
        if (in->dtype == T_F32 &&
            opengl_graph_average_pool2d_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                            o->shape[1], o->shape[2], ky, kx, sy, sx, py, px)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Embedding")) {
        if (try_gpu_graph_embedding(n, in, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Resize") || !strcmp(op, "ResizeNearest2D") || !strcmp(op, "UpsampleNearest2D")) {
        if (in && o && in->dtype == T_F32 && in->ndim == 4 && o->ndim == 4 &&
            in->shape[0] > 0 && o->shape[0] == in->shape[0]) {
            int mode = (!strcmp(op, "Resize") && strcmp(p_str(p, "mode", "bilinear"), "nearest")) ? 1 : 0;
            if (o->shape[3] == in->shape[3] && opengl_graph_resize_f32(in->data, o->data,
                    in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                    o->shape[1], o->shape[2], mode)) {
                *node_backend = "opengl-graph";
                return 1;
            }
        }
        return 0;
    }

    if (!strcmp(op, "MaxPool2D")) {
        if (!in || !o || in->dtype != T_F32 || in->ndim != 4 || o->ndim != 4 ||
            in->shape[0] <= 0 || o->shape[0] != in->shape[0] || o->shape[3] != in->shape[3]) return 0;
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx);
        p_pair(p, "stride", 1, &sy, &sx);
        p_spatial_pad_origin(p, &py, &px);
        if (opengl_graph_maxpool2d_f32(in->data, o->data, in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                                       o->shape[1], o->shape[2], ky, kx, sy, sx, py, px)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Transpose")) {
        if (!in || in->ndim <= 0 || in->ndim > 8) return 0;
        materialize_tensor_f32(in);
        int perm[8];
        for (int i = 0; i < in->ndim; i++) perm[i] = in->ndim - 1 - i;
        if (p && cJSON_GetObjectItem(p, "perm"))
            p_iarr(p, "perm", perm, in->ndim, 0);
        if (in->dtype == T_F32 && o->numel == in->numel &&
            opengl_graph_transpose_f32(in->data, o->data, in->shape, perm, in->ndim)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Cast")) {
        T* x = in ? in : nin(n, "data");
        if (!x) return 0;
        materialize_tensor_f32(x);
        if (x->dtype == T_F32 && x->numel == o->numel && opengl_graph_cast_copy_f32(x->data, o->data, o->numel)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Where") || !strcmp(op, "Mask")) {
        T* cond = nin(n, "cond"); if (!cond) cond = nin(n, "condition");
        T* a = nin(n, "x"); if (!a) a = nin(n, "a");
        T* b = nin(n, "y"); if (!b) b = nin(n, "b");
        if (!cond || !a || !b) return 0;
        materialize_tensor_f32(cond); materialize_tensor_f32(a); materialize_tensor_f32(b);
        if (cond->dtype == T_F32 && a->dtype == T_F32 && b->dtype == T_F32 &&
            cond->numel == o->numel && a->numel == o->numel && b->numel == o->numel &&
            opengl_graph_where_f32(cond->data, a->data, b->data, o->data, o->numel)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Expand") || !strcmp(op, "Broadcast")) {
        T* x = in ? in : nin(n, "data");
        if (!x) return 0;
        materialize_tensor_f32(x);
        if (x->dtype == T_F32 && opengl_graph_expand_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Gather")) {
        T* idx = nin(n, "indices");
        int outer, axis_size, inner, indices_elements, output_elements;
        int axis;
        if (!in || !idx || in->ndim <= 0) return 0;
        axis = normalize_axis(p_int(p, "axis", 0), in->ndim);
        materialize_tensor_f32(in);
        if (gather_i32_dimensions(in, idx, o, axis, &outer, &axis_size,
                                  &inner, &indices_elements,
                                  &output_elements) &&
            opengl_graph_gather_i32_f32(in->data,
                                        (const int32_t*)idx->data, o->data,
                                        outer, axis_size, inner,
                                        indices_elements, output_elements)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Pad")) {
        T* x = in ? in : nin(n, "data");
        if (!x || x->ndim > 4 || o->ndim > 4) return 0;
        materialize_tensor_f32(x);
        int pads[8] = {0}; p_iarr(p, "pads", pads, 8, 0);
        cJSON* pads_node = p ? cJSON_GetObjectItem(p, "pads") : NULL;
        int pads_len = pads_node && cJSON_IsArray(pads_node) ? cJSON_GetArraySize(pads_node) : 0;
        int pt = pads_len == 8 ? pads[1] : pads[0];
        int pl = pads_len == 8 ? pads[2] : pads[1];
        if (pt < 0 || pl < 0) return 0;
        if (x->dtype == T_F32 && opengl_graph_pad4d_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim,
                                                        pt, pl, p_flt(p, "value", 0.0f))) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Slice")) {
        T* x = in ? in : nin(n, "data");
        if (!x || x->ndim > 4 || o->ndim > 4) return 0;
        materialize_tensor_f32(x);
        int starts4[4] = {0, 0, 0, 0}, steps4[4] = {1, 1, 1, 1};
        int starts[8] = {0}, steps[8] = {1,1,1,1,1,1,1,1}, axes[8] = {0,1,2,3,4,5,6,7};
        p_iarr(p, "starts", starts, 8, 0); p_iarr(p, "steps", steps, 8, 1); p_iarr(p, "axes", axes, 8, 0);
        cJSON* axes_node = p ? cJSON_GetObjectItem(p, "axes") : NULL;
        int n_axes = axes_node && cJSON_IsArray(axes_node) ? cJSON_GetArraySize(axes_node) : x->ndim;
        int base = 4 - x->ndim;
        for (int i = 0; i < n_axes && i < 8; i++) {
            int ax = axes_node ? axes[i] : i;
            if (ax < 0) ax += x->ndim;
            if (ax < 0 || ax >= x->ndim || steps[i] <= 0) return 0;
            int padded = base + ax;
            int st = starts[i] < 0 ? starts[i] + x->shape[ax] : starts[i];
            if (st < 0) return 0;
            starts4[padded] = st;
            steps4[padded] = steps[i];
        }
        if (x->dtype == T_F32 && opengl_graph_slice4d_f32(x->data, o->data, x->shape, x->ndim, o->shape, o->ndim, starts4, steps4)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ConvTranspose2D")) {
        T* w = nin(n, "weight"); T* b = nin(n, "bias");
        if (!in || !w || in->ndim != 4 || o->ndim != 4) return 0;
        materialize_tensor_f32(in); materialize_tensor_f32(w); if (b) materialize_tensor_f32(b);
        int ky, kx, sy, sx, py, px;
        p_pair(p, "kernel", 1, &ky, &kx); p_pair(p, "stride", 1, &sy, &sx); p_pair(p, "padding", 0, &py, &px);
        if (in->dtype == T_F32 && w->dtype == T_F32 && (!b || b->dtype == T_F32) &&
            opengl_graph_conv_transpose2d_f32(in->data, w->data, b ? b->data : NULL, o->data,
                in->shape[0], in->shape[1], in->shape[2], in->shape[3],
                o->shape[1], o->shape[2], o->shape[3], ky, kx, sy, sx, py, px)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "MoERouter")) {
        if (try_gpu_graph_moe_router(n, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "MoELinear")) {
        if (try_gpu_graph_moe_linear(n, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SDPA")) {
        if (try_gpu_graph_sdpa(n, in, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossSDPA")) {
        if (try_gpu_graph_cross_sdpa(n, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossAttention")) {
        if (try_gpu_graph_cross_attention(n, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Split")) {
        if (try_gpu_graph_split(n, in, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SpatialSoftargmaxY") || !strcmp(op, "ProfileX") ||
        !strcmp(op, "ProfileY") || !strcmp(op, "MeanHeight")) {
        if (try_gpu_graph_profile_op(n, in, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "NonMaxSuppression")) {
        if (try_gpu_graph_nms(n, o, &g_opengl_f32_ops)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Interpolate1D")) {
        if (!in || in->ndim != 2 || o->ndim != 2) return 0;
        materialize_tensor_f32(in);
        int channels = in->shape[in->ndim - 2];
        int in_l = in->shape[in->ndim - 1];
        int out_l = o->shape[o->ndim - 1];
        if (in->dtype == T_F32 && opengl_graph_interp1d_f32(in->data, o->data, channels, in_l, out_l)) {
            *node_backend = "opengl-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Dropout")) {
        if (!in || in->dtype != T_F32 || in->numel != o->numel) return 0;
#if VOLVOXAI_ENABLE_TRAINING
        if (g_native_training_mode) {
            NativeDropoutParams dropout;
            if (native_dropout_params(n, idx, &dropout) != 0) return 0;
            if (opengl_graph_dropout_f32(in->data, o->data, o->numel, dropout.threshold,
                                         dropout.seed, dropout.counter, dropout.scale)) {
                *node_backend = "opengl-graph";
                return 1;
            }
        } else
#endif
        if (opengl_graph_alias_f32(in->data, o->data, o->numel)) {
            *node_backend = "opengl-graph-alias";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Reshape") || !strcmp(op, "Flatten") || !strcmp(op, "Squeeze")
            || !strcmp(op, "Unsqueeze") || !strcmp(op, "Identity")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            opengl_graph_alias_f32(in->data, o->data, o->numel)) {
            *node_backend = "opengl-graph-alias";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Concat")) {
        int axis = p_int(p, "axis", 1);
        if (axis < 0) axis += o->ndim;
        if (axis < 0 || axis >= o->ndim || o->shape[axis] <= 0) return 0;
        const float* ptrs[MAXIN];
        long sizes[MAXIN];
        int input_axes[MAXIN];
        long total = 0;
        long summed_axis = 0;
        for (int i = 0; i < n->nin; i++) {
            T* src = t_find(n->ins[i].name);
            if (!src || src->dtype != T_F32 || src->ndim != o->ndim || src->shape[axis] <= 0) return 0;
            for (int d = 0; d < o->ndim; d++) {
                if (d != axis && src->shape[d] != o->shape[d]) return 0;
            }
            materialize_tensor_f32(src);
            ptrs[i] = src->data;
            sizes[i] = src->numel;
            input_axes[i] = src->shape[axis];
            total += sizes[i];
            summed_axis += input_axes[i];
        }
        int sigmoid = p_int(p, "sigmoid", 0) ? 1 : 0;
        int inner = tensor_inner_after_axis(o, axis);
        int ok = total == o->numel && summed_axis == o->shape[axis] && inner > 0 &&
                 opengl_graph_concat_f32(ptrs, sizes, input_axes, n->nin, o->data,
                                         o->shape[axis], inner, sigmoid);
        if (ok) {
            *node_backend = sigmoid ? "opengl-graph+sigmoid" : "opengl-graph";
            return 1;
        }
        return 0;
    }

    return 0;
}
#endif

#include "engine_runtime_cuda_f32.inc"

#if VOLVOXAI_ENABLE_METAL
static int try_metal_graph_node(Node* n, int idx, const char** node_backend) {
    if (!n || g_active_row >= 0 || g_prefix_rows > 0) return 0;
    T* o = t_find(n->out);
    T* in = nin(n, "input");
    cJSON* p = n->params;
    const char* op = n->op;
    if (!o) return 0;
    if (try_gpu_graph_typed_control(n, in, o, &g_metal_f32_ops)) {
        *node_backend = "metal-graph";
        return 1;
    }
    if (try_gpu_graph_batch_matmul_f32(n, o, &g_metal_f32_ops)) {
        *node_backend = "metal-graph";
        return 1;
    }
    if (try_gpu_graph_linear_f32(n, in, o, &g_metal_f32_ops)) {
        *node_backend = "metal-graph";
        return 1;
    }
    if (o->dtype != T_F32) return 0;

    if (!strcmp(op, "Conv2D")) {
        T* weight = nin(n, "weight");
        T* bias = nin(n, "bias");
        if (!in || !weight || in->dtype != T_F32 || weight->dtype != T_F32 ||
            in->ndim != 4 || weight->ndim != 4 || o->ndim != 4 ||
            in->shape[0] <= 0 || o->shape[0] != in->shape[0]) return 0;
        int sy, sx, py, px, dy, dx;
        p_pair(p, "stride", 1, &sy, &sx);
        p_pair(p, "padding", 0, &py, &px);
        p_pair(p, "dilation", 1, &dy, &dx);
        int pads[4] = {py, px, py, px};
        if (p && cJSON_GetObjectItem(p, "pads")) p_iarr(p, "pads", pads, 4, 0);
        int groups = p_int(p, "groups", 1);
        int relu = n->fuse_relu6 ? 2 : p_int(p, "relu", 0);
        if (pads[0] < 0 || pads[1] < 0 || relu > 2) return 0;
        T weight_image;
        T* prepared_weight = prepare_conv_weight_image(idx, weight, p, in->shape[3],
                                                        o->shape[3], groups, &weight_image);
        if (!prepared_weight || prepared_weight->dtype != T_F32 ||
            (bias && bias->dtype != T_F32)) return 0;
        if (metal_graph_conv2d_f32(in->data, o->data, prepared_weight->data,
                                   bias ? bias->data : NULL, in->shape[0], in->shape[1],
                                   in->shape[2], in->shape[3], o->shape[3],
                                   prepared_weight->shape[0], prepared_weight->shape[1],
                                   o->shape[1], o->shape[2], sy, sx, pads[0], pads[1],
                                   groups, relu, dy, dx)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Add")) {
        if (p_int(p, "relu", 0) == 0 && try_gpu_graph_binary(n, in, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Sigmoid")) {
        if (!in || !tensor_same_shape(in, o)) return 0;
        materialize_tensor_f32(in);
        if (in->dtype == T_F32 &&
            metal_graph_sigmoid_f32(in->data, o->data, o->numel)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "GELU") || !strcmp(op, "SiLU")) {
        if (!in || in->dtype != T_F32 || in->numel != o->numel) return 0;
        int ok;
        if (!strcmp(op, "GELU")) {
            int approximate_tanh = 0;
            if (gelu_approximate_tanh(p, &approximate_tanh) != 0) return 0;
            ok = metal_graph_gelu_f32(in->data, o->data, o->numel, approximate_tanh);
        } else {
            ok = metal_graph_silu_f32(in->data, o->data, o->numel);
        }
        if (ok) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "LayerNorm")) {
        T* weight = nin(n, "weight");
        T* bias = nin(n, "bias");
        if (!in || !weight || !bias || in->dtype != T_F32 ||
            weight->dtype != T_F32 || bias->dtype != T_F32 || in->ndim <= 0 ||
            in->numel != o->numel) return 0;
        int d_model = p_int(p, "d_model", in->shape[in->ndim - 1]);
        long rows = d_model > 0 ? in->numel / d_model : 0;
        float epsilon = p_flt(p, "eps", 1.0e-6f);
        if (rows <= 0 || rows > INT32_MAX || weight->numel != d_model ||
            bias->numel != d_model || !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
        if (metal_graph_layernorm_f32(in->data, weight->data, bias->data, o->data,
                                      (int)rows, d_model, epsilon)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "GroupNorm")) {
        T* weight = nin(n, "weight");
        if (!weight) weight = nin(n, "scale");
        T* bias = nin(n, "bias");
        int groups = p_int(p, "num_groups", p_int(p, "groups", 1));
        float epsilon = p_flt(p, "eps", 1.0e-5f);
        int batch, height, width, channels;
        if (!bias || groupnorm_dimensions(in, o, weight, bias, groups, &batch, &height,
                                          &width, &channels) != 0 ||
            !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
        if (metal_graph_groupnorm_f32(in->data, weight->data, bias->data, o->data,
                                      batch, height, width, channels, groups, epsilon)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Softmax")) {
        long rows;
        int axis;
        int d;
        if (!in || in->ndim <= 0 || !tensor_same_shape(in, o)) return 0;
        axis = p_int(p, "axis", in->ndim - 1);
        if (axis < 0) axis += in->ndim;
        if (axis != in->ndim - 1) return 0;
        materialize_tensor_f32(in);
        d = in->shape[in->ndim - 1];
        if (in->dtype != T_F32 || d <= 0 || in->numel <= 0 ||
            in->numel % d != 0) return 0;
        rows = in->numel / d;
        if (rows <= 0 || rows > INT32_MAX) return 0;
        if (metal_graph_softmax_f32(in->data, o->data, (int)rows, d)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "ReduceSum") || !strcmp(op, "ReduceMean")) {
        T* source = NULL;
        long rows = 0;
        int width = 0;
        float scale = 1.0f;
        if (last_axis_reduction_dimensions(n, o, &source, &rows, &width, &scale) != 0 ||
            rows > INT32_MAX) return 0;
        if (metal_graph_reduce_f32(source->data, o->data, (int)rows, width, scale)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Transpose")) {
        if (!in || in->dtype != T_F32 || in->ndim <= 0 || in->ndim > 8 ||
            in->ndim != o->ndim || in->numel != o->numel) return 0;
        int perm[8];
        for (int dimension = 0; dimension < in->ndim; dimension++)
            perm[dimension] = in->ndim - 1 - dimension;
        if (p && cJSON_GetObjectItem(p, "perm"))
            p_iarr(p, "perm", perm, in->ndim, 0);
        if (metal_graph_transpose_f32(in->data, o->data, in->shape, perm, in->ndim)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Expand") || !strcmp(op, "Broadcast")) {
        T* source = in ? in : nin(n, "data");
        if (!source) return 0;
        materialize_tensor_f32(source);
        if (source->dtype == T_F32 &&
            metal_graph_expand_f32(source->data, o->data,
                                   source->shape, source->ndim,
                                   o->shape, o->ndim)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Gather")) {
        T* indices = nin(n, "indices");
        int outer;
        int axis_size;
        int inner;
        int indices_elements;
        int output_elements;
        int axis;
        int expected_rank;
        int output_dimension = 0;
        if (!in || !indices || in->ndim <= 0 ||
            indices->ndim < 0 || indices->ndim > 8)
            return 0;
        axis = normalize_axis(p_int(p, "axis", 0), in->ndim);
        expected_rank = in->ndim - 1 + indices->ndim;
        if (axis < 0 || axis >= in->ndim ||
            expected_rank < 0 || expected_rank > 8 ||
            o->ndim != expected_rank)
            return 0;
        for (int dimension = 0; dimension < axis; dimension++) {
            if (o->shape[output_dimension++] != in->shape[dimension])
                return 0;
        }
        for (int dimension = 0; dimension < indices->ndim; dimension++) {
            if (indices->shape[dimension] <= 0 ||
                o->shape[output_dimension++] != indices->shape[dimension])
                return 0;
        }
        for (int dimension = axis + 1; dimension < in->ndim; dimension++) {
            if (o->shape[output_dimension++] != in->shape[dimension])
                return 0;
        }
        materialize_tensor_f32(in);
        if (gather_i32_dimensions(in, indices, o, axis, &outer,
                                  &axis_size, &inner, &indices_elements,
                                  &output_elements) &&
            metal_graph_gather_i32_f32(
                in->data, (const int32_t*)indices->data, o->data,
                outer, axis_size, inner, indices_elements,
                output_elements)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Slice")) {
        T* source = in ? in : nin(n, "data");
        cJSON* starts_node =
            p ? cJSON_GetObjectItemCaseSensitive(p, "starts") : NULL;
        cJSON* axes_node =
            p ? cJSON_GetObjectItemCaseSensitive(p, "axes") : NULL;
        cJSON* steps_node =
            p ? cJSON_GetObjectItemCaseSensitive(p, "steps") : NULL;
        if (!source || source->ndim <= 0 || source->ndim > 4 ||
            o->ndim != source->ndim ||
            !cJSON_IsArray(starts_node) || !cJSON_IsArray(axes_node) ||
            !cJSON_IsArray(steps_node))
            return 0;
        int slice_axes = cJSON_GetArraySize(axes_node);
        if (slice_axes <= 0 || slice_axes > source->ndim ||
            cJSON_GetArraySize(starts_node) != slice_axes ||
            cJSON_GetArraySize(steps_node) != slice_axes)
            return 0;
        int starts[4] = {0, 0, 0, 0};
        int axes[4] = {0, 1, 2, 3};
        int steps[4] = {1, 1, 1, 1};
        int starts4[4] = {0, 0, 0, 0};
        int steps4[4] = {1, 1, 1, 1};
        unsigned selected_axes = 0u;
        p_iarr(p, "starts", starts, 4, 0);
        p_iarr(p, "axes", axes, 4, 0);
        p_iarr(p, "steps", steps, 4, 1);
        int base = 4 - source->ndim;
        for (int index = 0; index < slice_axes; index++) {
            int slice_axis = axes[index];
            if (slice_axis < 0) slice_axis += source->ndim;
            if (slice_axis < 0 || slice_axis >= source->ndim ||
                (selected_axes & (1u << slice_axis)) ||
                steps[index] <= 0)
                return 0;
            int start = starts[index] < 0
                ? starts[index] + source->shape[slice_axis]
                : starts[index];
            if (start < 0 || start >= source->shape[slice_axis])
                return 0;
            selected_axes |= 1u << slice_axis;
            starts4[base + slice_axis] = start;
            steps4[base + slice_axis] = steps[index];
        }
        for (int dimension = 0; dimension < source->ndim; dimension++) {
            if (!(selected_axes & (1u << dimension)) &&
                o->shape[dimension] != source->shape[dimension])
                return 0;
        }
        materialize_tensor_f32(source);
        if (source->dtype == T_F32 &&
            metal_graph_slice4d_f32(
                source->data, o->data, source->shape, source->ndim,
                o->shape, o->ndim, starts4, steps4)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Concat")) {
        int axis = p_int(p, "axis", 1);
        if (axis < 0) axis += o->ndim;
        if (axis < 0 || axis >= o->ndim || o->shape[axis] <= 0 ||
            p_int(p, "sigmoid", 0)) return 0;
        const float* inputs[MAXIN];
        long sizes[MAXIN];
        int input_axes[MAXIN];
        long total = 0, summed_axis = 0;
        for (int input_index = 0; input_index < n->nin; input_index++) {
            T* source = t_find(n->ins[input_index].name);
            if (!source || source->dtype != T_F32 || source->ndim != o->ndim ||
                source->shape[axis] <= 0) return 0;
            for (int dimension = 0; dimension < o->ndim; dimension++) {
                if (dimension != axis && source->shape[dimension] != o->shape[dimension])
                    return 0;
            }
            inputs[input_index] = source->data;
            sizes[input_index] = source->numel;
            input_axes[input_index] = source->shape[axis];
            total += source->numel;
            summed_axis += source->shape[axis];
        }
        int inner = tensor_inner_after_axis(o, axis);
        if (total == o->numel && summed_axis == o->shape[axis] && inner > 0 &&
            metal_graph_concat_f32(inputs, sizes, input_axes, n->nin, o->data,
                                   o->shape[axis], inner)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Dropout")) {
        if (!in || in->dtype != T_F32 || in->numel != o->numel) return 0;
#if VOLVOXAI_ENABLE_TRAINING
        if (g_native_training_mode) {
            NativeDropoutParams dropout;
            if (native_dropout_params(n, idx, &dropout) != 0) return 0;
            if (metal_graph_dropout_f32(in->data, o->data, o->numel, dropout.threshold,
                                        dropout.seed, dropout.counter, dropout.scale)) {
                *node_backend = "metal-graph";
                return 1;
            }
        } else
#endif
        if (metal_graph_alias_f32(in->data, o->data, o->numel)) {
            *node_backend = "metal-graph-alias";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Reshape") || !strcmp(op, "Flatten") || !strcmp(op, "Squeeze") ||
        !strcmp(op, "Unsqueeze") || !strcmp(op, "Identity")) {
        if (in && in->dtype == T_F32 && in->numel == o->numel &&
            metal_graph_alias_f32(in->data, o->data, o->numel)) {
            *node_backend = "metal-graph-alias";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Mul") || !strcmp(op, "Sub") || !strcmp(op, "Div")) {
        if (try_gpu_graph_binary(n, in, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Conv1D")) {
        if (try_gpu_graph_conv1d(n, in, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Embedding")) {
        if (try_gpu_graph_embedding(n, in, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "MoERouter")) {
        if (try_gpu_graph_moe_router(n, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "MoELinear")) {
        if (try_gpu_graph_moe_linear(n, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SDPA")) {
        if (try_gpu_graph_sdpa(n, in, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossSDPA")) {
        if (try_gpu_graph_cross_sdpa(n, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "CrossAttention")) {
        if (try_gpu_graph_cross_attention(n, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "Split")) {
        if (try_gpu_graph_split(n, in, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "SpatialSoftargmaxY") || !strcmp(op, "ProfileX") ||
        !strcmp(op, "ProfileY") || !strcmp(op, "MeanHeight")) {
        if (try_gpu_graph_profile_op(n, in, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    if (!strcmp(op, "NonMaxSuppression")) {
        if (try_gpu_graph_nms(n, o, &g_metal_f32_ops)) {
            *node_backend = "metal-graph";
            return 1;
        }
        return 0;
    }

    return 0;
}
#endif
