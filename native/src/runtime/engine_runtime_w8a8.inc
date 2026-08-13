static int quantized_linear_f32(int node_index, Node* node, const T* input, const T* weight,
                                const T* bias, T* output, int row_start, int row_count,
                                int d_in, int d_out, int out_in, int include_bias) {
    if (!node || !input || !weight || !output ||
        (weight->dtype != T_I8 && weight->dtype != T_U8) ||
        input->dtype != T_F32 || output->dtype != T_F32 || row_start < 0 ||
        row_count <= 0 || d_in <= 0 || d_out <= 0 ||
        ((long)row_start + row_count) * d_in > input->numel ||
        ((long)row_start + row_count) * d_out > output->numel) return -1;
    T* scale = nin(node, "weight_scale");
    if (!scale) scale = nin(node, "scale");
    T* zero_point = nin(node, "weight_zero_point");
    if (!zero_point) zero_point = nin(node, "zero_point");
    if (!scale || (scale->dtype != T_F32 && scale->dtype != T_F16) ||
        (scale->numel != 1 && scale->numel != d_out) ||
        (zero_point && zero_point->numel != 1 && zero_point->numel != d_out) ||
        weight->numel != (long)d_in * d_out) return -1;
    if (zero_point && zero_point->dtype != T_I8 && zero_point->dtype != T_U8 &&
        zero_point->dtype != T_I32) return -1;
    if (bias && (bias->dtype != T_F32 || bias->numel != d_out)) return -1;
    {
        if (scale->dtype == T_F16 &&
            (size_t)scale->numel > SIZE_MAX / sizeof(float)) return -1;
        float* converted_scales = scale->dtype == T_F16 ?
            (float*)malloc((size_t)scale->numel * sizeof(float)) : NULL;
        const float* kernel_scales = scale->dtype == T_F32 ?
            scale->data : converted_scales;
        if (!kernel_scales) return -1;
        for (long index = 0; index < scale->numel; index++) {
            float value = tensor_scalar_f32(scale, index);
            if (!(value > 0.0f) || !isfinite(value)) {
                free(converted_scales);
                return -1;
            }
            if (converted_scales) converted_scales[index] = value;
        }
        for (long index = 0; zero_point && index < zero_point->numel; index++) {
            float value = tensor_scalar_any(zero_point, index);
            if (!isfinite(value) || value != truncf(value) ||
                (weight->dtype == T_I8 && (value < -128.0f || value > 127.0f)) ||
                (weight->dtype == T_U8 && (value < 0.0f || value > 255.0f))) {
                free(converted_scales);
                return -1;
            }
        }
        uint32_t zero_point_dtype = !zero_point
            ? (uint32_t)VX_DTYPE_UNSPECIFIED
            : (uint32_t)zero_point->dtype;
        int ok = 0;
        if (node_index >= 0 && node_index < MAXN &&
            volvoxai_engine_tensor_is_model_weight_locked(weight->name)) {
            uint32_t packed_bytes = vx_packed_q8_weight_size((uint32_t)d_in,
                (uint32_t)d_out);
            if (packed_bytes && !g_q8wcache[node_index]) {
                void* packed = malloc(packed_bytes);
                if (packed && vx_pack_q8_weight(packed, packed_bytes, weight->data,
                        (uint32_t)d_in, (uint32_t)d_out,
                        physical_byte_dtype(weight->dtype), out_in ? 1u : 0u)) {
                    g_q8wcache[node_index] = packed;
                    g_q8wcache_bytes[node_index] = packed_bytes;
                } else {
                    free(packed);
                }
            }
            if (g_q8wcache[node_index] &&
                g_q8wcache_bytes[node_index] == packed_bytes) {
                ok = vx_matmul_quantized_f32_packed(
                    input->data + (long)row_start * d_in, g_q8wcache[node_index],
                    kernel_scales, zero_point ? zero_point->data : NULL,
                    include_bias && bias ? bias->data : NULL,
                    output->data + (long)row_start * d_out,
                    (uint32_t)row_count, (uint32_t)d_in, (uint32_t)d_out,
                    physical_byte_dtype(weight->dtype),
                    (uint32_t)scale->numel, zero_point_dtype,
                    zero_point ? (uint32_t)zero_point->numel : 0u);
            }
        }
        if (!ok && out_in) ok = matmul_quantized_f32(
            input->data + (long)row_start * d_in, weight->data, kernel_scales,
            zero_point ? zero_point->data : NULL,
            include_bias && bias ? bias->data : NULL,
            output->data + (long)row_start * d_out,
            (uint32_t)row_count, (uint32_t)d_in, (uint32_t)d_out,
            physical_byte_dtype(weight->dtype),
            (uint32_t)scale->numel, zero_point_dtype,
            zero_point ? (uint32_t)zero_point->numel : 0u);
        free(converted_scales);
        if (ok) return 0;
        if (out_in) return -1;
    }
    for (int row = 0; row < row_count; row++) {
        const float* x = input->data + (long)(row_start + row) * d_in;
        float* y = output->data + (long)(row_start + row) * d_out;
        for (int j = 0; j < d_out; j++) {
            float ws = tensor_scalar_f32(scale, scale->numel == 1 ? 0 : j);
            float wzp_value = zero_point ? tensor_scalar_any(zero_point, zero_point->numel == 1 ? 0 : j) : 0.0f;
            if (!isfinite(wzp_value) || wzp_value != truncf(wzp_value) ||
                (weight->dtype == T_I8 && (wzp_value < -128.0f || wzp_value > 127.0f)) ||
                (weight->dtype == T_U8 && (wzp_value < 0.0f || wzp_value > 255.0f))) return -1;
            int wzp = (int)wzp_value;
            if (!(ws > 0.0f) || !isfinite(ws)) return -1;
            float sum = include_bias && bias ? tensor_scalar_f32(bias, j) : 0.0f;
            for (int k = 0; k < d_in; k++) {
                size_t wi = out_in ? (size_t)j * d_in + k : (size_t)k * d_out + j;
                int q = weight->dtype == T_I8 ? ((const int8_t*)weight->data)[wi] : ((const uint8_t*)weight->data)[wi];
                sum += x[k] * ((float)(q - wzp) * ws);
            }
            y[j] = sum;
        }
    }
    return 0;
}

/* Explicit QLinear/QMatMul/QGemm is a physical-byte path. It deliberately
 * does not call materialize_tensor_f32(): input and output T.data hold their
 * declared I8/U8 bytes for the entire operation. */
static int qlinear_physical_forward(int node_index, Node* node, T* output,
                                    int is_last) {
    QLinearMetadata* meta;
    T* input;
    T* weight;
    T* bias;
    long rows;
    int row_start, row_count;
    uint32_t input_dtype, weight_dtype, output_dtype;
    const unsigned char* input_data;
    unsigned char* output_data;
    if (node_index < 0 || node_index >= MAXN || !node || !output) return -1;
    meta = &g_qlinear_meta[node_index];
    input = qlinear_node_input(node);
    weight = nin(node, "weight");
    bias = nin(node, "bias");
    if (!meta->valid || !input || !weight || !bias || output != t_find(node->out) ||
        input->dtype != meta->input_dtype || weight->dtype != meta->weight_dtype ||
        output->dtype != meta->output_dtype || !input->quantization.valid ||
        !output->quantization.valid || input->ndim <= 0 || output->ndim != input->ndim ||
        weight->ndim != 2 || bias->dtype != T_I32 || bias->numel != meta->d_out ||
        !meta->weight_scales || !meta->weight_zero_points || !meta->bias ||
        input->shape[input->ndim - 1] != meta->d_in ||
        output->shape[output->ndim - 1] != meta->d_out ||
        weight->shape[0] != meta->d_out || weight->shape[1] != meta->d_in ||
        weight->numel != (long)meta->d_out * meta->d_in) return -1;
    for (int dimension = 0; dimension < input->ndim - 1; dimension++) {
        if (input->shape[dimension] != output->shape[dimension]) return -1;
    }
    rows = input->numel / meta->d_in;
    if (rows <= 0 || input->numel != rows * meta->d_in ||
        output->numel != rows * meta->d_out || (unsigned long)rows > UINT32_MAX) return -1;
    seq_range(rows, &row_start, &row_count);
    if (is_last && g_active_row < 0 && g_execution_row >= 0 && rows > 1) {
        row_start = g_execution_row;
        row_count = 1;
    }
    if (row_start < 0 || row_count <= 0 || (long)row_start + row_count > rows) return -1;
    input_dtype = physical_byte_dtype(input->dtype);
    weight_dtype = physical_byte_dtype(weight->dtype);
    output_dtype = physical_byte_dtype(output->dtype);
    if (!input_dtype || !weight_dtype || !output_dtype) return -1;
    input_data = (const unsigned char*)input->data +
        (size_t)row_start * (size_t)meta->d_in * input->elem_size;
    output_data = (unsigned char*)output->data +
        (size_t)row_start * (size_t)meta->d_out * output->elem_size;
    /* Immutable multi-row x86 weights use the packed N-panel microkernel.
     * It partitions output panels through the same context-owned pool and
     * avoids rescanning raw N-by-K rows. M=1 retains the raw GEMV policy;
     * ARM keeps its runtime-gated NEON/SDOT dispatcher. */
    if (vx_packed_q8_preferred_for_native_w8a8((uint32_t)row_count,
            (uint32_t)meta->d_in, (uint32_t)meta->d_out, weight_dtype,
            meta->weight_zero_all_zero) &&
        meta->packed_weight &&
        vx_qlinear_i8u8_packed(input_data, meta->packed_weight,
            meta->bias, meta->weight_scales, meta->weight_zero_points, output_data,
            (uint32_t)row_count, (uint32_t)meta->d_in, (uint32_t)meta->d_out,
            input->quantization.scale, input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            input_dtype, weight_dtype, output_dtype) == 1) return 0;
    return vx_qlinear_i8u8_native(input_data, weight->data, meta->bias, meta->weight_scales,
                                   meta->weight_zero_points, output_data, (uint32_t)row_count,
                                   (uint32_t)meta->d_in, (uint32_t)meta->d_out,
                                   input->quantization.scale, input->quantization.zero_point,
                                   output->quantization.scale, output->quantization.zero_point,
                                   input_dtype, weight_dtype, output_dtype) == 1 ? 0 : -1;
}

#define VX_RUN_W8A8(OPS, MEMBER, ...) \
    do { \
        if ((OPS) && (OPS)->MEMBER && (OPS)->MEMBER(__VA_ARGS__)) { \
            *node_backend = backend_label; \
            return 1; \
        } \
    } while (0)

/* Vulkan/OpenGL/Metal keep their established CPU boundary for prefix/row
 * execution. Strict CUDA resolves canonical sequence slices as views of the
 * complete resident allocation and never crosses a hidden CPU boundary. */
static int physical_gpu_partial_execution_declined(void) {
    return (g_active_row >= 0 || g_prefix_rows > 0) && !g_use_cuda;
}

static int physical_sequence_window(const T* tensor, uint32_t width,
                                    int* row_start, int* row_count) {
    long rows;
    if (g_active_row < 0 && g_prefix_rows <= 0) return 0;
    if (!tensor || !width || !row_start || !row_count || tensor->ndim != 3 ||
        tensor->shape[0] != 1 || tensor->shape[1] <= 0 ||
        tensor->shape[2] != (int)width || tensor->numel % (long)width)
        return -1;
    rows = tensor->numel / (long)width;
    if (rows != tensor->shape[1] || rows > INT_MAX) return -1;
    seq_range(rows, row_start, row_count);
    return *row_start >= 0 && *row_count > 0 &&
           (long)*row_start + *row_count <= rows ? 1 : -1;
}

static int qlinear_physical_gpu_forward(int node_index, Node* node, T* output,
                                        int is_last,
                                        const VxW8A8Ops* ops,
                                        const char* backend_label,
                                        const char** node_backend) {
    QLinearMetadata* meta;
    T* input;
    T* weight;
    T* bias;
    long rows;
    int row_start = 0;
    int row_count = 0;
    int partial;
    const void* input_data;
    void* output_data;
    uint32_t input_dtype, weight_dtype, output_dtype;
    if (node_index < 0 || node_index >= MAXN || !node || !output || !node_backend ||
        physical_gpu_partial_execution_declined()) return 0;
    meta = &g_qlinear_meta[node_index];
    input = qlinear_node_input(node);
    weight = nin(node, "weight");
    bias = nin(node, "bias");
    if (!meta->valid || !input || !weight || !bias || output != t_find(node->out) ||
        input->dtype != meta->input_dtype || weight->dtype != meta->weight_dtype ||
        output->dtype != meta->output_dtype || !input->quantization.valid ||
        !output->quantization.valid || input->ndim <= 0 || output->ndim != input->ndim ||
        weight->ndim != 2 || bias->dtype != T_I32 || bias->numel != meta->d_out ||
        !meta->weight_scales || !meta->weight_zero_points || !meta->bias ||
        input->shape[input->ndim - 1] != meta->d_in ||
        output->shape[output->ndim - 1] != meta->d_out ||
        weight->shape[0] != meta->d_out || weight->shape[1] != meta->d_in ||
        weight->numel != (long)meta->d_out * meta->d_in || !input->data ||
        !weight->data || !output->data) return 0;
    for (int dimension = 0; dimension < input->ndim - 1; dimension++) {
        if (input->shape[dimension] != output->shape[dimension]) return 0;
    }
    rows = input->numel / meta->d_in;
    if (rows <= 0 || input->numel != rows * meta->d_in ||
        output->numel != rows * meta->d_out || rows > INT_MAX) return 0;
    partial = physical_sequence_window(input, (uint32_t)meta->d_in,
                                       &row_start, &row_count);
    if (partial < 0) return 0;
    if (!partial) row_count = (int)rows;
    if (partial && is_last && g_active_row < 0 && g_execution_row >= 0 &&
        rows > 1) {
        if (g_execution_row >= rows) return 0;
        row_start = g_execution_row;
        row_count = 1;
    }
    input_data = (const unsigned char*)input->data +
        (size_t)row_start * (size_t)meta->d_in * input->elem_size;
    output_data = (unsigned char*)output->data +
        (size_t)row_start * (size_t)meta->d_out * output->elem_size;
    input_dtype = physical_byte_dtype(input->dtype);
    weight_dtype = physical_byte_dtype(weight->dtype);
    output_dtype = physical_byte_dtype(output->dtype);
    if (!input_dtype || !weight_dtype || !output_dtype) return 0;
    if (ops && ops->qlinear_i8u8 && ops->qlinear_i8u8(
            input_data, weight->data, meta->weight_scales, meta->weight_zero_points,
            meta->bias, output_data, (uint32_t)row_count, (uint32_t)meta->d_in,
            (uint32_t)meta->d_out, input->quantization.scale,
            input->quantization.zero_point, output->quantization.scale,
            output->quantization.zero_point, input_dtype, weight_dtype, output_dtype)) {
        *node_backend = backend_label;
        return 1;
    }
    return 0;
}

typedef struct {
    QEmbeddingMetadata* meta;
    T* tokens;
    T* weight;
    uint32_t tokens_count;
    uint32_t weight_dtype;
    uint32_t output_dtype;
} QEmbeddingPhysicalDescriptor;

static int qembedding_physical_validate(int node_index, Node* node, T* output,
                                        int validate_host_values,
                                        QEmbeddingPhysicalDescriptor* descriptor) {
    QEmbeddingMetadata* meta;
    T* tokens;
    T* weight;
    uint64_t weight_elements;
    uint64_t output_elements;
    uint32_t weight_dtype;
    uint32_t output_dtype;
    if (node_index < 0 || node_index >= MAXN || !node || !output || !descriptor) return -1;
    meta = &g_qembedding_meta[node_index];
    tokens = qembedding_node_input(node);
    weight = nin(node, "weight");
    if (!meta->valid || !tokens || !weight || output != t_find(node->out) ||
        !tokens->data || !weight->data || !output->data ||
        tokens->dtype != T_I32 || tokens->elem_size != sizeof(int32_t) ||
        tokens->ndim <= 0 || tokens->numel <= 0 || tokens->numel > (long)UINT32_MAX ||
        weight->dtype != meta->weight_dtype || weight->elem_size != 1 || weight->ndim != 2 ||
        output->dtype != meta->output_dtype || output->elem_size != 1 ||
        output->ndim != tokens->ndim + 1 || !output->quantization.valid ||
        !isfinite(output->quantization.scale) || output->quantization.scale <= 0.0f ||
        !meta->vocab || !meta->hidden || !meta->weight_scales ||
        !meta->weight_zero_points || weight->shape[0] != (int)meta->vocab ||
        weight->shape[1] != (int)meta->hidden ||
        output->shape[tokens->ndim] != (int)meta->hidden) return -1;
    for (int dimension = 0; dimension < tokens->ndim; dimension++) {
        if (tokens->shape[dimension] <= 0 || output->shape[dimension] != tokens->shape[dimension])
            return -1;
    }
    weight_elements = (uint64_t)meta->vocab * meta->hidden;
    output_elements = (uint64_t)tokens->numel * meta->hidden;
    if (weight_elements > (uint64_t)LONG_MAX || output_elements > (uint64_t)LONG_MAX ||
        weight->numel != (long)weight_elements || output->numel != (long)output_elements)
        return -1;
    if (validate_host_values)
        for (long token = 0; token < tokens->numel; token++) {
            int32_t id;
            memcpy(&id,
                   (const unsigned char*)tokens->data +
                       (size_t)token * sizeof(id),
                   sizeof(id));
            if (id < 0 || (uint32_t)id >= meta->vocab) return -1;
        }
    weight_dtype = physical_byte_dtype(weight->dtype);
    output_dtype = physical_byte_dtype(output->dtype);
    if (!weight_dtype || !output_dtype ||
        output->quantization.zero_point < (output->dtype == T_I8 ? -128 : 0) ||
        output->quantization.zero_point > (output->dtype == T_I8 ? 127 : 255)) return -1;
    descriptor->meta = meta;
    descriptor->tokens = tokens;
    descriptor->weight = weight;
    descriptor->tokens_count = (uint32_t)tokens->numel;
    descriptor->weight_dtype = weight_dtype;
    descriptor->output_dtype = output_dtype;
    return 0;
}

/* QEmbedding gathers directly from an I8/U8 table. No F32 materialization
 * occurs between the table and output tensor. */
static int qembedding_physical_forward(int node_index, Node* node, T* output) {
    QEmbeddingPhysicalDescriptor descriptor;
    const int32_t* tokens;
    void* output_data;
    uint32_t token_count;
    if (qembedding_physical_validate(
            node_index, node, output, 1, &descriptor) != 0) return -1;
    tokens = (const int32_t*)descriptor.tokens->data;
    output_data = output->data;
    token_count = descriptor.tokens_count;
    if (g_active_row >= 0 ||
        (g_prefix_rows > 0 &&
         descriptor.tokens->numel == g_prefix_row_capacity)) {
        int row_start;
        int row_count;
        if (descriptor.tokens->ndim != 2 || descriptor.tokens->shape[0] != 1 ||
            output->ndim != 3 || output->shape[0] != 1) return -1;
        seq_range(descriptor.tokens->numel, &row_start, &row_count);
        if (row_start < 0 || row_count <= 0 ||
            (long)row_start + row_count > descriptor.tokens->numel)
            return -1;
        tokens += row_start;
        output_data = (unsigned char*)output_data +
            (size_t)row_start * descriptor.meta->hidden * output->elem_size;
        token_count = (uint32_t)row_count;
    }
    return qembedding_i8u8(tokens,
                           descriptor.weight->data, descriptor.meta->weight_scales,
                           descriptor.meta->weight_zero_points, output_data,
                           token_count, descriptor.meta->vocab,
                           descriptor.meta->hidden, output->quantization.scale,
                           output->quantization.zero_point, descriptor.weight_dtype,
                           descriptor.output_dtype) == 1 ? 0 : -1;
}

/* Keep canonical byte gather resident on the selected device. CUDA token and
 * output slices resolve into the full allocations registered at prefix start. */
static int qembedding_physical_gpu_forward(int node_index, Node* node, T* output,
                                           const VxW8A8Ops* ops,
                                           const char* backend_label,
                                           const char** node_backend) {
    QEmbeddingPhysicalDescriptor descriptor;
    const int32_t* tokens;
    void* output_data;
    uint32_t token_count;
    int row_start = 0;
    int row_count = 0;
    if (!node_backend || physical_gpu_partial_execution_declined() ||
        qembedding_physical_validate(
            node_index, node, output,
            !g_bounded_gpu_value_domain_proven, &descriptor) != 0) return 0;
    tokens = (const int32_t*)descriptor.tokens->data;
    output_data = output->data;
    token_count = descriptor.tokens_count;
    if (g_active_row >= 0 ||
        (g_prefix_rows > 0 &&
         descriptor.tokens->numel == g_prefix_row_capacity)) {
        if (descriptor.tokens->ndim != 2 || descriptor.tokens->shape[0] != 1 ||
            output->ndim != 3 || output->shape[0] != 1) return 0;
        seq_range(descriptor.tokens->numel, &row_start, &row_count);
        if (row_start < 0 || row_count <= 0 ||
            (long)row_start + row_count > descriptor.tokens->numel) return 0;
        tokens += row_start;
        output_data = (unsigned char*)output_data +
            (size_t)row_start * descriptor.meta->hidden * output->elem_size;
        token_count = (uint32_t)row_count;
    }
    if (ops && ops->qembedding_i8u8 && ops->qembedding_i8u8(
            tokens, descriptor.weight->data,
            descriptor.meta->weight_scales, descriptor.meta->weight_zero_points,
            output_data, token_count, descriptor.meta->vocab,
            descriptor.meta->hidden, output->quantization.scale,
            output->quantization.zero_point, descriptor.weight_dtype,
            descriptor.output_dtype)) {
        *node_backend = backend_label;
        return 1;
    }
    return 0;
}

typedef struct {
    QConv2DMetadata* meta;
    T* input;
    T* weight;
    uint32_t batch;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t input_dtype;
    uint32_t weight_dtype;
    uint32_t output_dtype;
} QConv2DPhysicalDescriptor;

/* QConv packing and affine metadata are invariant, but activation geometry is
 * a property of the current BoundExecutionGraph.  Never compare a rebound
 * tensor against the bootstrap shape cached while the package was loaded. */
static int qconv2d_physical_validate(
        int node_index, Node* node, T* output,
        QConv2DPhysicalDescriptor* descriptor) {
    QConv2DMetadata* meta;
    T* input;
    T* weight;
    T* bias;
    uint64_t effective_height;
    uint64_t effective_width;
    uint64_t padded_height;
    uint64_t padded_width;
    uint64_t expected_height;
    uint64_t expected_width;
    uint64_t input_elements;
    uint64_t output_elements;
    if (node_index < 0 || node_index >= MAXN || !node || !output ||
        !descriptor) return -1;
    meta = &g_qconv_meta[node_index];
    input = qconv2d_node_input(node);
    weight = nin(node, "weight");
    bias = nin(node, "bias");
    if (!meta->valid || !input || !weight || output != t_find(node->out) ||
        !input->data || !weight->data || !output->data ||
        input->dtype != meta->input_dtype || weight->dtype != meta->weight_dtype ||
        output->dtype != meta->output_dtype || !input->quantization.valid ||
        !output->quantization.valid || input->elem_size != 1 ||
        weight->elem_size != 1 || output->elem_size != 1 ||
        input->ndim != 4 || weight->ndim != 4 || output->ndim != 4 ||
        input->shape[0] <= 0 || input->shape[1] <= 0 ||
        input->shape[2] <= 0 || input->shape[3] != (int)meta->input_channels ||
        output->shape[0] != input->shape[0] || output->shape[1] <= 0 ||
        output->shape[2] <= 0 ||
        output->shape[3] != (int)meta->output_channels ||
        weight->shape[0] != (int)meta->output_channels ||
        weight->shape[1] != (int)meta->kernel_height ||
        weight->shape[2] != (int)meta->kernel_width ||
        weight->shape[3] != (int)meta->input_per_group ||
        !meta->weight_scales || !meta->weight_zero_points ||
        !meta->stride_y || !meta->stride_x || !meta->dilation_y ||
        !meta->dilation_x || !meta->kernel_height || !meta->kernel_width ||
        !meta->groups || ((bias != NULL) != (meta->bias != NULL)) ||
        (bias && (bias->dtype != T_I32 || bias->ndim != 1 ||
                  bias->shape[0] != (int)meta->output_channels ||
                  bias->numel != (long)meta->output_channels))) return -1;
    effective_height = (uint64_t)(meta->kernel_height - 1u) *
        meta->dilation_y + 1u;
    effective_width = (uint64_t)(meta->kernel_width - 1u) *
        meta->dilation_x + 1u;
    padded_height = (uint64_t)(uint32_t)input->shape[1] +
        meta->padding_top + meta->padding_bottom;
    padded_width = (uint64_t)(uint32_t)input->shape[2] +
        meta->padding_left + meta->padding_right;
    if (padded_height < effective_height || padded_width < effective_width)
        return -1;
    expected_height =
        (padded_height - effective_height) / meta->stride_y + 1u;
    expected_width =
        (padded_width - effective_width) / meta->stride_x + 1u;
    if (expected_height != (uint32_t)output->shape[1] ||
        expected_width != (uint32_t)output->shape[2]) return -1;
    input_elements = (uint32_t)input->shape[0];
    if (input_elements > UINT64_MAX / (uint32_t)input->shape[1]) return -1;
    input_elements *= (uint32_t)input->shape[1];
    if (input_elements > UINT64_MAX / (uint32_t)input->shape[2]) return -1;
    input_elements *= (uint32_t)input->shape[2];
    if (input_elements > UINT64_MAX / meta->input_channels) return -1;
    input_elements *= meta->input_channels;
    output_elements = (uint32_t)output->shape[0];
    if (output_elements > UINT64_MAX / (uint32_t)output->shape[1]) return -1;
    output_elements *= (uint32_t)output->shape[1];
    if (output_elements > UINT64_MAX / (uint32_t)output->shape[2]) return -1;
    output_elements *= (uint32_t)output->shape[2];
    if (output_elements > UINT64_MAX / meta->output_channels) return -1;
    output_elements *= meta->output_channels;
    if (input_elements > (uint64_t)LONG_MAX ||
        output_elements > (uint64_t)LONG_MAX ||
        input->numel != (long)input_elements ||
        output->numel != (long)output_elements) return -1;
    descriptor->meta = meta;
    descriptor->input = input;
    descriptor->weight = weight;
    descriptor->batch = (uint32_t)input->shape[0];
    descriptor->input_height = (uint32_t)input->shape[1];
    descriptor->input_width = (uint32_t)input->shape[2];
    descriptor->output_height = (uint32_t)output->shape[1];
    descriptor->output_width = (uint32_t)output->shape[2];
    descriptor->input_dtype = physical_byte_dtype(input->dtype);
    descriptor->weight_dtype = physical_byte_dtype(weight->dtype);
    descriptor->output_dtype = physical_byte_dtype(output->dtype);
    return descriptor->input_dtype && descriptor->weight_dtype &&
        descriptor->output_dtype ? 0 : -1;
}

/* Canonical QConv2D is the matching physical-byte dense spatial path. */
static int qconv2d_physical_forward(int node_index, Node* node, T* output) {
    QConv2DPhysicalDescriptor descriptor;
    QConv2DMetadata* meta;
    if (qconv2d_physical_validate(
            node_index, node, output, &descriptor) != 0) return -1;
    meta = descriptor.meta;
    return vx_qconv2d_i8u8_native_prepacked(
            descriptor.input->data, descriptor.weight->data, meta->bias,
            meta->weight_scales, meta->weight_zero_points, output->data,
            descriptor.batch, descriptor.input_height, descriptor.input_width,
            meta->input_channels, descriptor.output_height,
            descriptor.output_width, meta->output_channels,
            meta->kernel_height, meta->kernel_width, meta->input_per_group,
            meta->stride_y, meta->stride_x, meta->dilation_y, meta->dilation_x,
            meta->padding_top, meta->padding_left, meta->padding_bottom,
            meta->padding_right, meta->groups, meta->relu,
            descriptor.input->quantization.scale,
            descriptor.input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            descriptor.input_dtype, descriptor.weight_dtype,
            descriptor.output_dtype, meta->packed_weight,
            meta->small_c_packed_weight) == 1 ? 0 : -1;
}

/* Keep the canonical QConv2D activation chain on its selected device.  This
 * deliberately mirrors the CPU descriptor validation above, but returns 0 on
 * an unavailable backend so the already-strict CPU physical path can run. */
static int qconv2d_physical_gpu_forward(int node_index, Node* node, T* output,
                                        const VxW8A8Ops* ops,
                                        const char* backend_label,
                                        const char** node_backend) {
    QConv2DPhysicalDescriptor descriptor;
    QConv2DMetadata* meta;
    if (!node_backend || physical_gpu_partial_execution_declined() ||
        qconv2d_physical_validate(
            node_index, node, output, &descriptor) != 0) return 0;
    meta = descriptor.meta;
    if (ops && ops->qconv2d_i8u8 && ops->qconv2d_i8u8(
            descriptor.input->data, descriptor.weight->data,
            meta->weight_scales, meta->weight_zero_points,
            meta->bias, output->data, descriptor.batch,
            descriptor.input_height, descriptor.input_width,
            meta->input_channels, descriptor.output_height,
            descriptor.output_width, meta->output_channels, meta->kernel_height,
            meta->kernel_width, meta->input_per_group, meta->stride_y, meta->stride_x,
            meta->dilation_y, meta->dilation_x, meta->padding_top, meta->padding_left,
            meta->padding_bottom, meta->padding_right, meta->groups, meta->relu,
            descriptor.input->quantization.scale,
            descriptor.input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            descriptor.input_dtype, descriptor.weight_dtype,
            descriptor.output_dtype)) {
        *node_backend = backend_label;
        return 1;
    }
    return 0;
}

static T* physical_node_data_input(Node* node) {
    T* input = nin(node, "input");
    if (!input) input = nin(node, "x");
    if (!input) input = nin(node, "data");
    return input;
}

static int physical_byte_tensor(const T* tensor, uint32_t* dtype_out) {
    uint32_t dtype;
    if (!tensor || !tensor->data || tensor->numel <= 0 || tensor->numel > UINT32_MAX ||
        tensor->elem_size != 1 || !tensor->quantization.valid ||
        !isfinite(tensor->quantization.scale) || tensor->quantization.scale <= 0.0f ||
        (tensor->dtype == T_I8 && (tensor->quantization.zero_point < -128 ||
                                   tensor->quantization.zero_point > 127)) ||
        (tensor->dtype == T_U8 && (tensor->quantization.zero_point < 0 ||
                                   tensor->quantization.zero_point > 255))) return -1;
    dtype = physical_byte_dtype(tensor->dtype);
    if (!dtype) return -1;
    if (dtype_out) *dtype_out = dtype;
    return 0;
}

static int physical_same_shape(const T* left, const T* right) {
    if (!left || !right || left->ndim != right->ndim || left->numel != right->numel) return 0;
    for (int dimension = 0; dimension < left->ndim; dimension++) {
        if (left->shape[dimension] != right->shape[dimension]) return 0;
    }
    return 1;
}

/* Physical byte tensors can share a mapped SafeTensors blob or a planned
 * activation arena.  Pointer equality alone is therefore not sufficient for
 * a fail-closed output-alias check.  Treat an unrepresentable byte range as
 * overlap so validation never permits a partially overwritten source. */
static int physical_tensor_ranges_overlap(const T* left, const T* right) {
    size_t left_bytes;
    size_t right_bytes;
    uintptr_t left_start;
    uintptr_t right_start;
    if (!left || !right || !left->data || !right->data || left->numel <= 0 ||
        right->numel <= 0 || left->elem_size == 0 || right->elem_size == 0 ||
        (uint64_t)left->numel > SIZE_MAX / left->elem_size ||
        (uint64_t)right->numel > SIZE_MAX / right->elem_size) return 1;
    left_bytes = (size_t)left->numel * left->elem_size;
    right_bytes = (size_t)right->numel * right->elem_size;
    left_start = (uintptr_t)left->data;
    right_start = (uintptr_t)right->data;
    if (left_start > UINTPTR_MAX - left_bytes ||
        right_start > UINTPTR_MAX - right_bytes) return 1;
    return left_start < right_start + right_bytes &&
        right_start < left_start + left_bytes;
}

static int physical_relu_parameter(cJSON* params, uint32_t* relu) {
    cJSON* value = params ? cJSON_GetObjectItem(params, "relu") : NULL;
    if (!relu) return -1;
    if (!value || cJSON_IsNull(value)) {
        *relu = 0;
        return 0;
    }
    if (qconv2d_parse_u32(value, 0, relu) != 0 || *relu > 2) return -1;
    return 0;
}

/* QSiLU has no tunable attributes.  Keep its canonical byte route closed to
 * unknown parameters rather than silently accepting a configuration whose
 * requested activation semantics differ from SiLU. */
static int physical_no_parameters(const cJSON* params) {
    if (!params || cJSON_IsNull(params)) return 0;
    return cJSON_IsObject(params) && params->child == NULL ? 0 : -1;
}

typedef struct {
    uint32_t matrix_count;
    size_t a_matrix_elements;
    size_t b_matrix_elements;
    size_t output_matrix_elements;
} QBatchMatMulPhysicalRowDescriptor;

/* Incremental decoder execution computes one sequence row while retaining
 * every other row from the seed pass. Rank-3 accepts only a cached right
 * matrix. Rank-4 additionally covers the two canonical attention layouts:
 * [1,H,S,D] x [1,H,D,S] (the current key is a column) and
 * [1,H,S,S] x [1,H,S,D] (the current value is a row). A clean rank-4 right
 * matrix also covers cross attention. General ONNX batch/head broadcasting
 * remains a whole-tensor operation, and A/B aliasing is never a row route. */
static int qbatch_matmul_physical_row_validate(
        const QBatchMatMulPhysicalDescriptor* descriptor, int row,
        int b_dirty, QBatchMatMulPhysicalRowDescriptor* row_descriptor) {
    const T* a;
    const T* b;
    const T* output;
    QBatchMatMulPhysicalRowDescriptor parsed = {0};
    if (!descriptor || !row_descriptor || row < 0) return -1;
    a = descriptor->a;
    b = descriptor->b;
    output = descriptor->output;
    if (!a || !b || !output || physical_tensor_ranges_overlap(a, b)) return -1;
    if (a->ndim == 3) {
        if (b_dirty || a->shape[0] != 1 || a->shape[1] <= row ||
            a->shape[2] <= 0 || output->ndim != 3 ||
            output->shape[0] != 1 || output->shape[1] != a->shape[1] ||
            output->shape[1] <= row || output->shape[2] <= 0 ||
            descriptor->output_batch_count != 1u ||
            descriptor->m != (uint32_t)a->shape[1] ||
            descriptor->k != (uint32_t)a->shape[2] ||
            descriptor->n != (uint32_t)output->shape[2] ||
            !((b->ndim == 2 &&
               b->shape[0] == (int)descriptor->k &&
               b->shape[1] == (int)descriptor->n) ||
              (b->ndim == 3 && b->shape[0] == 1 &&
               b->shape[1] == (int)descriptor->k &&
               b->shape[2] == (int)descriptor->n)))
            return -1;
        parsed.matrix_count = 1u;
    } else if (a->ndim == 4) {
        int b_k_is_sequence;
        int b_n_is_sequence;
        if (b->ndim != 4 || output->ndim != 4 ||
            a->shape[0] != 1 || b->shape[0] != 1 ||
            output->shape[0] != 1 || a->shape[1] <= 0 ||
            b->shape[1] != a->shape[1] ||
            output->shape[1] != a->shape[1] ||
            a->shape[2] <= row || output->shape[2] != a->shape[2] ||
            output->shape[2] <= row || a->shape[3] <= 0 ||
            b->shape[2] != a->shape[3] || b->shape[3] <= 0 ||
            output->shape[3] != b->shape[3] ||
            descriptor->output_batch_count != (uint32_t)a->shape[1] ||
            descriptor->m != (uint32_t)a->shape[2] ||
            descriptor->k != (uint32_t)a->shape[3] ||
            descriptor->n != (uint32_t)b->shape[3])
            return -1;
        b_k_is_sequence = b->shape[2] == a->shape[2];
        b_n_is_sequence = b->shape[3] == a->shape[2];
        if (b_dirty && (b_k_is_sequence == b_n_is_sequence)) return -1;
        parsed.matrix_count = (uint32_t)a->shape[1];
    } else {
        return -1;
    }
    if ((size_t)descriptor->m > SIZE_MAX / descriptor->k ||
        (size_t)descriptor->k > SIZE_MAX / descriptor->n ||
        (size_t)descriptor->m > SIZE_MAX / descriptor->n)
        return -1;
    parsed.a_matrix_elements = (size_t)descriptor->m * descriptor->k;
    parsed.b_matrix_elements = (size_t)descriptor->k * descriptor->n;
    parsed.output_matrix_elements = (size_t)descriptor->m * descriptor->n;
    *row_descriptor = parsed;
    return 0;
}

static int qbatch_matmul_physical_forward(Node* node, T* output) {
    QBatchMatMulPhysicalDescriptor descriptor;
    uint32_t coordinates[6] = {0};
    size_t a_matrix_elements;
    size_t b_matrix_elements;
    size_t output_matrix_elements;
    size_t workspace_capacity = 0u;
    void* workspace = volvoxai_engine_cpu_typed_workspace(
        &workspace_capacity);
    if (qbatch_matmul_physical_validate(node, output, &descriptor) != 0)
        return -1;
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        const int b_index = t_index(descriptor.b);
        int row_start;
        int row_count;
        QBatchMatMulPhysicalRowDescriptor row_descriptor;
        seq_range(descriptor.m, &row_start, &row_count);
        if (g_active_row < 0 && row_count == (int)descriptor.m)
            goto full_matrix;
        if (qbatch_matmul_physical_row_validate(
                &descriptor, row_start,
                g_active_row >= 0 && b_index >= 0 && b_index < g_nt
                    ? g_dirty[b_index] : 0,
                &row_descriptor) != 0 ||
            b_index < 0 || b_index >= g_nt || row_count <= 0 ||
            (uint32_t)row_start + (uint32_t)row_count > descriptor.m)
            return -1;
        for (uint32_t matrix = 0; matrix < row_descriptor.matrix_count;
             matrix++) {
            size_t a_row_offset =
                (size_t)matrix * row_descriptor.a_matrix_elements +
                (size_t)row_start * descriptor.k;
            size_t b_matrix_offset =
                (size_t)matrix * row_descriptor.b_matrix_elements;
            size_t output_row_offset =
                (size_t)matrix * row_descriptor.output_matrix_elements +
                (size_t)row_start * descriptor.n;
            if (!vx_qbatch_matmul_i8u8_native_with_workspace(
                    (const uint8_t*)descriptor.a->data + a_row_offset,
                    (const uint8_t*)descriptor.b->data + b_matrix_offset,
                    (uint8_t*)descriptor.output->data + output_row_offset,
                    (uint32_t)row_count, descriptor.k, descriptor.n,
                    descriptor.a->quantization.scale,
                    descriptor.a->quantization.zero_point,
                    descriptor.b->quantization.scale,
                    descriptor.b->quantization.zero_point,
                    descriptor.output->quantization.scale,
                    descriptor.output->quantization.zero_point,
                    descriptor.a_dtype, descriptor.b_dtype,
                    descriptor.output_dtype,
                    workspace, workspace_capacity))
                return -1;
        }
        return 0;
    }
full_matrix:
    a_matrix_elements = (size_t)descriptor.m * descriptor.k;
    b_matrix_elements = (size_t)descriptor.k * descriptor.n;
    output_matrix_elements = (size_t)descriptor.m * descriptor.n;
    for (uint32_t batch = 0; batch < descriptor.output_batch_count; batch++) {
        uint32_t remainder = batch;
        size_t a_batch = 0u;
        size_t b_batch = 0u;
        int a_offset =
            (int)descriptor.batch_rank - (descriptor.a->ndim - 2);
        int b_offset =
            (int)descriptor.batch_rank - (descriptor.b->ndim - 2);
        for (int dimension = (int)descriptor.batch_rank - 1;
             dimension >= 0; dimension--) {
            uint32_t extent =
                (uint32_t)descriptor.output->shape[dimension];
            coordinates[dimension] = remainder % extent;
            remainder /= extent;
        }
        for (int dimension = 0; dimension < descriptor.a->ndim - 2;
             dimension++) {
            uint32_t extent = (uint32_t)descriptor.a->shape[dimension];
            uint32_t coordinate =
                extent == 1u ? 0u : coordinates[a_offset + dimension];
            a_batch = a_batch * extent + coordinate;
        }
        for (int dimension = 0; dimension < descriptor.b->ndim - 2;
             dimension++) {
            uint32_t extent = (uint32_t)descriptor.b->shape[dimension];
            uint32_t coordinate =
                extent == 1u ? 0u : coordinates[b_offset + dimension];
            b_batch = b_batch * extent + coordinate;
        }
        if (!vx_qbatch_matmul_i8u8_native_with_workspace(
                (const uint8_t*)descriptor.a->data +
                    a_batch * a_matrix_elements,
                (const uint8_t*)descriptor.b->data +
                    b_batch * b_matrix_elements,
                (uint8_t*)descriptor.output->data +
                    (size_t)batch * output_matrix_elements,
                descriptor.m, descriptor.k, descriptor.n,
                descriptor.a->quantization.scale,
                descriptor.a->quantization.zero_point,
                descriptor.b->quantization.scale,
                descriptor.b->quantization.zero_point,
                descriptor.output->quantization.scale,
                descriptor.output->quantization.zero_point,
                descriptor.a_dtype, descriptor.b_dtype,
                descriptor.output_dtype,
                workspace, workspace_capacity))
            return -1;
    }
    return 0;
}

static int qbatch_matmul_physical_gpu_forward(
        Node* node, T* output, const VxW8A8Ops* ops,
        const char* backend_label, const char** node_backend) {
    QBatchMatMulPhysicalDescriptor descriptor;
    if (!ops || !ops->qbatch_matmul_i8u8 || !backend_label ||
        !node_backend ||
        qbatch_matmul_physical_validate(node, output, &descriptor) != 0)
        return 0;
    if (!ops->qbatch_matmul_i8u8(
            descriptor.a->data,
            descriptor.a->shape, descriptor.a->ndim,
            descriptor.a->quantization.scale,
            descriptor.a->quantization.zero_point,
            descriptor.a_dtype,
            descriptor.b->data,
            descriptor.b->shape, descriptor.b->ndim,
            descriptor.b->quantization.scale,
            descriptor.b->quantization.zero_point,
            descriptor.b_dtype,
            descriptor.output->data,
            descriptor.output->shape, descriptor.output->ndim,
            descriptor.output->quantization.scale,
            descriptor.output->quantization.zero_point,
            descriptor.output_dtype))
        return 0;
    *node_backend = backend_label;
    return 1;
}

/* The QGELU ABI is permanently the fixed A-S erf form.  Accept the compact
 * parameter-free spelling and the explicit graph spelling `{approximate:
 * "none"}`, but reject tanh, duplicate keys, and every unrelated parameter. */
static int physical_qgelu_parameters(const cJSON* params) {
    const cJSON* item;
    if (!params || cJSON_IsNull(params)) return 0;
    if (!cJSON_IsObject(params)) return -1;
    item = params->child;
    if (!item) return 0;
    if (item->next || !item->string || strcmp(item->string, "approximate") ||
        !cJSON_IsString(item) || !item->valuestring || strcmp(item->valuestring, "none"))
        return -1;
    return 0;
}

/* QGroupNorm owns a deliberately narrow, typed contract.  Iterate children
 * rather than using cJSON_GetObjectItem() so duplicate keys cannot silently
 * choose one meaning.  Null eps/layout follow the JS `??` spelling; the
 * required num_groups never has a default. */
static int physical_qgroupnorm_parameters(const cJSON* params, uint32_t* groups_out,
                                          float* epsilon_out) {
    const cJSON* item;
    int saw_groups = 0;
    int saw_epsilon = 0;
    int saw_layout = 0;
    uint32_t groups = 0;
    float epsilon = 1.0e-5f;
    if (!params || !groups_out || !epsilon_out || !cJSON_IsObject(params)) return -1;
    for (item = params->child; item; item = item->next) {
        if (!item->string) return -1;
        if (!strcmp(item->string, "num_groups")) {
            double value;
            if (saw_groups || !cJSON_IsNumber(item)) return -1;
            saw_groups = 1;
            value = item->valuedouble;
            if (!isfinite(value) || value != trunc(value) || value < 1.0 ||
                value > (double)UINT32_MAX) return -1;
            groups = (uint32_t)value;
        } else if (!strcmp(item->string, "eps")) {
            float value;
            if (saw_epsilon) return -1;
            saw_epsilon = 1;
            if (cJSON_IsNull(item)) continue;
            if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
                item->valuedouble <= 0.0) return -1;
            value = (float)item->valuedouble;
            if (!isfinite(value) || value <= 0.0f) return -1;
            epsilon = value;
        } else if (!strcmp(item->string, "data_layout")) {
            if (saw_layout) return -1;
            saw_layout = 1;
            if (cJSON_IsNull(item)) continue;
            if (!cJSON_IsString(item) || !item->valuestring ||
                strcmp(item->valuestring, "NHWC")) return -1;
        } else {
            return -1;
        }
    }
    if (!saw_groups) return -1;
    *groups_out = groups;
    *epsilon_out = epsilon;
    return 0;
}

/* QLayerNorm is last-axis only.  It has no layout or group attributes: eps
 * defaults exactly as the portable/browser contract does, and an optional
 * d_model is merely a strict consistency assertion for the final dimension. */
static int physical_qlayernorm_parameters(const cJSON* params, uint32_t d_model,
                                          float* epsilon_out) {
    const cJSON* item;
    int saw_epsilon = 0;
    int saw_d_model = 0;
    float epsilon = 1.0e-5f;
    if (!d_model || !epsilon_out) return -1;
    if (!params || cJSON_IsNull(params)) {
        *epsilon_out = epsilon;
        return 0;
    }
    if (!cJSON_IsObject(params)) return -1;
    for (item = params->child; item; item = item->next) {
        if (!item->string) return -1;
        if (!strcmp(item->string, "eps")) {
            float value;
            if (saw_epsilon) return -1;
            saw_epsilon = 1;
            if (cJSON_IsNull(item)) continue;
            if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
                item->valuedouble <= 0.0) return -1;
            value = (float)item->valuedouble;
            if (!isfinite(value) || value <= 0.0f) return -1;
            epsilon = value;
        } else if (!strcmp(item->string, "d_model")) {
            double value;
            if (saw_d_model) return -1;
            saw_d_model = 1;
            if (cJSON_IsNull(item)) continue;
            if (!cJSON_IsNumber(item)) return -1;
            value = item->valuedouble;
            if (!isfinite(value) || value != trunc(value) || value < 1.0 ||
                value > (double)UINT32_MAX || (uint32_t)value != d_model) return -1;
        } else {
            return -1;
        }
    }
    *epsilon_out = epsilon;
    return 0;
}

/* QSDPA accepts only its three attention attributes. Heads and causal are
 * explicit because quantized attention must not inherit the F32 SDPA
 * defaults; scale alone follows the optional `1/sqrt(head_dim)` contract. */
static int physical_qsdpa_parameters(const cJSON* params, uint32_t d_model,
                                     uint32_t* heads_out, uint32_t* causal_out,
                                     float* scale_out) {
    const cJSON* item;
    int saw_heads = 0;
    int saw_causal = 0;
    int saw_scale = 0;
    uint32_t heads = 0;
    uint32_t causal = 0;
    float scale = 0.0f;
    uint32_t head_dim;
    if (!params || !d_model || !heads_out || !causal_out || !scale_out ||
        !cJSON_IsObject(params)) return -1;
    for (item = params->child; item; item = item->next) {
        if (!item->string) return -1;
        if (!strcmp(item->string, "heads")) {
            double value;
            if (saw_heads || !cJSON_IsNumber(item)) return -1;
            saw_heads = 1;
            value = item->valuedouble;
            if (!isfinite(value) || value != trunc(value) || value < 1.0 ||
                value > (double)UINT32_MAX) return -1;
            heads = (uint32_t)value;
        } else if (!strcmp(item->string, "causal")) {
            if (saw_causal || !cJSON_IsBool(item)) return -1;
            saw_causal = 1;
            causal = cJSON_IsTrue(item) ? 1u : 0u;
        } else if (!strcmp(item->string, "scale")) {
            float value;
            if (saw_scale) return -1;
            saw_scale = 1;
            if (cJSON_IsNull(item)) continue;
            if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
                item->valuedouble <= 0.0) return -1;
            value = (float)item->valuedouble;
            if (!isfinite(value) || value <= 0.0f) return -1;
            scale = value;
        } else {
            return -1;
        }
    }
    if (!saw_heads || !saw_causal || d_model % heads || d_model % 4u) return -1;
    head_dim = d_model / heads;
    if (!head_dim || head_dim % 4u || head_dim > 64u) return -1;
    if (!saw_scale || scale == 0.0f) scale = 1.0f / sqrtf((float)head_dim);
    if (!isfinite(scale) || scale <= 0.0f) return -1;
    *heads_out = heads;
    *causal_out = causal;
    *scale_out = scale;
    return 0;
}

/* QArgMax intentionally has exactly one required attribute. Normalizing here
 * keeps negative-axis handling out of CPU/GPU kernels and gives every backend
 * the same [outer, axis, inner] descriptor. */
static int physical_qargmax_axis(const cJSON* params, int rank,
                                 uint32_t* axis_out) {
    const cJSON* item;
    int saw_axis = 0;
    int axis = 0;
    if (!params || !axis_out || rank < 2 || rank > 8 ||
        !cJSON_IsObject(params)) return -1;
    for (item = params->child; item; item = item->next) {
        double value;
        if (!item->string || strcmp(item->string, "axis")) return -1;
        if (saw_axis || !cJSON_IsNumber(item)) return -1;
        saw_axis = 1;
        value = item->valuedouble;
        if (!isfinite(value) || value != trunc(value) ||
            value < -(double)rank || value >= (double)rank) return -1;
        axis = (int)value;
    }
    if (!saw_axis) return -1;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return -1;
    *axis_out = (uint32_t)axis;
    return 0;
}

static int physical_scalar_f32(const T* tensor, float* value) {
    if (!tensor || !value || !tensor->data || tensor->dtype != T_F32 || tensor->numel != 1 ||
        tensor->elem_size != sizeof(float)) return -1;
    memcpy(value, tensor->data, sizeof(*value));
    return isfinite(*value) && *value > 0.0f ? 0 : -1;
}

/* The portable Quantize/Dequantize ABI dereferences scalar pointers. Keep an
 * aligned local copy because SafeTensors permits arbitrary byte offsets. */
static int physical_zero_point_scalar(const T* tensor, int expected_dtype, int* value,
                                      int8_t* i8_copy, uint8_t* u8_copy,
                                      const void** pointer, uint32_t* dtype_out) {
    if (!value || !i8_copy || !u8_copy || !pointer || !dtype_out ||
        (expected_dtype != T_I8 && expected_dtype != T_U8)) return -1;
    *value = 0;
    *pointer = NULL;
    *dtype_out = (uint32_t)VX_DTYPE_UNSPECIFIED;
    if (!tensor) return 0;
    if (!tensor->data || tensor->dtype != expected_dtype || tensor->numel != 1 ||
        tensor->elem_size != 1) return -1;
    if (expected_dtype == T_I8) {
        memcpy(i8_copy, tensor->data, sizeof(*i8_copy));
        *value = *i8_copy;
        *pointer = i8_copy;
    } else {
        memcpy(u8_copy, tensor->data, sizeof(*u8_copy));
        *value = *u8_copy;
        *pointer = u8_copy;
    }
    *dtype_out = physical_byte_dtype(expected_dtype);
    return *dtype_out ? 0 : -1;
}

static int qadd_physical_candidate(Node* node, T* output) {
    T* a = nin(node, "a");
    T* b = nin(node, "b");
    if (!a) a = nin(node, "input");
    if (!a) a = nin(node, "x");
    if (!b) b = nin(node, "y");
    return (a && ((a->dtype == T_I8 || a->dtype == T_U8) || a->quantization.valid)) ||
           (b && ((b->dtype == T_I8 || b->dtype == T_U8) || b->quantization.valid)) ||
           (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                       output->quantization.valid));
}

static int qsilu_physical_candidate(Node* node, T* output) {
    T* input = physical_node_data_input(node);
    return (input && ((input->dtype == T_I8 || input->dtype == T_U8) ||
                      input->quantization.valid)) ||
           (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                       output->quantization.valid));
}

static int qgelu_physical_candidate(Node* node, T* output) {
    T* input = physical_node_data_input(node);
    return (input && ((input->dtype == T_I8 || input->dtype == T_U8) ||
                      input->quantization.valid)) ||
           (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                       output->quantization.valid));
}

static int qgroupnorm_physical_candidate(Node* node, T* output) {
    T* input = node ? nin(node, "input") : NULL;
    return (input && ((input->dtype == T_I8 || input->dtype == T_U8) ||
                      input->quantization.valid)) ||
           (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                       output->quantization.valid));
}

static int qlayernorm_physical_candidate(Node* node, T* output) {
    T* input = node ? nin(node, "input") : NULL;
    return (input && ((input->dtype == T_I8 || input->dtype == T_U8) ||
                      input->quantization.valid)) ||
           (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                       output->quantization.valid));
}

static int qsdpa_physical_candidate(Node* node, T* output) {
    T* q = node ? nin(node, "q") : NULL;
    T* k = node ? nin(node, "k") : NULL;
    T* v = node ? nin(node, "v") : NULL;
    return (q && ((q->dtype == T_I8 || q->dtype == T_U8) || q->quantization.valid)) ||
           (k && ((k->dtype == T_I8 || k->dtype == T_U8) || k->quantization.valid)) ||
           (v && ((v->dtype == T_I8 || v->dtype == T_U8) || v->quantization.valid)) ||
           (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                       output->quantization.valid));
}

/* QArgMax is a dedicated typed reduction, not a fallback form of F32
 * ArgMax. Always route its spelling through the strict validator so malformed
 * byte descriptors fail before an output write. */
static int qargmax_physical_candidate(Node* node, T* output) {
    return node != NULL && output != NULL;
}

/* QMaskedMean is another strict, standalone byte-domain spelling.  Its
 * validator owns the exact input names and rejects any F32 reduction
 * form before an output can be written. */
static int qmaskedmean_physical_candidate(Node* node, T* output) {
    return node != NULL && output != NULL;
}

static int requantize_physical_candidate(Node* node, T* output) {
    T* input = physical_node_data_input(node);
    return (input && ((input->dtype == T_I8 || input->dtype == T_U8) ||
                      input->quantization.valid)) ||
           (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                       output->quantization.valid));
}

static int quantize_physical_candidate(T* output) {
    return output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                      output->quantization.valid);
}

static int dequantize_physical_candidate(Node* node) {
    T* input = physical_node_data_input(node);
    return input && ((input->dtype == T_I8 || input->dtype == T_U8) ||
                     input->quantization.valid);
}

/* The incremental token-row contract is unambiguous only for [S,D], [1,S,D],
 * and the sequence-major singleton-batch spelling [S,1,D]. Multi-head and
 * other higher-rank layouts keep their existing complete-tensor execution:
 * flattening their leading dimensions would refresh only one head rather than
 * one token across every head. */
static int physical_qdq_row_span(const T* input, const T* output, int row,
                                 size_t* offset_out,
                                 uint32_t* elements_out) {
    long sequence;
    long width;
    size_t offset;
    if (!input || !output || !offset_out || !elements_out || row < 0 ||
        !physical_same_shape(input, output)) return -1;
    if (input->ndim == 2) {
        sequence = input->shape[0];
        width = input->shape[1];
    } else if (input->ndim == 3 &&
               (input->shape[0] == 1 || input->shape[1] == 1)) {
        sequence = input->shape[0] == 1
            ? input->shape[1] : input->shape[0];
        width = input->shape[2];
    } else {
        return 0;
    }
    if (sequence <= row || width <= 0 || width > UINT32_MAX ||
        input->numel <= 0 || input->numel % sequence ||
        input->numel / sequence != width ||
        (size_t)row > SIZE_MAX / (size_t)width) return -1;
    offset = (size_t)row * (size_t)width;
    if (offset > (size_t)input->numel ||
        (size_t)width > (size_t)input->numel - offset) return -1;
    *offset_out = offset;
    *elements_out = (uint32_t)width;
    return 1;
}

static int physical_incremental_tensor_dirty(const T* tensor) {
    int index;
    if (!tensor || g_active_row < 0) return 0;
    index = t_index(tensor);
    return index >= 0 && index < g_nt && g_dirty[index] != 0;
}

static int quantize_physical_row_compatible(Node* node, T* output, int row) {
    T* input = physical_node_data_input(node);
    T* scale = nin(node, "scale");
    T* zero_point = nin(node, "zero_point");
    float scale_value;
    int zero_value;
    int8_t zero_i8;
    uint8_t zero_u8;
    const void* zero_data;
    uint32_t zero_dtype, output_dtype;
    size_t offset;
    uint32_t elements;
    return node && input && output && input->data && input->dtype == T_F32 &&
        input->elem_size == sizeof(float) && input->numel > 0 &&
        input->numel <= UINT32_MAX &&
        physical_byte_tensor(output, &output_dtype) == 0 &&
        physical_scalar_f32(scale, &scale_value) == 0 &&
        physical_zero_point_scalar(zero_point, output->dtype, &zero_value,
                                   &zero_i8, &zero_u8, &zero_data,
                                   &zero_dtype) == 0 &&
        scale_value == output->quantization.scale &&
        zero_value == output->quantization.zero_point &&
        physical_qdq_row_span(input, output, row, &offset, &elements) == 1;
}

static int dequantize_physical_row_compatible(Node* node, T* output, int row) {
    T* input = physical_node_data_input(node);
    T* scale = nin(node, "scale");
    T* zero_point = nin(node, "zero_point");
    float scale_value;
    int zero_value;
    int8_t zero_i8;
    uint8_t zero_u8;
    const void* zero_data;
    uint32_t zero_dtype, input_dtype;
    size_t offset;
    uint32_t elements;
    return node && input && output && output->data && output->dtype == T_F32 &&
        output->elem_size == sizeof(float) && output->numel > 0 &&
        output->numel <= INT_MAX &&
        physical_byte_tensor(input, &input_dtype) == 0 &&
        physical_scalar_f32(scale, &scale_value) == 0 &&
        physical_zero_point_scalar(zero_point, input->dtype, &zero_value,
                                   &zero_i8, &zero_u8, &zero_data,
                                   &zero_dtype) == 0 &&
        scale_value == input->quantization.scale &&
        zero_value == input->quantization.zero_point &&
        physical_qdq_row_span(input, output, row, &offset, &elements) == 1;
}

static int qadd_physical_forward(Node* node, T* output) {
    T* a = nin(node, "a");
    T* b = nin(node, "b");
    const void* a_data;
    const void* b_data;
    void* output_data;
    uint32_t elements;
    uint32_t a_dtype, b_dtype, output_dtype, relu;
    if (!a) a = nin(node, "input");
    if (!a) a = nin(node, "x");
    if (!b) b = nin(node, "y");
    if (!node || !output || physical_byte_tensor(a, &a_dtype) != 0 ||
        physical_byte_tensor(b, &b_dtype) != 0 || physical_byte_tensor(output, &output_dtype) != 0 ||
        !physical_same_shape(a, b) || !physical_same_shape(a, output) ||
        physical_relu_parameter(node->params, &relu) != 0) return -1;
    a_data = a->data;
    b_data = b->data;
    output_data = output->data;
    elements = (uint32_t)output->numel;
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        const uint32_t width = output->ndim > 0
            ? (uint32_t)output->shape[output->ndim - 1] : 0u;
        int row_start;
        int row_count;
        int partial;
        size_t offset;
        partial = physical_sequence_window(output, width,
                                           &row_start, &row_count);
        if (partial <= 0) return -1;
        offset = (size_t)row_start * width;
        a_data = (const unsigned char*)a_data + offset * a->elem_size;
        b_data = (const unsigned char*)b_data + offset * b->elem_size;
        output_data = (unsigned char*)output_data + offset * output->elem_size;
        elements = (uint32_t)row_count * width;
    }
    return qadd_i8u8(a_data, b_data, output_data, elements,
                     a->quantization.scale, a->quantization.zero_point,
                     b->quantization.scale, b->quantization.zero_point,
                     output->quantization.scale, output->quantization.zero_point,
                     a_dtype, b_dtype, output_dtype, relu) == 1 ? 0 : -1;
}

static int qadd_physical_gpu_forward(Node* node, T* output, const VxW8A8Ops* ops,
                                     const char* backend_label, const char** node_backend) {
    T* a;
    T* b;
    const void* a_data;
    const void* b_data;
    void* output_data;
    uint32_t elements;
    uint32_t sequence_width;
    int row_start = 0;
    int row_count = 0;
    int partial;
    uint32_t a_dtype, b_dtype, output_dtype, relu;
    if (!node || !output || !node_backend ||
        physical_gpu_partial_execution_declined()) return 0;
    a = nin(node, "a");
    b = nin(node, "b");
    if (!a) a = nin(node, "input");
    if (!a) a = nin(node, "x");
    if (!b) b = nin(node, "y");
    if (physical_byte_tensor(a, &a_dtype) != 0 || physical_byte_tensor(b, &b_dtype) != 0 ||
        physical_byte_tensor(output, &output_dtype) != 0 || !physical_same_shape(a, b) ||
        !physical_same_shape(a, output) || physical_relu_parameter(node->params, &relu) != 0) return 0;
    sequence_width = output->ndim > 0
        ? (uint32_t)output->shape[output->ndim - 1] : 0u;
    partial = physical_sequence_window(output, sequence_width,
                                       &row_start, &row_count);
    if (partial < 0) return 0;
    elements = partial
        ? (uint32_t)row_count * sequence_width
        : (uint32_t)output->numel;
    size_t offset = partial
        ? (size_t)row_start * sequence_width : 0;
    a_data = (const unsigned char*)a->data + offset * a->elem_size;
    b_data = (const unsigned char*)b->data + offset * b->elem_size;
    output_data = (unsigned char*)output->data + offset * output->elem_size;
    VX_RUN_W8A8(ops, qadd_i8u8,
            a_data, elements, b_data, elements, output_data, elements,
            a->quantization.scale,
            a->quantization.zero_point, b->quantization.scale, b->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            a_dtype, b_dtype, output_dtype, relu);
    return 0;
}

/* QSiLU is a scalar byte-domain transform, so it needs no persistent parsed
 * node metadata.  Its two activation descriptors are nevertheless validated
 * as strictly as QAdd before either a CPU or GPU route can touch storage. */
static int qsilu_physical_validate(Node* node, T* output, T** input_out,
                                   uint32_t* input_dtype_out,
                                   uint32_t* output_dtype_out) {
    T* input;
    uint32_t input_dtype, output_dtype;
    if (!node || !output || !input_out || !input_dtype_out || !output_dtype_out ||
        node->nin != 1 || physical_no_parameters(node->params) != 0) return -1;
    input = physical_node_data_input(node);
    if (physical_byte_tensor(input, &input_dtype) != 0 ||
        physical_byte_tensor(output, &output_dtype) != 0 ||
        !physical_same_shape(input, output) || input->data == output->data) return -1;
    *input_out = input;
    *input_dtype_out = input_dtype;
    *output_dtype_out = output_dtype;
    return 0;
}

static int qsilu_physical_forward(Node* node, T* output) {
    T* input;
    const void* input_data;
    void* output_data;
    uint32_t elements;
    uint32_t input_dtype, output_dtype;
    if (qsilu_physical_validate(node, output, &input, &input_dtype, &output_dtype) != 0)
        return -1;
    input_data = input->data;
    output_data = output->data;
    elements = (uint32_t)output->numel;
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        const uint32_t width = output->ndim > 0
            ? (uint32_t)output->shape[output->ndim - 1] : 0u;
        int row_start;
        int row_count;
        int partial;
        size_t offset;
        partial = physical_sequence_window(output, width,
                                           &row_start, &row_count);
        if (partial <= 0) return -1;
        offset = (size_t)row_start * width;
        input_data = (const unsigned char*)input_data +
            offset * input->elem_size;
        output_data = (unsigned char*)output_data +
            offset * output->elem_size;
        elements = (uint32_t)row_count * width;
    }
    return vx_qsilu_i8u8_native_validated(
                       input_data, output_data, elements,
                       input->quantization.scale, input->quantization.zero_point,
                       output->quantization.scale, output->quantization.zero_point,
                       input_dtype, output_dtype) == 1 ? 0 : -1;
}

/* Keep a sequence of canonical QSiLU activations resident on the selected
 * backend. Row/prefix execution retains its explicit CPU synchronization boundary,
 * identical to the other physical-byte GPU routes. */
static int qsilu_physical_gpu_forward(Node* node, T* output,
                                      const VxW8A8Ops* ops,
                                      const char* backend_label,
                                      const char** node_backend) {
    T* input;
    const void* input_data;
    void* output_data;
    uint32_t elements;
    uint32_t width;
    int row_start = 0;
    int row_count = 0;
    int partial;
    uint32_t input_dtype, output_dtype;
    if (!node_backend || physical_gpu_partial_execution_declined() ||
        qsilu_physical_validate(node, output, &input, &input_dtype, &output_dtype) != 0)
        return 0;
    width = output->ndim > 0
        ? (uint32_t)output->shape[output->ndim - 1] : 0u;
    partial = physical_sequence_window(output, width, &row_start,
                                       &row_count);
    if (partial < 0) return 0;
    elements = partial ? (uint32_t)row_count * width
                       : (uint32_t)output->numel;
    size_t offset = partial ? (size_t)row_start * width : 0;
    input_data = (const unsigned char*)input->data + offset * input->elem_size;
    output_data = (unsigned char*)output->data + offset * output->elem_size;
    VX_RUN_W8A8(ops, qsilu_i8u8,
            input_data, output_data, elements,
            input->quantization.scale, input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            input_dtype, output_dtype);
    return 0;
}

/* QGELU has the same scalar byte-domain shape contract as QSiLU, but its
 * permitted parameter spelling is the fixed approximate='none' convention.
 * It remains stateless: immutable tensor descriptors are all the runtime
 * metadata needed to invoke qgelu_i8u8. */
static int qgelu_physical_validate(Node* node, T* output, T** input_out,
                                   uint32_t* input_dtype_out,
                                   uint32_t* output_dtype_out) {
    T* input;
    uint32_t input_dtype, output_dtype;
    if (!node || !output || !input_out || !input_dtype_out || !output_dtype_out ||
        node->nin != 1 || physical_qgelu_parameters(node->params) != 0) return -1;
    input = physical_node_data_input(node);
    if (physical_byte_tensor(input, &input_dtype) != 0 ||
        physical_byte_tensor(output, &output_dtype) != 0 ||
        !physical_same_shape(input, output) || input->data == output->data) return -1;
    *input_out = input;
    *input_dtype_out = input_dtype;
    *output_dtype_out = output_dtype;
    return 0;
}

static int qgelu_physical_forward(Node* node, T* output) {
    T* input;
    const void* input_data;
    void* output_data;
    uint32_t elements;
    uint32_t input_dtype, output_dtype;
    if (qgelu_physical_validate(node, output, &input, &input_dtype, &output_dtype) != 0)
        return -1;
    input_data = input->data;
    output_data = output->data;
    elements = (uint32_t)output->numel;
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        const uint32_t width = output->ndim > 0
            ? (uint32_t)output->shape[output->ndim - 1] : 0u;
        int row_start;
        int row_count;
        int partial;
        size_t offset;
        partial = physical_sequence_window(output, width,
                                           &row_start, &row_count);
        if (partial <= 0) return -1;
        offset = (size_t)row_start * width;
        input_data = (const unsigned char*)input_data + offset * input->elem_size;
        output_data = (unsigned char*)output_data + offset * output->elem_size;
        elements = (uint32_t)row_count * width;
    }
    return qgelu_i8u8(input_data, output_data, elements,
                       input->quantization.scale, input->quantization.zero_point,
                       output->quantization.scale, output->quantization.zero_point,
                       input_dtype, output_dtype) == 1 ? 0 : -1;
}

static int qgelu_physical_gpu_forward(Node* node, T* output,
                                      const VxW8A8Ops* ops,
                                      const char* backend_label,
                                      const char** node_backend) {
    T* input;
    const void* input_data;
    void* output_data;
    uint32_t elements;
    uint32_t width;
    int row_start = 0;
    int row_count = 0;
    int partial;
    uint32_t input_dtype, output_dtype;
    if (!node_backend || physical_gpu_partial_execution_declined() ||
        qgelu_physical_validate(node, output, &input, &input_dtype, &output_dtype) != 0)
        return 0;
    width = output->ndim > 0
        ? (uint32_t)output->shape[output->ndim - 1] : 0u;
    partial = physical_sequence_window(output, width, &row_start,
                                       &row_count);
    if (partial < 0) return 0;
    elements = partial ? (uint32_t)row_count * width
                       : (uint32_t)output->numel;
    size_t offset = partial ? (size_t)row_start * width : 0;
    input_data = (const unsigned char*)input->data + offset * input->elem_size;
    output_data = (unsigned char*)output->data + offset * output->elem_size;
    VX_RUN_W8A8(ops, qgelu_i8u8,
            input_data, output_data, elements,
            input->quantization.scale, input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            input_dtype, output_dtype);
    return 0;
}

typedef struct {
    T* input;
    T* weight;
    T* bias;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t groups;
    uint32_t input_dtype;
    uint32_t output_dtype;
    float epsilon;
} QGroupNormPhysicalDescriptor;

static int qnorm_exact_inputs(Node* node, T** input_out, T** weight_out,
                              T** bias_out) {
    T* input = NULL;
    T* weight = NULL;
    T* bias = NULL;
    if (!node || !input_out || !weight_out || !bias_out || node->nin != 3) return -1;
    for (int index = 0; index < node->nin; index++) {
        T* tensor = t_find(node->ins[index].name);
        if (!strcmp(node->ins[index].key, "input")) {
            if (input) return -1;
            input = tensor;
        } else if (!strcmp(node->ins[index].key, "weight")) {
            if (weight) return -1;
            weight = tensor;
        } else if (!strcmp(node->ins[index].key, "bias")) {
            if (bias) return -1;
            bias = tensor;
        } else {
            return -1;
        }
    }
    if (!input || !weight || !bias) return -1;
    *input_out = input;
    *weight_out = weight;
    *bias_out = bias;
    return 0;
}

/* QSDPA is one canonical three-input attention operation with an optional
 * binary I32 keep-mask.  Do not inherit CrossSDPA aliases here: accepting an
 * alternate spelling would weaken its byte-domain descriptor contract. */
static int qsdpa_exact_inputs(Node* node, T** q_out, T** k_out, T** v_out,
                              T** mask_out) {
    T* q = NULL;
    T* k = NULL;
    T* v = NULL;
    T* mask = NULL;
    int saw_q = 0;
    int saw_k = 0;
    int saw_v = 0;
    int saw_mask = 0;
    if (!node || !q_out || !k_out || !v_out || !mask_out ||
        node->nin < 3 || node->nin > 4) return -1;
    for (int index = 0; index < node->nin; index++) {
        if (!strcmp(node->ins[index].key, "q")) {
            if (saw_q) return -1;
            saw_q = 1;
            q = t_find(node->ins[index].name);
        } else if (!strcmp(node->ins[index].key, "k")) {
            if (saw_k) return -1;
            saw_k = 1;
            k = t_find(node->ins[index].name);
        } else if (!strcmp(node->ins[index].key, "v")) {
            if (saw_v) return -1;
            saw_v = 1;
            v = t_find(node->ins[index].name);
        } else if (!strcmp(node->ins[index].key, "mask")) {
            if (saw_mask) return -1;
            saw_mask = 1;
            mask = t_find(node->ins[index].name);
        } else {
            return -1;
        }
    }
    if (!saw_q || !saw_k || !saw_v || !q || !k || !v ||
        (saw_mask && !mask)) return -1;
    *q_out = q;
    *k_out = k;
    *v_out = v;
    *mask_out = mask;
    return 0;
}

static int qargmax_exact_input(Node* node, T** input_out) {
    T* input;
    if (!node || !input_out || node->nin != 1 ||
        strcmp(node->ins[0].key, "input")) return -1;
    input = t_find(node->ins[0].name);
    if (!input) return -1;
    *input_out = input;
    return 0;
}

static int qmaskedmean_exact_inputs(Node* node, T** input_out, T** mask_out) {
    T* input = NULL;
    T* mask = NULL;
    if (!node || !input_out || !mask_out || node->nin != 2) return -1;
    for (int index = 0; index < node->nin; index++) {
        if (!strcmp(node->ins[index].key, "input")) {
            if (input) return -1;
            input = t_find(node->ins[index].name);
        } else if (!strcmp(node->ins[index].key, "mask")) {
            if (mask) return -1;
            mask = t_find(node->ins[index].name);
        } else {
            return -1;
        }
    }
    if (!input || !mask) return -1;
    *input_out = input;
    *mask_out = mask;
    return 0;
}

/* QGroupNorm's affine inputs remain F32 parameters, but activation storage
 * must remain physical I8/U8 bytes.  Validate the complete three-input NHWC
 * contract before CPU or GPU code can allocate/write an output buffer. */
static int qgroupnorm_physical_validate(Node* node, T* output,
                                        int validate_host_values,
                                        QGroupNormPhysicalDescriptor* descriptor) {
    T* input;
    T* weight;
    T* bias;
    uint32_t groups;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    float epsilon;
    uint64_t elements;
    if (!node || !output || !descriptor || node->nout != 1 ||
        output != t_find(node->out) ||
        qnorm_exact_inputs(node, &input, &weight, &bias) != 0 ||
        physical_qgroupnorm_parameters(node->params, &groups, &epsilon) != 0 ||
        physical_byte_tensor(input, &input_dtype) != 0 ||
        physical_byte_tensor(output, &output_dtype) != 0 ||
        input->ndim != 4 || output->ndim != 4 ||
        !physical_same_shape(input, output) || input->data == output->data ||
        input->data == weight->data || input->data == bias->data ||
        output->data == weight->data || output->data == bias->data ||
        weight->dtype != T_F32 || bias->dtype != T_F32 ||
        weight->elem_size != sizeof(float) || bias->elem_size != sizeof(float) ||
        weight->ndim != 1 || bias->ndim != 1) return -1;
    if (input->shape[0] <= 0 || input->shape[1] <= 0 || input->shape[2] <= 0 ||
        input->shape[3] <= 0) return -1;
    batch = (uint32_t)input->shape[0];
    height = (uint32_t)input->shape[1];
    width = (uint32_t)input->shape[2];
    channels = (uint32_t)input->shape[3];
    if (!groups || channels % groups || weight->shape[0] != (int)channels ||
        bias->shape[0] != (int)channels || weight->numel != (long)channels ||
        bias->numel != (long)channels || batch > UINT32_MAX / groups) return -1;
    elements = batch;
    if (elements > UINT64_MAX / height) return -1;
    elements *= height;
    if (elements > UINT64_MAX / width) return -1;
    elements *= width;
    if (elements > UINT64_MAX / channels) return -1;
    elements *= channels;
    if (elements > UINT32_MAX || input->numel != (long)elements ||
        output->numel != (long)elements) return -1;
    if (validate_host_values)
        for (uint32_t channel = 0; channel < channels; channel++) {
            if (!isfinite(tensor_scalar_f32(weight, channel)) ||
                !isfinite(tensor_scalar_f32(bias, channel))) return -1;
        }
    descriptor->input = input;
    descriptor->weight = weight;
    descriptor->bias = bias;
    descriptor->batch = batch;
    descriptor->height = height;
    descriptor->width = width;
    descriptor->channels = channels;
    descriptor->groups = groups;
    descriptor->input_dtype = input_dtype;
    descriptor->output_dtype = output_dtype;
    descriptor->epsilon = epsilon;
    return 0;
}

static int qgroupnorm_physical_forward(Node* node, T* output) {
    QGroupNormPhysicalDescriptor descriptor;
    if (qgroupnorm_physical_validate(
            node, output, 1, &descriptor) != 0) return -1;
    return vx_qgroupnorm_i8u8_native_validated(
                           descriptor.input->data, descriptor.weight->data,
                           descriptor.bias->data, output->data, descriptor.batch,
                           descriptor.height, descriptor.width, descriptor.channels,
                           descriptor.groups, descriptor.input->quantization.scale,
                           descriptor.input->quantization.zero_point,
                           output->quantization.scale,
                           output->quantization.zero_point, descriptor.epsilon,
                           descriptor.input_dtype,
                           descriptor.output_dtype) == 1 ? 0 : -1;
}

/* The paired GPU passes keep byte activations resident.  The short-lived F32
 * group-statistics buffer is backend-private scratch, never an activation
 * tensor, and row/prefix execution retains the established CPU boundary. */
static int qgroupnorm_physical_gpu_forward(Node* node, T* output,
                                           const VxW8A8Ops* ops,
                                           const char* backend_label,
                                           const char** node_backend) {
    QGroupNormPhysicalDescriptor descriptor;
    if (!node_backend || physical_gpu_partial_execution_declined() ||
        qgroupnorm_physical_validate(
            node, output, !g_bounded_gpu_value_domain_proven,
            &descriptor) != 0) return 0;
    VX_RUN_W8A8(ops, qgroupnorm_i8u8,
            descriptor.input->data, descriptor.weight->data, descriptor.bias->data,
            output->data, descriptor.batch, descriptor.height, descriptor.width,
            descriptor.channels, descriptor.groups,
            descriptor.input->quantization.scale,
            descriptor.input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            descriptor.epsilon, descriptor.input_dtype, descriptor.output_dtype);
    return 0;
}

typedef struct {
    T* input;
    T* weight;
    T* bias;
    uint32_t rows;
    uint32_t d_model;
    uint32_t input_dtype;
    uint32_t output_dtype;
    float epsilon;
} QLayerNormPhysicalDescriptor;

/* QLayerNorm shares QGroupNorm's exact three-input spelling, but normalizes
 * each contiguous final-axis row of an otherwise arbitrary rank-1..8 byte
 * tensor.  Validate row shape, affine metadata, and all scalar attributes
 * before CPU/GPU allocation or an output write can occur. */
static int qlayernorm_physical_validate(Node* node, T* output,
                                        int validate_host_values,
                                        QLayerNormPhysicalDescriptor* descriptor) {
    T* input;
    T* weight;
    T* bias;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t d_model;
    uint32_t rows;
    float epsilon;
    uint64_t elements;
    if (!node || !output || !descriptor || node->nout != 1 ||
        output != t_find(node->out) ||
        qnorm_exact_inputs(node, &input, &weight, &bias) != 0 ||
        physical_byte_tensor(input, &input_dtype) != 0 ||
        physical_byte_tensor(output, &output_dtype) != 0 ||
        input->ndim < 1 || input->ndim > 8 || output->ndim != input->ndim ||
        !physical_same_shape(input, output) || input->data == output->data ||
        input->data == weight->data || input->data == bias->data ||
        output->data == weight->data || output->data == bias->data ||
        weight->dtype != T_F32 || bias->dtype != T_F32 ||
        weight->elem_size != sizeof(float) || bias->elem_size != sizeof(float) ||
        weight->ndim != 1 || bias->ndim != 1 || input->numel > UINT32_MAX) return -1;
    for (int dimension = 0; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0) return -1;
    }
    d_model = (uint32_t)input->shape[input->ndim - 1];
    if (!d_model || input->numel <= 0 || input->numel % (long)d_model) return -1;
    elements = (uint64_t)input->numel;
    rows = (uint32_t)(elements / d_model);
    if (!rows || (uint64_t)rows * d_model != elements ||
        physical_qlayernorm_parameters(node->params, d_model, &epsilon) != 0 ||
        weight->shape[0] != (int)d_model || bias->shape[0] != (int)d_model ||
        weight->numel != (long)d_model || bias->numel != (long)d_model) return -1;
    if (validate_host_values)
        for (uint32_t channel = 0; channel < d_model; channel++) {
            if (!isfinite(tensor_scalar_f32(weight, channel)) ||
                !isfinite(tensor_scalar_f32(bias, channel))) return -1;
        }
    descriptor->input = input;
    descriptor->weight = weight;
    descriptor->bias = bias;
    descriptor->rows = rows;
    descriptor->d_model = d_model;
    descriptor->input_dtype = input_dtype;
    descriptor->output_dtype = output_dtype;
    descriptor->epsilon = epsilon;
    return 0;
}

static int qlayernorm_physical_forward(Node* node, T* output) {
    QLayerNormPhysicalDescriptor descriptor;
    const void* input_data;
    void* output_data;
    uint32_t rows;
    if (qlayernorm_physical_validate(
            node, output, 1, &descriptor) != 0) return -1;
    input_data = descriptor.input->data;
    output_data = output->data;
    rows = descriptor.rows;
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        int row_start;
        int row_count;
        int partial;
        size_t offset;
        partial = physical_sequence_window(descriptor.input,
            descriptor.d_model, &row_start, &row_count);
        if (partial <= 0) return -1;
        offset = (size_t)row_start * descriptor.d_model;
        input_data = (const unsigned char*)input_data + offset * descriptor.input->elem_size;
        output_data = (unsigned char*)output_data + offset * output->elem_size;
        rows = (uint32_t)row_count;
    }
    return vx_qlayernorm_i8u8_native_validated(
                           input_data, descriptor.weight->data,
                           descriptor.bias->data, output_data, rows,
                           descriptor.d_model, descriptor.input->quantization.scale,
                           descriptor.input->quantization.zero_point,
                           output->quantization.scale,
                           output->quantization.zero_point, descriptor.epsilon,
                           descriptor.input_dtype, descriptor.output_dtype) == 1 ? 0 : -1;
}

/* Each GPU row keeps only a mean/inverse-stddev F32 pair in backend-local
 * scratch. CUDA launches only the selected resident sequence rows. */
static int qlayernorm_physical_gpu_forward(Node* node, T* output,
                                           const VxW8A8Ops* ops,
                                           const char* backend_label,
                                           const char** node_backend) {
    QLayerNormPhysicalDescriptor descriptor;
    const void* input_data;
    void* output_data;
    uint32_t rows;
    int row_start = 0;
    int row_count = 0;
    int partial;
    if (!node_backend || physical_gpu_partial_execution_declined() ||
        qlayernorm_physical_validate(
            node, output, !g_bounded_gpu_value_domain_proven,
            &descriptor) != 0) return 0;
    partial = physical_sequence_window(descriptor.input,
        descriptor.d_model, &row_start, &row_count);
    if (partial < 0) return 0;
    rows = partial ? (uint32_t)row_count : descriptor.rows;
    input_data = (const unsigned char*)descriptor.input->data +
        (size_t)row_start * descriptor.d_model * descriptor.input->elem_size;
    output_data = (unsigned char*)output->data +
        (size_t)row_start * descriptor.d_model * output->elem_size;
    VX_RUN_W8A8(ops, qlayernorm_i8u8,
            input_data, descriptor.weight->data, descriptor.bias->data,
            output_data, rows, descriptor.d_model,
            descriptor.input->quantization.scale,
            descriptor.input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            descriptor.epsilon, descriptor.input_dtype, descriptor.output_dtype);
    return 0;
}

typedef struct {
    T* q;
    T* k;
    T* v;
    T* mask;
    uint32_t batch;
    uint32_t seq_q;
    uint32_t seq_kv;
    uint32_t d_model;
    uint32_t heads;
    uint32_t causal;
    uint32_t mask_mode;
    uint32_t q_dtype;
    uint32_t k_dtype;
    uint32_t v_dtype;
    uint32_t output_dtype;
    float attention_scale;
} QSDPAPhysicalDescriptor;

static int qsdpa_centered_magnitude(uint32_t dtype, int32_t zero_point,
                                    uint64_t* magnitude_out) {
    int64_t low;
    int64_t high;
    uint64_t magnitude;
    if (!magnitude_out) return 0;
    if (dtype == VX_DTYPE_I8) {
        low = -128 - (int64_t)zero_point;
        high = 127 - (int64_t)zero_point;
    } else if (dtype == VX_DTYPE_U8) {
        low = -(int64_t)zero_point;
        high = 255 - (int64_t)zero_point;
    } else {
        return 0;
    }
    magnitude = (uint64_t)(low < 0 ? -low : low);
    if ((uint64_t)(high < 0 ? -high : high) > magnitude)
        magnitude = (uint64_t)(high < 0 ? -high : high);
    *magnitude_out = magnitude;
    return 1;
}

/* This mirrors qsdpa_i8u8 and the fixed WebGPU ABI: finite input scalars are
 * not enough when an extreme centered dot can still overflow F32.  Reject the
 * descriptor before a score is computed so all online softmax paths stay in
 * the finite domain. */
static int qsdpa_score_scale_valid(const T* q, const T* k, uint32_t q_dtype,
                                   uint32_t k_dtype, uint32_t head_dim,
                                   float attention_scale) {
    uint64_t q_magnitude;
    uint64_t k_magnitude;
    uint64_t maximum_dot;
    float qk_scale;
    float score_scale;
    if (!q || !k || !head_dim || !isfinite(attention_scale) ||
        attention_scale <= 0.0f ||
        !qsdpa_centered_magnitude(q_dtype, q->quantization.zero_point,
                                  &q_magnitude) ||
        !qsdpa_centered_magnitude(k_dtype, k->quantization.zero_point,
                                  &k_magnitude)) return 0;
    qk_scale = q->quantization.scale * k->quantization.scale;
    score_scale = qk_scale * attention_scale;
    if (!isfinite(qk_scale) || qk_scale <= 0.0f ||
        !isfinite(score_scale) || score_scale <= 0.0f) return 0;
    maximum_dot = q_magnitude * k_magnitude * head_dim;
    return isfinite((float)maximum_dot * score_scale);
}

/* QSDPA is deliberately separate from F32 CrossSDPA. It accepts only
 * canonical rank-2/3 physical bytes, owns the strict q/k/v/mask spelling, and
 * keeps all score/value intermediates local to the kernel/backend. */
static int qsdpa_physical_validate(Node* node, T* output,
                                   QSDPAPhysicalDescriptor* descriptor) {
    T* q;
    T* k;
    T* v;
    T* mask;
    uint32_t q_dtype;
    uint32_t k_dtype;
    uint32_t v_dtype;
    uint32_t output_dtype;
    uint32_t batch;
    uint32_t seq_q;
    uint32_t seq_kv;
    uint32_t d_model;
    uint32_t heads;
    uint32_t causal;
    uint32_t mask_mode;
    int mask_mode_value;
    uint64_t q_elements;
    uint64_t kv_elements;
    uint64_t mask_elements = 0;
    float attention_scale;
    if (!node || !output || !descriptor || node->nout != 1 ||
        output != t_find(node->out) ||
        qsdpa_exact_inputs(node, &q, &k, &v, &mask) != 0 ||
        physical_byte_tensor(q, &q_dtype) != 0 ||
        physical_byte_tensor(k, &k_dtype) != 0 ||
        physical_byte_tensor(v, &v_dtype) != 0 ||
        physical_byte_tensor(output, &output_dtype) != 0 ||
        (q->ndim != 2 && q->ndim != 3) || k->ndim != q->ndim ||
        v->ndim != q->ndim || output->ndim != q->ndim ||
        !physical_same_shape(q, output) || !physical_same_shape(k, v)) return -1;
    for (int dimension = 0; dimension < q->ndim; dimension++) {
        if (q->shape[dimension] <= 0 || k->shape[dimension] <= 0 ||
            v->shape[dimension] <= 0 || output->shape[dimension] <= 0) return -1;
    }
    d_model = (uint32_t)q->shape[q->ndim - 1];
    batch = q->ndim == 3 ? (uint32_t)q->shape[0] : 1u;
    seq_q = (uint32_t)q->shape[q->ndim - 2];
    seq_kv = (uint32_t)k->shape[k->ndim - 2];
    if (!d_model || !batch || !seq_q || !seq_kv ||
        k->shape[k->ndim - 1] != (int)d_model ||
        v->shape[v->ndim - 1] != (int)d_model ||
        v->shape[v->ndim - 2] != (int)seq_kv ||
        (q->ndim == 3 && (k->shape[0] != (int)batch ||
                           v->shape[0] != (int)batch ||
                           output->shape[0] != (int)batch))) return -1;
    q_elements = batch;
    if (q_elements > UINT32_MAX / seq_q) return -1;
    q_elements *= seq_q;
    if (q_elements > UINT32_MAX / d_model) return -1;
    q_elements *= d_model;
    kv_elements = batch;
    if (kv_elements > UINT32_MAX / seq_kv) return -1;
    kv_elements *= seq_kv;
    if (kv_elements > UINT32_MAX / d_model) return -1;
    kv_elements *= d_model;
    if (q_elements == 0 || kv_elements == 0 || q->numel != (long)q_elements ||
        output->numel != (long)q_elements || k->numel != (long)kv_elements ||
        v->numel != (long)kv_elements ||
        physical_qsdpa_parameters(node->params, d_model, &heads, &causal,
                                  &attention_scale) != 0 ||
        !qsdpa_score_scale_valid(q, k, q_dtype, k_dtype, d_model / heads,
                                 attention_scale)) return -1;
    mask_mode_value = attention_mask_mode(mask, (int)batch, (int)seq_q,
                                          (int)seq_kv);
    if (mask_mode_value < 0) return -1;
    mask_mode = (uint32_t)mask_mode_value;
    if (mask_mode == ATTN_MASK_KEY) mask_elements = seq_kv;
    else if (mask_mode == ATTN_MASK_BATCH_KEY) {
        mask_elements = seq_kv;
        if (mask_elements > UINT32_MAX / batch) return -1;
        mask_elements *= batch;
    } else if (mask_mode == ATTN_MASK_QUERY_KEY) {
        mask_elements = seq_kv;
        if (mask_elements > UINT32_MAX / seq_q) return -1;
        mask_elements *= seq_q;
    } else if (mask_mode == ATTN_MASK_BATCH_QUERY_KEY) {
        mask_elements = seq_kv;
        if (mask_elements > UINT32_MAX / seq_q) return -1;
        mask_elements *= seq_q;
        if (mask_elements > UINT32_MAX / batch) return -1;
        mask_elements *= batch;
    }
    if ((mask && (mask->dtype != T_I32 || mask->elem_size != sizeof(int32_t) ||
                  mask->numel != (long)mask_elements || mask_elements > UINT32_MAX)) ||
        physical_tensor_ranges_overlap(output, q) ||
        physical_tensor_ranges_overlap(output, k) ||
        physical_tensor_ranges_overlap(output, v) ||
        (mask && physical_tensor_ranges_overlap(output, mask))) return -1;
    descriptor->q = q;
    descriptor->k = k;
    descriptor->v = v;
    descriptor->mask = mask;
    descriptor->batch = batch;
    descriptor->seq_q = seq_q;
    descriptor->seq_kv = seq_kv;
    descriptor->d_model = d_model;
    descriptor->heads = heads;
    descriptor->causal = causal;
    descriptor->mask_mode = mask_mode;
    descriptor->q_dtype = q_dtype;
    descriptor->k_dtype = k_dtype;
    descriptor->v_dtype = v_dtype;
    descriptor->output_dtype = output_dtype;
    descriptor->attention_scale = attention_scale;
    return 0;
}

static int qsdpa_physical_forward(Node* node, T* output) {
    QSDPAPhysicalDescriptor descriptor;
    int query_start = 0;
    int query_count = 0;
    if (qsdpa_physical_validate(node, output, &descriptor) != 0) return -1;
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        if (descriptor.batch != 1 ||
            (descriptor.q->ndim != 2 && descriptor.q->ndim != 3)) return -1;
        seq_range(descriptor.seq_q, &query_start, &query_count);
        if (query_start < 0 || query_count <= 0 ||
            (uint32_t)query_start + (uint32_t)query_count > descriptor.seq_q)
            return -1;
        return vx_qsdpa_i8u8_native_range_validated(
                    descriptor.q->data, descriptor.k->data, descriptor.v->data,
                    descriptor.mask ? (const int32_t*)descriptor.mask->data : NULL,
                    output->data, descriptor.batch, descriptor.seq_q,
                    descriptor.seq_kv, descriptor.d_model, descriptor.heads,
                    descriptor.q->quantization.scale,
                    descriptor.q->quantization.zero_point,
                    descriptor.k->quantization.scale,
                    descriptor.k->quantization.zero_point,
                    descriptor.v->quantization.scale,
                    descriptor.v->quantization.zero_point,
                    output->quantization.scale,
                    output->quantization.zero_point,
                    descriptor.attention_scale, descriptor.q_dtype,
                    descriptor.k_dtype, descriptor.v_dtype,
                    descriptor.output_dtype, descriptor.causal,
                    descriptor.mask_mode, (uint32_t)query_start,
                    (uint32_t)query_count) == 1 ? 0 : -1;
    }
    return vx_qsdpa_i8u8_native_validated(
                      descriptor.q->data, descriptor.k->data, descriptor.v->data,
                      descriptor.mask ? (const int32_t*)descriptor.mask->data : NULL,
                      output->data, descriptor.batch, descriptor.seq_q,
                      descriptor.seq_kv, descriptor.d_model, descriptor.heads,
                      descriptor.q->quantization.scale,
                      descriptor.q->quantization.zero_point,
                      descriptor.k->quantization.scale,
                      descriptor.k->quantization.zero_point,
                      descriptor.v->quantization.scale,
                      descriptor.v->quantization.zero_point,
                      output->quantization.scale,
                      output->quantization.zero_point,
                      descriptor.attention_scale, descriptor.q_dtype,
                      descriptor.k_dtype, descriptor.v_dtype,
                      descriptor.output_dtype, descriptor.causal,
                      descriptor.mask_mode) == 1 ? 0 : -1;
}

/* QSDPA keeps scores and dequantized values kernel-local. CUDA's optional
 * range entry maps selected logical queries to their absolute sequence rows,
 * so resident K/V prefixes remain usable without an activation transfer. */
static int qsdpa_physical_gpu_forward(Node* node, T* output,
                                      const VxW8A8Ops* ops,
                                      const char* backend_label,
                                      const char** node_backend) {
    QSDPAPhysicalDescriptor descriptor;
    int query_start = 0;
    int query_count = 0;
    if (!node_backend || physical_gpu_partial_execution_declined() ||
        qsdpa_physical_validate(node, output, &descriptor) != 0) return 0;
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        if (!ops || !ops->qsdpa_range_i8u8 || descriptor.batch != 1 ||
            descriptor.q->ndim != 3) return 0;
        if (!descriptor.causal && g_prefix_rows > 0 &&
            g_prefix_rows < (int)descriptor.seq_q) {
            /* This prefix has not produced the future K/V rows required by
             * non-causal attention; the complete kernel below is correct. */
        } else {
            seq_range(descriptor.seq_q, &query_start, &query_count);
            if (query_start < 0 || query_count <= 0 ||
                (uint32_t)query_start + (uint32_t)query_count >
                    descriptor.seq_q) return 0;
            if (ops->qsdpa_range_i8u8(
                    descriptor.q->data, descriptor.k->data,
                    descriptor.v->data,
                    descriptor.mask
                        ? (const int32_t*)descriptor.mask->data : NULL,
                    output->data, descriptor.batch, descriptor.seq_q,
                    descriptor.seq_kv, descriptor.d_model, descriptor.heads,
                    descriptor.q->quantization.scale,
                    descriptor.q->quantization.zero_point,
                    descriptor.k->quantization.scale,
                    descriptor.k->quantization.zero_point,
                    descriptor.v->quantization.scale,
                    descriptor.v->quantization.zero_point,
                    output->quantization.scale,
                    output->quantization.zero_point,
                    descriptor.attention_scale, descriptor.q_dtype,
                    descriptor.k_dtype, descriptor.v_dtype,
                    descriptor.output_dtype, descriptor.causal,
                    descriptor.mask_mode, (uint32_t)query_start,
                    (uint32_t)query_count)) {
                *node_backend = backend_label;
                return 1;
            }
            return 0;
        }
    }
    VX_RUN_W8A8(ops, qsdpa_i8u8,
            descriptor.q->data, descriptor.k->data, descriptor.v->data,
            descriptor.mask ? (const int32_t*)descriptor.mask->data : NULL,
            output->data, descriptor.batch, descriptor.seq_q, descriptor.seq_kv,
            descriptor.d_model, descriptor.heads, descriptor.q->quantization.scale,
            descriptor.q->quantization.zero_point, descriptor.k->quantization.scale,
            descriptor.k->quantization.zero_point, descriptor.v->quantization.scale,
            descriptor.v->quantization.zero_point, output->quantization.scale,
            output->quantization.zero_point, descriptor.attention_scale,
            descriptor.q_dtype, descriptor.k_dtype, descriptor.v_dtype,
            descriptor.output_dtype, descriptor.causal, descriptor.mask_mode);
    return 0;
}

typedef struct {
    T* input;
    uint32_t outer;
    uint32_t axis_size;
    uint32_t inner;
    uint32_t input_dtype;
} QArgMaxPhysicalDescriptor;

/* QArgMax consumes exactly one per-tensor I8/U8 activation and emits ordinary
 * I32 indices with the selected axis removed. The reduction compares raw
 * typed bytes only, so scale/zero point cannot introduce an F32 boundary. */
static int qargmax_physical_validate(Node* node, T* output,
                                     QArgMaxPhysicalDescriptor* descriptor) {
    T* input;
    uint32_t input_dtype;
    uint32_t axis;
    uint64_t outer = 1;
    uint64_t inner = 1;
    uint64_t input_elements = 1;
    uint64_t output_elements;
    if (!node || !output || !descriptor || node->nout != 1 ||
        output != t_find(node->out) ||
        qargmax_exact_input(node, &input) != 0 ||
        physical_byte_tensor(input, &input_dtype) != 0 ||
        !output->data || output->dtype != T_I32 ||
        output->elem_size != sizeof(int32_t) ||
        input->ndim < 2 || input->ndim > 8 ||
        output->ndim != input->ndim - 1 ||
        physical_qargmax_axis(node->params, input->ndim, &axis) != 0) return -1;
    for (int dimension = 0; dimension < input->ndim; dimension++) {
        uint64_t extent;
        int output_dimension;
        if (input->shape[dimension] <= 0) return -1;
        extent = (uint32_t)input->shape[dimension];
        if (input_elements > UINT32_MAX / extent) return -1;
        input_elements *= extent;
        if (dimension < (int)axis) {
            if (outer > UINT32_MAX / extent) return -1;
            outer *= extent;
            output_dimension = dimension;
        } else if (dimension > (int)axis) {
            if (inner > UINT32_MAX / extent) return -1;
            inner *= extent;
            output_dimension = dimension - 1;
        } else {
            continue;
        }
        if (output->shape[output_dimension] <= 0 ||
            output->shape[output_dimension] != input->shape[dimension]) return -1;
    }
    output_elements = outer;
    if (output_elements > UINT32_MAX / inner) return -1;
    output_elements *= inner;
    if (!outer || !inner || !input_elements || !output_elements ||
        input->shape[axis] > INT32_MAX ||
        input->numel != (long)input_elements ||
        output->numel != (long)output_elements ||
        physical_tensor_ranges_overlap(output, input)) return -1;
    descriptor->input = input;
    descriptor->outer = (uint32_t)outer;
    descriptor->axis_size = (uint32_t)input->shape[axis];
    descriptor->inner = (uint32_t)inner;
    descriptor->input_dtype = input_dtype;
    return 0;
}

static int qargmax_physical_forward(Node* node, T* output) {
    QArgMaxPhysicalDescriptor descriptor;
    if (qargmax_physical_validate(node, output, &descriptor) != 0) return -1;
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        int row_start;
        int row_count;
        size_t input_offset;
        if (descriptor.input->ndim != 3 || descriptor.input->shape[0] != 1 ||
            output->ndim != 2 || output->shape[0] != 1 || descriptor.inner != 1 ||
            output->shape[1] <= 0) return -1;
        seq_range(descriptor.outer, &row_start, &row_count);
        if (row_start < 0 || row_count <= 0 ||
            (uint32_t)row_start + (uint32_t)row_count > descriptor.outer)
            return -1;
        input_offset = (size_t)row_start * descriptor.axis_size;
        return qargmax_i8u8((const unsigned char*)descriptor.input->data +
                                input_offset * descriptor.input->elem_size,
                            (int32_t*)output->data + row_start,
                            (uint32_t)row_count,
                            descriptor.axis_size, 1, descriptor.input_dtype) == 1 ? 0 : -1;
    }
    return qargmax_i8u8(descriptor.input->data, (int32_t*)output->data,
                        descriptor.outer, descriptor.axis_size,
                        descriptor.inner, descriptor.input_dtype) == 1 ? 0 : -1;
}

/* QArgMax's canonical decoder layout reduces the vocabulary axis independently
 * for each sequence row, so CUDA can launch only the requested resident rows. */
static int qargmax_physical_gpu_forward(Node* node, T* output,
                                        const VxW8A8Ops* ops,
                                        const char* backend_label,
                                        const char** node_backend) {
    QArgMaxPhysicalDescriptor descriptor;
    const void* input_data;
    int32_t* output_data;
    uint32_t outer;
    int row_start = 0;
    int row_count = 0;
    if (!node_backend || physical_gpu_partial_execution_declined() ||
        qargmax_physical_validate(node, output, &descriptor) != 0) return 0;
    input_data = descriptor.input->data;
    output_data = (int32_t*)output->data;
    outer = descriptor.outer;
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        if (descriptor.input->ndim != 3 || descriptor.input->shape[0] != 1 ||
            output->ndim != 2 || output->shape[0] != 1 ||
            descriptor.inner != 1) return 0;
        seq_range(descriptor.outer, &row_start, &row_count);
        if (row_start < 0 || row_count <= 0 ||
            (uint32_t)row_start + (uint32_t)row_count > descriptor.outer)
            return 0;
        input_data = (const unsigned char*)descriptor.input->data +
            (size_t)row_start * descriptor.axis_size *
            descriptor.input->elem_size;
        output_data += row_start;
        outer = (uint32_t)row_count;
    }
    VX_RUN_W8A8(ops, qargmax_i8u8,
            input_data, output_data, outer,
            descriptor.axis_size, descriptor.inner, descriptor.input_dtype);
    return 0;
}

typedef struct {
    T* input;
    T* mask;
    uint32_t batch;
    uint32_t sequence;
    uint32_t width;
    uint32_t input_dtype;
    uint32_t output_dtype;
} QMaskedMeanPhysicalDescriptor;

/* Keep the portable C and shader reductions in their shared I32 domain.  The
 * largest centered byte magnitude is at most 255, but honoring the actual
 * zero point makes this bound precise and fail-closed for every I8/U8 form. */
static int qmaskedmean_centered_sum_i32_valid(uint32_t dtype,
                                              int32_t zero_point,
                                              uint32_t sequence) {
    int64_t low;
    int64_t high;
    uint64_t magnitude;
    if (dtype == VX_DTYPE_I8) {
        low = -128 - (int64_t)zero_point;
        high = 127 - (int64_t)zero_point;
    } else if (dtype == VX_DTYPE_U8) {
        low = -(int64_t)zero_point;
        high = 255 - (int64_t)zero_point;
    } else {
        return 0;
    }
    magnitude = (uint64_t)(low < 0 ? -low : low);
    if ((uint64_t)(high < 0 ? -high : high) > magnitude)
        magnitude = (uint64_t)(high < 0 ? -high : high);
    return magnitude == 0 || (uint64_t)sequence <= (uint64_t)INT32_MAX / magnitude;
}

/* QMaskedMean has no scalar parameters.  Its exact [B,S,D] and [B,S]
 * descriptors make the router's token-padding semantics explicit, and its
 * output is the quantized [B,D] reduction rather than a hidden F32 boundary. */
static int qmaskedmean_physical_validate(Node* node, T* output,
                                         QMaskedMeanPhysicalDescriptor* descriptor) {
    T* input;
    T* mask;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t batch;
    uint32_t sequence;
    uint32_t width;
    uint64_t input_elements;
    uint64_t mask_elements;
    uint64_t output_elements;
    float multiplier;
    if (!node || !output || !descriptor || node->nout != 1 ||
        output != t_find(node->out) ||
        qmaskedmean_exact_inputs(node, &input, &mask) != 0 ||
        physical_no_parameters(node->params) != 0 ||
        physical_byte_tensor(input, &input_dtype) != 0 ||
        physical_byte_tensor(output, &output_dtype) != 0 ||
        !mask->data || mask->dtype != T_I32 ||
        mask->elem_size != sizeof(int32_t) || mask->quantization.valid ||
        input->ndim != 3 || mask->ndim != 2 || output->ndim != 2 ||
        input->shape[0] <= 0 || input->shape[1] <= 0 || input->shape[2] <= 0 ||
        mask->shape[0] != input->shape[0] || mask->shape[1] != input->shape[1] ||
        output->shape[0] != input->shape[0] || output->shape[1] != input->shape[2])
        return -1;
    batch = (uint32_t)input->shape[0];
    sequence = (uint32_t)input->shape[1];
    width = (uint32_t)input->shape[2];
    input_elements = batch;
    if (input_elements > UINT32_MAX / sequence) return -1;
    input_elements *= sequence;
    if (input_elements > UINT32_MAX / width) return -1;
    input_elements *= width;
    mask_elements = (uint64_t)batch * sequence;
    output_elements = (uint64_t)batch * width;
    multiplier = input->quantization.scale / output->quantization.scale;
    if (!batch || !sequence || !width || !input_elements || !mask_elements ||
        !output_elements || mask_elements > UINT32_MAX || output_elements > UINT32_MAX ||
        input->numel != (long)input_elements || mask->numel != (long)mask_elements ||
        output->numel != (long)output_elements ||
        !isfinite(multiplier) || multiplier <= 0.0f ||
        !qmaskedmean_centered_sum_i32_valid(input_dtype,
                                             input->quantization.zero_point,
                                             sequence) ||
        physical_tensor_ranges_overlap(output, input) ||
        physical_tensor_ranges_overlap(output, mask)) return -1;
    descriptor->input = input;
    descriptor->mask = mask;
    descriptor->batch = batch;
    descriptor->sequence = sequence;
    descriptor->width = width;
    descriptor->input_dtype = input_dtype;
    descriptor->output_dtype = output_dtype;
    return 0;
}

static int qmaskedmean_physical_forward(Node* node, T* output) {
    QMaskedMeanPhysicalDescriptor descriptor;
    if (qmaskedmean_physical_validate(node, output, &descriptor) != 0) return -1;
    return qmaskedmean_i8u8(descriptor.input->data,
                            (const int32_t*)descriptor.mask->data, output->data,
                            descriptor.batch, descriptor.sequence, descriptor.width,
                            descriptor.input->quantization.scale,
                            descriptor.input->quantization.zero_point,
                            output->quantization.scale,
                            output->quantization.zero_point,
                            descriptor.input_dtype, descriptor.output_dtype) == 1 ? 0 : -1;
}

/* Accelerator execution is intentionally ordinary-forward-only, as for the
 * other canonical physical-byte routes.  The mask and activation bytes remain
 * backend-resident and the scalar mean never becomes a graph tensor. */
static int qmaskedmean_physical_gpu_forward(Node* node, T* output,
                                            const VxW8A8Ops* ops,
                                            const char* backend_label,
                                            const char** node_backend) {
    QMaskedMeanPhysicalDescriptor descriptor;
    if (!node_backend || physical_gpu_partial_execution_declined() ||
        qmaskedmean_physical_validate(node, output, &descriptor) != 0) return 0;
    VX_RUN_W8A8(ops, qmaskedmean_i8u8,
            descriptor.input->data, (const int32_t*)descriptor.mask->data,
            output->data, descriptor.batch, descriptor.sequence, descriptor.width,
            descriptor.input->quantization.scale,
            descriptor.input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            descriptor.input_dtype, descriptor.output_dtype);
    return 0;
}

static int requantize_physical_forward(Node* node, T* output) {
    T* input = physical_node_data_input(node);
    uint32_t input_dtype, output_dtype;
    if (!node || !output || node->nin != 1 || physical_byte_tensor(input, &input_dtype) != 0 ||
        physical_byte_tensor(output, &output_dtype) != 0 || !physical_same_shape(input, output)) return -1;
    return requantize_linear_i8u8(input->data, output->data, (uint32_t)output->numel,
                                  input->quantization.scale, input->quantization.zero_point,
                                  output->quantization.scale, output->quantization.zero_point,
                                  input_dtype, output_dtype) == 1 ? 0 : -1;
}

static int requantize_physical_gpu_forward(Node* node, T* output,
                                            const VxW8A8Ops* ops,
                                            const char* backend_label,
                                            const char** node_backend) {
    T* input;
    uint32_t input_dtype, output_dtype;
    if (!node || !output || !node_backend ||
        physical_gpu_partial_execution_declined()) return 0;
    input = physical_node_data_input(node);
    if (node->nin != 1 || physical_byte_tensor(input, &input_dtype) != 0 ||
        physical_byte_tensor(output, &output_dtype) != 0 || !physical_same_shape(input, output)) return 0;
    VX_RUN_W8A8(ops, requantize_linear_i8u8,
            input->data, (uint32_t)input->numel, output->data, (uint32_t)output->numel,
            input->quantization.scale, input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            input_dtype, output_dtype);
    return 0;
}

static int quantize_physical_forward(Node* node, T* output) {
    T* input = physical_node_data_input(node);
    T* scale = nin(node, "scale");
    T* zero_point = nin(node, "zero_point");
    float scale_value;
    int zero_value;
    int8_t zero_i8;
    uint8_t zero_u8;
    const void* zero_data;
    uint32_t zero_dtype, output_dtype;
    size_t offset = 0;
    uint32_t elements;
    int row_status = 0;
    if (!node || !input || !output || !input->data || input->dtype != T_F32 ||
        input->elem_size != sizeof(float) || input->numel <= 0 || input->numel > UINT32_MAX ||
        physical_byte_tensor(output, &output_dtype) != 0 || !physical_same_shape(input, output) ||
        physical_scalar_f32(scale, &scale_value) != 0 ||
        physical_zero_point_scalar(zero_point, output->dtype, &zero_value, &zero_i8, &zero_u8,
                                   &zero_data, &zero_dtype) != 0 ||
        scale_value != output->quantization.scale || zero_value != output->quantization.zero_point) return -1;
    elements = (uint32_t)output->numel;
    if (g_active_row >= 0 && !physical_incremental_tensor_dirty(scale) &&
        !physical_incremental_tensor_dirty(zero_point)) {
        row_status = physical_qdq_row_span(input, output, g_active_row,
                                            &offset, &elements);
        if (row_status < 0) return -1;
    }
    if (row_status == 0) {
        offset = 0;
        elements = (uint32_t)output->numel;
    }
    return quantize_linear_typed((const float*)input->data + offset,
                                 &scale_value, zero_data, zero_dtype,
                                 (uint8_t*)output->data + offset,
                                 output_dtype, elements) == 1 ? 0 : -1;
}

/* QuantizeLinear uses the output descriptor as immutable device metadata.
 * A scale/zero-point computed by an earlier node is deliberately left on the
 * CPU path: validating its host value is then the explicit synchronization
 * boundary. Static weights and graph-input scalars remain safe to read here
 * without interrupting the F32 -> I8/U8 device chain. */
static int quantize_physical_gpu_forward(Node* node, T* output,
                                         const VxW8A8Ops* ops,
                                         const char* backend_label,
                                         const char** node_backend) {
    T* input = physical_node_data_input(node);
    T* scale = nin(node, "scale");
    T* zero_point = nin(node, "zero_point");
    float scale_value;
    int zero_value;
    int8_t zero_i8;
    uint8_t zero_u8;
    const void* zero_data;
    uint32_t zero_dtype, output_dtype;
    if (!node || !input || !output || !node_backend ||
        physical_gpu_partial_execution_declined() ||
        !input->data || input->dtype != T_F32 || input->elem_size != sizeof(float) ||
        input->numel <= 0 || input->numel > UINT32_MAX ||
        (scale && scale->owns && !scale->is_graph_input) ||
        (zero_point && zero_point->owns && !zero_point->is_graph_input) ||
        physical_byte_tensor(output, &output_dtype) != 0 || !physical_same_shape(input, output) ||
        physical_scalar_f32(scale, &scale_value) != 0 ||
        physical_zero_point_scalar(zero_point, output->dtype, &zero_value, &zero_i8, &zero_u8,
                                   &zero_data, &zero_dtype) != 0 ||
        scale_value != output->quantization.scale ||
        zero_value != output->quantization.zero_point) return 0;
    (void)zero_data;
    VX_RUN_W8A8(ops, quantize_typed_f32_i8u8,
            input->data, (uint32_t)input->numel, output->data,
            output->quantization.scale, output->quantization.zero_point, output_dtype);
    return 0;
}

static int dequantize_physical_forward(Node* node, T* output) {
    T* input = physical_node_data_input(node);
    T* scale = nin(node, "scale");
    T* zero_point = nin(node, "zero_point");
    float scale_value;
    int zero_value;
    int8_t zero_i8;
    uint8_t zero_u8;
    const void* zero_data;
    uint32_t zero_dtype, input_dtype;
    size_t offset = 0;
    uint32_t elements;
    int row_status = 0;
    if (!node || !input || !output || !output->data || output->dtype != T_F32 ||
        output->elem_size != sizeof(float) || output->numel <= 0 || output->numel > INT_MAX ||
        physical_byte_tensor(input, &input_dtype) != 0 || !physical_same_shape(input, output) ||
        physical_scalar_f32(scale, &scale_value) != 0 ||
        physical_zero_point_scalar(zero_point, input->dtype, &zero_value, &zero_i8, &zero_u8,
                                   &zero_data, &zero_dtype) != 0 ||
        scale_value != input->quantization.scale || zero_value != input->quantization.zero_point) return -1;
    elements = (uint32_t)output->numel;
    if (g_active_row >= 0 && !physical_incremental_tensor_dirty(scale) &&
        !physical_incremental_tensor_dirty(zero_point)) {
        row_status = physical_qdq_row_span(input, output, g_active_row,
                                            &offset, &elements);
        if (row_status < 0) return -1;
    }
    if (row_status == 0) {
        offset = 0;
        elements = (uint32_t)output->numel;
    }
    return dequantize_linear_typed((const uint8_t*)input->data + offset,
                                   (int)input_dtype, &scale_value, zero_data,
                                   (int)zero_dtype,
                                   (float*)output->data + offset,
                                   (int)elements) == 1 ? 0 : -1;
}

static int dequantize_physical_gpu_forward(Node* node, T* output,
                                           const VxW8A8Ops* ops,
                                           const char* backend_label,
                                           const char** node_backend) {
    T* input = physical_node_data_input(node);
    T* scale = nin(node, "scale");
    T* zero_point = nin(node, "zero_point");
    float scale_value;
    int zero_value;
    int8_t zero_i8;
    uint8_t zero_u8;
    const void* zero_data;
    uint32_t zero_dtype, input_dtype;
    if (!node || !input || !output || !node_backend ||
        physical_gpu_partial_execution_declined() ||
        !output->data || output->dtype != T_F32 || output->elem_size != sizeof(float) ||
        output->numel <= 0 || output->numel > UINT32_MAX ||
        (scale && scale->owns && !scale->is_graph_input) ||
        (zero_point && zero_point->owns && !zero_point->is_graph_input) ||
        physical_byte_tensor(input, &input_dtype) != 0 || !physical_same_shape(input, output) ||
        physical_scalar_f32(scale, &scale_value) != 0 ||
        physical_zero_point_scalar(zero_point, input->dtype, &zero_value, &zero_i8, &zero_u8,
                                   &zero_data, &zero_dtype) != 0 ||
        scale_value != input->quantization.scale ||
        zero_value != input->quantization.zero_point) return 0;
    (void)zero_data;
    VX_RUN_W8A8(ops, dequantize_typed_i8u8_f32,
            input->data, (uint32_t)input->numel, input->quantization.scale,
            input->quantization.zero_point, input_dtype, output->data);
    return 0;
}

/* Physical byte shape operators carry no arithmetic quantization transform.
 * They are valid only when every edge keeps the exact I8/U8 storage domain. */
static int physical_byte_hint(const T* tensor) {
    return tensor && ((tensor->dtype == T_I8 || tensor->dtype == T_U8) ||
                      tensor->quantization.valid);
}

static int physical_same_byte_domain(const T* left, const T* right,
                                     uint32_t* dtype_out) {
    uint32_t left_dtype, right_dtype;
    if (physical_byte_tensor(left, &left_dtype) != 0 ||
        physical_byte_tensor(right, &right_dtype) != 0 || left_dtype != right_dtype ||
        left->quantization.scale != right->quantization.scale ||
        left->quantization.zero_point != right->quantization.zero_point) return -1;
    if (dtype_out) *dtype_out = left_dtype;
    return 0;
}

static int physical_shape_copy_op(const char* op) {
    return op && (!strcmp(op, "Reshape") || !strcmp(op, "Flatten") ||
                  !strcmp(op, "Squeeze") || !strcmp(op, "Unsqueeze") ||
                  !strcmp(op, "Identity"));
}

static int physical_shape_supported_op(const char* op) {
    return physical_shape_copy_op(op) || (op && (!strcmp(op, "MaxPool2D") ||
        !strcmp(op, "ResizeNearest2D") || !strcmp(op, "Resize") ||
        !strcmp(op, "Transpose") || !strcmp(op, "Expand") ||
        !strcmp(op, "Concat")));
}

/* A byte hint selects the strict physical route. Complete F32 nodes remain on
 * their dedicated implementation, while a malformed typed node fails
 * closed rather than being reinterpreted as F32 storage. */
static int physical_shape_candidate(Node* node, T* output) {
    T* input;
    if (!node || !physical_shape_supported_op(node->op)) return 0;
    if (!strcmp(node->op, "Concat")) {
        if (physical_byte_hint(output)) return 1;
        for (int index = 0; index < node->nin; index++) {
            if (physical_byte_hint(t_find(node->ins[index].name))) return 1;
        }
        return 0;
    }
    input = physical_node_data_input(node);
    return physical_byte_hint(input) || physical_byte_hint(output);
}

static int physical_false_or_absent(cJSON* params, const char* name) {
    cJSON* value = params ? cJSON_GetObjectItem(params, name) : NULL;
    if (!value || cJSON_IsNull(value) || cJSON_IsFalse(value)) return 0;
    if (cJSON_IsNumber(value) && isfinite(value->valuedouble) &&
        value->valuedouble == 0.0) return 0;
    return -1;
}

static int physical_resize_nearest_parameters(const Node* node) {
    cJSON* mode;
    cJSON* transform;
    cJSON* unsupported_transform;
    cJSON* nearest_mode;
    if (!node) return -1;
    mode = node->params ? cJSON_GetObjectItem(node->params, "mode") : NULL;
    if (!strcmp(node->op, "Resize")) {
        if (!mode || !cJSON_IsString(mode) || !mode->valuestring ||
            strcmp(mode->valuestring, "nearest")) return -1;
    } else if (mode && !cJSON_IsNull(mode) &&
               (!cJSON_IsString(mode) || !mode->valuestring || strcmp(mode->valuestring, "nearest"))) {
        return -1;
    }
    /* The byte kernel implements the canonical asymmetric floor mapping. */
    transform = node->params ? cJSON_GetObjectItem(node->params, "coordinate_transformation_mode") : NULL;
    unsupported_transform = node->params ? cJSON_GetObjectItem(node->params, "coordinate_transform_mode") : NULL;
    if (unsupported_transform) return -1;
    if (transform && !cJSON_IsNull(transform) &&
        (!cJSON_IsString(transform) || !transform->valuestring ||
         strcmp(transform->valuestring, "asymmetric"))) return -1;
    nearest_mode = node->params ? cJSON_GetObjectItem(node->params, "nearest_mode") : NULL;
    if (nearest_mode && !cJSON_IsNull(nearest_mode) &&
        (!cJSON_IsString(nearest_mode) || !nearest_mode->valuestring ||
         strcmp(nearest_mode->valuestring, "floor"))) return -1;
    if (physical_false_or_absent(node->params, "align_corners") != 0 ||
        physical_false_or_absent(node->params, "antialias") != 0) return -1;
    return 0;
}

typedef struct {
    T* input;
    uint32_t dtype;
} PhysicalByteCopyDescriptor;

typedef struct {
    T* input;
    uint32_t dtype;
    uint32_t input_shape[8];
    uint32_t output_shape[8];
    uint32_t input_rank;
    uint32_t output_rank;
    uint32_t elements;
} PhysicalByteExpandDescriptor;

typedef struct {
    T* input;
    uint32_t dtype;
    uint32_t input_shape[8];
    uint32_t permutation[8];
    uint32_t rank;
    uint32_t elements;
} PhysicalByteTransposeDescriptor;

typedef struct {
    T* input;
    uint32_t dtype;
    uint32_t input_shape[4];
    uint32_t output_shape[4];
} PhysicalByteResizeDescriptor;

typedef struct {
    T* input;
    uint32_t dtype;
    uint32_t input_shape[4];
    uint32_t output_shape[4];
    uint32_t kernel[2];
    uint32_t stride[2];
    uint32_t pads[4];
} PhysicalByteMaxPoolDescriptor;

static int physical_shape_copy_validate(Node* node, T* output,
                                        PhysicalByteCopyDescriptor* descriptor) {
    T* input = physical_node_data_input(node);
    uint32_t dtype;
    if (!node || !output || !descriptor || node->nin != 1 ||
        physical_same_byte_domain(input, output, &dtype) != 0 ||
        input->numel != output->numel) return -1;
    descriptor->input = input;
    descriptor->dtype = dtype;
    return 0;
}

static int physical_shape_copy_forward(Node* node, T* output) {
    PhysicalByteCopyDescriptor descriptor;
    if (physical_shape_copy_validate(node, output, &descriptor) != 0) return -1;
    return copy_i8u8((const uint8_t*)descriptor.input->data, (uint8_t*)output->data,
                     (uint32_t)output->numel, descriptor.dtype) == 1 ? 0 : -1;
}

static int physical_shape_expand_validate(
        Node* node, T* output, PhysicalByteExpandDescriptor* descriptor) {
    T* input = physical_node_data_input(node);
    uint64_t input_elements = 1;
    uint64_t output_elements = 1;
    uint32_t dtype;
    if (!node || !output || !descriptor || node->nin != 1 ||
        physical_same_byte_domain(input, output, &dtype) != 0 ||
        input->data == output->data || input->ndim <= 0 || input->ndim > 8 ||
        output->ndim < input->ndim || output->ndim > 8 ||
        input->numel <= 0 || output->numel <= 0 ||
        (unsigned long)output->numel > UINT32_MAX ||
        (node->params && node->params->child)) return -1;
    for (int dimension = 0; dimension < input->ndim; dimension++) {
        if (input->shape[dimension] <= 0 ||
            input_elements > UINT32_MAX / (uint32_t)input->shape[dimension])
            return -1;
        input_elements *= (uint32_t)input->shape[dimension];
        descriptor->input_shape[dimension] = (uint32_t)input->shape[dimension];
    }
    for (int dimension = 0; dimension < output->ndim; dimension++) {
        int input_dimension = dimension - (output->ndim - input->ndim);
        uint32_t source = input_dimension < 0
            ? 1u : (uint32_t)input->shape[input_dimension];
        uint32_t target;
        if (output->shape[dimension] <= 0 ||
            output_elements > UINT32_MAX / (uint32_t)output->shape[dimension])
            return -1;
        target = (uint32_t)output->shape[dimension];
        if (source != 1u && source != target) return -1;
        output_elements *= target;
        descriptor->output_shape[dimension] = target;
    }
    if (input_elements != (uint64_t)input->numel ||
        output_elements != (uint64_t)output->numel ||
        output_elements < input_elements) return -1;
    descriptor->input = input;
    descriptor->dtype = dtype;
    descriptor->input_rank = (uint32_t)input->ndim;
    descriptor->output_rank = (uint32_t)output->ndim;
    descriptor->elements = (uint32_t)output_elements;
    return 0;
}

static int physical_shape_expand_forward(Node* node, T* output) {
    PhysicalByteExpandDescriptor descriptor;
    int row_start;
    int row_count;
    if (physical_shape_expand_validate(node, output, &descriptor) != 0)
        return -1;
    if (output->ndim == 3) {
        seq_range(output->shape[1], &row_start, &row_count);
        if (row_count != output->shape[1]) {
            uint32_t row_output_shape[8] = {0};
            uint32_t row_elements;
            if (row_count <= 0 || descriptor.input_rank != 3 ||
                descriptor.output_rank != 3 ||
                descriptor.input_shape[0] != 1u ||
                descriptor.input_shape[1] != 1u ||
                descriptor.output_shape[0] != 1u ||
                descriptor.input_shape[2] != descriptor.output_shape[2] ||
                row_start < 0 || row_start >= output->shape[1] ||
                row_count > output->shape[1] - row_start ||
                descriptor.output_shape[2] >
                    UINT32_MAX / (uint32_t)row_count) return -1;
            memcpy(row_output_shape, descriptor.output_shape,
                   descriptor.output_rank * sizeof(uint32_t));
            row_output_shape[1] = (uint32_t)row_count;
            row_elements = (uint32_t)row_count * descriptor.output_shape[2];
            return expand_nd_i8u8(
                (const uint8_t*)descriptor.input->data,
                (uint8_t*)output->data +
                    (size_t)row_start * descriptor.output_shape[2],
                descriptor.input_shape, row_output_shape,
                descriptor.input_rank, descriptor.output_rank,
                row_elements) == 1 ? 0 : -1;
        }
    }
    return expand_nd_i8u8(
        (const uint8_t*)descriptor.input->data, (uint8_t*)output->data,
        descriptor.input_shape, descriptor.output_shape,
        descriptor.input_rank, descriptor.output_rank,
        descriptor.elements) == 1 ? 0 : -1;
}

static int physical_shape_transpose_validate(
        Node* node, T* output, PhysicalByteTransposeDescriptor* descriptor) {
    T* input = physical_node_data_input(node);
    cJSON* permutation_value;
    uint32_t seen = 0;
    uint64_t elements = 1;
    uint32_t dtype;
    if (!node || !output || !descriptor || node->nin != 1 ||
        physical_same_byte_domain(input, output, &dtype) != 0 ||
        input->data == output->data || input->ndim <= 0 || input->ndim > 8 ||
        output->ndim != input->ndim || input->numel <= 0 ||
        input->numel != output->numel || (unsigned long)input->numel > UINT32_MAX)
        return -1;
    for (cJSON* parameter = node->params ? node->params->child : NULL;
         parameter; parameter = parameter->next) {
        if (!parameter->string || strcmp(parameter->string, "perm")) return -1;
    }
    permutation_value = node->params
        ? cJSON_GetObjectItem(node->params, "perm") : NULL;
    if (permutation_value &&
        (!cJSON_IsArray(permutation_value) ||
         cJSON_GetArraySize(permutation_value) != input->ndim)) return -1;
    for (int dimension = 0; dimension < input->ndim; dimension++) {
        int source = input->ndim - 1 - dimension;
        if (input->shape[dimension] <= 0 ||
            elements > UINT32_MAX / (uint32_t)input->shape[dimension]) return -1;
        elements *= (uint32_t)input->shape[dimension];
        descriptor->input_shape[dimension] = (uint32_t)input->shape[dimension];
        if (permutation_value) {
            cJSON* item = cJSON_GetArrayItem(permutation_value, dimension);
            if (!item || !cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
                item->valuedouble != trunc(item->valuedouble) ||
                item->valuedouble < 0.0 ||
                item->valuedouble >= (double)input->ndim) return -1;
            source = (int)item->valuedouble;
        }
        if (seen & (1u << (unsigned)source)) return -1;
        seen |= 1u << (unsigned)source;
        if (output->shape[dimension] != input->shape[source]) return -1;
        descriptor->permutation[dimension] = (uint32_t)source;
    }
    if (elements != (uint64_t)input->numel) return -1;
    descriptor->input = input;
    descriptor->dtype = dtype;
    descriptor->rank = (uint32_t)input->ndim;
    descriptor->elements = (uint32_t)elements;
    return 0;
}

static int physical_shape_transpose_forward(Node* node, T* output) {
    PhysicalByteTransposeDescriptor descriptor;
    if (physical_shape_transpose_validate(node, output, &descriptor) != 0)
        return -1;
    return vx_transpose_nd_i8u8_native_validated(
        (const uint8_t*)descriptor.input->data, (uint8_t*)output->data,
        descriptor.input_shape, descriptor.permutation, descriptor.rank,
        descriptor.elements, descriptor.dtype) == 1 ? 0 : -1;
}

static int physical_shape_model_validate(Node* node, T* output) {
    if (!node) return 0;
    if (!strcmp(node->op, "Transpose")) {
        PhysicalByteTransposeDescriptor descriptor;
        return physical_shape_transpose_validate(node, output, &descriptor);
    }
    if (!strcmp(node->op, "Expand")) {
        PhysicalByteExpandDescriptor descriptor;
        return physical_shape_expand_validate(node, output, &descriptor);
    }
    return 0;
}

static int physical_shape_resize_validate(Node* node, T* output,
                                          PhysicalByteResizeDescriptor* descriptor) {
    T* input = physical_node_data_input(node);
    uint32_t dtype;
    if (!node || !output || !descriptor || node->nin != 1 ||
        physical_same_byte_domain(input, output, &dtype) != 0 ||
        qconv2d_shape4(input, descriptor->input_shape) != 0 ||
        qconv2d_shape4(output, descriptor->output_shape) != 0 ||
        descriptor->input_shape[0] != descriptor->output_shape[0] ||
        descriptor->input_shape[3] != descriptor->output_shape[3] ||
        physical_resize_nearest_parameters(node) != 0) return -1;
    descriptor->input = input;
    descriptor->dtype = dtype;
    return 0;
}

static int physical_shape_resize_forward(Node* node, T* output) {
    PhysicalByteResizeDescriptor descriptor;
    if (physical_shape_resize_validate(node, output, &descriptor) != 0) return -1;
    if (descriptor.input->data == output->data) {
        if (!physical_same_shape(descriptor.input, output)) return -1;
        return copy_i8u8((const uint8_t*)descriptor.input->data, (uint8_t*)output->data,
                         (uint32_t)output->numel, descriptor.dtype) == 1 ? 0 : -1;
    }
    return resize_nearest2d_i8u8((const uint8_t*)descriptor.input->data, (uint8_t*)output->data,
                                 descriptor.input_shape[0], descriptor.input_shape[1],
                                 descriptor.input_shape[2], descriptor.input_shape[3],
                                 descriptor.output_shape[1], descriptor.output_shape[2],
                                 descriptor.dtype) == 1 ? 0 : -1;
}

static int physical_shape_maxpool_validate(Node* node, T* output,
                                           PhysicalByteMaxPoolDescriptor* descriptor) {
    T* input = physical_node_data_input(node);
    uint32_t dtype;
    uint32_t dilation[2];
    uint32_t padding[2];
    uint64_t padded_height;
    uint64_t padded_width;
    uint64_t expected_height;
    uint64_t expected_width;
    cJSON* kernel_value;
    if (!node || !output || !descriptor || node->nin != 1 ||
        physical_same_byte_domain(input, output, &dtype) != 0 ||
        qconv2d_shape4(input, descriptor->input_shape) != 0 ||
        qconv2d_shape4(output, descriptor->output_shape) != 0 ||
        descriptor->input_shape[0] != descriptor->output_shape[0] ||
        descriptor->input_shape[3] != descriptor->output_shape[3] ||
        input->data == output->data || physical_false_or_absent(node->params, "ceil_mode") != 0) return -1;
    kernel_value = node->params ? cJSON_GetObjectItem(node->params, "kernel") : NULL;
    if (!kernel_value || cJSON_IsNull(kernel_value) ||
        qconv2d_pair_parameter(node->params, "kernel", 1, 1, descriptor->kernel) != 0 ||
        qconv2d_pair_parameter(node->params, "stride", 1, 1, descriptor->stride) != 0 ||
        qconv2d_pair_parameter(node->params, "dilation", 1, 1, dilation) != 0 ||
        qconv2d_pair_parameter(node->params, "padding", 0, 0, padding) != 0 ||
        qconv2d_pads_parameter(node->params, padding, descriptor->pads) != 0 ||
        dilation[0] != 1 || dilation[1] != 1) return -1;
    padded_height = (uint64_t)descriptor->input_shape[1] + descriptor->pads[0] + descriptor->pads[2];
    padded_width = (uint64_t)descriptor->input_shape[2] + descriptor->pads[1] + descriptor->pads[3];
    if (padded_height < descriptor->kernel[0] || padded_width < descriptor->kernel[1]) return -1;
    expected_height = (padded_height - descriptor->kernel[0]) / descriptor->stride[0] + 1u;
    expected_width = (padded_width - descriptor->kernel[1]) / descriptor->stride[1] + 1u;
    if (expected_height != descriptor->output_shape[1] || expected_width != descriptor->output_shape[2]) return -1;
    descriptor->input = input;
    descriptor->dtype = dtype;
    return 0;
}

static int physical_shape_maxpool_forward(Node* node, T* output) {
    PhysicalByteMaxPoolDescriptor descriptor;
    if (physical_shape_maxpool_validate(node, output, &descriptor) != 0) return -1;
    return maxpool2d_i8u8((const uint8_t*)descriptor.input->data, (uint8_t*)output->data,
                          descriptor.input_shape[0], descriptor.input_shape[1],
                          descriptor.input_shape[2], descriptor.input_shape[3],
                          descriptor.output_shape[1], descriptor.output_shape[2],
                          descriptor.kernel[0], descriptor.kernel[1],
                          descriptor.stride[0], descriptor.stride[1],
                          descriptor.pads[0], descriptor.pads[1], descriptor.dtype) == 1 ? 0 : -1;
}

static int physical_concat_numeric_input_key(const char* key, uint64_t* value) {
    uint64_t parsed = 0;
    const char* cursor;
    int overflow = 0;
    if (!key || strncmp(key, "input", 5) || !key[5]) return 0;
    for (cursor = key + 5; *cursor; cursor++) {
        unsigned digit;
        if (*cursor < '0' || *cursor > '9') return 0;
        digit = (unsigned)(*cursor - '0');
        if (parsed > (UINT64_MAX - digit) / 10u) overflow = 1;
        else if (!overflow) parsed = parsed * 10u + digit;
    }
    if (value) *value = overflow ? UINT64_MAX : parsed;
    return 1;
}

static int physical_concat_key_compare(const char* left, const char* right) {
    uint64_t left_index;
    uint64_t right_index;
    int left_numeric = physical_concat_numeric_input_key(left, &left_index);
    int right_numeric = physical_concat_numeric_input_key(right, &right_index);
    if (left_numeric && right_numeric) {
        if (left_index < right_index) return -1;
        if (left_index > right_index) return 1;
        return 0;  /* JavaScript's stable sort retains declaration order on ties. */
    }
    return strcmp(left, right);
}

static int physical_concat_order(const Node* node, int order[MAXIN], int* count) {
    static const char* const preferred[] = {"input", "a", "b", "c", "d", "e", "f", "g", "h"};
    int selected[MAXIN] = {0};
    int entries = 0;
    if (!node || !order || !count || node->nin <= 0 || node->nin > MAXIN) return -1;
    for (int left = 0; left < node->nin; left++) {
        if (!node->ins[left].key[0]) return -1;
        for (int right = left + 1; right < node->nin; right++) {
            if (!strcmp(node->ins[left].key, node->ins[right].key)) return -1;
        }
    }
    for (size_t preferred_index = 0;
         preferred_index < sizeof(preferred) / sizeof(preferred[0]); preferred_index++) {
        for (int input_index = 0; input_index < node->nin; input_index++) {
            if (!strcmp(node->ins[input_index].key, preferred[preferred_index])) {
                selected[input_index] = 1;
                order[entries++] = input_index;
                break;
            }
        }
    }
    int first_sorted = entries;
    for (int input_index = 0; input_index < node->nin; input_index++) {
        if (!selected[input_index]) order[entries++] = input_index;
    }
    for (int input_index = first_sorted + 1; input_index < entries; input_index++) {
        int current = order[input_index];
        int insert = input_index;
        while (insert > first_sorted &&
               physical_concat_key_compare(node->ins[current].key,
                                           node->ins[order[insert - 1]].key) < 0) {
            order[insert] = order[insert - 1];
            insert--;
        }
        order[insert] = current;
    }
    *count = entries;
    return 0;
}

static int physical_concat_axis(const Node* node, int rank, int* axis) {
    cJSON* value = node && node->params ? cJSON_GetObjectItem(node->params, "axis") : NULL;
    double parsed = 0.0;
    int normalized;
    if (!axis || rank <= 0) return -1;
    if (value && !cJSON_IsNull(value)) {
        if (!cJSON_IsNumber(value) || !isfinite(value->valuedouble) ||
            value->valuedouble != trunc(value->valuedouble) ||
            value->valuedouble < (double)INT_MIN || value->valuedouble > (double)INT_MAX) return -1;
        parsed = value->valuedouble;
    }
    normalized = (int)parsed;
    if (normalized < 0) {
        if (normalized < -rank) return -1;
        normalized += rank;
    }
    if (normalized < 0 || normalized >= rank) return -1;
    *axis = normalized;
    return 0;
}

typedef struct {
    T* sources[MAXIN];
    uint32_t source_axes[MAXIN];
    uint32_t dtype;
    uint32_t outer;
    uint32_t inner;
    uint32_t output_axis;
    int source_count;
} PhysicalByteConcatDescriptor;

static int physical_shape_concat_validate(Node* node, int node_index, T* output,
                                          PhysicalByteConcatDescriptor* descriptor) {
    uint64_t outer = 1;
    uint64_t inner = 1;
    uint64_t axis_sum = 0;
    int order[MAXIN];
    int source_count;
    int axis;
    if (!node || !output || !descriptor ||
        physical_byte_tensor(output, &descriptor->dtype) != 0 ||
        physical_concat_order(node, order, &source_count) != 0 ||
        physical_concat_axis(node, output->ndim, &axis) != 0 ||
        physical_false_or_absent(node->params, "sigmoid") != 0 ||
        (node_index >= 0 && node_index < MAXN && g_concat_sigmoid_fuse[node_index] >= 0)) return -1;
    for (int dimension = 0; dimension < axis; dimension++) {
        if (output->shape[dimension] <= 0 || outer > UINT32_MAX / (uint32_t)output->shape[dimension]) return -1;
        outer *= (uint32_t)output->shape[dimension];
    }
    for (int dimension = axis + 1; dimension < output->ndim; dimension++) {
        if (output->shape[dimension] <= 0 || inner > UINT32_MAX / (uint32_t)output->shape[dimension]) return -1;
        inner *= (uint32_t)output->shape[dimension];
    }
    if (output->shape[axis] <= 0 || outer == 0 || inner == 0 ||
        outer > UINT32_MAX / (uint32_t)output->shape[axis] ||
        outer * (uint32_t)output->shape[axis] > UINT32_MAX / inner ||
        outer * (uint32_t)output->shape[axis] * inner != (uint64_t)output->numel) return -1;
    for (int index = 0; index < source_count; index++) {
        T* source = t_find(node->ins[order[index]].name);
        uint32_t source_dtype;
        if (physical_byte_tensor(source, &source_dtype) != 0 || source_dtype != descriptor->dtype ||
            source->dtype != output->dtype || source->quantization.scale != output->quantization.scale ||
            source->quantization.zero_point != output->quantization.zero_point ||
            source->data == output->data || source->ndim != output->ndim ||
            source->shape[axis] <= 0) return -1;
        for (int dimension = 0; dimension < output->ndim; dimension++) {
            if (dimension != axis && source->shape[dimension] != output->shape[dimension]) return -1;
        }
        descriptor->source_axes[index] = (uint32_t)source->shape[axis];
        if ((uint64_t)outer * descriptor->source_axes[index] > UINT32_MAX / inner ||
            source->numel != (long)((uint64_t)outer * descriptor->source_axes[index] * inner) ||
            axis_sum > UINT32_MAX - descriptor->source_axes[index]) return -1;
        axis_sum += descriptor->source_axes[index];
        descriptor->sources[index] = source;
    }
    if (axis_sum != (uint32_t)output->shape[axis]) return -1;
    descriptor->outer = (uint32_t)outer;
    descriptor->inner = (uint32_t)inner;
    descriptor->output_axis = (uint32_t)output->shape[axis];
    descriptor->source_count = source_count;
    return 0;
}

static int physical_shape_concat_forward(Node* node, int node_index, T* output) {
    PhysicalByteConcatDescriptor descriptor;
    if (physical_shape_concat_validate(node, node_index, output, &descriptor) != 0) return -1;
    uint32_t axis_offset = 0;
    for (int index = 0; index < descriptor.source_count; index++) {
        if (concat_slice_i8u8((const uint8_t*)descriptor.sources[index]->data, (uint8_t*)output->data,
                              descriptor.outer, descriptor.source_axes[index],
                              descriptor.output_axis, descriptor.inner,
                              axis_offset, descriptor.dtype) != 1) return -1;
        axis_offset += descriptor.source_axes[index];
    }
    return 0;
}

static int physical_shape_forward(Node* node, int node_index, T* output) {
    if (!node || !output) return -1;
    if (physical_shape_copy_op(node->op)) return physical_shape_copy_forward(node, output);
    if (!strcmp(node->op, "Transpose"))
        return physical_shape_transpose_forward(node, output);
    if (!strcmp(node->op, "Expand"))
        return physical_shape_expand_forward(node, output);
    if (!strcmp(node->op, "ResizeNearest2D") || !strcmp(node->op, "Resize"))
        return physical_shape_resize_forward(node, output);
    if (!strcmp(node->op, "MaxPool2D")) return physical_shape_maxpool_forward(node, output);
    if (!strcmp(node->op, "Concat")) return physical_shape_concat_forward(node, node_index, output);
    return -1;
}

static int physical_shape_gpu_alias_forward(const char* backend_label,
                                            const char** node_backend) {
    if (!backend_label || !node_backend) return 0;
    *node_backend = backend_label;
    return 1;
}

static int physical_shape_copy_gpu_forward(Node* node, T* output,
                                           const VxW8A8Ops* ops,
                                           const char* alias_label,
                                           const char* backend_label,
                                           const char** node_backend) {
    PhysicalByteCopyDescriptor descriptor;
    if (physical_shape_copy_validate(node, output, &descriptor) != 0) return 0;
    if (descriptor.input->data == output->data)
        return physical_shape_gpu_alias_forward(alias_label, node_backend);
    VX_RUN_W8A8(ops, copy_i8u8,
            descriptor.input->data, (uint32_t)descriptor.input->numel,
            output->data, (uint32_t)output->numel,
            descriptor.input->quantization.scale, descriptor.input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            descriptor.dtype, descriptor.dtype);
    return 0;
}

static int physical_shape_resize_gpu_forward(Node* node, T* output,
                                             const VxW8A8Ops* ops,
                                             const char* alias_label,
                                             const char* backend_label,
                                             const char** node_backend) {
    PhysicalByteResizeDescriptor descriptor;
    if (physical_shape_resize_validate(node, output, &descriptor) != 0) return 0;
    if (descriptor.input->data == output->data) {
        if (!physical_same_shape(descriptor.input, output)) return 0;
        return physical_shape_gpu_alias_forward(alias_label, node_backend);
    }
    VX_RUN_W8A8(ops, resize_nearest_i8u8,
            descriptor.input->data, output->data,
            descriptor.input_shape[0], descriptor.input_shape[1], descriptor.input_shape[2],
            descriptor.input_shape[3], descriptor.output_shape[1], descriptor.output_shape[2],
            descriptor.input->quantization.scale, descriptor.input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            descriptor.dtype, descriptor.dtype);
    return 0;
}

static int physical_shape_transpose_gpu_forward(
        Node* node, T* output, const VxW8A8Ops* ops,
        const char* backend_label, const char** node_backend) {
    PhysicalByteTransposeDescriptor descriptor;
    if (physical_shape_transpose_validate(node, output, &descriptor) != 0)
        return 0;
    VX_RUN_W8A8(ops, transpose_i8u8,
            descriptor.input->data, output->data, descriptor.input_shape,
            descriptor.permutation, descriptor.rank, descriptor.elements,
            descriptor.input->quantization.scale,
            descriptor.input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            descriptor.dtype, descriptor.dtype);
    return 0;
}

static int physical_shape_maxpool_gpu_forward(Node* node, T* output,
                                              const VxW8A8Ops* ops,
                                              const char* backend_label,
                                              const char** node_backend) {
    PhysicalByteMaxPoolDescriptor descriptor;
    if (physical_shape_maxpool_validate(node, output, &descriptor) != 0) return 0;
    VX_RUN_W8A8(ops, maxpool2d_i8u8,
            descriptor.input->data, output->data,
            descriptor.input_shape[0], descriptor.input_shape[1], descriptor.input_shape[2],
            descriptor.input_shape[3], descriptor.output_shape[1], descriptor.output_shape[2],
            descriptor.kernel[0], descriptor.kernel[1], descriptor.stride[0], descriptor.stride[1],
            descriptor.pads[0], descriptor.pads[1], descriptor.pads[2], descriptor.pads[3],
            descriptor.input->quantization.scale, descriptor.input->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            descriptor.dtype, descriptor.dtype);
    return 0;
}

static int physical_shape_concat_gpu_forward(Node* node, int node_index, T* output,
                                             const VxW8A8Ops* ops,
                                             const char* backend_label,
                                             const char** node_backend) {
    PhysicalByteConcatDescriptor descriptor;
    const void* inputs[MAXIN];
    uint32_t input_elements[MAXIN];
    float input_scales[MAXIN];
    int32_t input_zero_points[MAXIN];
    uint32_t input_dtypes[MAXIN];
    if (physical_shape_concat_validate(node, node_index, output, &descriptor) != 0) return 0;
    for (int index = 0; index < descriptor.source_count; index++) {
        inputs[index] = descriptor.sources[index]->data;
        input_elements[index] = (uint32_t)descriptor.sources[index]->numel;
        input_scales[index] = descriptor.sources[index]->quantization.scale;
        input_zero_points[index] = (int32_t)descriptor.sources[index]->quantization.zero_point;
        input_dtypes[index] = descriptor.dtype;
    }
    VX_RUN_W8A8(ops, concat_i8u8,
            inputs, input_elements, descriptor.source_axes, input_scales, input_zero_points,
            input_dtypes, (uint32_t)descriptor.source_count, output->data,
            (uint32_t)output->numel, descriptor.output_axis, descriptor.inner,
            output->quantization.scale, output->quantization.zero_point, descriptor.dtype);
    return 0;
}

static int physical_shape_gpu_forward(Node* node, int node_index, T* output,
                                      const VxW8A8Ops* ops,
                                      const char* alias_label,
                                      const char* copy_label,
                                      const char* transpose_label,
                                      const char* resize_label,
                                      const char* maxpool_label,
                                      const char* concat_label,
                                      const char** node_backend) {
    if (!node || !output || !node_backend ||
        physical_gpu_partial_execution_declined()) return 0;
    if (physical_shape_copy_op(node->op))
        return physical_shape_copy_gpu_forward(node, output, ops, alias_label, copy_label, node_backend);
    if (!strcmp(node->op, "Transpose"))
        return physical_shape_transpose_gpu_forward(
            node, output, ops, transpose_label, node_backend);
    if (!strcmp(node->op, "ResizeNearest2D") || !strcmp(node->op, "Resize"))
        return physical_shape_resize_gpu_forward(node, output, ops, alias_label, resize_label, node_backend);
    if (!strcmp(node->op, "MaxPool2D"))
        return physical_shape_maxpool_gpu_forward(node, output, ops, maxpool_label, node_backend);
    if (!strcmp(node->op, "Concat"))
        return physical_shape_concat_gpu_forward(node, node_index, output, ops, concat_label, node_backend);
    return 0;
}
