/* Vulkan bounded-forward graph paths for the W8A8 quantized operators.
 *
 * Fragment of vulkan_engine.c: these reach the file-static device, pipeline
 * cache and slot arena, so they stay in that translation unit. */

int vk_graph_qlinear_i8u8(const void* input, const void* weight,
                          const float* weight_scales, const int32_t* weight_zero_points,
                          const int32_t* bias, void* output,
                          uint32_t rows, uint32_t d_in, uint32_t d_out,
                          float input_scale, int32_t input_zero_point,
                          float output_scale, int32_t output_zero_point,
                          uint32_t input_dtype, uint32_t weight_dtype,
                          uint32_t output_dtype) {
    size_t input_bytes, weight_bytes, output_bytes;
    size_t multiplier_bytes;
    float* multipliers;
    if (!qlinear_gpu_args_valid(input, weight, weight_scales, weight_zero_points, bias, output,
                                rows, d_in, d_out, input_scale, input_zero_point,
                                output_scale, output_zero_point, input_dtype, weight_dtype,
                                output_dtype, &input_bytes, &weight_bytes, &output_bytes)) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    multiplier_bytes = (size_t)d_out * sizeof(*multipliers);
    multipliers = (float*)malloc(multiplier_bytes);
    if (!multipliers ||
        !vx_qlinear_build_multipliers(
            input_scale, weight_scales, output_scale, d_out, multipliers)) {
        free(multipliers);
        return 0;
    }
    graph_scratch_begin();
    /* Activations may be a row window; weights never are. */
    VkGraphWindow src, dst;
    int resolved = graph_window_packed_bytes(input, input_bytes, 0, &src) &&
        graph_window_output_packed(output, output_bytes, &dst);
    VkTensorSlot* wt = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    VkTensorSlot* zero_points = graph_ensure_device(weight_zero_points,
                                                     (size_t)d_out * sizeof(*weight_zero_points), 1);
    VkTensorSlot* biases = graph_ensure_device(bias, (size_t)d_out * sizeof(*bias), 1);
    if (!resolved || !wt || !zero_points || !biases) {
        free(multipliers);
        return 0;
    }
    size_t multiplier_offset =
        graph_scratch_upload(multipliers, multiplier_bytes);
    free(multipliers);
    if (multiplier_offset == SIZE_MAX) return 0;
    VkQLinearParams params = {
        rows, d_in, d_out,
        input_dtype,
        weight_dtype,
        output_dtype, 0u, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[7] = {
        {src.offset, src.bytes}, {wt->offset, wt->bytes},
        {multiplier_offset, multiplier_bytes},
        {zero_points->offset, zero_points->bytes},
        {biases->offset, biases->bytes}, {dst.offset, dst.bytes},
        {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    int tiled = rows > 1u && d_in >= 16u && d_out >= 32u && (d_out & 3u) == 0u;
    uint32_t groups_x = tiled ? ((d_out / 4u + 7u) / 8u) : ((packed_words + 63u) / 64u);
    uint32_t groups_y = tiled ? ((rows + 7u) / 8u) : 1u;
    int dispatched = 0;
    if (vk_packed_dot_enabled()) {
        VkKernel* dot_kernel = tiled
            ? &k_qlinear_int8_dot_tiled : &k_qlinear_int8_dot;
        dispatched = vk_dispatch_kernel(dot_kernel, binds,
                                        groups_x, groups_y, 1u);
        if (!dispatched) vk_disable_packed_dot_after_failure(dot_kernel->name);
        else vk_context_current()->qlinear_dot_dispatches++;
    }
    if (!dispatched) {
        VkKernel* kernel = tiled ? &k_qlinear_int8_tiled : &k_qlinear_int8;
        if (!vk_dispatch_kernel(kernel, binds, groups_x, groups_y, 1u)) return 0;
        if (tiled) vk_context_current()->qlinear_tiled_dispatches++;
        else vk_context_current()->qlinear_scalar_dispatches++;
    }
    graph_mark_device(dst.slot);
    return 1;
}

typedef struct {
    uint32_t tokens;
    uint32_t vocab;
    uint32_t hidden;
    uint32_t weight_type;
    uint32_t output_type;
    int32_t output_zero_point;
    float output_scale;
    uint32_t token_offset;
} VkQEmbeddingParams;

_Static_assert(sizeof(VkQEmbeddingParams) == 32, "qEmbeddingInt8 uniform ABI");

/*
 * Where in the ids tensor this dispatch starts, as a token index.
 *
 * Every other activation here binds as a window, because a row's byte offset is
 * `row * stride` and that is a multiple of the storage alignment whenever the
 * stride is. A token row is a single i32. Four bytes never divide 256, so an ids
 * window resolves for row 0 and for nothing else, and a decode row would take
 * this node to the host on every step while its kernel sat unused.
 *
 * So the ids bind whole and the row is named as a scalar instead, where the
 * alignment rule does not apply. The offset is in tokens rather than bytes
 * because that is what the shader indexes with.
 */
static int graph_token_window(const int32_t* tokens, size_t token_bytes,
                              VkTensorSlot** slot, uint32_t* token_offset) {
    const void* base = tokens;
    size_t base_bytes = token_bytes;
    size_t inner = 0;
    int index;
    if (!tokens || !token_bytes || !slot || !token_offset) return 0;
    *token_offset = 0;
    index = graph_find_containing_slot(tokens, token_bytes, &inner);
    if (index >= 0 && inner) {
        /* A misaligned interior pointer is not a token boundary, so it is not
         * a row of this tensor and nothing here can address it. */
        if (inner % sizeof(int32_t)) return 0;
        if (inner / sizeof(int32_t) > UINT32_MAX) return 0;
        base = graph_slots[index].host;
        base_bytes = graph_slots[index].bytes;
        *token_offset = (uint32_t)(inner / sizeof(int32_t));
    }
    *slot = graph_ensure_device(base, base_bytes, 0);
    return *slot != NULL;
}

static int qembedding_gpu_args_valid(const int32_t* tokens, const void* weight,
                                     const float* weight_scales,
                                     const int32_t* weight_zero_points, void* output,
                                     uint32_t token_count, uint32_t vocab, uint32_t hidden,
                                     float output_scale, int32_t output_zero_point,
                                     uint32_t weight_dtype, uint32_t output_dtype,
                                     size_t* token_bytes, size_t* weight_bytes,
                                     size_t* output_bytes) {
    int64_t distance = 0;
    uint64_t weight_elements;
    uint64_t output_elements;
    if (!tokens || !weight || !weight_scales || !weight_zero_points || !output ||
        !token_count || !vocab || !hidden || !isfinite(output_scale) ||
        output_scale <= 0.0f || !qlinear_dtype_bounds(output_dtype, output_zero_point,
                                                        &distance) ||
        (weight_dtype != VX_DTYPE_I8 && weight_dtype != VX_DTYPE_U8) ||
        token_count > UINT32_MAX / hidden || vocab > UINT32_MAX / hidden) return 0;
    weight_elements = (uint64_t)vocab * hidden;
    output_elements = (uint64_t)token_count * hidden;
    if (weight_elements > SIZE_MAX || output_elements > SIZE_MAX ||
        (size_t)token_count > SIZE_MAX / sizeof(*tokens) ||
        (size_t)vocab > SIZE_MAX / sizeof(*weight_scales) ||
        (size_t)vocab > SIZE_MAX / sizeof(*weight_zero_points)) return 0;
    for (uint32_t row = 0; row < vocab; row++) {
        if (!isfinite(weight_scales[row]) || weight_scales[row] <= 0.0f ||
            !qlinear_dtype_bounds(weight_dtype, weight_zero_points[row], &distance)) return 0;
    }
    for (uint32_t token = 0; token < token_count; token++) {
        if (tokens[token] < 0 || (uint32_t)tokens[token] >= vocab) return 0;
    }
    *token_bytes = (size_t)token_count * sizeof(*tokens);
    *weight_bytes = (size_t)weight_elements;
    *output_bytes = (size_t)output_elements;
    return 1;
}

int vk_graph_qembedding_i8u8(const int32_t* tokens, const void* weight,
                             const float* weight_scales,
                             const int32_t* weight_zero_points, void* output,
                             uint32_t token_count, uint32_t vocab, uint32_t hidden,
                             float output_scale, int32_t output_zero_point,
                             uint32_t weight_dtype, uint32_t output_dtype) {
    size_t token_bytes;
    size_t weight_bytes;
    size_t output_bytes;
    size_t packed_output_bytes;
    if (!qembedding_gpu_args_valid(tokens, weight, weight_scales, weight_zero_points,
                                   output, token_count, vocab, hidden, output_scale,
                                   output_zero_point, weight_dtype, output_dtype,
                                   &token_bytes, &weight_bytes, &output_bytes) ||
        !graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkGraphWindow dst;
    VkTensorSlot* ids = NULL;
    uint32_t token_offset = 0;
    int resolved = graph_token_window(tokens, token_bytes, &ids, &token_offset) &&
        graph_window_output_packed(output, output_bytes, &dst);
    VkTensorSlot* table = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    VkTensorSlot* scales = graph_ensure_device(weight_scales,
                                                (size_t)vocab * sizeof(*weight_scales), 1);
    VkTensorSlot* zero_points = graph_ensure_device(weight_zero_points,
                                                     (size_t)vocab * sizeof(*weight_zero_points), 1);
    if (!resolved || !table || !scales || !zero_points) return 0;
    VkQEmbeddingParams params = {
        token_count, vocab, hidden,
        weight_dtype,
        output_dtype, output_zero_point,
        output_scale, token_offset
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[6] = {
        {ids->offset, ids->bytes}, {table->offset, table->bytes},
        {scales->offset, scales->bytes}, {zero_points->offset, zero_points->bytes},
        {dst.offset, dst.bytes}, {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_qembedding_int8, binds,
                            (packed_words + 63u) / 64u, 1u, 1u)) return 0;
    graph_mark_device(dst.slot);
    return 1;
}

typedef struct {
    uint32_t elements;
    uint32_t a_type;
    uint32_t b_type;
    uint32_t output_type;
    int32_t a_zero_point;
    int32_t b_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    float a_scale;
    float b_scale;
    float output_scale;
    uint32_t relu;
} VkQAddParams;

_Static_assert(sizeof(VkQAddParams) == 48, "qAdd uniform ABI");

typedef struct {
    uint32_t batch_rank;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t output_elements;
    uint32_t a_type;
    uint32_t b_type;
    uint32_t output_type;
    int32_t a_zero_point;
    int32_t b_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    float a_scale;
    float b_scale;
    float output_scale;
    float pad1;
} VkQBatchMatMulParams;

_Static_assert(sizeof(VkQBatchMatMulParams) == 64,
               "qBatchMatMul uniform ABI");

typedef struct {
    uint32_t elements;
    uint32_t input_type;
    uint32_t output_type;
    uint32_t pad0;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad1;
    int32_t pad2;
    float input_scale;
    float output_scale;
    float pad3;
    float pad4;
} VkQByteUnaryParams;

_Static_assert(sizeof(VkQByteUnaryParams) == 48,
               "qSiLUInt8/qGELUInt8 uniform ABI");

typedef struct {
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t groups;
    uint32_t input_type;
    uint32_t output_type;
    uint32_t pad0;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad1;
    int32_t pad2;
    float input_scale;
    float output_scale;
    float epsilon;
    float pad3;
} VkQGroupNormParams;

_Static_assert(sizeof(VkQGroupNormParams) == 64,
               "qGroupNormStats/qGroupNormApply uniform ABI");

typedef struct {
    uint32_t rows;
    uint32_t d_model;
    uint32_t input_type;
    uint32_t output_type;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    int32_t pad1;
    float input_scale;
    float output_scale;
    float epsilon;
    float pad2;
} VkQLayerNormParams;

_Static_assert(sizeof(VkQLayerNormParams) == 48,
               "qLayerNormStats/qLayerNormApply uniform ABI");

/* Five tightly packed 16-byte blocks shared verbatim with qSDPAInt8.wgsl.
 * The four canonical protobuf dtype values occupy byte lanes in one u32 so
 * every backend uses the same 80-byte uniform layout. */
typedef struct {
    uint32_t seq_q;
    uint32_t seq_kv;
    uint32_t d_model;
    uint32_t heads;
    uint32_t batch;
    uint32_t mask_mode;
    uint32_t causal;
    uint32_t dtypes;
    int32_t q_zero_point;
    int32_t k_zero_point;
    int32_t v_zero_point;
    int32_t output_zero_point;
    float q_scale;
    float k_scale;
    float v_scale;
    float output_scale;
    float attention_scale;
    float pad0;
    float pad1;
    float pad2;
} VkQSDPAParams;

_Static_assert(sizeof(VkQSDPAParams) == 80, "qSDPAInt8 uniform ABI");

typedef struct {
    uint32_t outer;
    uint32_t axis_size;
    uint32_t inner;
    uint32_t input_dtype;
} VkQArgMaxParams;

_Static_assert(sizeof(VkQArgMaxParams) == 16, "qArgMaxInt8 uniform ABI");

/* Eight u32 fields followed by two I32 zero points and two F32 scales.  This
 * 48-byte block is shared verbatim with qMaskedMeanInt8.wgsl and the browser
 * WebGPU dispatcher. */
typedef struct {
    uint32_t batch;
    uint32_t sequence;
    uint32_t width;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    int32_t input_zero_point;
    int32_t output_zero_point;
    float input_scale;
    float output_scale;
} VkQMaskedMeanParams;

_Static_assert(sizeof(VkQMaskedMeanParams) == 48,
               "qMaskedMeanInt8 uniform ABI");

typedef struct {
    uint32_t elements;
    uint32_t input_type;
    uint32_t output_type;
    uint32_t pad0;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad1;
    int32_t pad2;
    float multiplier;
    float pad3;
    float pad4;
    float pad5;
} VkRequantizeLinearParams;

_Static_assert(sizeof(VkRequantizeLinearParams) == 48,
               "requantizeLinearTyped uniform ABI");

static int qbyte_dtype_zero_point_valid(uint32_t dtype, int32_t zero_point) {
    if (dtype == VX_DTYPE_I8) return zero_point >= -128 && zero_point <= 127;
    if (dtype == VX_DTYPE_U8) return zero_point >= 0 && zero_point <= 255;
    return 0;
}

static int qadd_gpu_args_valid(const void* a, uint32_t a_elements,
                               const void* b, uint32_t b_elements,
                               void* output, uint32_t output_elements,
                               float a_scale, int32_t a_zero_point,
                               float b_scale, int32_t b_zero_point,
                               float output_scale, int32_t output_zero_point,
                               uint32_t a_dtype, uint32_t b_dtype,
                               uint32_t output_dtype, uint32_t relu,
                               size_t* logical_bytes) {
    if (!a || !b || !output || output == a || output == b ||
        a_elements == 0 || a_elements != b_elements ||
        a_elements != output_elements || relu > 2u ||
        !isfinite(a_scale) || !isfinite(b_scale) || !isfinite(output_scale) ||
        a_scale <= 0.0f || b_scale <= 0.0f || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(a_dtype, a_zero_point) ||
        !qbyte_dtype_zero_point_valid(b_dtype, b_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point)) return 0;
    if ((uint64_t)a_elements > SIZE_MAX) return 0;
    *logical_bytes = (size_t)a_elements;
    return 1;
}

static int qbyte_unary_gpu_args_valid(const void* input, void* output, uint32_t elements,
                                      float input_scale, int32_t input_zero_point,
                                      float output_scale, int32_t output_zero_point,
                                      uint32_t input_dtype, uint32_t output_dtype,
                                      size_t* logical_bytes) {
    if (!input || !output || input == output || !elements ||
        !isfinite(input_scale) || !isfinite(output_scale) ||
        input_scale <= 0.0f || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        (uint64_t)elements > SIZE_MAX) return 0;
    *logical_bytes = (size_t)elements;
    return 1;
}

static float qnorm_f32_at(const float* values, uint32_t index) {
    float value;
    memcpy(&value, (const unsigned char*)values + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

static int qgroupnorm_gpu_args_valid(const void* input, const float* weight,
                                     const float* bias, void* output,
                                     uint32_t batch, uint32_t height,
                                     uint32_t width, uint32_t channels,
                                     uint32_t groups, float input_scale,
                                     int32_t input_zero_point, float output_scale,
                                     int32_t output_zero_point, float epsilon,
                                     uint32_t input_dtype, uint32_t output_dtype,
                                     size_t* logical_bytes, size_t* affine_bytes,
                                     size_t* stats_bytes) {
    uint64_t elements;
    uint64_t group_count;
    if (!input || !weight || !bias || !output || output == input ||
        output == (const void*)weight || output == (const void*)bias ||
        !batch || !height || !width || !channels || !groups || channels % groups ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        batch > UINT32_MAX / groups) return 0;
    elements = batch;
    if (elements > UINT32_MAX / height) return 0;
    elements *= height;
    if (elements > UINT32_MAX / width) return 0;
    elements *= width;
    if (elements > UINT32_MAX / channels) return 0;
    elements *= channels;
    group_count = (uint64_t)batch * groups;
    if (elements == 0 || elements > SIZE_MAX || group_count == 0 ||
        group_count > SIZE_MAX / (2u * sizeof(float)) ||
        (size_t)channels > SIZE_MAX / sizeof(float)) return 0;
    for (uint32_t channel = 0; channel < channels; channel++) {
        if (!isfinite(qnorm_f32_at(weight, channel)) ||
            !isfinite(qnorm_f32_at(bias, channel))) return 0;
    }
    *logical_bytes = (size_t)elements;
    *affine_bytes = (size_t)channels * sizeof(float);
    *stats_bytes = (size_t)group_count * 2u * sizeof(float);
    return 1;
}

static int qlayernorm_gpu_args_valid(const void* input, const float* weight,
                                     const float* bias, void* output,
                                     uint32_t rows, uint32_t d_model,
                                     float input_scale, int32_t input_zero_point,
                                     float output_scale, int32_t output_zero_point,
                                     float epsilon, uint32_t input_dtype,
                                     uint32_t output_dtype, size_t* logical_bytes,
                                     size_t* affine_bytes, size_t* stats_bytes) {
    uint64_t elements;
    size_t row_count;
    if (!input || !weight || !bias || !output || output == input ||
        output == (const void*)weight || output == (const void*)bias ||
        !rows || !d_model || !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        rows > UINT32_MAX / d_model) return 0;
    elements = (uint64_t)rows * d_model;
    row_count = (size_t)rows;
    if (elements == 0 || elements > SIZE_MAX ||
        row_count > SIZE_MAX / (2u * sizeof(float)) ||
        (size_t)d_model > SIZE_MAX / sizeof(float)) return 0;
    for (uint32_t channel = 0; channel < d_model; channel++) {
        if (!isfinite(qnorm_f32_at(weight, channel)) ||
            !isfinite(qnorm_f32_at(bias, channel))) return 0;
    }
    *logical_bytes = (size_t)elements;
    *affine_bytes = (size_t)d_model * sizeof(float);
    *stats_bytes = row_count * 2u * sizeof(float);
    return 1;
}

static int qsdpa_ranges_overlap(const void* left, size_t left_bytes,
                                const void* right, size_t right_bytes) {
    uintptr_t left_start = (uintptr_t)left;
    uintptr_t right_start = (uintptr_t)right;
    if (left_start > UINTPTR_MAX - left_bytes ||
        right_start > UINTPTR_MAX - right_bytes) return 1;
    return left_start < right_start + right_bytes &&
        right_start < left_start + left_bytes;
}

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

static int qsdpa_mask_bytes(uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
                            uint32_t mask_mode, size_t* mask_bytes) {
    uint64_t elements = seq_kv;
    if (!mask_bytes) return 0;
    if (mask_mode == 0u) {
        *mask_bytes = 0;
        return 1;
    }
    if (mask_mode == 2u) {
        if (elements > UINT32_MAX / batch) return 0;
        elements *= batch;
    } else if (mask_mode == 3u) {
        if (elements > UINT32_MAX / seq_q) return 0;
        elements *= seq_q;
    } else if (mask_mode == 4u) {
        if (elements > UINT32_MAX / seq_q) return 0;
        elements *= seq_q;
        if (elements > UINT32_MAX / batch) return 0;
        elements *= batch;
    } else if (mask_mode != 1u) {
        return 0;
    }
    if (elements == 0 || elements > UINT32_MAX ||
        elements > SIZE_MAX / sizeof(int32_t)) return 0;
    *mask_bytes = (size_t)elements * sizeof(int32_t);
    return 1;
}

/* Validate all descriptor arithmetic before residency/upload.  The finite
 * worst-case score test guarantees qSDPAInt8's online recurrence never sees
 * +/-infinity and therefore cannot evaluate an invalid infinity subtraction. */
static int qsdpa_gpu_args_valid(const void* q, const void* k, const void* v,
                                const int32_t* mask, void* output,
                                uint32_t batch, uint32_t seq_q,
                                uint32_t seq_kv, uint32_t d_model,
                                uint32_t heads, float q_scale,
                                int32_t q_zero_point, float k_scale,
                                int32_t k_zero_point, float v_scale,
                                int32_t v_zero_point, float output_scale,
                                int32_t output_zero_point, float attention_scale,
                                uint32_t q_dtype, uint32_t k_dtype,
                                uint32_t v_dtype, uint32_t output_dtype,
                                uint32_t causal, uint32_t mask_mode,
                                size_t* q_bytes, size_t* kv_bytes,
                                size_t* mask_bytes) {
    uint64_t q_elements = batch;
    uint64_t kv_elements = batch;
    uint64_t q_magnitude;
    uint64_t k_magnitude;
    uint64_t maximum_dot;
    uint32_t head_dim;
    float qk_scale;
    float score_scale;
    if (!q || !k || !v || !output || !batch || !seq_q || !seq_kv ||
        !d_model || !heads || causal > 1u || d_model % heads ||
        d_model % 4u || !isfinite(q_scale) || q_scale <= 0.0f ||
        !isfinite(k_scale) || k_scale <= 0.0f || !isfinite(v_scale) ||
        v_scale <= 0.0f || !isfinite(output_scale) || output_scale <= 0.0f ||
        !isfinite(attention_scale) || attention_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(q_dtype, q_zero_point) ||
        !qbyte_dtype_zero_point_valid(k_dtype, k_zero_point) ||
        !qbyte_dtype_zero_point_valid(v_dtype, v_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        !q_bytes || !kv_bytes || !mask_bytes) return 0;
    head_dim = d_model / heads;
    if (!head_dim || head_dim % 4u || head_dim > 64u ||
        q_elements > UINT32_MAX / seq_q) return 0;
    q_elements *= seq_q;
    if (q_elements > UINT32_MAX / d_model) return 0;
    q_elements *= d_model;
    if (kv_elements > UINT32_MAX / seq_kv) return 0;
    kv_elements *= seq_kv;
    if (kv_elements > UINT32_MAX / d_model) return 0;
    kv_elements *= d_model;
    if (q_elements == 0 || kv_elements == 0 || q_elements > SIZE_MAX ||
        kv_elements > SIZE_MAX || !qsdpa_mask_bytes(batch, seq_q, seq_kv,
                                                     mask_mode, mask_bytes) ||
        (mask_mode == 0u ? mask != NULL : mask == NULL)) return 0;
    qk_scale = q_scale * k_scale;
    score_scale = qk_scale * attention_scale;
    if (!isfinite(qk_scale) || qk_scale <= 0.0f ||
        !isfinite(score_scale) || score_scale <= 0.0f ||
        !qsdpa_centered_magnitude(q_dtype, q_zero_point, &q_magnitude) ||
        !qsdpa_centered_magnitude(k_dtype, k_zero_point, &k_magnitude)) return 0;
    maximum_dot = q_magnitude * k_magnitude * head_dim;
    if (!isfinite((float)maximum_dot * score_scale)) return 0;
    *q_bytes = (size_t)q_elements;
    *kv_bytes = (size_t)kv_elements;
    if (qsdpa_ranges_overlap(output, *q_bytes, q, *q_bytes) ||
        qsdpa_ranges_overlap(output, *q_bytes, k, *kv_bytes) ||
        qsdpa_ranges_overlap(output, *q_bytes, v, *kv_bytes) ||
        (*mask_bytes && qsdpa_ranges_overlap(output, *q_bytes, mask,
                                             *mask_bytes))) return 0;
    return 1;
}

static int qargmax_gpu_args_valid(const void* input, int32_t* output,
                                  uint32_t outer, uint32_t axis_size,
                                  uint32_t inner, uint32_t input_dtype,
                                  size_t* input_bytes, size_t* output_bytes,
                                  uint32_t* output_elements_out) {
    uint64_t input_elements = outer;
    uint64_t output_elements = outer;
    if (!input || !output || !outer || !axis_size || !inner ||
        axis_size > (uint32_t)INT32_MAX ||
        (input_dtype != VX_DTYPE_I8 && input_dtype != VX_DTYPE_U8) ||
        !input_bytes || !output_bytes || !output_elements_out ||
        input_elements > UINT32_MAX / axis_size) return 0;
    input_elements *= axis_size;
    if (input_elements > UINT32_MAX / inner ||
        output_elements > UINT32_MAX / inner) return 0;
    input_elements *= inner;
    output_elements *= inner;
    if (!input_elements || !output_elements || input_elements > SIZE_MAX ||
        output_elements > UINT32_MAX ||
        output_elements > SIZE_MAX / sizeof(*output)) return 0;
    *input_bytes = (size_t)input_elements;
    *output_bytes = (size_t)output_elements * sizeof(*output);
    *output_elements_out = (uint32_t)output_elements;
    return !qsdpa_ranges_overlap(output, *output_bytes, input, *input_bytes);
}

/* Validate the exact scalar byte-reduction envelope before allocating device
 * residency.  qMaskedMeanInt8 accumulates centered values in I32, so sequence
 * length is bounded by the declared signed/unsigned zero point rather than
 * relying on an implementation-defined overflow. */
static int qmaskedmean_gpu_args_valid(const void* input, const int32_t* mask,
                                      void* output, uint32_t batch,
                                      uint32_t sequence, uint32_t width,
                                      float input_scale, int32_t input_zero_point,
                                      float output_scale, int32_t output_zero_point,
                                      uint32_t input_dtype, uint32_t output_dtype,
                                      size_t* input_bytes, size_t* mask_bytes,
                                      size_t* output_bytes) {
    uint64_t input_elements = batch;
    uint64_t mask_elements = batch;
    uint64_t output_elements = batch;
    int64_t low;
    int64_t high;
    uint64_t magnitude;
    float multiplier;
    if (!input || !mask || !output || !batch || !sequence || !width ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        !input_bytes || !mask_bytes || !output_bytes ||
        input_elements > UINT32_MAX / sequence) return 0;
    input_elements *= sequence;
    if (input_elements > UINT32_MAX / width ||
        mask_elements > UINT32_MAX / sequence ||
        output_elements > UINT32_MAX / width) return 0;
    input_elements *= width;
    mask_elements *= sequence;
    output_elements *= width;
    if (!input_elements || !mask_elements || !output_elements ||
        input_elements > SIZE_MAX || mask_elements > SIZE_MAX / sizeof(*mask) ||
        output_elements > SIZE_MAX) return 0;
    if (input_dtype == VX_DTYPE_I8) {
        low = -128 - (int64_t)input_zero_point;
        high = 127 - (int64_t)input_zero_point;
    } else {
        low = -(int64_t)input_zero_point;
        high = 255 - (int64_t)input_zero_point;
    }
    magnitude = (uint64_t)(low < 0 ? -low : low);
    if ((uint64_t)(high < 0 ? -high : high) > magnitude)
        magnitude = (uint64_t)(high < 0 ? -high : high);
    multiplier = input_scale / output_scale;
    if ((magnitude != 0 && (uint64_t)sequence > (uint64_t)INT32_MAX / magnitude) ||
        !isfinite(multiplier) || multiplier <= 0.0f) return 0;
    *input_bytes = (size_t)input_elements;
    *mask_bytes = (size_t)mask_elements * sizeof(*mask);
    *output_bytes = (size_t)output_elements;
    return !qsdpa_ranges_overlap(output, *output_bytes, input, *input_bytes) &&
        !qsdpa_ranges_overlap(output, *output_bytes, mask, *mask_bytes);
}

static int requantize_gpu_args_valid(const void* input, uint32_t input_elements,
                                     void* output, uint32_t output_elements,
                                     float input_scale, int32_t input_zero_point,
                                     float output_scale, int32_t output_zero_point,
                                     uint32_t input_dtype, uint32_t output_dtype,
                                     float* multiplier, size_t* logical_bytes) {
    if (!input || !output || input == output || input_elements == 0 ||
        input_elements != output_elements || !isfinite(input_scale) ||
        !isfinite(output_scale) || input_scale <= 0.0f || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point)) return 0;
    float ratio = input_scale / output_scale;
    if (!isfinite(ratio) || ratio <= 0.0f || (uint64_t)input_elements > SIZE_MAX) return 0;
    *multiplier = ratio;
    *logical_bytes = (size_t)input_elements;
    return 1;
}

int vk_graph_qbatch_matmul_i8u8(
        const void* a, const int* a_shape, int a_rank,
        float a_scale, int32_t a_zero_point, uint32_t a_dtype,
        const void* b, const int* b_shape, int b_rank,
        float b_scale, int32_t b_zero_point, uint32_t b_dtype,
        void* output, const int* output_shape, int output_rank,
        float output_scale, int32_t output_zero_point,
        uint32_t output_dtype) {
    VxQBatchMatMulDevicePlan plan;
    uint32_t metadata[24] = {0};
    uint32_t metadata_words;
    size_t metadata_bytes;
    size_t a_packed_bytes;
    size_t b_packed_bytes;
    size_t output_packed_bytes;
    if (!vx_qbatch_matmul_device_plan(
            a, a_shape, a_rank, a_scale, a_zero_point, a_dtype,
            b, b_shape, b_rank, b_scale, b_zero_point, b_dtype,
            output, output_shape, output_rank, output_scale,
            output_zero_point, output_dtype, &plan) ||
        !graph_packed_bytes(plan.a_bytes, &a_packed_bytes) ||
        !graph_packed_bytes(plan.b_bytes, &b_packed_bytes) ||
        !graph_packed_bytes(plan.output_bytes, &output_packed_bytes))
        return 0;
    graph_scratch_begin();
    VkTensorSlot* a_slot =
        graph_ensure_packed_bytes(a, plan.a_bytes, 0);
    VkTensorSlot* b_slot =
        graph_ensure_packed_bytes(b, plan.b_bytes, 0);
    VkTensorSlot* output_slot =
        graph_output_packed_bytes(output, plan.output_bytes);
    if (!a_slot || !b_slot || !output_slot) return 0;
    memcpy(metadata, plan.output_batch_strides,
           plan.batch_rank * sizeof(uint32_t));
    memcpy(metadata + plan.batch_rank, plan.a_batch_strides,
           plan.batch_rank * sizeof(uint32_t));
    memcpy(metadata + 2u * plan.batch_rank, plan.b_batch_strides,
           plan.batch_rank * sizeof(uint32_t));
    metadata_words = plan.batch_rank ? 3u * plan.batch_rank : 1u;
    metadata_bytes = (size_t)metadata_words * sizeof(uint32_t);
    VkQBatchMatMulParams params = {
        plan.batch_rank, plan.m, plan.k, plan.n,
        plan.output_elements, a_dtype, b_dtype, output_dtype,
        a_zero_point, b_zero_point, output_zero_point, 0,
        a_scale, b_scale, output_scale, 0.0f,
    };
    size_t metadata_offset =
        graph_scratch_upload(metadata, metadata_bytes);
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (metadata_offset == SIZE_MAX || params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {a_slot->offset, a_packed_bytes},
        {b_slot->offset, b_packed_bytes},
        {output_slot->offset, output_packed_bytes},
        {metadata_offset, metadata_bytes},
        {params_offset, sizeof(params)},
    };
    uint32_t packed_words =
        (uint32_t)(output_packed_bytes / sizeof(uint32_t));
    int dispatched = 0;
    uint32_t groups = (packed_words + 63u) / 64u;
    if (vk_packed_dot_enabled()) {
        dispatched = vk_dispatch_kernel(
            &k_qbatch_matmul_i8u8_dot, binds, groups, 1u, 1u);
        if (!dispatched) {
            vk_disable_packed_dot_after_failure(
                k_qbatch_matmul_i8u8_dot.name);
        }
        else vk_context_current()->qbatch_dot_dispatches++;
    }
    if (!dispatched && !vk_dispatch_kernel(
            &k_qbatch_matmul_i8u8, binds, groups, 1u, 1u))
        return 0;
    if (!dispatched) vk_context_current()->qbatch_scalar_dispatches++;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_qadd_i8u8(const void* a, uint32_t a_elements,
                       const void* b, uint32_t b_elements,
                       void* output, uint32_t output_elements,
                       float a_scale, int32_t a_zero_point,
                       float b_scale, int32_t b_zero_point,
                       float output_scale, int32_t output_zero_point,
                       uint32_t a_dtype, uint32_t b_dtype,
                       uint32_t output_dtype, uint32_t relu) {
    size_t logical_bytes;
    if (!qadd_gpu_args_valid(a, a_elements, b, b_elements, output, output_elements,
                             a_scale, a_zero_point, b_scale, b_zero_point,
                             output_scale, output_zero_point, a_dtype, b_dtype,
                             output_dtype, relu, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkGraphWindow a_slot, b_slot, output_slot;
    if (!graph_window_packed_bytes(a, logical_bytes, 0, &a_slot) ||
        !graph_window_packed_bytes(b, logical_bytes, 0, &b_slot) ||
        !graph_window_output_packed(output, logical_bytes, &output_slot)) return 0;
    VkQAddParams params = {
        a_elements, a_dtype,
        b_dtype,
        output_dtype,
        a_zero_point, b_zero_point, output_zero_point, 0,
        a_scale, b_scale, output_scale, relu
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {a_slot.offset, a_slot.bytes}, {b_slot.offset, b_slot.bytes},
        {output_slot.offset, output_slot.bytes}, {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_qadd_i8u8, binds, (packed_words + 63u) / 64u,
                            1u, 1u)) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

int vk_graph_qsilu_i8u8(const void* input, void* output, uint32_t elements,
                        float input_scale, int32_t input_zero_point,
                        float output_scale, int32_t output_zero_point,
                        uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    if (!qbyte_unary_gpu_args_valid(input, output, elements, input_scale,
                                    input_zero_point, output_scale, output_zero_point,
                                    input_dtype, output_dtype, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkGraphWindow input_slot, output_slot;
    if (!graph_window_packed_bytes(input, logical_bytes, 0, &input_slot) ||
        !graph_window_output_packed(output, logical_bytes, &output_slot)) return 0;
    VkQByteUnaryParams params = {
        elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot.offset, input_slot.bytes},
        {output_slot.offset, output_slot.bytes}, {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_qsilu_i8u8, binds, (packed_words + 63u) / 64u,
                            1u, 1u)) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

int vk_graph_qgelu_i8u8(const void* input, void* output, uint32_t elements,
                        float input_scale, int32_t input_zero_point,
                        float output_scale, int32_t output_zero_point,
                        uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    if (!qbyte_unary_gpu_args_valid(input, output, elements, input_scale,
                                    input_zero_point, output_scale, output_zero_point,
                                    input_dtype, output_dtype, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkGraphWindow input_slot, output_slot;
    if (!graph_window_packed_bytes(input, logical_bytes, 0, &input_slot) ||
        !graph_window_output_packed(output, logical_bytes, &output_slot)) return 0;
    VkQByteUnaryParams params = {
        elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot.offset, input_slot.bytes},
        {output_slot.offset, output_slot.bytes}, {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_qgelu_i8u8, binds, (packed_words + 63u) / 64u,
                            1u, 1u)) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

/* The statistics pass owns one [batch, group] workgroup; the apply pass owns
 * complete packed output words.  Keeping them separate prevents byte-lane
 * races whenever a channel-group boundary crosses a u32 word. */
int vk_graph_qgroupnorm_i8u8(const void* input, const float* weight,
                             const float* bias, void* output, uint32_t batch,
                             uint32_t height, uint32_t width, uint32_t channels,
                             uint32_t groups, float input_scale,
                             int32_t input_zero_point, float output_scale,
                             int32_t output_zero_point, float epsilon,
                             uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    size_t affine_bytes;
    size_t stats_bytes;
    size_t packed_bytes;
    if (!qgroupnorm_gpu_args_valid(input, weight, bias, output, batch, height,
                                   width, channels, groups, input_scale,
                                   input_zero_point, output_scale,
                                   output_zero_point, epsilon, input_dtype,
                                   output_dtype, &logical_bytes, &affine_bytes,
                                   &stats_bytes) ||
        !graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkGraphWindow input_slot, output_slot;
    VkTensorSlot* weight_slot = graph_ensure_device(weight, affine_bytes, 1);
    VkTensorSlot* bias_slot = graph_ensure_device(bias, affine_bytes, 1);
    if (!graph_window_packed_bytes(input, logical_bytes, 0, &input_slot) ||
        !graph_window_output_packed(output, logical_bytes, &output_slot) ||
        !weight_slot || !bias_slot) return 0;
    VulkanContextState* context = vk_context_current();
    size_t stats_offset;
    if (context && context->domain_enforced) {
        if (!context->domain_qgroupnorm_stats_bytes ||
            stats_bytes > context->domain_qgroupnorm_stats_bytes)
            return 0;
        stats_offset = context->domain_qgroupnorm_stats_offset;
    } else {
        stats_offset = graph_scratch_alloc(stats_bytes);
        if (stats_offset == SIZE_MAX) return 0;
    }
    VkQGroupNormParams params = {
        batch, height, width, channels, groups,
        input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, epsilon, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding stats_binds[3] = {
        {input_slot.offset, input_slot.bytes}, {stats_offset, stats_bytes},
        {params_offset, sizeof(params)}
    };
    VkGraphBinding apply_binds[6] = {
        {input_slot.offset, input_slot.bytes},
        {weight_slot->offset, weight_slot->bytes}, {bias_slot->offset, bias_slot->bytes},
        {stats_offset, stats_bytes}, {output_slot.offset, output_slot.bytes},
        {params_offset, sizeof(params)}
    };
    uint32_t total_groups = batch * groups;
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    uint32_t apply_groups = packed_words / 64u + (packed_words % 64u != 0u);
    if (!vk_dispatch_kernel(&k_qgroupnorm_stats, stats_binds, total_groups, 1u, 1u) ||
        !vk_dispatch_kernel(&k_qgroupnorm_apply, apply_binds, apply_groups, 1u, 1u)) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

/* One 64-lane workgroup produces a [mean_raw, inverse_stddev] pair for each
 * final-axis row; the apply pass then owns whole packed destination words. */
int vk_graph_qlayernorm_i8u8(const void* input, const float* weight,
                             const float* bias, void* output, uint32_t rows,
                             uint32_t d_model, float input_scale,
                             int32_t input_zero_point, float output_scale,
                             int32_t output_zero_point, float epsilon,
                             uint32_t input_dtype, uint32_t output_dtype) {
    size_t logical_bytes;
    size_t affine_bytes;
    size_t stats_bytes;
    size_t packed_bytes;
    if (!qlayernorm_gpu_args_valid(input, weight, bias, output, rows, d_model,
                                   input_scale, input_zero_point, output_scale,
                                   output_zero_point, epsilon, input_dtype,
                                   output_dtype, &logical_bytes, &affine_bytes,
                                   &stats_bytes) ||
        !graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkGraphWindow input_slot, output_slot;
    VkTensorSlot* weight_slot = graph_ensure_device(weight, affine_bytes, 1);
    VkTensorSlot* bias_slot = graph_ensure_device(bias, affine_bytes, 1);
    if (!graph_window_packed_bytes(input, logical_bytes, 0, &input_slot) ||
        !graph_window_output_packed(output, logical_bytes, &output_slot) ||
        !weight_slot || !bias_slot) return 0;
    VulkanContextState* context = vk_context_current();
    size_t stats_offset;
    if (context && context->domain_enforced) {
        if (!context->domain_qlayernorm_stats_bytes ||
            stats_bytes > context->domain_qlayernorm_stats_bytes)
            return 0;
        stats_offset = context->domain_qlayernorm_stats_offset;
    } else {
        stats_offset = graph_scratch_alloc(stats_bytes);
        if (stats_offset == SIZE_MAX) return 0;
    }
    VkQLayerNormParams params = {
        rows, d_model, input_dtype,
        output_dtype,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, epsilon, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding stats_binds[3] = {
        {input_slot.offset, input_slot.bytes}, {stats_offset, stats_bytes},
        {params_offset, sizeof(params)}
    };
    VkGraphBinding apply_binds[6] = {
        {input_slot.offset, input_slot.bytes},
        {weight_slot->offset, weight_slot->bytes}, {bias_slot->offset, bias_slot->bytes},
        {stats_offset, stats_bytes}, {output_slot.offset, output_slot.bytes},
        {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    uint32_t apply_groups = packed_words / 64u + (packed_words % 64u != 0u);
    if (!vk_dispatch_kernel(&k_qlayernorm_stats, stats_binds, rows, 1u, 1u) ||
        !vk_dispatch_kernel(&k_qlayernorm_apply, apply_binds, apply_groups, 1u, 1u)) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

/* One [query, head, batch] workgroup performs its own online softmax and
 * writes only its head's bytes.  No graph-visible F32 score/value storage is
 * allocated; absent masks bind the dedicated static I32 dummy buffer. */
int vk_graph_qsdpa_i8u8(const void* q, const void* k, const void* v,
                        const int32_t* mask, void* output, uint32_t batch,
                        uint32_t seq_q, uint32_t seq_kv, uint32_t d_model,
                        uint32_t heads, float q_scale, int32_t q_zero_point,
                        float k_scale, int32_t k_zero_point, float v_scale,
                        int32_t v_zero_point, float output_scale,
                        int32_t output_zero_point, float attention_scale,
                        uint32_t q_dtype, uint32_t k_dtype, uint32_t v_dtype,
                        uint32_t output_dtype, uint32_t causal,
                        uint32_t mask_mode) {
    size_t q_bytes;
    size_t kv_bytes;
    size_t mask_bytes;
    size_t packed_output_bytes;
    if (!qsdpa_gpu_args_valid(q, k, v, mask, output, batch, seq_q, seq_kv,
                               d_model, heads, q_scale, q_zero_point, k_scale,
                               k_zero_point, v_scale, v_zero_point, output_scale,
                               output_zero_point, attention_scale, q_dtype, k_dtype,
                               v_dtype, output_dtype, causal, mask_mode, &q_bytes,
                               &kv_bytes, &mask_bytes) ||
        !graph_packed_bytes(q_bytes, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkGraphWindow q_slot, k_slot, v_slot, mask_slot, output_slot;
    if (!graph_window_packed_bytes(q, q_bytes, 0, &q_slot) ||
        !graph_window_packed_bytes(k, kv_bytes, 0, &k_slot) ||
        !graph_window_packed_bytes(v, kv_bytes, 0, &v_slot) ||
        !(mask_mode ? graph_window_device(mask, mask_bytes, 0, &mask_slot)
                    : graph_window_device(qsdpa_dummy_mask, sizeof(qsdpa_dummy_mask), 1,
                                          &mask_slot)) ||
        !graph_window_output_packed(output, q_bytes, &output_slot)) return 0;
    VkQSDPAParams params = {
        seq_q, seq_kv, d_model, heads,
        batch, mask_mode, causal,
        q_dtype |
            (k_dtype << 8u) |
            (v_dtype << 16u) |
            (output_dtype << 24u),
        q_zero_point, k_zero_point, v_zero_point, output_zero_point,
        q_scale, k_scale, v_scale, output_scale,
        attention_scale, 0.0f, 0.0f, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[6] = {
        {q_slot.offset, q_slot.bytes}, {k_slot.offset, k_slot.bytes},
        {v_slot.offset, v_slot.bytes}, {mask_slot.offset, mask_slot.bytes},
        {output_slot.offset, output_slot.bytes}, {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_qsdpa_int8, binds, seq_q, heads, batch)) return 0;
    graph_mark_device(output_slot.slot);
    return 1;
}

typedef struct {
    uint32_t rows;
    uint32_t row_words;
    uint32_t indexed_rows;
    uint32_t mode;
} VkRowIndexTransferParams;

_Static_assert(sizeof(VkRowIndexTransferParams) == 16, "rowIndexTransfer uniform ABI");

/*
 * Move a set of rows between two resident buffers, on the device.
 *
 * A batched decode names its rows as a set rather than a span -- lane b writes
 * `b * S + position[b]`, and a paged lane's prefix is whatever physical slots
 * its page table holds. The host resolves such a set by copying; a backend that
 * keeps its activations on the device has to do the same copy there, or the
 * step reads a host mirror the device has already moved past.
 *
 * The index array is uploaded through the scratch arena rather than bound as a
 * tensor: it is different every step, and a slot keyed on the host pointer
 * would be clean after its first upload and serve stale rows from then on --
 * which is precisely the bug this exists to prevent, arriving by another door.
 */
int vk_graph_row_index_transfer(const void* source, size_t source_bytes,
                                const int32_t* indices, uint32_t rows,
                                void* destination, size_t destination_bytes,
                                uint32_t row_words, uint32_t indexed_rows,
                                uint32_t mode) {
    size_t index_bytes;
    size_t index_offset;
    size_t params_offset;
    uint32_t invocations;
    if (!source || !destination || !indices || !rows || !row_words ||
        !indexed_rows || mode > 1u || !source_bytes || !destination_bytes ||
        rows > UINT32_MAX / row_words) return 0;
    invocations = rows * row_words;
    if ((size_t)rows > SIZE_MAX / sizeof(int32_t)) return 0;
    index_bytes = (size_t)rows * sizeof(int32_t);
    graph_scratch_begin();
    VkGraphWindow src, dst;
    if (!graph_window_packed_bytes(source, source_bytes, 0, &src) ||
        !graph_window_output_packed(destination, destination_bytes, &dst)) return 0;
    index_offset = graph_scratch_upload(indices, index_bytes);
    if (index_offset == SIZE_MAX) return 0;
    VkRowIndexTransferParams params = { rows, row_words, indexed_rows, mode };
    params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {src.offset, src.bytes}, {index_offset, index_bytes},
        {dst.offset, dst.bytes}, {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_row_index_transfer, binds,
                            (invocations + 63u) / 64u, 1u, 1u)) return 0;
    graph_mark_device(dst.slot);
    return 1;
}

/* qArgMaxInt8 owns one output index per [outer,inner] coordinate. Input stays
 * byte-packed, output is conventional I32 storage, and the 16-byte uniform
 * exactly mirrors the WebGPU ABI. */
int vk_graph_qargmax_i8u8(const void* input, int32_t* output,
                          uint32_t outer, uint32_t axis_size,
                          uint32_t inner, uint32_t input_dtype) {
    size_t input_bytes;
    size_t output_bytes;
    uint32_t output_elements;
    if (!qargmax_gpu_args_valid(input, output, outer, axis_size, inner,
                                input_dtype, &input_bytes, &output_bytes,
                                &output_elements)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    VkTensorSlot* output_slot = graph_output_slot(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    VkQArgMaxParams params = {
        outer, axis_size, inner, input_dtype
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes},
        {output_slot->offset, output_slot->bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_qargmax_int8, binds,
                            output_elements / 64u + (output_elements % 64u != 0u),
                            1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

/* qMaskedMeanInt8 owns complete packed output words, so byte lanes never
 * race while producing [B,D] output from the [B,S,D] activation and I32
 * [B,S] keep mask.  The reduction's F32 arithmetic is shader-private. */
int vk_graph_qmaskedmean_i8u8(const void* input, const int32_t* mask,
                              void* output, uint32_t batch,
                              uint32_t sequence, uint32_t width,
                              float input_scale, int32_t input_zero_point,
                              float output_scale, int32_t output_zero_point,
                              uint32_t input_dtype, uint32_t output_dtype) {
    size_t input_bytes;
    size_t mask_bytes;
    size_t output_bytes;
    size_t packed_output_bytes;
    if (!qmaskedmean_gpu_args_valid(input, mask, output, batch, sequence, width,
                                    input_scale, input_zero_point, output_scale,
                                    output_zero_point, input_dtype, output_dtype,
                                    &input_bytes, &mask_bytes, &output_bytes) ||
        !graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    VkTensorSlot* mask_slot = graph_ensure_device(mask, mask_bytes, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_bytes);
    if (!input_slot || !mask_slot || !output_slot) return 0;
    VkQMaskedMeanParams params = {
        batch, sequence, width, input_dtype,
        output_dtype, 0u, 0u, 0u,
        input_zero_point, output_zero_point, input_scale, output_scale
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[4] = {
        {input_slot->offset, input_slot->bytes},
        {mask_slot->offset, mask_slot->bytes},
        {output_slot->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_qmaskedmean_int8, binds,
                            (packed_words + 63u) / 64u, 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_requantize_linear_i8u8(const void* input, uint32_t input_elements,
                                    void* output, uint32_t output_elements,
                                    float input_scale, int32_t input_zero_point,
                                    float output_scale, int32_t output_zero_point,
                                    uint32_t input_dtype, uint32_t output_dtype) {
    float multiplier;
    size_t logical_bytes;
    if (!requantize_gpu_args_valid(input, input_elements, output, output_elements,
                                   input_scale, input_zero_point,
                                   output_scale, output_zero_point,
                                   input_dtype, output_dtype,
                                   &multiplier, &logical_bytes)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes(logical_bytes, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, logical_bytes, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, logical_bytes);
    if (!input_slot || !output_slot) return 0;
    VkRequantizeLinearParams params = {
        input_elements, input_dtype,
        output_dtype, 0u,
        input_zero_point, output_zero_point, 0, 0,
        multiplier, 0.0f, 0.0f, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes},
        {output_slot->offset, packed_bytes}, {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    if (!vk_dispatch_kernel(&k_requantize_linear_i8u8, binds,
                            (packed_words + 63u) / 64u, 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

typedef struct {
    uint32_t batch;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t input_channels;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t output_channels;
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t groups;
    uint32_t input_type;
    uint32_t weight_type;
    uint32_t output_type;
    uint32_t relu;
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t pad0;
    int32_t pad1;
    float input_scale;
    float output_scale;
    float pad2;
    float pad3;
} VkQConv2DParams;

_Static_assert(sizeof(VkQConv2DParams) == 112, "qConv2DInt8 uniform ABI");

static const int32_t* qconv_zero_bias_get(uint32_t output_channels) {
    if (output_channels == 0 ||
        (size_t)output_channels > SIZE_MAX / sizeof(int32_t)) return NULL;
    for (QConvZeroBiasBacking* block = qconv_zero_bias_backings; block;
         block = block->next) {
        /* A graph weight slot records the exact readable byte range retained
         * at domain bind. Distinct channel widths therefore need distinct
         * durable keys; reusing a larger block for a later smaller node would
         * shrink that slot and make the earlier node invalid after bind. */
        if (block->elements == output_channels) return block->values;
    }
    QConvZeroBiasBacking* block = (QConvZeroBiasBacking*)calloc(1, sizeof(*block));
    if (!block) return NULL;
    block->values = (int32_t*)calloc((size_t)output_channels, sizeof(*block->values));
    if (!block->values) {
        free(block);
        return NULL;
    }
    block->elements = output_channels;
    block->next = qconv_zero_bias_backings;
    qconv_zero_bias_backings = block;
    return block->values;
}

static int qconv_mul_u64(uint64_t left, uint64_t right, uint64_t* out) {
    if (!out || (left != 0 && right > UINT64_MAX / left)) return 0;
    *out = left * right;
    return 1;
}

static int32_t qconv_i32_at(const int32_t* values, uint32_t index) {
    int32_t value;
    memcpy(&value, (const unsigned char*)values + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

static float qconv_f32_at(const float* values, uint32_t index) {
    float value;
    memcpy(&value, (const unsigned char*)values + (size_t)index * sizeof(value),
           sizeof(value));
    return value;
}

static int qconv_gpu_args_valid(const void* input, const void* weight,
                                const float* weight_scales,
                                const int32_t* weight_zero_points,
                                const int32_t* bias, void* output,
                                uint32_t batch, uint32_t input_height,
                                uint32_t input_width, uint32_t input_channels,
                                uint32_t output_height, uint32_t output_width,
                                uint32_t output_channels, uint32_t kernel_height,
                                uint32_t kernel_width, uint32_t input_per_group,
                                uint32_t stride_y, uint32_t stride_x,
                                uint32_t dilation_y, uint32_t dilation_x,
                                uint32_t padding_top, uint32_t padding_left,
                                uint32_t padding_bottom, uint32_t padding_right,
                                uint32_t groups, uint32_t relu,
                                float input_scale, int32_t input_zero_point,
                                float output_scale, int32_t output_zero_point,
                                uint32_t input_dtype, uint32_t weight_dtype,
                                uint32_t output_dtype,
                                size_t* input_bytes, size_t* weight_bytes,
                                size_t* output_bytes, size_t* metadata_bytes) {
    uint64_t input_elements = 1, weight_elements = 1, output_elements = 1;
    uint64_t terms = 1, padded_height, padded_width;
    uint64_t effective_height, effective_width;
    uint64_t expected_height, expected_width;
    uint64_t input_magnitude;
    if (!input || !weight || !weight_scales || !weight_zero_points || !output ||
        output == input || output == weight || !batch || !input_height ||
        !input_width || !input_channels || !output_height || !output_width ||
        !output_channels || !kernel_height || !kernel_width || !input_per_group ||
        !stride_y || !stride_x || !dilation_y || !dilation_x || !groups || relu > 2u ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        !qbyte_dtype_zero_point_valid(input_dtype, input_zero_point) ||
        !qbyte_dtype_zero_point_valid(output_dtype, output_zero_point) ||
        (weight_dtype != VX_DTYPE_I8 && weight_dtype != VX_DTYPE_U8) ||
        input_channels % groups || output_channels % groups ||
        (uint64_t)input_per_group * groups != input_channels) return 0;
    if (!qconv_mul_u64(input_elements, batch, &input_elements) ||
        !qconv_mul_u64(input_elements, input_height, &input_elements) ||
        !qconv_mul_u64(input_elements, input_width, &input_elements) ||
        !qconv_mul_u64(input_elements, input_channels, &input_elements) ||
        !qconv_mul_u64(weight_elements, output_channels, &weight_elements) ||
        !qconv_mul_u64(weight_elements, kernel_height, &weight_elements) ||
        !qconv_mul_u64(weight_elements, kernel_width, &weight_elements) ||
        !qconv_mul_u64(weight_elements, input_per_group, &weight_elements) ||
        !qconv_mul_u64(output_elements, batch, &output_elements) ||
        !qconv_mul_u64(output_elements, output_height, &output_elements) ||
        !qconv_mul_u64(output_elements, output_width, &output_elements) ||
        !qconv_mul_u64(output_elements, output_channels, &output_elements) ||
        !qconv_mul_u64(terms, kernel_height, &terms) ||
        !qconv_mul_u64(terms, kernel_width, &terms) ||
        !qconv_mul_u64(terms, input_per_group, &terms) ||
        input_elements > UINT32_MAX || weight_elements > UINT32_MAX ||
        output_elements > UINT32_MAX || input_elements > SIZE_MAX ||
        weight_elements > SIZE_MAX || output_elements > SIZE_MAX ||
        (size_t)output_channels > SIZE_MAX / sizeof(float) ||
        (size_t)output_channels > SIZE_MAX / sizeof(int32_t)) return 0;
    if ((uint64_t)(kernel_height - 1u) > (UINT64_MAX - 1u) / dilation_y ||
        (uint64_t)(kernel_width - 1u) > (UINT64_MAX - 1u) / dilation_x) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    effective_height = (uint64_t)(kernel_height - 1u) * dilation_y + 1u;
    effective_width = (uint64_t)(kernel_width - 1u) * dilation_x + 1u;
    if (padded_height < effective_height || padded_width < effective_width) return 0;
    expected_height = (padded_height - effective_height) / stride_y + 1u;
    expected_width = (padded_width - effective_width) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width) return 0;
    {
        int64_t low = (int64_t)(input_dtype == VX_DTYPE_I8 ? -128 : 0) -
            input_zero_point;
        int64_t high = (int64_t)(input_dtype == VX_DTYPE_I8 ? 127 : 255) -
            input_zero_point;
        uint64_t low_magnitude = (uint64_t)(low < 0 ? -low : low);
        uint64_t high_magnitude = (uint64_t)(high < 0 ? -high : high);
        input_magnitude = low_magnitude > high_magnitude ? low_magnitude : high_magnitude;
    }
    for (uint32_t output_channel = 0; output_channel < output_channels;
         output_channel++) {
        const float weight_scale = qconv_f32_at(weight_scales, output_channel);
        const int32_t weight_zero_point = qconv_i32_at(weight_zero_points, output_channel);
        int64_t low, high;
        uint64_t weight_magnitude, accumulator_bound, bias_magnitude = 0;
        if (!isfinite(weight_scale) || weight_scale <= 0.0f ||
            !qbyte_dtype_zero_point_valid(weight_dtype, weight_zero_point)) return 0;
        low = (int64_t)(weight_dtype == VX_DTYPE_I8 ? -128 : 0) -
            weight_zero_point;
        high = (int64_t)(weight_dtype == VX_DTYPE_I8 ? 127 : 255) -
            weight_zero_point;
        weight_magnitude = (uint64_t)(low < 0 ? -low : low);
        {
            uint64_t high_magnitude = (uint64_t)(high < 0 ? -high : high);
            if (high_magnitude > weight_magnitude) weight_magnitude = high_magnitude;
        }
        if (input_magnitude && weight_magnitude &&
            terms > (uint64_t)INT32_MAX / input_magnitude / weight_magnitude) return 0;
        accumulator_bound = input_magnitude * weight_magnitude * terms;
        if (bias) {
            int64_t bias_value = qconv_i32_at(bias, output_channel);
            bias_magnitude = (uint64_t)(bias_value < 0 ? -bias_value : bias_value);
        }
        if (accumulator_bound > (uint64_t)INT32_MAX ||
            bias_magnitude > (uint64_t)INT32_MAX - accumulator_bound) return 0;
    }
    *input_bytes = (size_t)input_elements;
    *weight_bytes = (size_t)weight_elements;
    *output_bytes = (size_t)output_elements;
    *metadata_bytes = (size_t)output_channels * sizeof(int32_t);
    return 1;
}

int vk_graph_qconv2d_i8u8(const void* input, const void* weight,
                           const float* weight_scales,
                           const int32_t* weight_zero_points,
                           const int32_t* bias, void* output,
                           uint32_t batch, uint32_t input_height,
                           uint32_t input_width, uint32_t input_channels,
                           uint32_t output_height, uint32_t output_width,
                           uint32_t output_channels, uint32_t kernel_height,
                           uint32_t kernel_width, uint32_t input_per_group,
                           uint32_t stride_y, uint32_t stride_x,
                           uint32_t dilation_y, uint32_t dilation_x,
                           uint32_t padding_top, uint32_t padding_left,
                           uint32_t padding_bottom, uint32_t padding_right,
                           uint32_t groups, uint32_t relu,
                           float input_scale, int32_t input_zero_point,
                           float output_scale, int32_t output_zero_point,
                           uint32_t input_dtype, uint32_t weight_dtype,
                           uint32_t output_dtype) {
    size_t input_bytes, weight_bytes, output_bytes, metadata_bytes;
    if (!qconv_gpu_args_valid(input, weight, weight_scales, weight_zero_points,
                              bias, output, batch, input_height, input_width,
                              input_channels, output_height, output_width,
                              output_channels, kernel_height, kernel_width,
                              input_per_group, stride_y, stride_x, dilation_y,
                              dilation_x, padding_top, padding_left,
                              padding_bottom, padding_right, groups, relu,
                              input_scale, input_zero_point, output_scale,
                              output_zero_point, input_dtype, weight_dtype,
                              output_dtype, &input_bytes, &weight_bytes,
                              &output_bytes, &metadata_bytes)) return 0;
    const int32_t* bound_bias = bias ? bias : qconv_zero_bias_get(output_channels);
    if (!bound_bias) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    VkTensorSlot* weight_slot = graph_ensure_packed_bytes(weight, weight_bytes, 1);
    VkTensorSlot* scales_slot = graph_ensure_device(weight_scales,
                                                     (size_t)output_channels * sizeof(float), 1);
    VkTensorSlot* zero_points_slot = graph_ensure_device(weight_zero_points, metadata_bytes, 1);
    VkTensorSlot* bias_slot = graph_ensure_device(bound_bias, metadata_bytes, 1);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_bytes);
    if (!input_slot || !weight_slot || !scales_slot || !zero_points_slot ||
        !bias_slot || !output_slot) return 0;
    VkQConv2DParams params = {
        batch, input_height, input_width, input_channels,
        output_height, output_width, output_channels, kernel_height,
        kernel_width, stride_y, stride_x, dilation_y,
        dilation_x, padding_top, padding_left, groups,
        input_dtype,
        weight_dtype,
        output_dtype, relu,
        input_zero_point, output_zero_point, 0, 0,
        input_scale, output_scale, 0.0f, 0.0f
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[7] = {
        {input_slot->offset, input_slot->bytes}, {weight_slot->offset, weight_slot->bytes},
        {scales_slot->offset, scales_slot->bytes}, {zero_points_slot->offset, zero_points_slot->bytes},
        {bias_slot->offset, bias_slot->bytes}, {output_slot->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    uint32_t packed_words = (uint32_t)(packed_output_bytes / sizeof(uint32_t));
    const uint64_t reduction_size =
        (uint64_t)kernel_height * kernel_width * input_per_group;
    const int dot_tiled = groups == 1u &&
        (output_channels & 3u) == 0u &&
        vk_workgroup_shape_supported(8u, 4u, 1u);
    const int tiled = groups == 1u && output_channels >= 32u &&
        reduction_size >= 16u && (output_channels & 3u) == 0u &&
        vk_workgroup_shape_supported(8u, 4u, 1u);
    int dispatched = 0;
    if (vk_packed_dot_enabled() && dot_tiled) {
        const uint32_t spatial = batch * output_height * output_width;
        const uint32_t output_words_per_spatial = output_channels / 4u;
        const uint32_t groups_x = (output_words_per_spatial + 7u) / 8u;
        const uint32_t groups_y = (spatial + 3u) / 4u;
        dispatched = vk_dispatch_kernel(&k_qconv2d_int8_dot_tiled, binds,
                                        groups_x, groups_y, 1u);
        if (!dispatched) {
            vk_disable_packed_dot_after_failure(k_qconv2d_int8_dot_tiled.name);
        }
        else {
            vk_context_current()->qconv_dot_tiled_dispatches++;
        }
    }
    if (!dispatched && tiled) {
        const uint32_t spatial = batch * output_height * output_width;
        const uint32_t output_words_per_spatial = output_channels / 4u;
        dispatched = vk_dispatch_kernel(
            &k_qconv2d_int8_tiled, binds,
            (output_words_per_spatial + 7u) / 8u,
            (spatial + 3u) / 4u, 1u);
        if (dispatched)
            vk_context_current()->qconv_tiled_dispatches++;
    }
    if (!dispatched && !vk_dispatch_kernel(&k_qconv2d_int8, binds,
                                            (packed_words + 63u) / 64u,
                                            1u, 1u)) return 0;
    if (!dispatched)
        vk_context_current()->qconv_scalar_dispatches++;
    graph_mark_device(output_slot);
    return 1;
}

typedef struct { uint32_t elements, pad0, pad1, pad2; } VkTypedCopyParams;
typedef struct {
    uint32_t size, axis_offset, input_axis, output_axis;
    uint32_t inner, pad0, pad1, pad2;
} VkTypedConcatParams;
typedef struct {
    uint32_t n, h, w, c, out_h, out_w, ky, kx;
    uint32_t sy, sx, py, px, dtype, pad0, pad1, pad2;
} VkTypedMaxPoolParams;
typedef struct { uint32_t n, h, w, c, out_h, out_w, pad0, pad1; } VkTypedResizeParams;
typedef struct { uint32_t elements, output_type, zero_point_type, has_zero_point; } VkTypedQuantizeParams;
typedef struct {
    uint32_t size, input_type, scale_type, zero_point_type;
    uint32_t output_type, has_zero_point, pad0, pad1;
} VkTypedDequantizeParams;

_Static_assert(sizeof(VkTypedCopyParams) == 16, "copyTyped uniform ABI");
_Static_assert(sizeof(VkTypedConcatParams) == 32, "concatCopyTyped uniform ABI");
_Static_assert(sizeof(VkTypedMaxPoolParams) == 64, "maxPool2DTyped uniform ABI");
_Static_assert(sizeof(VkTypedResizeParams) == 32, "resizeNearestTyped uniform ABI");
_Static_assert(sizeof(VkTypedQuantizeParams) == 16, "quantizeLinearTyped uniform ABI");
_Static_assert(sizeof(VkTypedDequantizeParams) == 32, "dequantizeLinearTyped uniform ABI");

static int typed_shape_qdesc_valid(float scale, int32_t zero_point, uint32_t dtype) {
    return isfinite(scale) && scale > 0.0f &&
        qbyte_dtype_zero_point_valid(dtype, zero_point);
}

static int typed_shape_qdesc_same(float input_scale, int32_t input_zero_point,
                                  uint32_t input_dtype, float output_scale,
                                  int32_t output_zero_point, uint32_t output_dtype) {
    return typed_shape_qdesc_valid(input_scale, input_zero_point, input_dtype) &&
        typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype) &&
        input_dtype == output_dtype && input_zero_point == output_zero_point &&
        input_scale == output_scale;
}

static int typed_shape_nhwc_elements(uint32_t n, uint32_t h, uint32_t w,
                                     uint32_t c, uint32_t* elements) {
    uint64_t value = n;
    if (!n || !h || !w || !c || !elements ||
        !qconv_mul_u64(value, h, &value) || !qconv_mul_u64(value, w, &value) ||
        !qconv_mul_u64(value, c, &value) || value > UINT32_MAX) return 0;
    *elements = (uint32_t)value;
    return 1;
}

static uint32_t typed_shape_groups(uint32_t elements) {
    return elements / 64u + (elements % 64u != 0u);
}

static uint32_t typed_shape_packed_groups(size_t packed_bytes) {
    uint32_t words = (uint32_t)(packed_bytes / sizeof(uint32_t));
    return typed_shape_groups(words);
}

static uint32_t typed_shape_zero_word(int32_t zero_point, uint32_t dtype) {
    return dtype == VX_DTYPE_I8 ? (uint32_t)(uint8_t)(int8_t)zero_point :
        (uint32_t)(uint8_t)zero_point;
}

int vk_graph_quantize_typed_f32_i8u8(const float* input, uint32_t elements,
                                     void* output, float output_scale,
                                     int32_t output_zero_point, uint32_t output_dtype) {
    if (!input || !output || input == output || elements == 0 ||
        !typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype) ||
        (uint64_t)elements > SIZE_MAX / sizeof(float)) return 0;
    size_t output_bytes = (size_t)elements;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_bytes, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_device(input, (size_t)elements * sizeof(float), 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    uint32_t zero_word = typed_shape_zero_word(output_zero_point, output_dtype);
    VkTypedQuantizeParams params = {
        elements, output_dtype, output_dtype, 1u
    };
    size_t scale_offset = graph_scratch_upload(&output_scale, sizeof(output_scale));
    size_t zero_offset = graph_scratch_upload(&zero_word, sizeof(zero_word));
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (scale_offset == SIZE_MAX || zero_offset == SIZE_MAX || params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {input_slot->offset, input_slot->bytes}, {scale_offset, sizeof(output_scale)},
        {zero_offset, sizeof(zero_word)}, {output_slot->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_quantize_typed_i8u8, binds,
                            typed_shape_packed_groups(packed_output_bytes), 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_dequantize_typed_i8u8_f32(const void* input, uint32_t elements,
                                       float input_scale, int32_t input_zero_point,
                                       uint32_t input_dtype, float* output) {
    if (!input || !output || input == output || elements == 0 ||
        !typed_shape_qdesc_valid(input_scale, input_zero_point, input_dtype) ||
        (uint64_t)elements > SIZE_MAX / sizeof(float)) return 0;
    size_t input_bytes = (size_t)elements;
    size_t packed_input_bytes;
    if (!graph_packed_bytes(input_bytes, &packed_input_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_bytes, 0);
    VkTensorSlot* output_slot = graph_output_slot(output, (size_t)elements * sizeof(float));
    if (!input_slot || !output_slot) return 0;
    uint32_t zero_word = typed_shape_zero_word(input_zero_point, input_dtype);
    VkTypedDequantizeParams params = {
        elements, input_dtype, VX_DTYPE_F32, input_dtype,
        VX_DTYPE_F32, 1u, 0u, 0u
    };
    size_t scale_offset = graph_scratch_upload(&input_scale, sizeof(input_scale));
    size_t zero_offset = graph_scratch_upload(&zero_word, sizeof(zero_word));
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (scale_offset == SIZE_MAX || zero_offset == SIZE_MAX || params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[5] = {
        {input_slot->offset, packed_input_bytes}, {scale_offset, sizeof(input_scale)},
        {zero_offset, sizeof(zero_word)}, {output_slot->offset, output_slot->bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_dequantize_typed_i8u8, binds,
                            typed_shape_groups(elements), 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_copy_i8u8(const void* input, uint32_t input_elements,
                       void* output, uint32_t output_elements,
                       float input_scale, int32_t input_zero_point,
                       float output_scale, int32_t output_zero_point,
                       uint32_t input_dtype, uint32_t output_dtype) {
    if (!input || !output || input == output || input_elements == 0 ||
        input_elements != output_elements || !typed_shape_qdesc_same(input_scale,
        input_zero_point, input_dtype, output_scale, output_zero_point, output_dtype)) return 0;
    size_t packed_bytes;
    if (!graph_packed_bytes((size_t)input_elements, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    VkTypedCopyParams params = {input_elements, 0u, 0u, 0u};
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes}, {output_slot->offset, packed_bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_copy_typed_i8u8, binds,
                            typed_shape_packed_groups(packed_bytes), 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_transpose_i8u8(const void* input, void* output,
                            const uint32_t* input_shape,
                            const uint32_t* permutation, uint32_t rank,
                            uint32_t elements, float input_scale,
                            int32_t input_zero_point, float output_scale,
                            int32_t output_zero_point, uint32_t input_dtype,
                            uint32_t output_dtype) {
    uint32_t input_strides[8] = {0};
    uint32_t output_shape[8] = {0};
    uint32_t output_strides[8] = {0};
    uint32_t metadata[18] = {0};
    uint32_t seen = 0;
    uint64_t product = 1;
    uint64_t stride = 1;
    size_t packed_bytes;
    if (!input || !output || !input_shape || !permutation ||
        rank == 0 || rank > 8 || elements == 0 ||
        !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype,
                                output_scale, output_zero_point, output_dtype) ||
        qsdpa_ranges_overlap(output, elements, input, elements)) return 0;
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
        output_strides[reverse] = (uint32_t)stride;
        stride *= output_shape[reverse];
    }
    if (!graph_packed_bytes(elements, &packed_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot =
        graph_ensure_packed_bytes(input, elements, 0);
    VkTensorSlot* output_slot =
        graph_output_packed_bytes(output, elements);
    if (!input_slot || !output_slot) return 0;
    metadata[0] = elements;
    metadata[1] = rank;
    for (uint32_t dimension = 0; dimension < rank; dimension++) {
        metadata[2 + dimension] = output_strides[dimension];
        metadata[2 + rank + dimension] =
            input_strides[permutation[dimension]];
    }
    size_t metadata_bytes = (size_t)(2 + 2 * rank) * sizeof(uint32_t);
    size_t metadata_offset =
        graph_scratch_upload(metadata, metadata_bytes);
    if (metadata_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, packed_bytes},
        {output_slot->offset, packed_bytes},
        {metadata_offset, metadata_bytes},
    };
    if (!vk_dispatch_kernel(&k_transpose_typed_i8u8, binds,
                            typed_shape_packed_groups(packed_bytes),
                            1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_concat_i8u8(const void* const* inputs, const uint32_t* input_elements,
                         const uint32_t* input_axes, const float* input_scales,
                         const int32_t* input_zero_points, const uint32_t* input_dtypes,
                         uint32_t input_count, void* output, uint32_t output_elements,
                         uint32_t output_axis, uint32_t inner,
                         float output_scale, int32_t output_zero_point,
                         uint32_t output_dtype) {
    if (!inputs || !input_elements || !input_axes || !input_scales ||
        !input_zero_points || !input_dtypes || !output || !input_count ||
        !output_elements || !output_axis || !inner ||
        !typed_shape_qdesc_valid(output_scale, output_zero_point, output_dtype)) return 0;
    uint64_t output_width = (uint64_t)output_axis * inner;
    if (!output_width || output_width > output_elements || output_elements % output_width) return 0;
    uint64_t outer = output_elements / output_width;
    uint64_t axis_sum = 0;
    for (uint32_t index = 0; index < input_count; index++) {
        uint64_t expected = 0;
        if (!qconv_mul_u64(outer, input_axes[index], &expected) ||
            !qconv_mul_u64(expected, inner, &expected)) {
            return 0;
        }
        if (!inputs[index] || inputs[index] == output || !input_axes[index] ||
            expected != input_elements[index] ||
            !typed_shape_qdesc_same(input_scales[index], input_zero_points[index],
                                    input_dtypes[index], output_scale,
                                    output_zero_point, output_dtype) ||
            axis_sum > UINT32_MAX - input_axes[index]) return 0;
        axis_sum += input_axes[index];
    }
    if (axis_sum != output_axis) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!output_slot) return 0;
    if (!vk_staging_zero(output_slot->offset, packed_output_bytes)) return 0;
    output_slot->host_dirty = 0;
    output_slot->device_dirty = 0;
    uint32_t axis_offset = 0;
    for (uint32_t index = 0; index < input_count; index++) {
        size_t packed_input_bytes;
        if (!graph_packed_bytes(input_elements[index], &packed_input_bytes)) return 0;
        VkTensorSlot* input_slot = graph_ensure_packed_bytes(inputs[index], input_elements[index], 0);
        if (!input_slot) return 0;
        VkTypedConcatParams params = {
            input_elements[index], axis_offset, input_axes[index], output_axis,
            inner, 0u, 0u, 0u
        };
        size_t params_offset = graph_scratch_upload(&params, sizeof(params));
        if (params_offset == SIZE_MAX) return 0;
        VkGraphBinding binds[3] = {
            {input_slot->offset, packed_input_bytes}, {output_slot->offset, packed_output_bytes},
            {params_offset, sizeof(params)}
        };
        if (!vk_dispatch_kernel(&k_concat_typed_i8u8, binds,
                                typed_shape_groups(input_elements[index]), 1u, 1u)) return 0;
        axis_offset += input_axes[index];
    }
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_maxpool2d_i8u8(const void* input, void* output,
                            uint32_t batch, uint32_t input_height,
                            uint32_t input_width, uint32_t channels,
                            uint32_t output_height, uint32_t output_width,
                            uint32_t kernel_y, uint32_t kernel_x,
                            uint32_t stride_y, uint32_t stride_x,
                            uint32_t padding_top, uint32_t padding_left,
                            uint32_t padding_bottom, uint32_t padding_right,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t output_dtype) {
    uint32_t input_elements, output_elements;
    uint64_t padded_height, padded_width, expected_height, expected_width;
    if (!input || !output || input == output || !kernel_y || !kernel_x ||
        !stride_y || !stride_x || !typed_shape_qdesc_same(input_scale,
        input_zero_point, input_dtype, output_scale, output_zero_point, output_dtype) ||
        !typed_shape_nhwc_elements(batch, input_height, input_width, channels, &input_elements) ||
        !typed_shape_nhwc_elements(batch, output_height, output_width, channels, &output_elements)) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    if (padded_height < kernel_y || padded_width < kernel_x) return 0;
    expected_height = (padded_height - kernel_y) / stride_y + 1u;
    expected_width = (padded_width - kernel_x) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    VkTypedMaxPoolParams params = {
        batch, input_height, input_width, channels, output_height, output_width,
        kernel_y, kernel_x, stride_y, stride_x, padding_top, padding_left,
        input_dtype, 0u, 0u, 0u
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes}, {output_slot->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_maxpool_typed_i8u8, binds,
                            typed_shape_packed_groups(packed_output_bytes), 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_resize_nearest_i8u8(const void* input, void* output,
                                 uint32_t batch, uint32_t input_height,
                                 uint32_t input_width, uint32_t channels,
                                 uint32_t output_height, uint32_t output_width,
                                 float input_scale, int32_t input_zero_point,
                                 float output_scale, int32_t output_zero_point,
                                 uint32_t input_dtype, uint32_t output_dtype) {
    uint32_t input_elements, output_elements;
    if (!input || !output || input == output ||
        !typed_shape_qdesc_same(input_scale, input_zero_point, input_dtype,
                                output_scale, output_zero_point, output_dtype) ||
        !typed_shape_nhwc_elements(batch, input_height, input_width, channels, &input_elements) ||
        !typed_shape_nhwc_elements(batch, output_height, output_width, channels, &output_elements)) return 0;
    size_t packed_output_bytes;
    if (!graph_packed_bytes(output_elements, &packed_output_bytes)) return 0;
    graph_scratch_begin();
    VkTensorSlot* input_slot = graph_ensure_packed_bytes(input, input_elements, 0);
    VkTensorSlot* output_slot = graph_output_packed_bytes(output, output_elements);
    if (!input_slot || !output_slot) return 0;
    VkTypedResizeParams params = {
        batch, input_height, input_width, channels, output_height, output_width, 0u, 0u
    };
    size_t params_offset = graph_scratch_upload(&params, sizeof(params));
    if (params_offset == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {input_slot->offset, input_slot->bytes}, {output_slot->offset, packed_output_bytes},
        {params_offset, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_resize_nearest_typed_i8u8, binds,
                            typed_shape_packed_groups(packed_output_bytes), 1u, 1u)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int vk_graph_quantize_linear_i8(const float* in, signed char* out, long n,
                                float input_scale, int input_zp,
                                float output_scale, int output_zp) {
    if (!in || !out || n <= 0 || output_scale <= 0.0f) return 0;
    size_t in_bytes = (size_t)n * sizeof(float);
    size_t packed_words = ((size_t)n + 3u) / 4u;
    size_t out_bytes = packed_words * sizeof(uint32_t);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    struct {
        uint32_t size;
        int32_t input_zp;
        int32_t output_zp;
        uint32_t has_input_scale;
        float input_scale;
        float output_scale;
        uint32_t pad0;
        uint32_t pad1;
    } params = {
        (uint32_t)n, (int32_t)input_zp, (int32_t)output_zp,
        input_scale > 0.0f ? 1u : 0u, input_scale, output_scale, 0u, 0u
    };
    size_t p_off = graph_scratch_upload(&params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {
        {src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}
    };
    if (!vk_dispatch_kernel(&k_quantize, binds, (uint32_t)((packed_words + 63u) / 64u), 1, 1)) return 0;
    graph_mark_device(dst);
    return vk_graph_sync_host(out, (size_t)n, 0);
}

static int vk_graph_profile_common(VkKernel* kernel, const float* in, float* out,
                                   int n, int h, int w, int c, long out_elems_per_batch,
                                   uint32_t gx, uint32_t gy) {
    if (!kernel || !in || !out || n <= 0 || h <= 0 || w <= 0 || c <= 0 || out_elems_per_batch <= 0) return 0;
    size_t in_bytes = (size_t)n * (size_t)h * (size_t)w * (size_t)c * sizeof(float);
    size_t out_bytes = (size_t)n * (size_t)out_elems_per_batch * sizeof(float);
    graph_scratch_begin();
    VkTensorSlot* src = graph_ensure_device(in, in_bytes, 0);
    VkTensorSlot* dst = graph_output_slot(out, out_bytes);
    if (!src || !dst) return 0;
    uint32_t params[4] = {(uint32_t)h, (uint32_t)w, (uint32_t)c, (uint32_t)n};
    size_t p_off = graph_scratch_upload(params, sizeof(params));
    if (p_off == SIZE_MAX) return 0;
    VkGraphBinding binds[3] = {{src->offset, in_bytes}, {dst->offset, out_bytes}, {p_off, sizeof(params)}};
    if (!vk_dispatch_kernel(kernel, binds, gx, gy, (uint32_t)n)) return 0;
    graph_mark_device(dst);
    return 1;
}

int vk_graph_spatial_softargmax_y_f32(const float* in, float* out, int n, int h, int w, int c) {
    return vk_graph_profile_common(&k_spatial_softargmax_y, in, out, n, h, w, c,
                                   (long)c * w, ((uint32_t)w + 63u) / 64u, (uint32_t)c);
}
