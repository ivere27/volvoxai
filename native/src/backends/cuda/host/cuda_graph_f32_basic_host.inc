int cuda_graph_copy_f32(const float* input, float* output, long elements) {
    size_t bytes;
    uint32_t count;
    CudaTensorView source;
    CudaTensorView destination;
    void* args[3];
    if (!input || !output || elements <= 0 || (unsigned long)elements > UINT32_MAX ||
        (size_t)elements > SIZE_MAX / sizeof(float)) return 0;
    count = (uint32_t)elements;
    bytes = (size_t)count * sizeof(float);
    if (!graph_ensure_input_view(input, bytes, 0, &source) ||
        !graph_output_view(output, bytes, &destination)) return 0;
    args[0] = &source.device;
    args[1] = &destination.device;
    args[2] = &count;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_copy), count, args)) return 0;
    graph_mark_device(destination.slot);
    return 1;
}

int cuda_graph_alias_f32(const float* input, float* output, long elements) {
    return cuda_graph_copy_f32(input, output, elements);
}

#include "cuda_graph_rope_host.inc"
/* The shared RoPE power-cache helper is defined by the preceding fragment. */
int cuda_graph_head_rms_rope_row_f32(const float* input, const float* weight,
    float* normalized, float* flat, float* output, int sequence, int row,
    int heads, int rotary_width, float epsilon, float theta,
    int32_t position, int interleaved) {
    CudaContextState* state = cuda_context_state_get();
    VxEngineState* owner = vx_engine_state_current();
    CudaTensorView views[6];
    uint32_t full_count;
    size_t full_bytes;
    if (!state || !state->ready || !owner || !cuda_domain_enforced ||
        !cuda_forward_scope_active || !cuda_forward_context_ready || cuda_forward_context_failed ||
        !input || !weight || !normalized || !flat || !output ||
        sequence <= 0 || row < 0 || row >= sequence || row == INT32_MAX ||
        heads <= 0 || rotary_width <= 0 || rotary_width > 128 || (rotary_width & 1) ||
        position < 0 || !isfinite(epsilon) || epsilon <= 0.0f ||
        !isfinite(theta) || theta < 1.0f || (interleaved != 0 && interleaved != 1) ||
        !owner->decode_metadata_stage ||
        (uintptr_t)owner->decode_metadata_stage % _Alignof(int32_t) ||
        (size_t)row + 2u > owner->decode_metadata_stage_bytes / sizeof(int32_t) ||
        !checked_product4_bytes(1u, (uint32_t)sequence, (uint32_t)heads, 128u,
            sizeof(float), &full_count, &full_bytes) ||
        !CUDA_FORWARD_FUNCTION(k_head_rms_rope) ||
        /* Preserve the original cooperative RMS128 route's 32-byte resource
         * proof as well as this entry's 16 bytes of static shared storage. */
        !cuda_launch_dimensions_valid((uint32_t)heads, 1u, 1u, 128u, 1u, 1u, 32u) ||
        !cuda_launch_dimensions_valid((uint32_t)heads, 1u, 1u, 128u, 1u, 1u, 16u)) return 0;
    uint32_t elements = full_count / (uint32_t)sequence;
    const void* hosts[6] = {input, weight, normalized, flat, output, owner->decode_metadata_stage};
    size_t extents[6] = {full_bytes, 128u * sizeof(float), full_bytes,
        full_bytes, full_bytes, owner->decode_metadata_stage_bytes};
    /* Resolve only existing live domain owners. No allocation, upload,
     * promotion or logical growth can precede a pure alias refusal. */
    for (int i = 0; i < 6; i++) {
        int index = graph_find_containing_slot(hosts[i], extents[i]);
        if (index < 0) return 0;
        CudaTensorSlot* slot = &graph_slots[index];
        size_t offset = (uintptr_t)hosts[i] - (uintptr_t)slot->host;
        if (!slot->device || offset > slot->bytes || extents[i] > slot->bytes - offset ||
            slot->device > UINTPTR_MAX - offset ||
            slot->device + offset > UINTPTR_MAX - extents[i] ||
            (i == 1 && !slot->is_weight)) return 0;
        views[i] = (CudaTensorView){slot, slot->device + offset};
    }
    if (!state->rope_powers_device ||
        state->rope_powers_device > UINTPTR_MAX - CUDA_ROPE_POWER_STORAGE_BYTES) return 0;
    for (int i = 2; i < 5; i++) {
        for (int j = 0; j < 6; j++) {
            if (j == i || (j >= 2 && j < i)) continue;
            /* Normalized and flat are the same contiguous view only when both
             * resolved byte spans agree. The kernel writes both at the same
             * head/lane index with the same value, so this adds no race. */
            if (i == 2 && j == 3 && hosts[2] == hosts[3] &&
                extents[2] == extents[3] && views[2].device == views[3].device) continue;
            if (ranges_overlap(hosts[i], extents[i], hosts[j], extents[j]) ||
                ranges_overlap((const void*)(uintptr_t)views[i].device, extents[i],
                    (const void*)(uintptr_t)views[j].device, extents[j])) return 0;
        }
        if (ranges_overlap(hosts[i], extents[i], state->rope_powers_host, sizeof(state->rope_powers_host)) ||
            ranges_overlap((const void*)(uintptr_t)views[i].device, extents[i],
                (const void*)(uintptr_t)state->rope_powers_device, CUDA_ROPE_POWER_STORAGE_BYTES)) return 0;
    }
    /* Coherence starts here. A later failure must never retry this triple as
     * ordinary work, since a current upload or kernel may have been queued. */
    CUdeviceptr powers;
    if (!cuda_rope_powers_view(state, rotary_width, theta, &powers)) return -1;
    int32_t* positions = (int32_t*)owner->decode_metadata_stage + (size_t)row + 1u;
    int changed = !owner->decode_cuda_positions_valid ||
        owner->decode_cuda_positions_offset != (size_t)row + 1u ||
        owner->decode_cuda_positions_count != 1u || *positions != position;
    if (*positions != position) {
        *positions = position;
        owner->decode_cuda_mask_valid = 0;
    }
    if (changed) {
        views[5].slot->host_dirty = 1;
        views[5].slot->device_dirty = 0;
    }
    owner->decode_cuda_positions_offset = (size_t)row + 1u;
    owner->decode_cuda_positions_count = 1u;
    owner->decode_cuda_positions_valid = 1;
    for (int i = 0; i < 6; i++)
        if (!graph_ensure_input_view(hosts[i], extents[i], i == 1, &views[i])) return -1;
    CUdeviceptr offset = (CUdeviceptr)(uint32_t)row * elements * sizeof(float);
    CUdeviceptr devices[6] = {views[0].device + offset, views[1].device,
        views[2].device + offset, views[3].device + offset,
        views[5].device + ((size_t)row + 1u) * sizeof(int32_t), views[4].device + offset};
    uint32_t rows = 1, narrow_sequence = 1, head_count = (uint32_t)heads;
    uint32_t rotary = (uint32_t)rotary_width, position_offset = 0;
    uint32_t order = (uint32_t)interleaved, mode = 2;
    void* args[15] = {&devices[0], &devices[1], &devices[2], &devices[3],
        &devices[4], &devices[5], &rows, &narrow_sequence, &head_count, &rotary,
        &powers, &epsilon, &position_offset, &order, &mode};
    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_head_rms_rope), (uint32_t)heads,
            1u, 1u, 128u, 1u, 1u, 0u, args)) return -1;
    for (int i = 2; i < 5; i++) graph_mark_device(views[i].slot);
    return 1;
}


int cuda_graph_add_relu_f32(const float* a, const float* b, float* output,
                            long elements, int relu) {
    CudaTensorView a_view;
    CudaTensorView b_view;
    CudaTensorView output_view;
    uint32_t count;
    uint32_t activation;
    size_t bytes;
    void* args[5];
    if (!a || !b || !output || elements <= 0 || relu < 0 || relu > 2 ||
        !checked_elements_bytes((uint64_t)elements, sizeof(float),
                                &count, &bytes)) return 0;
    activation = (uint32_t)relu;
    if (!graph_ensure_input_view(a, bytes, 0, &a_view) ||
        !graph_ensure_input_view(b, bytes, 0, &b_view) ||
        !graph_output_view(output, bytes, &output_view)) return 0;
    args[0] = &a_view.device;
    args[1] = &b_view.device;
    args[2] = &output_view.device;
    args[3] = &count;
    args[4] = &activation;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_add_relu), count, args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}


int cuda_graph_add3_relu_f32(const float* a, const float* b, const float* c,
                             float* output, long elements, int relu) {
    CudaTensorView a_view;
    CudaTensorView b_view;
    CudaTensorView c_view;
    CudaTensorView output_view;
    uint32_t count;
    uint32_t activation;
    size_t bytes;
    void* args[6];
    if (!a || !b || !c || !output || elements <= 0 || relu < 0 || relu > 2 ||
        !checked_elements_bytes((uint64_t)elements, sizeof(float),
                                &count, &bytes) ||
        ranges_overlap(output, bytes, a, bytes) ||
        ranges_overlap(output, bytes, b, bytes) ||
        ranges_overlap(output, bytes, c, bytes)) return 0;
    activation = (uint32_t)relu;
    if (!graph_ensure_input_view(a, bytes, 0, &a_view) ||
        !graph_ensure_input_view(b, bytes, 0, &b_view) ||
        !graph_ensure_input_view(c, bytes, 0, &c_view) ||
        !graph_output_view(output, bytes, &output_view)) return 0;
    args[0] = &a_view.device;
    args[1] = &b_view.device;
    args[2] = &c_view.device;
    args[3] = &output_view.device;
    args[4] = &count;
    args[5] = &activation;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_add3_relu), count, args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}

int cuda_graph_add_f32(const float* a, const float* b, float* output,
                       long elements) {
    return cuda_graph_add_relu_f32(a, b, output, elements, 0);
}

int cuda_graph_binary_f32(const float* a, long a_elements,
                          const float* b, long b_elements,
                          float* output, long output_elements,
                          const uint32_t* output_strides,
                          const uint32_t* a_strides,
                          const uint32_t* b_strides,
                          int rank, int op) {
    CudaTensorSlot* a_slot;
    CudaTensorSlot* b_slot;
    CudaTensorSlot* output_slot;
    VxCudaBroadcastParams params;
    uint32_t elements;
    int32_t operation = op;
    void* args[6];
    if (!a || !b || !output || !output_strides || !a_strides || !b_strides ||
        a_elements <= 0 || b_elements <= 0 || output_elements <= 0 ||
        (unsigned long)a_elements > UINT32_MAX ||
        (unsigned long)b_elements > UINT32_MAX ||
        (unsigned long)output_elements > UINT32_MAX ||
        (size_t)a_elements > SIZE_MAX / sizeof(float) ||
        (size_t)b_elements > SIZE_MAX / sizeof(float) ||
        (size_t)output_elements > SIZE_MAX / sizeof(float) ||
        rank <= 0 || rank > 8 ||
        op < 0 || op > 3) return 0;
    elements = (uint32_t)output_elements;
    a_slot = graph_ensure_input(a, (size_t)a_elements * sizeof(float), 0);
    b_slot = graph_ensure_input(b, (size_t)b_elements * sizeof(float), 0);
    output_slot = graph_output(output, (size_t)elements * sizeof(float));
    if (!a_slot || !b_slot || !output_slot) return 0;
    memset(&params, 0, sizeof(params));
    memcpy(params.output_strides, output_strides, (size_t)rank * sizeof(uint32_t));
    memcpy(params.a_strides, a_strides, (size_t)rank * sizeof(uint32_t));
    memcpy(params.b_strides, b_strides, (size_t)rank * sizeof(uint32_t));
    params.rank = (uint32_t)rank;
    args[0] = &a_slot->device;
    args[1] = &b_slot->device;
    args[2] = &output_slot->device;
    args[3] = &elements;
    args[4] = &params;
    args[5] = &operation;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_binary), elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

enum {
    CUDA_UNARY_RELU = 0,
    CUDA_UNARY_SIGMOID = 1,
    CUDA_UNARY_GELU = 2,
    CUDA_UNARY_SILU = 3,
    CUDA_UNARY_TANH = 4,
    CUDA_UNARY_HARDSWISH = 5,
    CUDA_UNARY_HARDSIGMOID = 6,
    CUDA_UNARY_LEAKY_RELU = 7,
    CUDA_UNARY_CLIP = 8,
};

static int cuda_graph_unary(const float* input, float* output, long elements,
                            int operation, float alpha, float beta) {
    CudaTensorView source;
    CudaTensorView destination;
    uint32_t count;
    int32_t op = operation;
    size_t bytes;
    void* args[6];
    if (!input || !output || elements <= 0 || (unsigned long)elements > UINT32_MAX ||
        (size_t)elements > SIZE_MAX / sizeof(float)) return 0;
    count = (uint32_t)elements;
    bytes = (size_t)count * sizeof(float);
    if (!graph_ensure_input_view(input, bytes, 0, &source) ||
        !graph_output_view(output, bytes, &destination)) return 0;
    args[0] = &source.device;
    args[1] = &destination.device;
    args[2] = &count;
    args[3] = &op;
    args[4] = &alpha;
    args[5] = &beta;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_unary), count, args)) return 0;
    graph_mark_device(destination.slot);
    return 1;
}

int cuda_graph_relu_f32(const float* input, float* output, long elements) {
    return cuda_graph_unary(input, output, elements, CUDA_UNARY_RELU, 0.0f, 0.0f);
}
int cuda_graph_sigmoid_f32(const float* input, float* output, long elements) {
    return cuda_graph_unary(input, output, elements, CUDA_UNARY_SIGMOID, 0.0f, 0.0f);
}
int cuda_graph_gelu_f32(const float* input, float* output, long elements,
                        int approximate_tanh) {
    if (approximate_tanh != 0 && approximate_tanh != 1) return 0;
    return cuda_graph_unary(input, output, elements, CUDA_UNARY_GELU,
                            approximate_tanh ? 1.0f : 0.0f, 0.0f);
}
int cuda_graph_silu_f32(const float* input, float* output, long elements) {
    return cuda_graph_unary(input, output, elements, CUDA_UNARY_SILU, 0.0f, 0.0f);
}

int cuda_graph_silu_mul_dual_f32(const float* input, const float* other,
                                float* activation, float* output,
                                long full_elements, long offset, long elements) {
    CudaTensorView input_view, other_view, activation_view, output_view;
    uint32_t full_count, count;
    size_t full_bytes, bytes;
    CUdeviceptr input_device, other_device, activation_device, output_device;
    void* args[5];
    if (!input || !other || !activation || !output || full_elements <= 0 ||
        offset < 0 || elements <= 0 || offset > full_elements ||
        elements > full_elements - offset ||
        !checked_elements_bytes((uint64_t)full_elements, sizeof(float), &full_count, &full_bytes) ||
        !checked_elements_bytes((uint64_t)elements, sizeof(float), &count, &bytes) ||
        ranges_overlap(activation, full_bytes, output, full_bytes) ||
        ranges_overlap(activation, full_bytes, input, full_bytes) ||
        ranges_overlap(activation, full_bytes, other, full_bytes) ||
        ranges_overlap(output, full_bytes, input, full_bytes) ||
        ranges_overlap(output, full_bytes, other, full_bytes)) return 0;
    if (!graph_ensure_input_view(input, full_bytes, 0, &input_view) ||
        !graph_ensure_input_view(other, full_bytes, 0, &other_view)) return 0;
    /* A partial write preserves every earlier/later row. Resolve and make
     * both full output owners current before enqueuing the first write. */
    if (offset || count != full_count) {
        if (!graph_ensure_input_view(activation, full_bytes, 0, &activation_view) ||
            !graph_ensure_input_view(output, full_bytes, 0, &output_view)) return 0;
    } else if (!graph_output_view(activation, full_bytes, &activation_view) ||
               !graph_output_view(output, full_bytes, &output_view)) return 0;
    CUdeviceptr device_offset = (CUdeviceptr)(uint64_t)offset * sizeof(float);
    input_device = input_view.device + device_offset;
    other_device = other_view.device + device_offset;
    activation_device = activation_view.device + device_offset;
    output_device = output_view.device + device_offset;
    args[0] = &input_device;
    args[1] = &other_device;
    args[2] = &activation_device;
    args[3] = &output_device;
    args[4] = &count;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_silu_mul_dual), count, args)) return 0;
    graph_mark_device(activation_view.slot);
    graph_mark_device(output_view.slot);
    return 1;
}
int cuda_graph_tanh_f32(const float* input, float* output, long elements) {
    return cuda_graph_unary(input, output, elements, CUDA_UNARY_TANH, 0.0f, 0.0f);
}
int cuda_graph_hardswish_f32(const float* input, float* output, long elements) {
    return cuda_graph_unary(input, output, elements, CUDA_UNARY_HARDSWISH, 0.0f, 0.0f);
}
int cuda_graph_hardsigmoid_f32(const float* input, float* output, long elements) {
    return cuda_graph_unary(input, output, elements, CUDA_UNARY_HARDSIGMOID, 0.0f, 0.0f);
}
int cuda_graph_leaky_relu_f32(const float* input, float* output, long elements,
                              float alpha) {
    if (!isfinite(alpha)) return 0;
    return cuda_graph_unary(input, output, elements, CUDA_UNARY_LEAKY_RELU, alpha, 0.0f);
}

int cuda_graph_prelu_f32(const float* input, const float* weight,
                         float* output, long elements, int channels) {
    CudaTensorSlot* input_slot;
    CudaTensorSlot* weight_slot;
    CudaTensorSlot* output_slot;
    uint32_t count;
    uint32_t channel_count;
    size_t bytes;
    void* args[5];
    if (!input || !weight || !output || elements <= 0 || channels <= 0 ||
        !checked_elements_bytes((uint64_t)elements, sizeof(float),
                                &count, &bytes) ||
        (uint64_t)(uint32_t)channels * sizeof(float) > SIZE_MAX) return 0;
    channel_count = (uint32_t)channels;
    input_slot = graph_ensure_input(input, bytes, 0);
    weight_slot = graph_ensure_input(weight,
        (size_t)channel_count * sizeof(float), 1);
    output_slot = graph_output(output, bytes);
    if (!input_slot || !weight_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &weight_slot->device;
    args[2] = &output_slot->device;
    args[3] = &count;
    args[4] = &channel_count;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_prelu), count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}
int cuda_graph_clip_f32(const float* input, float* output, long elements,
                        float minimum, float maximum) {
    if (isnan(minimum) || isnan(maximum) || minimum > maximum) return 0;
    return cuda_graph_unary(input, output, elements, CUDA_UNARY_CLIP, minimum, maximum);
}

static int cuda_graph_norm(const float* input, const float* weight,
                           const float* bias, float* output,
                           int rows, int width, float epsilon, int rms) {
    CudaTensorView input_view;
    CudaTensorSlot* weight_slot;
    CudaTensorSlot* bias_slot = NULL;
    CudaTensorView output_view;
    uint32_t row_count;
    uint32_t row_width;
    size_t bytes;
    void* args[7];
    CUdeviceptr no_bias = 0;
    CUdeviceptr bias_device;
    if (!input || !weight || !output || rows <= 0 || width <= 0 ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !size_product_u32((uint32_t)rows, (uint32_t)width, sizeof(float), &bytes)) return 0;
    row_count = (uint32_t)rows;
    row_width = (uint32_t)width;
    if (!graph_ensure_input_view(input, bytes, 0, &input_view)) return 0;
    weight_slot = graph_ensure_input(weight, (size_t)width * sizeof(float), 1);
    if (!rms && bias)
        bias_slot = graph_ensure_input(bias, (size_t)width * sizeof(float), 1);
    if (!graph_output_view(output, bytes, &output_view) || !weight_slot ||
        (!rms && bias && !bias_slot)) return 0;
    args[0] = &input_view.device;
    args[1] = &weight_slot->device;
    if (rms) {
        unsigned int block = 32u;
        while (block < row_width && block < 256u) block *= 2u;
        args[2] = &output_view.device;
        args[3] = &row_count;
        args[4] = &row_width;
        args[5] = &epsilon;
        /* Keep the scalar route for unsupported block resources, very large
         * row grids and overlapping spans other than exact in-place input. */
        if ((input == output || !ranges_overlap(output, bytes, input, bytes)) &&
            !ranges_overlap(output, bytes, weight, (size_t)width * sizeof(float)) &&
            cuda_launch_dimensions_valid(row_count, 1u, 1u, block, 1u, 1u, 32u)) {
            if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_rmsnorm_cooperative),
                    row_count, 1u, 1u, block, 1u, 1u, 0u, args)) return 0;
        } else if (!cuda_launch_1d(
                CUDA_FORWARD_FUNCTION(k_rmsnorm), row_count, args)) return 0;
    } else {
        bias_device = bias_slot ? bias_slot->device : no_bias;
        args[2] = &bias_device;
        args[3] = &output_view.device;
        args[4] = &row_count;
        args[5] = &row_width;
        args[6] = &epsilon;
        if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_layernorm), row_count, args)) return 0;
    }
    graph_mark_device(output_view.slot);
    return 1;
}

int cuda_graph_layernorm_f32(const float* input, const float* weight,
                             const float* bias, float* output,
                             int rows, int width, float epsilon) {
    return cuda_graph_norm(input, weight, bias, output, rows, width, epsilon, 0);
}
int cuda_graph_rmsnorm_f32(const float* input, const float* weight,
                           float* output, int rows, int width, float epsilon) {
    return cuda_graph_norm(input, weight, NULL, output, rows, width, epsilon, 1);
}

static int cuda_graph_softmax(const float* input, float* output,
                              int rows, int width, int logarithmic) {
    CudaTensorSlot* input_slot;
    CudaTensorSlot* output_slot;
    uint32_t row_count;
    uint32_t row_width;
    int32_t log_flag = logarithmic;
    size_t bytes;
    void* args[6];
    if (!input || !output || rows <= 0 || width <= 0 ||
        !size_product_u32((uint32_t)rows, (uint32_t)width, sizeof(float), &bytes)) return 0;
    row_count = (uint32_t)rows;
    row_width = (uint32_t)width;
    input_slot = graph_ensure_input(input, bytes, 0);
    output_slot = graph_output(output, bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &row_count;
    args[3] = &row_width;
    args[4] = &log_flag;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_softmax), row_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_softmax_f32(const float* input, float* output,
                           int rows, int width) {
    return cuda_graph_softmax(input, output, rows, width, 0);
}
int cuda_graph_logsoftmax_f32(const float* input, float* output,
                              int rows, int width) {
    return cuda_graph_softmax(input, output, rows, width, 1);
}

int cuda_graph_reduce_f32(const float* input, float* output,
                          int rows, int width, float scale) {
    CudaTensorSlot* input_slot;
    CudaTensorSlot* output_slot;
    uint32_t row_count;
    uint32_t column_count;
    uint32_t input_count;
    size_t input_bytes;
    size_t output_bytes;
    void* args[5];
    if (!input || !output || rows <= 0 || width <= 0 || !isfinite(scale) ||
        !checked_elements_bytes((uint64_t)(uint32_t)rows * (uint32_t)width,
                                sizeof(float), &input_count, &input_bytes) ||
        (uint64_t)(uint32_t)rows * sizeof(float) > SIZE_MAX) return 0;
    row_count = (uint32_t)rows;
    column_count = (uint32_t)width;
    output_bytes = (size_t)row_count * sizeof(float);
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &row_count;
    args[3] = &column_count;
    args[4] = &scale;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_reduce), row_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_global_average_pool_f32(const float* input, float* output,
                                       int batch, int height, int width,
                                       int channels) {
    CudaTensorSlot* input_slot;
    CudaTensorSlot* output_slot;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t n, h, w, c;
    size_t input_bytes;
    size_t output_bytes;
    void* args[6];
    if (!input || !output || batch <= 0 || height <= 0 || width <= 0 ||
        channels <= 0 || !checked_product4_bytes((uint32_t)batch,
            (uint32_t)height, (uint32_t)width, (uint32_t)channels,
            sizeof(float), &input_count, &input_bytes) ||
        !checked_elements_bytes((uint64_t)(uint32_t)batch *
            (uint32_t)channels, sizeof(float), &output_count,
            &output_bytes)) return 0;
    n = (uint32_t)batch;
    h = (uint32_t)height;
    w = (uint32_t)width;
    c = (uint32_t)channels;
    input_slot = graph_ensure_input(input, input_bytes, 0);
    output_slot = graph_output(output, output_bytes);
    if (!input_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &output_slot->device;
    args[2] = &n;
    args[3] = &h;
    args[4] = &w;
    args[5] = &c;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_global_average_pool), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_average_pool2d_f32(const float* input, float* output,
                                  int batch, int height, int width,
                                  int channels, int output_height,
                                  int output_width, int kernel_y, int kernel_x,
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
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_average_pool), output_count, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_batchnorm2d_f32(const float* input, const float* weight,
                               const float* bias, const float* mean,
                               const float* variance, float* output,
                               int batch, int height, int width, int channels,
                               float epsilon) {
    CudaTensorSlot *input_slot, *weight_slot, *bias_slot, *mean_slot,
                   *variance_slot, *output_slot;
    uint32_t elements;
    uint32_t params[4];
    size_t bytes;
    size_t metadata_bytes;
    void* args[11];
    if (!input || !weight || !bias || !mean || !variance || !output ||
        batch <= 0 || height <= 0 || width <= 0 || channels <= 0 ||
        !(epsilon > 0.0f) || !isfinite(epsilon) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)height,
            (uint32_t)width, (uint32_t)channels, sizeof(float),
            &elements, &bytes) ||
        (uint64_t)(uint32_t)channels * sizeof(float) > SIZE_MAX) return 0;
    metadata_bytes = (size_t)(uint32_t)channels * sizeof(float);
    params[0] = (uint32_t)batch;
    params[1] = (uint32_t)height;
    params[2] = (uint32_t)width;
    params[3] = (uint32_t)channels;
    input_slot = graph_ensure_input(input, bytes, 0);
    weight_slot = graph_ensure_input(weight, metadata_bytes, 1);
    bias_slot = graph_ensure_input(bias, metadata_bytes, 1);
    mean_slot = graph_ensure_input(mean, metadata_bytes, 1);
    variance_slot = graph_ensure_input(variance, metadata_bytes, 1);
    output_slot = graph_output(output, bytes);
    if (!input_slot || !weight_slot || !bias_slot || !mean_slot ||
        !variance_slot || !output_slot) return 0;
    args[0] = &input_slot->device;
    args[1] = &weight_slot->device;
    args[2] = &bias_slot->device;
    args[3] = &mean_slot->device;
    args[4] = &variance_slot->device;
    args[5] = &output_slot->device;
    for (int index = 0; index < 4; index++) args[index + 6] = &params[index];
    args[10] = &epsilon;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_batchnorm), elements, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_groupnorm_f32(const float* input, const float* weight,
                             const float* bias, float* output,
                             int batch, int height, int width, int channels,
                             int groups, float epsilon) {
    CudaTensorSlot *input_slot, *weight_slot, *bias_slot = NULL, *output_slot;
    uint32_t elements;
    uint32_t tasks;
    uint32_t params[5];
    size_t bytes;
    size_t metadata_bytes;
    CUdeviceptr no_bias = 0;
    CUdeviceptr bias_device;
    void* args[11];
    if (!input || !weight || !output || batch <= 0 || height <= 0 ||
        width <= 0 || channels <= 0 || groups <= 0 || channels % groups ||
        !(epsilon > 0.0f) || !isfinite(epsilon) ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)height,
            (uint32_t)width, (uint32_t)channels, sizeof(float),
            &elements, &bytes) ||
        !checked_elements_bytes((uint64_t)(uint32_t)batch *
            (uint32_t)groups, 1, &tasks, &metadata_bytes) ||
        tasks > (uint32_t)INT_MAX) return 0;
    metadata_bytes = (size_t)(uint32_t)channels * sizeof(float);
    params[0] = (uint32_t)batch;
    params[1] = (uint32_t)height;
    params[2] = (uint32_t)width;
    params[3] = (uint32_t)channels;
    params[4] = (uint32_t)groups;
    input_slot = graph_ensure_input(input, bytes, 0);
    weight_slot = graph_ensure_input(weight, metadata_bytes, 1);
    if (bias) bias_slot = graph_ensure_input(bias, metadata_bytes, 1);
    output_slot = graph_output(output, bytes);
    if (!input_slot || !weight_slot || (bias && !bias_slot) || !output_slot)
        return 0;
    bias_device = bias_slot ? bias_slot->device : no_bias;
    args[0] = &input_slot->device;
    args[1] = &weight_slot->device;
    args[2] = &bias_device;
    args[3] = &output_slot->device;
    for (int index = 0; index < 5; index++) args[index + 4] = &params[index];
    args[9] = &epsilon;
    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_groupnorm), tasks, 1u, 1u,
                     64u, 1u, 1u, 0u, args)) return 0;
    graph_mark_device(output_slot);
    return 1;
}

int cuda_graph_embedding_f32(const int32_t* tokens, const float* weight,
                             float* output, int token_count, int width,
                             int vocab_size) {
    CudaTensorView token_view;
    CudaTensorSlot* weight_slot;
    CudaTensorView output_view;
    uint32_t count;
    uint32_t row_width;
    size_t weight_bytes;
    size_t output_bytes;
    void* args[5];
    if (!tokens || !weight || !output || token_count <= 0 || width <= 0 ||
        vocab_size <= 0 ||
        !size_product_u32((uint32_t)vocab_size, (uint32_t)width,
                          sizeof(float), &weight_bytes) ||
        !size_product_u32((uint32_t)token_count, (uint32_t)width,
                          sizeof(float), &output_bytes)) return 0;
    /* Public bounded-GPU execution proves every producer chain at compile
     * time and preflights public IDs before committing a run. Device-produced
     * IDs can therefore have a deliberately stale host mirror here. Raw
     * private-engine callers keep the legacy host validation below. */
    if (!vx_engine_state_current()->bounded_gpu_value_domain_proven)
        for (int i = 0; i < token_count; i++)
            if (tokens[i] < 0 || tokens[i] >= vocab_size) return 0;
    count = (uint32_t)token_count;
    row_width = (uint32_t)width;
    if (!graph_ensure_input_view(tokens,
            (size_t)token_count * sizeof(int32_t), 0, &token_view)) return 0;
    weight_slot = graph_ensure_input(weight, weight_bytes, 1);
    if (!weight_slot || !graph_output_view(output, output_bytes, &output_view))
        return 0;
    args[0] = &token_view.device;
    args[1] = &weight_slot->device;
    args[2] = &output_view.device;
    args[3] = &count;
    args[4] = &row_width;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_embedding), count * row_width, args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}
