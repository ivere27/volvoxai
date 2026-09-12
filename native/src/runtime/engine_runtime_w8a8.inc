static int quantized_linear_f32(int node_index, Node* node, const T* input, const T* weight,
                                const T* bias, T* output, int row_start, int row_count,
                                int d_in, int d_out, int out_in, int include_bias) {
    if (!node || !input || !weight || !output ||
        (weight->dtype != T_I8 && weight->dtype != T_U8) ||
        input->dtype != T_F32 || output->dtype != T_F32 || row_start < 0 ||
        row_count <= 0 || d_in <= 0 || d_out <= 0 ||
        ((long)row_start + row_count) * d_in > input->numel ||
        ((long)row_start + row_count) * d_out > output->numel) return -1;
    T* scale = nin(node, VX_PORT_WEIGHT_SCALE);
    if (!scale) scale = nin(node, VX_PORT_SCALE);
    T* zero_point = nin(node, VX_PORT_WEIGHT_ZERO_POINT);
    if (!zero_point) zero_point = nin(node, VX_PORT_ZERO_POINT);
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
        if (node_index >= 0 && (size_t)node_index < g_node_capacity &&
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

/*
 * One QLinear over a declared batch's rows.
 *
 * The lanes are gathered into `lanes` adjacent rows, the ordinary kernel runs
 * over them as if they had always been adjacent, and the results are scattered
 * back to the rows each lane owns. A batched row step has to be bit-identical
 * to the same lanes decoded one at a time, and the surest way to get that is
 * for the arithmetic to be the same arithmetic on the same bytes in the same
 * order.
 *
 * The kernel preference is the scalar path's, repeated rather than avoided.
 * That preference is a function of the row count -- `M=1 retains the raw GEMV
 * policy` -- so a one-lane decode and a four-lane decode do go through
 * different kernels, and the first version of this function called the raw
 * kernel unconditionally to keep bit-identity resting on there being one
 * kernel rather than on two agreeing. That was over-cautious: the two agreeing
 * is an invariant across supported shapes, dtypes and zero points. Declining
 * to use it cost
 * a measured 2.4x at two lanes and 1024 wide -- 0.205 against 0.085 ms per
 * lane-step -- which is most of what batching was supposed to buy.
 *
 * A parked lane still occupies a staged row, because the operand's shape says
 * `lanes` rows exist and a short batch is a different call. Its result is
 * dropped by the scatter.
 */
static int qlinear_physical_forward_batch(QLinearMetadata* meta, T* input,
                                          T* weight, T* output, long rows) {
    int* input_rows = g_decode_rows.scratch_indices[0];
    int* output_rows = g_decode_rows.scratch_indices[1];
    const int lanes = g_decode_lanes;
    size_t input_row_bytes;
    size_t output_row_bytes;
    size_t input_total_bytes;
    size_t output_total_bytes;
    size_t input_bytes;
    size_t output_bytes;
    unsigned char* stage;
    unsigned char* staged_input;
    unsigned char* staged_output;
    uint32_t input_dtype = physical_byte_dtype(input->dtype);
    uint32_t weight_dtype = physical_byte_dtype(weight->dtype);
    uint32_t output_dtype = physical_byte_dtype(output->dtype);
    if (!input_dtype || !weight_dtype || !output_dtype) return -1;
    /* Row extents come from the dtype rather than from `elem_size`, so a
     * sub-byte activation is addressed correctly or refused, never rounded. */
    if (!tensor_span_bytes(input, meta->d_in, &input_row_bytes) ||
        !tensor_span_bytes(output, meta->d_out, &output_row_bytes) ||
        !tensor_total_bytes(input, &input_total_bytes) ||
        !tensor_total_bytes(output, &output_total_bytes) ||
        (size_t)lanes > SIZE_MAX / (input_row_bytes ? input_row_bytes : 1) ||
        (size_t)lanes > SIZE_MAX / (output_row_bytes ? output_row_bytes : 1)) return -1;
    input_bytes = (size_t)lanes * input_row_bytes;
    output_bytes = (size_t)lanes * output_row_bytes;
    if (decode_lane_rows(input, meta->d_in, input_rows) != lanes) return -1;
    /*
     * A paged output is a K/V pool, and the row a lane writes there is the slot
     * its page table names -- not `lane * S + position`, which is where the row
     * would be if the tensor carried one slice per lane. The pool is shared, so
     * that address belongs to whichever lane happens to own it.
     *
     * The mapping comes from `vx_decode_row_set_write_rows` rather than from
     * page arithmetic written out here, so the native path and its TypeScript
     * and WebGPU twins keep sharing one implementation and one corpus. A lane
     * stride of zero is deliberate: the paged answer does not use one, and
     * passing the sequence length would suggest it did.
     */
    if (vx_paged_tensor_locked(output)) {
        if (!vx_paged_bound_locked() || vx_paged_lanes_locked() < lanes ||
            g_decode_rows.unpaged) return -1;
        if (vx_decode_row_set_write_rows(&g_decode_rows, 0, 1, output_rows) !=
            VX_DECODE_ROW_SET_OK) return -1;
        for (int lane = 0; lane < lanes; lane++) {
            if (input_rows[lane] >= rows || output_rows[lane] < 0 ||
                (long)output_rows[lane] >= output->numel / meta->d_out) return -1;
        }
    } else {
        if (decode_lane_rows(output, meta->d_out, output_rows) != lanes) return -1;
        /* Both operands must name the same lanes of the same batch; the widths
         * differ but the row each lane owns does not. */
        for (int lane = 0; lane < lanes; lane++) {
            if (input_rows[lane] >= rows || output_rows[lane] >= rows) return -1;
        }
    }
    stage = (unsigned char*)decode_stage_bytes(input_bytes + output_bytes);
    if (!stage) return -1;
    staged_input = stage;
    staged_output = stage + input_bytes;
    if (vx_decode_row_gather(staged_input, input->data, input_total_bytes,
                             input_row_bytes,
                             input_rows, lanes) != VX_DECODE_ROW_SET_OK) return -1;
    {
        const VxW8A8QLinearCall raw_call = {
            .input = staged_input,
            .weight = weight->data,
            .bias = meta->bias,
            .weight_scales = meta->weight_scales,
            .weight_zero_points = meta->weight_zero_points,
            .output = staged_output,
            .rows = (uint32_t)lanes,
            .d_in = (uint32_t)meta->d_in,
            .d_out = (uint32_t)meta->d_out,
            .input_scale = input->quantization.scale,
            .input_zero_point = input->quantization.zero_point,
            .output_scale = output->quantization.scale,
            .output_zero_point = output->quantization.zero_point,
            .input_dtype = input_dtype,
            .weight_dtype = weight_dtype,
            .output_dtype = output_dtype,
        };
        if (!(vx_packed_q8_preferred_for_native_w8a8(
                  (uint32_t)lanes, (uint32_t)meta->d_in,
                  (uint32_t)meta->d_out, weight_dtype,
                  meta->weight_zero_all_zero) &&
              meta->packed_weight &&
              vx_qlinear_i8u8_packed(
                  staged_input, meta->packed_weight, meta->bias,
                  meta->weight_scales, meta->weight_zero_points,
                  staged_output, (uint32_t)lanes, (uint32_t)meta->d_in,
                  (uint32_t)meta->d_out, input->quantization.scale,
                  input->quantization.zero_point,
                  output->quantization.scale,
                  output->quantization.zero_point,
                  input_dtype, weight_dtype, output_dtype) == 1) &&
            vx_w8a8_qlinear_native_call(&raw_call) != 1) return -1;
    }
    return vx_decode_row_scatter(output->data, output_total_bytes,
                                 staged_output, output_row_bytes,
                                 output_rows, lanes,
                                 g_decode_rows.parked) == VX_DECODE_ROW_SET_OK
        ? 0 : -1;
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
    size_t input_offset;
    size_t output_row_bytes;
    const unsigned char* input_data;
    unsigned char* output_data;
    if (node_index < 0 || (size_t)node_index >= g_node_capacity ||
        !node || !output) return -1;
    meta = &g_qlinear_meta[node_index];
    input = qlinear_node_input(node);
    weight = nin(node, VX_PORT_WEIGHT);
    bias = nin(node, VX_PORT_BIAS);
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
    /* A declared batch's rows are not one span, so they are staged rather than
     * pointed at. The kernel below is untouched: it still receives `row_count`
     * adjacent rows, which is exactly what staging makes true. */
    if (g_decode_lanes > 1) {
        return qlinear_physical_forward_batch(meta, input, weight, output, rows);
    }
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
    if (!tensor_span_bytes(input, (long)row_start * meta->d_in, &input_offset) ||
        !tensor_span_bytes(output, meta->d_out, &output_row_bytes)) return -1;
    input_data = (const unsigned char*)input->data + input_offset;
    /*
     * The output row is resolved per tensor, the input row is not: a K/V
     * projection reads its logical token row and writes the physical slot the
     * page table maps it to. Identity for every tensor outside the page pool,
     * so a context that does not page runs the arithmetic it always ran.
     *
     * More than one row at a time is refused rather than mapped: consecutive
     * logical rows of a paged tensor need not be consecutive physical rows, and
     * the kernel below writes `row_count` adjacent ones. A decode row step asks
     * for exactly one, which is the case this exists for.
     */
    {
        const int out_row = vx_paged_row_locked(output, row_start);
        /* The page budget and the tensor backing the pool are configured
         * separately, so a mapping that runs past the tensor is a caller error
         * this has to catch rather than write through. */
        if (out_row < 0 || (out_row != row_start && row_count != 1) ||
            (long)out_row + row_count > output->numel / meta->d_out) return -1;
        output_data = (unsigned char*)output->data +
            (size_t)out_row * output_row_bytes;
    }
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
    {
        const VxW8A8QLinearCall raw_call = {
            .input = input_data,
            .weight = weight->data,
            .bias = meta->bias,
            .weight_scales = meta->weight_scales,
            .weight_zero_points = meta->weight_zero_points,
            .output = output_data,
            .rows = (uint32_t)row_count,
            .d_in = (uint32_t)meta->d_in,
            .d_out = (uint32_t)meta->d_out,
            .input_scale = input->quantization.scale,
            .input_zero_point = input->quantization.zero_point,
            .output_scale = output->quantization.scale,
            .output_zero_point = output->quantization.zero_point,
            .input_dtype = input_dtype,
            .weight_dtype = weight_dtype,
            .output_dtype = output_dtype,
        };
        return vx_w8a8_qlinear_native_call(&raw_call) == 1 ? 0 : -1;
    }
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
/* A row window is executable wherever an interior pointer has a device
 * address. This used to name CUDA directly, which read as "only CUDA can do
 * this" when what it meant was "only CUDA resolves a subspan". */
static int physical_gpu_partial_execution_declined(void) {
    return (g_active_row >= 0 || g_prefix_rows > 0) &&
        !vx_runtime_backend_has_device_rows();
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
    size_t input_row_bytes;
    size_t output_row_bytes;
    size_t input_offset;
    const void* input_data;
    void* output_data;
    uint32_t input_dtype, weight_dtype, output_dtype;
    if (node_index < 0 || (size_t)node_index >= g_node_capacity ||
        !node || !output || !node_backend ||
        physical_gpu_partial_execution_declined()) return 0;
    meta = &g_qlinear_meta[node_index];
    input = qlinear_node_input(node);
    weight = nin(node, VX_PORT_WEIGHT);
    bias = nin(node, VX_PORT_BIAS);
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
    input_dtype = physical_byte_dtype(input->dtype);
    weight_dtype = physical_byte_dtype(weight->dtype);
    output_dtype = physical_byte_dtype(output->dtype);
    if (!input_dtype || !weight_dtype || !output_dtype) return 0;
    /*
     * A declared batch dispatches once per contiguous run of lane rows.
     *
     * Unlike the host path this stages nothing: every run is already a
     * contiguous span of the resident activation, which is precisely what a
     * device window can bind. Two adjacent lanes therefore cost one dispatch
     * and two scattered ones cost two -- `vx_decode_row_copy_runs` decides
     * which, per step, from the positions the scheduler produced.
     *
     * Either every run binds or none does: a row's window offset is
     * `row * width * element size`, so when the width clears the device's
     * binding alignment every row does, and when it does not, no row does.
     * That is what makes a partial dispatch not a case to handle here.
     */
    if (g_decode_lanes > 1) {
        VxDecodeRowRun* runs = g_decode_rows.scratch_runs;
        int count;
        if (!ops || !ops->qlinear_i8u8) return 0;
        if (!tensor_span_bytes(input, meta->d_in, &input_row_bytes) ||
            !tensor_span_bytes(output, meta->d_out, &output_row_bytes)) return 0;
        /*
         * A paged output has no runs to find. Its rows are the slots the lanes'
         * page tables name, which are not the input's rows and not contiguous
         * with each other, so each lane is its own dispatch: read the lane's
         * logical row, write the physical one. Coalescing is what runs buy, and
         * a page table is precisely the thing that stops them coalescing.
         */
        if (vx_paged_tensor_locked(output)) {
            int* in_rows = g_decode_rows.scratch_indices[2];
            int* out_rows = g_decode_rows.scratch_indices[3];
            const long output_rows = output->numel / meta->d_out;
            if (!vx_paged_bound_locked() ||
                vx_paged_lanes_locked() < g_decode_lanes ||
                g_decode_rows.unpaged ||
                decode_lane_rows(input, meta->d_in, in_rows) != g_decode_lanes ||
                vx_decode_row_set_write_rows(&g_decode_rows, 0, 1, out_rows) !=
                    VX_DECODE_ROW_SET_OK) return 0;
            for (int lane = 0; lane < g_decode_lanes; lane++) {
                if (g_decode_rows.parked[lane]) continue;
                if (in_rows[lane] < 0 || (long)in_rows[lane] >= rows ||
                    out_rows[lane] < 0 || (long)out_rows[lane] >= output_rows)
                    return 0;
                if (!ops->qlinear_i8u8(
                        (const unsigned char*)input->data +
                            (size_t)in_rows[lane] * input_row_bytes,
                        weight->data, meta->weight_scales,
                        meta->weight_zero_points, meta->bias,
                        (unsigned char*)output->data +
                            (size_t)out_rows[lane] * output_row_bytes,
                        1u, (uint32_t)meta->d_in, (uint32_t)meta->d_out,
                        input->quantization.scale, input->quantization.zero_point,
                        output->quantization.scale, output->quantization.zero_point,
                        input_dtype, weight_dtype, output_dtype)) return 0;
            }
            *node_backend = backend_label;
            return 1;
        }
        count = decode_lane_runs(input, meta->d_in, runs);
        if (count <= 0) return 0;
        for (int index = 0; index < count; index++) {
            const size_t row = (size_t)runs[index].source;
            if (!ops->qlinear_i8u8(
                    (const unsigned char*)input->data + row * input_row_bytes,
                    weight->data, meta->weight_scales, meta->weight_zero_points,
                    meta->bias,
                    (unsigned char*)output->data + row * output_row_bytes,
                    (uint32_t)runs[index].rows, (uint32_t)meta->d_in,
                    (uint32_t)meta->d_out, input->quantization.scale,
                    input->quantization.zero_point, output->quantization.scale,
                    output->quantization.zero_point,
                    input_dtype, weight_dtype, output_dtype)) return 0;
        }
        *node_backend = backend_label;
        return 1;
    }
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
    if (!tensor_span_bytes(input, (long)row_start * meta->d_in, &input_offset) ||
        !tensor_span_bytes(output, meta->d_out, &output_row_bytes)) return 0;
    input_data = (const unsigned char*)input->data + input_offset;
    /* The paged substitution the CPU path above explains. The interior pointer
     * this produces resolves to the pool's device slot at that slot's offset,
     * which is the same window every other activation row binds through. */
    {
        const int out_row = vx_paged_row_locked(output, row_start);
        if (out_row < 0 || (out_row != row_start && row_count != 1) ||
            (long)out_row + row_count > output->numel / meta->d_out) return 0;
        output_data = (unsigned char*)output->data +
            (size_t)out_row * output_row_bytes;
    }
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
    if (node_index < 0 || (size_t)node_index >= g_node_capacity ||
        !node || !output || !descriptor) return -1;
    meta = &g_qembedding_meta[node_index];
    tokens = qembedding_node_input(node);
    weight = nin(node, VX_PORT_WEIGHT);
    if (!meta->valid || !tokens || !weight || output != t_find(node->out) ||
        !tokens->data || !weight->data || !output->data ||
        tokens->dtype != T_I32 || tokens->elem_size != sizeof(int32_t) ||
        tokens->ndim <= 0 || tokens->numel <= 0 || (uint64_t)tokens->numel > UINT32_MAX ||
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
/*
 * One QEmbedding over a declared batch's rows.
 *
 * The token operand is `[lanes,S]` and the output `[lanes,S,hidden]`, so both
 * name the same rows -- `lane * S + position[lane]` -- and differ only in how
 * wide a row is. Resolving the indices once and applying them at two widths is
 * what keeps the token a lane reads and the row that lane writes from drifting
 * apart, which is the failure this would have if each operand did its own
 * arithmetic.
 */
static int qembedding_physical_forward_batch(
        const QEmbeddingPhysicalDescriptor* descriptor, T* output) {
    int* rows = g_decode_rows.scratch_indices[0];
    const int lanes = g_decode_lanes;
    const T* tokens = descriptor->tokens;
    size_t hidden_bytes;
    size_t token_total_bytes;
    size_t output_total_bytes;
    size_t token_bytes = (size_t)lanes * sizeof(int32_t);
    size_t output_bytes;
    unsigned char* stage;
    if (!tensor_span_bytes(output, descriptor->meta->hidden, &hidden_bytes) ||
        !tensor_total_bytes(tokens, &token_total_bytes) ||
        !tensor_total_bytes(output, &output_total_bytes) ||
        (size_t)lanes > SIZE_MAX / (hidden_bytes ? hidden_bytes : 1)) return -1;
    output_bytes = (size_t)lanes * hidden_bytes;
    if (decode_lane_rows(output, descriptor->meta->hidden, rows) != lanes)
        return -1;
    /* The token operand carries the same lanes with the same stride; saying so
     * here rather than deriving a second set of indices from its own shape. */
    if (tokens->ndim != 2 || tokens->shape[0] != lanes ||
        tokens->shape[1] != output->shape[1]) return -1;
    stage = (unsigned char*)decode_stage_bytes(token_bytes + output_bytes);
    if (!stage) return -1;
    if (vx_decode_row_gather(stage, tokens->data, token_total_bytes,
                             sizeof(int32_t), rows, lanes) != VX_DECODE_ROW_SET_OK)
        return -1;
    if (qembedding_i8u8((const int32_t*)stage, descriptor->weight->data,
                        descriptor->meta->weight_scales,
                        descriptor->meta->weight_zero_points,
                        stage + token_bytes, (uint32_t)lanes,
                        descriptor->meta->vocab, descriptor->meta->hidden,
                        output->quantization.scale, output->quantization.zero_point,
                        descriptor->weight_dtype, descriptor->output_dtype) != 1)
        return -1;
    return vx_decode_row_scatter(output->data, output_total_bytes,
                                 stage + token_bytes, hidden_bytes,
                                 rows, lanes, g_decode_rows.parked) ==
        VX_DECODE_ROW_SET_OK ? 0 : -1;
}

static int qembedding_physical_forward(int node_index, Node* node, T* output) {
    QEmbeddingPhysicalDescriptor descriptor;
    const int32_t* tokens;
    void* output_data;
    uint32_t token_count;
    size_t row_offset;
    if (qembedding_physical_validate(
            node_index, node, output, 1, &descriptor) != 0) return -1;
    if (g_decode_lanes > 1)
        return qembedding_physical_forward_batch(&descriptor, output);
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
        if (!tensor_span_bytes(output, (long)row_start * descriptor.meta->hidden,
                               &row_offset)) return -1;
        output_data = (unsigned char*)output_data + row_offset;
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
    size_t hidden_bytes;
    if (!node_backend || physical_gpu_partial_execution_declined() ||
        qembedding_physical_validate(
            node_index, node, output,
            !g_bounded_gpu_value_domain_proven, &descriptor) != 0) return 0;
    tokens = (const int32_t*)descriptor.tokens->data;
    output_data = output->data;
    token_count = descriptor.tokens_count;
    /*
     * A declared batch dispatches once per contiguous run of lane rows, and
     * stages nothing -- the same shape QLinear takes above, for the same
     * reason: a run is already a contiguous span of both operands.
     *
     * The two operands have different widths and the same rows. Resolving the
     * runs once from the output and applying `runs[i].source` to the tokens at
     * one element per row is what keeps the token a lane reads and the row it
     * writes from drifting apart; deriving a second set of indices from the
     * token shape is the way that goes wrong.
     *
     * Note this is where the token binding earns its keep. A run starts at row
     * `source`, so the ids pointer handed down is interior whenever that is not
     * zero -- which no alignment rule would let a descriptor name, and which
     * `graph_token_window` now expresses as a shader-side token offset instead.
     */
    if (g_decode_lanes > 1) {
        VxDecodeRowRun* runs = g_decode_rows.scratch_runs;
        const T* token_tensor = descriptor.tokens;
        const int count = decode_lane_runs(output, descriptor.meta->hidden, runs);
        if (count <= 0 || !ops || !ops->qembedding_i8u8) return 0;
        /* The token operand carries the same lanes over the same sequence;
         * said here rather than inferred from its own leading extent. */
        if (token_tensor->ndim != 2 || token_tensor->shape[0] != g_decode_lanes ||
            token_tensor->shape[1] != output->shape[1] ||
            token_tensor->elem_size != sizeof(int32_t)) return 0;
        if (!tensor_span_bytes(output, descriptor.meta->hidden, &hidden_bytes))
            return 0;
        for (int index = 0; index < count; index++) {
            const size_t row = (size_t)runs[index].source;
            if (!ops->qembedding_i8u8(
                    (const int32_t*)token_tensor->data + row,
                    descriptor.weight->data, descriptor.meta->weight_scales,
                    descriptor.meta->weight_zero_points,
                    (unsigned char*)output->data + row * hidden_bytes,
                    (uint32_t)runs[index].rows, descriptor.meta->vocab,
                    descriptor.meta->hidden, output->quantization.scale,
                    output->quantization.zero_point, descriptor.weight_dtype,
                    descriptor.output_dtype)) return 0;
        }
        *node_backend = backend_label;
        return 1;
    }
    if (g_active_row >= 0 ||
        (g_prefix_rows > 0 &&
         descriptor.tokens->numel == g_prefix_row_capacity)) {
        if (descriptor.tokens->ndim != 2 || descriptor.tokens->shape[0] != 1 ||
            output->ndim != 3 || output->shape[0] != 1) return 0;
        seq_range(descriptor.tokens->numel, &row_start, &row_count);
        if (row_start < 0 || row_count <= 0 ||
            (long)row_start + row_count > descriptor.tokens->numel) return 0;
        tokens += row_start;
        if (!tensor_span_bytes(output, (long)row_start * descriptor.meta->hidden,
                               &hidden_bytes)) return 0;
        output_data = (unsigned char*)output_data + hidden_bytes;
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
    if (node_index < 0 || (size_t)node_index >= g_node_capacity ||
        !node || !output ||
        !descriptor) return -1;
    meta = &g_qconv_meta[node_index];
    input = qconv2d_node_input(node);
    weight = nin(node, VX_PORT_WEIGHT);
    bias = nin(node, VX_PORT_BIAS);
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
    T* input = nin(node, VX_PORT_INPUT);
    if (!input) input = nin(node, VX_PORT_X);
    if (!input) input = nin(node, VX_PORT_DATA);
    return input;
}

/*
 * The gate every W8A8 operand passes, and the one place a narrower dtype has to
 * be let in.
 *
 * `elem_size != 1` is what keeps a sub-byte activation out. That is deliberate
 * and not an oversight: these kernels index one byte per element, so a 4-bit
 * tensor reaching them would read two values as one.
 *
 * What an int4 kernel would have to change, stated so it does not have to be
 * rediscovered:
 *
 *   - this predicate and `physical_byte_dtype` below it;
 *   - the elementwise operators (QAdd, QSiLU, QGELU, the norms), which turn an
 *     element offset into a byte offset by multiplying by `elem_size`. Correct
 *     only while that is one, which this gate guarantees today;
 *   - nothing in the row, page, staging or gather addressing. Those ask
 *     `tensor_span_bytes`, which answers from the dtype's bit width and refuses
 *     a row that does not end on a byte boundary rather than rounding it into
 *     its neighbour.
 */
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

static int physical_parameters_exact(
        const Node* node, const VxNodeParamKey* allowed,
        size_t allowed_count) {
    unsigned present = 0;
    if (!node || node->parsed_params.parsed != 1 ||
        (!allowed && allowed_count)) return -1;
    for (size_t index = 0; index < allowed_count; index++) {
        const VxCachedNodeParam* value = vx_node_param(node, allowed[index]);
        if (value && value->kind != VX_NODE_PARAM_ABSENT) present++;
    }
    return node->parsed_params.parameter_count == present &&
        node->parsed_params.cached_parameter_count == present ? 0 : -1;
}

static int physical_relu_parameter(const Node* node, uint32_t* relu) {
    static const VxNodeParamKey allowed[] = {VX_NODE_PARAM_RELU};
    const VxCachedNodeParam* value = vx_node_param(node, VX_NODE_PARAM_RELU);
    if (!relu || physical_parameters_exact(
            node, allowed, sizeof(allowed) / sizeof(allowed[0])) != 0)
        return -1;
    if (!value || value->kind == VX_NODE_PARAM_ABSENT ||
        value->kind == VX_NODE_PARAM_NULL) {
        *relu = 0;
        return 0;
    }
    if (value->kind != VX_NODE_PARAM_I32 || value->value.i32 < 0 ||
        value->value.i32 > 2) return -1;
    *relu = (uint32_t)value->value.i32;
    return 0;
}

/* QSiLU has no tunable attributes.  Keep its canonical byte route closed to
 * unknown parameters rather than silently accepting a configuration whose
 * requested activation semantics differ from SiLU. */
static int physical_no_parameters(const Node* node) {
    if (!node || node->parsed_params.parsed != 1) return -1;
    return node->parsed_params.parameter_count == 0 ? 0 : -1;
}

typedef struct {
    uint32_t matrix_count;
    size_t a_matrix_elements;
    size_t b_matrix_elements;
    size_t output_matrix_elements;
} QBatchMatMulPhysicalRowDescriptor;

/* Incremental decoder execution computes one sequence row while retaining
 * every other row from the prefill pass. Rank-3 accepts only a cached right
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
static int physical_qgelu_parameters(const Node* node) {
    const VxCachedNodeParam* approximate;
    if (!node || node->parsed_params.parsed != 1) return -1;
    if (node->parsed_params.parameter_count == 0) return 0;
    approximate = vx_node_param(node, VX_NODE_PARAM_APPROXIMATE);
    if (node->parsed_params.parameter_count != 1 ||
        node->parsed_params.cached_parameter_count != 1 || !approximate ||
        approximate->kind != VX_NODE_PARAM_SYMBOL ||
        approximate->symbol != VX_NODE_SYMBOL_NONE) return -1;
    return 0;
}

/* QGroupNorm owns a deliberately narrow, typed contract.  Iterate children
 * rather than using cJSON_GetObjectItem() so duplicate keys cannot silently
 * choose one meaning.  Null eps/layout follow the JS `??` spelling; the
 * required num_groups never has a default. */
static int physical_qgroupnorm_parameters(const Node* node, uint32_t* groups_out,
                                          float* epsilon_out) {
    if (!node || !groups_out || !epsilon_out ||
        node->parsed_params.parsed != 1 ||
        !node->parsed_params.qgroupnorm_params_valid) return -1;
    *groups_out = node->parsed_params.strict_num_groups;
    *epsilon_out = node->parsed_params.eps;
    return 0;
}

/* QLayerNorm is last-axis only.  It has no layout or group attributes: eps
 * defaults exactly as the portable/browser contract does, and an optional
 * d_model is merely a strict consistency assertion for the final dimension. */
static int physical_qlayernorm_parameters(const Node* node, uint32_t d_model,
                                          float* epsilon_out) {
    const VxCachedNodeParam* eps;
    const VxCachedNodeParam* declared_d_model;
    unsigned present_count = 0;
    float epsilon = 1.0e-5f;
    if (!node || node->parsed_params.parsed != 1 || !d_model || !epsilon_out)
        return -1;
    eps = vx_node_param(node, VX_NODE_PARAM_EPS);
    declared_d_model = vx_node_param(node, VX_NODE_PARAM_D_MODEL);
    if (eps && eps->kind != VX_NODE_PARAM_ABSENT) present_count++;
    if (declared_d_model &&
        declared_d_model->kind != VX_NODE_PARAM_ABSENT) present_count++;
    if (node->parsed_params.parameter_count != present_count ||
        node->parsed_params.cached_parameter_count != present_count) return -1;
    if (eps && eps->kind != VX_NODE_PARAM_ABSENT &&
        eps->kind != VX_NODE_PARAM_NULL) {
        if (eps->kind != VX_NODE_PARAM_F32 || !(eps->value.f32 > 0.0f) ||
            !isfinite(eps->value.f32)) return -1;
        epsilon = eps->value.f32;
    }
    if (declared_d_model && declared_d_model->kind != VX_NODE_PARAM_ABSENT &&
        declared_d_model->kind != VX_NODE_PARAM_NULL &&
        (declared_d_model->kind != VX_NODE_PARAM_I32 ||
         declared_d_model->value.i32 <= 0 ||
         (uint32_t)declared_d_model->value.i32 != d_model)) return -1;
    *epsilon_out = epsilon;
    return 0;
}

/* QSDPA accepts only its three attention attributes. Heads and causal are
 * explicit because quantized attention must not inherit the F32 SDPA
 * defaults; scale alone follows the optional `1/sqrt(head_dim)` contract. */
static int physical_qsdpa_parameters(const Node* node, uint32_t d_model,
                                     uint32_t* heads_out, uint32_t* causal_out,
                                     float* scale_out) {
    uint32_t heads;
    uint32_t head_dim;
    if (!node || !d_model || !heads_out || !causal_out || !scale_out ||
        node->parsed_params.parsed != 1 ||
        !node->parsed_params.qsdpa_params_valid) return -1;
    heads = node->parsed_params.strict_heads;
    if (!heads || d_model % heads || d_model % 4u) return -1;
    head_dim = d_model / heads;
    if (!head_dim || head_dim % 4u || head_dim > 64u) return -1;
    *heads_out = heads;
    *causal_out = node->parsed_params.causal ? 1u : 0u;
    *scale_out = node->parsed_params.has_scale
        ? node->parsed_params.scale : 1.0f / sqrtf((float)head_dim);
    if (!isfinite(*scale_out) || *scale_out <= 0.0f) return -1;
    return 0;
}

/* QArgMax intentionally has exactly one required attribute. Normalizing here
 * keeps negative-axis handling out of CPU/GPU kernels and gives every backend
 * the same [outer, axis, inner] descriptor. */
static int physical_qargmax_axis(const Node* node, int rank,
                                 uint32_t* axis_out) {
    int axis;
    if (!node || !axis_out || rank < 2 || rank > 8 ||
        node->parsed_params.parsed != 1 ||
        !node->parsed_params.qargmax_params_valid) return -1;
    axis = node->parsed_params.axis;
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
    T* a = nin(node, VX_PORT_A);
    T* b = nin(node, VX_PORT_B);
    if (!a) a = nin(node, VX_PORT_INPUT);
    if (!a) a = nin(node, VX_PORT_X);
    if (!b) b = nin(node, VX_PORT_Y);
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
    T* input = node ? nin(node, VX_PORT_INPUT) : NULL;
    return (input && ((input->dtype == T_I8 || input->dtype == T_U8) ||
                      input->quantization.valid)) ||
           (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                       output->quantization.valid));
}

static int qlayernorm_physical_candidate(Node* node, T* output) {
    T* input = node ? nin(node, VX_PORT_INPUT) : NULL;
    return (input && ((input->dtype == T_I8 || input->dtype == T_U8) ||
                      input->quantization.valid)) ||
           (output && ((output->dtype == T_I8 || output->dtype == T_U8) ||
                       output->quantization.valid));
}

static int qsdpa_physical_candidate(Node* node, T* output) {
    T* q = node ? nin(node, VX_PORT_Q) : NULL;
    T* k = node ? nin(node, VX_PORT_K) : NULL;
    T* v = node ? nin(node, VX_PORT_V) : NULL;
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
    T* scale = nin(node, VX_PORT_SCALE);
    T* zero_point = nin(node, VX_PORT_ZERO_POINT);
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
    T* scale = nin(node, VX_PORT_SCALE);
    T* zero_point = nin(node, VX_PORT_ZERO_POINT);
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
    T* a = nin(node, VX_PORT_A);
    T* b = nin(node, VX_PORT_B);
    const void* a_data;
    const void* b_data;
    void* output_data;
    uint32_t elements;
    uint32_t a_dtype, b_dtype, output_dtype, relu;
    if (!a) a = nin(node, VX_PORT_INPUT);
    if (!a) a = nin(node, VX_PORT_X);
    if (!b) b = nin(node, VX_PORT_Y);
    if (!node || !output || physical_byte_tensor(a, &a_dtype) != 0 ||
        physical_byte_tensor(b, &b_dtype) != 0 || physical_byte_tensor(output, &output_dtype) != 0 ||
        !physical_same_shape(a, b) || !physical_same_shape(a, output) ||
        physical_relu_parameter(node, &relu) != 0) return -1;
    a_data = a->data;
    b_data = b->data;
    output_data = output->data;
    elements = (uint32_t)output->numel;
    if (g_decode_lanes > 1) {
        const long width = output->ndim > 0 ? output->shape[output->ndim - 1] : 0;
        VxDecodeRowRun* runs = g_decode_rows.scratch_runs;
        const int count = decode_lane_runs(output, width, runs);
        if (count < 0) return -1;
        for (int index = 0; index < count; index++) {
            const size_t offset = (size_t)runs[index].source * (size_t)width;
            const VxW8A8QAddCall call = {
                .left = (const unsigned char*)a_data + offset * a->elem_size,
                .right = (const unsigned char*)b_data + offset * b->elem_size,
                .output = (unsigned char*)output_data +
                    offset * output->elem_size,
                .elements = (uint32_t)((long)runs[index].rows * width),
                .left_scale = a->quantization.scale,
                .left_zero_point = a->quantization.zero_point,
                .right_scale = b->quantization.scale,
                .right_zero_point = b->quantization.zero_point,
                .output_scale = output->quantization.scale,
                .output_zero_point = output->quantization.zero_point,
                .left_dtype = a_dtype,
                .right_dtype = b_dtype,
                .output_dtype = output_dtype,
                .relu = relu,
            };
            if (vx_w8a8_qadd_portable_call(&call) != 1) return -1;
        }
        return 0;
    }
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
    {
        const VxW8A8QAddCall call = {
            .left = a_data,
            .right = b_data,
            .output = output_data,
            .elements = elements,
            .left_scale = a->quantization.scale,
            .left_zero_point = a->quantization.zero_point,
            .right_scale = b->quantization.scale,
            .right_zero_point = b->quantization.zero_point,
            .output_scale = output->quantization.scale,
            .output_zero_point = output->quantization.zero_point,
            .left_dtype = a_dtype,
            .right_dtype = b_dtype,
            .output_dtype = output_dtype,
            .relu = relu,
        };
        return vx_w8a8_qadd_portable_call(&call) == 1 ? 0 : -1;
    }
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
    a = nin(node, VX_PORT_A);
    b = nin(node, VX_PORT_B);
    if (!a) a = nin(node, VX_PORT_INPUT);
    if (!a) a = nin(node, VX_PORT_X);
    if (!b) b = nin(node, VX_PORT_Y);
    if (physical_byte_tensor(a, &a_dtype) != 0 || physical_byte_tensor(b, &b_dtype) != 0 ||
        physical_byte_tensor(output, &output_dtype) != 0 || !physical_same_shape(a, b) ||
        !physical_same_shape(a, output) ||
        physical_relu_parameter(node, &relu) != 0) return 0;
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
        node->nin != 1 || physical_no_parameters(node) != 0) return -1;
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
    if (g_decode_lanes > 1) {
        const long width = output->ndim > 0 ? output->shape[output->ndim - 1] : 0;
        VxDecodeRowRun* runs = g_decode_rows.scratch_runs;
        const int count = decode_lane_runs(output, width, runs);
        if (count < 0) return -1;
        for (int index = 0; index < count; index++) {
            const size_t offset = (size_t)runs[index].source * (size_t)width;
            const VxW8A8UnaryCall call = {
                .input = (const unsigned char*)input_data +
                    offset * input->elem_size,
                .output = (unsigned char*)output_data +
                    offset * output->elem_size,
                .elements = (uint32_t)((long)runs[index].rows * width),
                .input_scale = input->quantization.scale,
                .input_zero_point = input->quantization.zero_point,
                .output_scale = output->quantization.scale,
                .output_zero_point = output->quantization.zero_point,
                .input_dtype = input_dtype,
                .output_dtype = output_dtype,
            };
            if (vx_w8a8_qsilu_native_call(&call) != 1) return -1;
        }
        return 0;
    }
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
    {
        const VxW8A8UnaryCall call = {
            .input = input_data,
            .output = output_data,
            .elements = elements,
            .input_scale = input->quantization.scale,
            .input_zero_point = input->quantization.zero_point,
            .output_scale = output->quantization.scale,
            .output_zero_point = output->quantization.zero_point,
            .input_dtype = input_dtype,
            .output_dtype = output_dtype,
        };
        return vx_w8a8_qsilu_native_call(&call) == 1 ? 0 : -1;
    }
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
        node->nin != 1 || physical_qgelu_parameters(node) != 0) return -1;
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
    if (g_decode_lanes > 1) {
        const long width = output->ndim > 0 ? output->shape[output->ndim - 1] : 0;
        VxDecodeRowRun* runs = g_decode_rows.scratch_runs;
        const int count = decode_lane_runs(output, width, runs);
        if (count < 0) return -1;
        for (int index = 0; index < count; index++) {
            const size_t offset = (size_t)runs[index].source * (size_t)width;
            const VxW8A8UnaryCall call = {
                .input = (const unsigned char*)input_data +
                    offset * input->elem_size,
                .output = (unsigned char*)output_data +
                    offset * output->elem_size,
                .elements = (uint32_t)((long)runs[index].rows * width),
                .input_scale = input->quantization.scale,
                .input_zero_point = input->quantization.zero_point,
                .output_scale = output->quantization.scale,
                .output_zero_point = output->quantization.zero_point,
                .input_dtype = input_dtype,
                .output_dtype = output_dtype,
            };
            if (vx_w8a8_qgelu_portable_call(&call) != 1) return -1;
        }
        return 0;
    }
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
    {
        const VxW8A8UnaryCall call = {
            .input = input_data,
            .output = output_data,
            .elements = elements,
            .input_scale = input->quantization.scale,
            .input_zero_point = input->quantization.zero_point,
            .output_scale = output->quantization.scale,
            .output_zero_point = output->quantization.zero_point,
            .input_dtype = input_dtype,
            .output_dtype = output_dtype,
        };
        return vx_w8a8_qgelu_portable_call(&call) == 1 ? 0 : -1;
    }
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
        if (node->ins[index].port == VX_PORT_INPUT) {
            if (input) return -1;
            input = tensor;
        } else if (node->ins[index].port == VX_PORT_WEIGHT) {
            if (weight) return -1;
            weight = tensor;
        } else if (node->ins[index].port == VX_PORT_BIAS) {
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
        if (node->ins[index].port == VX_PORT_Q) {
            if (saw_q) return -1;
            saw_q = 1;
            q = t_find(node->ins[index].name);
        } else if (node->ins[index].port == VX_PORT_K) {
            if (saw_k) return -1;
            saw_k = 1;
            k = t_find(node->ins[index].name);
        } else if (node->ins[index].port == VX_PORT_V) {
            if (saw_v) return -1;
            saw_v = 1;
            v = t_find(node->ins[index].name);
        } else if (node->ins[index].port == VX_PORT_MASK) {
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
        (node->ins[0].port != VX_PORT_INPUT)) return -1;
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
        if (node->ins[index].port == VX_PORT_INPUT) {
            if (input) return -1;
            input = t_find(node->ins[index].name);
        } else if (node->ins[index].port == VX_PORT_MASK) {
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
        physical_qgroupnorm_parameters(node, &groups, &epsilon) != 0 ||
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
        physical_qlayernorm_parameters(node, d_model, &epsilon) != 0 ||
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
    if (g_decode_lanes > 1) {
        VxDecodeRowRun* runs = g_decode_rows.scratch_runs;
        const int count = decode_lane_runs(descriptor.input,
                                           (long)descriptor.d_model, runs);
        if (count < 0) return -1;
        for (int index = 0; index < count; index++) {
            const size_t offset = (size_t)runs[index].source * descriptor.d_model;
            if (vx_qlayernorm_i8u8_native_validated(
                    (const unsigned char*)input_data +
                        offset * descriptor.input->elem_size,
                    descriptor.weight->data, descriptor.bias->data,
                    (unsigned char*)output_data + offset * output->elem_size,
                    (uint32_t)runs[index].rows, descriptor.d_model,
                    descriptor.input->quantization.scale,
                    descriptor.input->quantization.zero_point,
                    output->quantization.scale, output->quantization.zero_point,
                    descriptor.epsilon, descriptor.input_dtype,
                    descriptor.output_dtype) != 1) return -1;
        }
        return 0;
    }
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
        physical_qsdpa_parameters(node, d_model, &heads, &causal,
                                  &attention_scale) != 0 ||
        !qsdpa_score_scale_valid(q, k, q_dtype, k_dtype, d_model / heads,
                                 attention_scale)) return -1;
    /*
     * One key axis, the operand's, whether or not the K/V are paged.
     *
     * Paging changes what a mask index *means* -- a lane-local logical token
     * rather than a row of this operand -- but not how wide the axis is. It
     * cannot: the graph declares the mask's shape once and the same graph runs
     * both with a cache bound and without (a prefill forward binds nothing), so an
     * extent only a bound cache could supply would make the unbound execution
     * of that graph refuse. How far into the axis a lane may read is the lane's
     * published length, which the paged branches bound below.
     */
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

/*
 * One QSDPA over a declared batch's lanes.
 *
 * The kernel already takes a batch axis and a query range, but the range is one
 * range for every lane, and lanes at different positions want different query
 * rows. So the queries are staged: each lane's row is gathered into a
 * `[lanes,1,D]` operand, and the kernel runs one query per lane.
 *
 * K and V are *not* staged -- unless they are paged. Lane b's keys ordinarily
 * occupy rows `b*S .. b*S+S-1` of a `[lanes,S,D]` tensor, which is exactly the
 * layout the kernel's `batch_index * seq_kv * d_model` addressing expects, and
 * staging them would copy the entire prefix of every lane on every token to
 * reproduce bytes that were already in the right place.
 *
 * A paged pool breaks that and nothing cheaper is available: lane b's prefix is
 * the physical slots its page table names, in an order shared with no other
 * lane, so there is no addressing the kernel could be given instead. The prefix
 * is gathered into `[lanes, key_capacity, D]` logical order --
 * `vx_decode_row_set_prefix_rows` says which source row each staged row takes,
 * and pads the tail of a lane shorter than the capacity -- and the keep mask
 * below already stops any lane reading into that padding.
 *
 * Causality moves into the keep mask, and this is the part that has to be
 * argued rather than assumed. The kernel implements causal by truncating the
 * key extent to `query + 1`; a staged query sits at index 0 and no longer knows
 * its absolute position, so truncation would admit key 0 alone. What makes the
 * substitution exact is that `vx_pf_qsdpa_key_allowed` returning zero makes the
 * loop `continue`: a masked key is not scored, does not enter the running
 * maximum, and does not enter the sum. A key excluded by the mask is therefore
 * indistinguishable from a key the extent never reached. `kv_lengths[lane]` is
 * `position + 1` -- the causal extent -- so a mask clamped to it admits keys
 * `0..position` in ascending order, which is the same set the one-lane decode
 * visits in the same order, through the same online-softmax recurrence.
 */
static int qsdpa_physical_forward_batch(const QSDPAPhysicalDescriptor* descriptor,
                                        T* output) {
    const int lanes = g_decode_lanes;
    T* q = descriptor->q;
    const uint32_t sequence = descriptor->seq_q;
    size_t row_bytes;
    const int paged = vx_paged_tensor_locked(descriptor->k);
    /* A paged pool has no lane axis, so its extent is its own row count and the
     * key axis the kernel sees is the staged capacity instead. */
    const uint32_t key_extent = paged
        ? (uint32_t)g_decode_rows.key_capacity : descriptor->seq_kv;
    size_t kv_row_bytes;
    size_t q_total_bytes;
    size_t output_total_bytes;
    int* rows = g_decode_rows.scratch_indices[0];
    VxDecodeKeepMask graph_mask = {0};
    size_t query_bytes;
    size_t mask_bytes = (size_t)lanes * key_extent * sizeof(int32_t);
    size_t kv_bytes;
    /* The staged prefix's source rows live in the same scratch as the rows
     * themselves. On the stack they would have had to be sized for the largest
     * context this could ever see, which is the sequence length -- and a fixed
     * bound there would refuse a long decode rather than a wrong one. */
    size_t prefix_row_bytes = paged
        ? (size_t)lanes * key_extent * sizeof(int) : 0;
    const void* keys;
    const void* values;
    unsigned char* stage;
    unsigned char* staged_q;
    unsigned char* staged_output;
    int32_t* staged_mask;
    /* Q and the output share the sequence axis; K and V carry the whole
     * prefix, so their extent is the tensor's and not the batch's. */
    if (q->ndim != 3 || (uint32_t)descriptor->batch != (uint32_t)lanes ||
        (!paged && descriptor->seq_kv != sequence)) return -1;
    /* Row extents come from the dtype, so a sub-byte activation is addressed
     * exactly or refused -- never rounded into its neighbour's row. */
    if (!tensor_span_bytes(q, descriptor->d_model, &row_bytes) ||
        !tensor_span_bytes(descriptor->k, descriptor->d_model, &kv_row_bytes) ||
        !tensor_total_bytes(q, &q_total_bytes) ||
        !tensor_total_bytes(output, &output_total_bytes) ||
        (size_t)lanes > SIZE_MAX / (row_bytes ? row_bytes : 1)) return -1;
    query_bytes = (size_t)lanes * row_bytes;
    kv_bytes = 0;
    if (paged) {
        if (key_extent == 0 || (size_t)lanes * key_extent >
                SIZE_MAX / (kv_row_bytes ? kv_row_bytes : 1)) return -1;
        kv_bytes = (size_t)lanes * key_extent * kv_row_bytes;
    }
    if (decode_lane_rows(q, descriptor->d_model, rows) != lanes ||
        decode_lane_rows(output, descriptor->d_model, rows) != lanes) return -1;
    if (paged) {
        if (!vx_paged_tensor_locked(descriptor->v) ||
            descriptor->v->elem_size != descriptor->k->elem_size ||
            descriptor->v->numel != descriptor->k->numel ||
            vx_paged_lanes_locked() < lanes || g_decode_rows.unpaged ||
            key_extent == 0) return -1;
    }
    if (descriptor->mask) {
        /*
         * `keep_mask_reads` indexes `(lane, key)` with `key` a lane-local
         * logical token, which is what a staged paged prefix already is:
         * staged row `lane * key_capacity + token` holds that lane's logical
         * token `token`. So the graph mask joins by naming its own key stride
         * and nothing else has to move.
         *
         * The stride is the operand's declared key axis, the one validate
         * classified the mask against. `key_extent` is what this step actually
         * reads -- `key_capacity`, the widest live lane -- and it has to lie
         * inside that axis, or a lane's row would be read past the mask
         * validate sized.
         */
        if (key_extent > descriptor->seq_kv) return -1;
        graph_mask.layout = (VxDecodeKeepMaskLayout)descriptor->mask_mode;
        graph_mask.values = (const int*)descriptor->mask->data;
        graph_mask.keys = (int)descriptor->seq_kv;
        graph_mask.queries = (int)sequence;
    }
    stage = (unsigned char*)decode_stage_bytes(query_bytes * 2 + mask_bytes +
                                               kv_bytes * 2 + prefix_row_bytes);
    if (!stage) return -1;
    staged_q = stage;
    staged_output = stage + query_bytes;
    staged_mask = (int32_t*)(stage + query_bytes * 2);
    keys = descriptor->k->data;
    values = descriptor->v->data;
    if (vx_decode_row_gather(staged_q, q->data, q_total_bytes, row_bytes,
                             rows, lanes) != VX_DECODE_ROW_SET_OK) return -1;
    if (paged) {
        unsigned char* staged_k = stage + query_bytes * 2 + mask_bytes;
        unsigned char* staged_v = staged_k + kv_bytes;
        int* prefix_rows = (int*)(staged_v + kv_bytes);
        size_t pool_bytes;
        if (!tensor_total_bytes(descriptor->k, &pool_bytes)) return -1;
        /* `lane_stride` is unused by the paged answer, and a K/V pool has no
         * lane stride to pass; the page table is the whole of the mapping. */
        if (vx_decode_row_set_prefix_rows(&g_decode_rows, 0, 1, prefix_rows) !=
            VX_DECODE_ROW_SET_OK) return -1;
        if (vx_decode_row_gather(staged_k, descriptor->k->data, pool_bytes,
                                 kv_row_bytes, prefix_rows,
                                 lanes * (int)key_extent) != VX_DECODE_ROW_SET_OK ||
            vx_decode_row_gather(staged_v, descriptor->v->data, pool_bytes,
                                 kv_row_bytes, prefix_rows,
                                 lanes * (int)key_extent) != VX_DECODE_ROW_SET_OK)
            return -1;
        keys = staged_k;
        values = staged_v;
    }
    /* Clamped only when the node is causal, or whenever the keys are staged --
     * a staged lane's rows past its own length are padding, and the clamp is
     * what keeps every lane out of them. A non-causal contiguous attention has
     * no per-lane visible length to publish: every lane reads the memory whole,
     * and clamping would silently shorten it. */
    if (vx_decode_row_set_keep_mask(&g_decode_rows, (int)key_extent,
                                    (descriptor->causal || paged) ? 1 : 0,
                                    descriptor->mask ? &graph_mask : NULL,
                                    (int*)staged_mask) != VX_DECODE_ROW_SET_OK)
        return -1;
    if (vx_qsdpa_i8u8_native_range_validated(
            staged_q, keys, values, staged_mask,
            staged_output, (uint32_t)lanes, 1u, key_extent,
            descriptor->d_model, descriptor->heads,
            q->quantization.scale, q->quantization.zero_point,
            descriptor->k->quantization.scale,
            descriptor->k->quantization.zero_point,
            descriptor->v->quantization.scale,
            descriptor->v->quantization.zero_point,
            output->quantization.scale, output->quantization.zero_point,
            descriptor->attention_scale, descriptor->q_dtype,
            descriptor->k_dtype, descriptor->v_dtype, descriptor->output_dtype,
            0u, ATTN_MASK_BATCH_KEY, 0u, 1u) != 1) return -1;
    return vx_decode_row_scatter(output->data, output_total_bytes,
                                 staged_output, row_bytes, rows, lanes,
                                 g_decode_rows.parked) == VX_DECODE_ROW_SET_OK
        ? 0 : -1;
}

/*
 * One lane's paged prefix, gathered into logical order.
 *
 * A gather rather than a page-table-aware kernel, for the reason the F32 path
 * gives: the kernel runs an online-softmax recurrence whose accumulation order
 * a rewrite would not reproduce, and "paged K/V is numerically identical to
 * contiguous K/V" would then hold only to a tolerance. Handing the *same*
 * kernel the same bytes in the same order makes it identical by construction.
 *
 * Both operands or neither, and the two gathers use different slots so the
 * second does not overwrite the first.
 */
static int qsdpa_paged_prefix(const QSDPAPhysicalDescriptor* descriptor, int lane,
                              const void** keys, const void** values,
                              uint32_t* kv_length) {
    size_t row_bytes;
    const int pool_rows = (int)(descriptor->k->numel / descriptor->d_model);
    const int length = vx_paged_kv_length_lane_locked(lane);
    if (descriptor->v->elem_size != descriptor->k->elem_size ||
        descriptor->v->numel != descriptor->k->numel ||
        !tensor_span_bytes(descriptor->k, descriptor->d_model, &row_bytes)) return -1;
    /* A published length longer than the tensor would gather past the pool:
     * the cache's page budget and the tensor backing it are configured
     * separately. */
    if (length <= 0 || length > pool_rows) return -1;
    *keys = vx_paged_gather_lane_bytes_locked(descriptor->k->data, pool_rows,
                                              row_bytes, length, 0, lane);
    *values = vx_paged_gather_lane_bytes_locked(descriptor->v->data, pool_rows,
                                                row_bytes, length, 1, lane);
    if (!*keys || !*values) return -1;
    *kv_length = (uint32_t)length;
    return 0;
}

static int qsdpa_physical_forward(Node* node, T* output) {
    QSDPAPhysicalDescriptor descriptor;
    int query_start = 0;
    int query_count = 0;
    if (qsdpa_physical_validate(node, output, &descriptor) != 0) return -1;
    if (g_decode_lanes > 1) return qsdpa_physical_forward_batch(&descriptor, output);
    if (g_active_row >= 0 || g_prefix_rows > 0) {
        const void* keys = descriptor.k->data;
        const void* values = descriptor.v->data;
        uint32_t seq_kv = descriptor.seq_kv;
        uint32_t causal = descriptor.causal;
        if (descriptor.batch != 1 ||
            (descriptor.q->ndim != 2 && descriptor.q->ndim != 3)) return -1;
        seq_range(descriptor.seq_q, &query_start, &query_count);
        if (query_start < 0 || query_count <= 0 ||
            (uint32_t)query_start + (uint32_t)query_count > descriptor.seq_q)
            return -1;
        const int32_t* mask_data = descriptor.mask
            ? (const int32_t*)descriptor.mask->data : NULL;
        uint32_t mask_mode = descriptor.mask_mode;
        if (vx_paged_tensor_locked(descriptor.k)) {
            /*
             * The gathered prefix is `kv_length` rows long and already ends at
             * this row's last visible key, so the extent *is* the causal bound
             * and the kernel must not truncate again: query `query_start` sits
             * at its absolute index but the keys no longer do, and `query + 1`
             * would admit a prefix of a prefix.
             */
            if (!vx_paged_tensor_locked(descriptor.v) || query_count != 1 ||
                qsdpa_paged_prefix(&descriptor, vx_engine_state_current()->paged.lane,
                                   &keys, &values, &seq_kv) != 0) return -1;
            causal = 0u;
            if (descriptor.mask) {
                /*
                 * The mask joins by being sliced, not staged.
                 *
                 * The gather replaced the key extent with this lane's length,
                 * and the kernel indexes the mask with whatever extent it is
                 * handed -- so the four layouts collapse to the one the kernel
                 * can read at that extent: the single key row belonging to this
                 * (batch, query) pair. `attention_mask_query_view` names it,
                 * and keys `0..kv_length-1` of that row are exactly this lane's
                 * logical tokens -- the gather put them in logical order, so
                 * the mask's own indexing needs no translation.
                 *
                 * The batch index is zero and not the bound cache lane. This
                 * path admits only `batch == 1`, so the mask has one row along
                 * its batch axis however many lanes the cache holds; indexing
                 * it by the cache lane would read another operand's bytes as
                 * soon as a scheduler bound anything but lane zero.
                 */
                int local_mode = 0;
                /* The gathered length is how many entries of the resolved row
                 * the kernel reads; it has to fit inside the declared axis. */
                if (seq_kv > descriptor.seq_kv) return -1;
                mask_data = attention_mask_query_view(
                    descriptor.mask, (int)descriptor.mask_mode, 0, query_start,
                    (int)descriptor.seq_q, (int)descriptor.seq_kv,
                    &local_mode);
                if (!mask_data || local_mode != VX_ATTN_MASK_LOCAL_KEY) return -1;
                mask_mode = ATTN_MASK_KEY;
            }
        }
        return vx_qsdpa_i8u8_native_range_validated(
                    descriptor.q->data, keys, values,
                    mask_data,
                    output->data, descriptor.batch, descriptor.seq_q,
                    seq_kv, descriptor.d_model, descriptor.heads,
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
                    descriptor.output_dtype, causal,
                    mask_mode, (uint32_t)query_start,
                    (uint32_t)query_count) == 1 ? 0 : -1;
    }
    /* The whole-sequence path reads K/V linearly, which is what a page table
     * exists to prevent: it would attend over whatever physical rows sit at
     * `[0, seq_kv)`, some of them another lane's. There is no correct linear
     * reading of a scattered mapping to fall back to. */
    if (vx_paged_tensor_locked(descriptor.k) ||
        vx_paged_tensor_locked(descriptor.v)) return -1;
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

/*
 * One QSDPA over a declared batch, on the device.
 *
 * The same staging the host path does, done where the bytes already are. That
 * is the whole reason this exists rather than deferring to the host one: on a
 * device-row backend the activations are resident, so a host gather would read
 * a mirror the device has moved past, and a host scatter would write one the
 * device is about to overwrite.
 *
 * Four dispatches in the contiguous case and six when the K/V are paged:
 * gather the queries, gather each lane's prefix if it is scattered, attend,
 * scatter the answers back. The keep mask is small and built on the host --
 * it is `lanes * key_extent` int32 and depends on the scheduler's positions,
 * not on any resident tensor -- so it is uploaded rather than computed here,
 * and marked host-dirty every step because its slot would otherwise be clean
 * from the first upload onward and serve a previous step's lengths.
 *
 * Everything is measured in words, so nothing here knows what a row contains.
 * A row whose bytes are not a whole number of words is refused to the host
 * path instead: the transfer kernel owns whole words to keep two lanes' rows
 * from sharing one, and a partial word would need read-modify-write.
 */
static int qsdpa_physical_gpu_forward_batch(const QSDPAPhysicalDescriptor* descriptor,
                                            T* output, const VxW8A8Ops* ops) {
    const int lanes = g_decode_lanes;
    T* q = descriptor->q;
    const uint32_t sequence = descriptor->seq_q;
    const int paged = vx_paged_tensor_locked(descriptor->k);
    const uint32_t key_extent = paged
        ? (uint32_t)g_decode_rows.key_capacity : descriptor->seq_kv;
    size_t row_bytes;
    size_t kv_row_bytes;
    size_t q_total_bytes;
    size_t output_total_bytes;
    size_t pool_bytes = 0;
    size_t query_bytes;
    size_t mask_bytes = (size_t)lanes * key_extent * sizeof(int32_t);
    size_t kv_bytes = 0;
    size_t prefix_row_bytes = 0;
    int* rows = g_decode_rows.scratch_indices[0];
    int* scatter_rows = g_decode_rows.scratch_indices[1];
    unsigned char* stage;
    unsigned char* staged_q;
    unsigned char* staged_output;
    int32_t* staged_mask;
    const void* keys;
    const void* values;
    uint32_t q_rows;
    uint32_t output_rows;
    uint32_t row_words;
    uint32_t kv_row_words = 0;

    if (!ops || !ops->qsdpa_i8u8 || !ops->row_index_transfer) return -1;
    if (q->ndim != 3 || (uint32_t)descriptor->batch != (uint32_t)lanes ||
        (!paged && descriptor->seq_kv != sequence)) return -1;
    if (!tensor_span_bytes(q, descriptor->d_model, &row_bytes) ||
        !tensor_span_bytes(descriptor->k, descriptor->d_model, &kv_row_bytes) ||
        !tensor_total_bytes(q, &q_total_bytes) ||
        !tensor_total_bytes(output, &output_total_bytes) ||
        row_bytes % sizeof(uint32_t) || kv_row_bytes % sizeof(uint32_t) ||
        !row_bytes || (size_t)lanes > SIZE_MAX / row_bytes) return -1;
    row_words = (uint32_t)(row_bytes / sizeof(uint32_t));
    kv_row_words = (uint32_t)(kv_row_bytes / sizeof(uint32_t));
    query_bytes = (size_t)lanes * row_bytes;
    q_rows = (uint32_t)(q->numel / descriptor->d_model);
    output_rows = (uint32_t)(output->numel / descriptor->d_model);
    if (!q_rows || !output_rows) return -1;
    if (decode_lane_rows(q, descriptor->d_model, rows) != lanes ||
        decode_lane_rows(output, descriptor->d_model, scatter_rows) != lanes)
        return -1;
    /* A parked lane occupies a row so the shape holds, and takes none back. */
    for (int lane = 0; lane < lanes; lane++)
        if (g_decode_rows.parked[lane]) scatter_rows[lane] = -1;
    if (paged) {
        if (!vx_paged_tensor_locked(descriptor->v) ||
            descriptor->v->elem_size != descriptor->k->elem_size ||
            descriptor->v->numel != descriptor->k->numel ||
            vx_paged_lanes_locked() < lanes || g_decode_rows.unpaged ||
            !key_extent || !kv_row_words ||
            !tensor_total_bytes(descriptor->k, &pool_bytes) ||
            (size_t)lanes * key_extent > SIZE_MAX / kv_row_bytes) return -1;
        kv_bytes = (size_t)lanes * key_extent * kv_row_bytes;
        prefix_row_bytes = (size_t)lanes * key_extent * sizeof(int);
    }
    stage = (unsigned char*)decode_stage_bytes(query_bytes * 2 + mask_bytes +
                                               kv_bytes * 2 + prefix_row_bytes);
    if (!stage) return -1;
    staged_q = stage;
    staged_output = stage + query_bytes;
    staged_mask = (int32_t*)(stage + query_bytes * 2);
    keys = descriptor->k->data;
    values = descriptor->v->data;

    if (!ops->row_index_transfer(q->data, q_total_bytes, rows, (uint32_t)lanes,
                                 staged_q, query_bytes, row_words, q_rows, 0u))
        return -1;
    if (paged) {
        unsigned char* staged_k = stage + query_bytes * 2 + mask_bytes;
        unsigned char* staged_v = staged_k + kv_bytes;
        int* prefix_rows = (int*)(staged_v + kv_bytes);
        const uint32_t staged_rows = (uint32_t)lanes * key_extent;
        const uint32_t pool_rows = (uint32_t)(descriptor->k->numel / descriptor->d_model);
        if (vx_decode_row_set_prefix_rows(&g_decode_rows, 0, 1, prefix_rows) !=
            VX_DECODE_ROW_SET_OK) return -1;
        if (!ops->row_index_transfer(descriptor->k->data, pool_bytes, prefix_rows,
                                     staged_rows, staged_k, kv_bytes,
                                     kv_row_words, pool_rows, 0u) ||
            !ops->row_index_transfer(descriptor->v->data, pool_bytes, prefix_rows,
                                     staged_rows, staged_v, kv_bytes,
                                     kv_row_words, pool_rows, 0u)) return -1;
        keys = staged_k;
        values = staged_v;
    }
    {
        VxDecodeKeepMask graph_mask = {0};
        if (descriptor->mask) {
            /* The same stride and the same bound as the host batch above: the
             * keep mask is built on the host either way, so the two paths
             * differ only in where the result is consumed. */
            if (key_extent > descriptor->seq_kv) return -1;
            graph_mask.layout = (VxDecodeKeepMaskLayout)descriptor->mask_mode;
            graph_mask.values = (const int*)descriptor->mask->data;
            graph_mask.keys = (int)descriptor->seq_kv;
            graph_mask.queries = (int)sequence;
        }
        if (vx_decode_row_set_keep_mask(&g_decode_rows, (int)key_extent,
                                        (descriptor->causal || paged) ? 1 : 0,
                                        descriptor->mask ? &graph_mask : NULL,
                                        (int*)staged_mask) != VX_DECODE_ROW_SET_OK)
            return -1;
        vx_runtime_backend_mark_host(staged_mask, mask_bytes, 0);
    }
    if (!ops->qsdpa_i8u8(staged_q, keys, values, staged_mask, staged_output,
                         (uint32_t)lanes, 1u, key_extent, descriptor->d_model,
                         descriptor->heads, q->quantization.scale,
                         q->quantization.zero_point,
                         descriptor->k->quantization.scale,
                         descriptor->k->quantization.zero_point,
                         descriptor->v->quantization.scale,
                         descriptor->v->quantization.zero_point,
                         output->quantization.scale, output->quantization.zero_point,
                         descriptor->attention_scale, descriptor->q_dtype,
                         descriptor->k_dtype, descriptor->v_dtype,
                         descriptor->output_dtype, 0u, ATTN_MASK_BATCH_KEY))
        return -1;
    return ops->row_index_transfer(staged_output, query_bytes, scatter_rows,
                                   (uint32_t)lanes, output->data,
                                   output_total_bytes, row_words, output_rows, 1u)
        ? 0 : -1;
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
    if (g_decode_lanes > 1) {
        if (qsdpa_physical_gpu_forward_batch(&descriptor, output, ops) != 0) return 0;
        *node_backend = backend_label;
        return 1;
    }
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
        physical_qargmax_axis(node, input->ndim, &axis) != 0) return -1;
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
    /*
     * Row-local like the elementwise operators, but with two widths: a row of
     * the input is `axis_size` bytes and a row of the output is one int32. The
     * row *indices* are the same for both -- they are the same lanes at the
     * same positions -- so they are resolved once, from the input, and applied
     * at each operand's own width. Resolving them twice is how an argmax ends
     * up writing lane 0's token id into lane 1's slot.
     */
    if (g_decode_lanes > 1) {
        VxDecodeRowRun* runs = g_decode_rows.scratch_runs;
        int count;
        if (output->ndim != 2 || output->shape[0] != g_decode_lanes ||
            descriptor.inner != 1) return -1;
        count = decode_lane_runs(descriptor.input, (long)descriptor.axis_size, runs);
        if (count < 0) return -1;
        for (int index = 0; index < count; index++) {
            const size_t row = (size_t)runs[index].source;
            if (qargmax_i8u8((const unsigned char*)descriptor.input->data +
                                 row * descriptor.axis_size *
                                     descriptor.input->elem_size,
                             (int32_t*)output->data + row,
                             (uint32_t)runs[index].rows, descriptor.axis_size,
                             1, descriptor.input_dtype) != 1) return -1;
        }
        return 0;
    }
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
        physical_no_parameters(node) != 0 ||
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
    T* scale = nin(node, VX_PORT_SCALE);
    T* zero_point = nin(node, VX_PORT_ZERO_POINT);
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
    T* scale = nin(node, VX_PORT_SCALE);
    T* zero_point = nin(node, VX_PORT_ZERO_POINT);
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
    T* scale = nin(node, VX_PORT_SCALE);
    T* zero_point = nin(node, VX_PORT_ZERO_POINT);
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
    T* scale = nin(node, VX_PORT_SCALE);
    T* zero_point = nin(node, VX_PORT_ZERO_POINT);
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

static int physical_shape_copy_op(VxOperatorKind op) {
    return op && ((op == VX_OP_RESHAPE) || (op == VX_OP_FLATTEN) ||
                  (op == VX_OP_SQUEEZE) || (op == VX_OP_UNSQUEEZE) ||
                  (op == VX_OP_IDENTITY));
}

static int physical_shape_supported_op(VxOperatorKind op) {
    return physical_shape_copy_op(op) || (op && ((op == VX_OP_MAX_POOL_2D) ||
        (op == VX_OP_RESIZE_NEAREST_2D) || (op == VX_OP_RESIZE) ||
        (op == VX_OP_TRANSPOSE) || (op == VX_OP_EXPAND) ||
        (op == VX_OP_CONCAT)));
}

/* A byte hint selects the strict physical route. Complete F32 nodes remain on
 * their dedicated implementation, while a malformed typed node fails
 * closed rather than being reinterpreted as F32 storage. */
static int physical_shape_candidate(Node* node, T* output) {
    T* input;
    if (!node || !physical_shape_supported_op(node->operator_kind)) return 0;
    if (node->operator_kind == VX_OP_CONCAT) {
        if (physical_byte_hint(output)) return 1;
        for (int index = 0; index < node->nin; index++) {
            if (physical_byte_hint(t_find(node->ins[index].name))) return 1;
        }
        return 0;
    }
    input = physical_node_data_input(node);
    return physical_byte_hint(input) || physical_byte_hint(output);
}

static int physical_false_or_absent(
        const Node* node, VxNodeParamKey key) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    if (!value || value->kind == VX_NODE_PARAM_ABSENT ||
        value->kind == VX_NODE_PARAM_NULL) return 0;
    if ((value->kind == VX_NODE_PARAM_I32 ||
         value->kind == VX_NODE_PARAM_BOOL) && !value->value.i32) return 0;
    return -1;
}

static int physical_resize_nearest_parameters(const Node* node) {
    static const VxNodeParamKey allowed[] = {
        VX_NODE_PARAM_MODE,
        VX_NODE_PARAM_COORDINATE_TRANSFORM_MODE,
        VX_NODE_PARAM_COORDINATE_TRANSFORMATION_MODE,
        VX_NODE_PARAM_NEAREST_MODE,
        VX_NODE_PARAM_ALIGN_CORNERS,
        VX_NODE_PARAM_ANTIALIAS,
        VX_NODE_PARAM_DATA_LAYOUT,
    };
    const VxCachedNodeParam* mode_value;
    const VxCachedNodeParam* unsupported_transform;
    VxNodeParamSymbol mode;
    VxNodeParamSymbol transform;
    VxNodeParamSymbol nearest_mode;
    VxNodeParamSymbol layout;
    if (!node || physical_parameters_exact(
            node, allowed, sizeof(allowed) / sizeof(allowed[0])) != 0)
        return -1;
    mode_value = vx_node_param(node, VX_NODE_PARAM_MODE);
    mode = vx_node_param_symbol(
        node, VX_NODE_PARAM_MODE, VX_NODE_SYMBOL_INVALID);
    if (node->operator_kind == VX_OP_RESIZE) {
        if (!mode_value || mode_value->kind != VX_NODE_PARAM_SYMBOL ||
            mode != VX_NODE_SYMBOL_NEAREST) return -1;
    } else if (mode_value && mode_value->kind != VX_NODE_PARAM_ABSENT &&
               mode_value->kind != VX_NODE_PARAM_NULL &&
               mode != VX_NODE_SYMBOL_NEAREST) {
        return -1;
    }
    /* The byte kernel implements the canonical asymmetric floor mapping. */
    unsupported_transform = vx_node_param(
        node, VX_NODE_PARAM_COORDINATE_TRANSFORM_MODE);
    if (unsupported_transform &&
        unsupported_transform->kind != VX_NODE_PARAM_ABSENT) return -1;
    transform = vx_node_param_symbol(
        node, VX_NODE_PARAM_COORDINATE_TRANSFORMATION_MODE,
        VX_NODE_SYMBOL_ASYMMETRIC);
    if (transform != VX_NODE_SYMBOL_ASYMMETRIC) return -1;
    nearest_mode = vx_node_param_symbol(
        node, VX_NODE_PARAM_NEAREST_MODE, VX_NODE_SYMBOL_FLOOR);
    if (nearest_mode != VX_NODE_SYMBOL_FLOOR) return -1;
    layout = vx_node_param_symbol(
        node, VX_NODE_PARAM_DATA_LAYOUT, VX_NODE_SYMBOL_NHWC);
    if (layout != VX_NODE_SYMBOL_NHWC) return -1;
    if (physical_false_or_absent(
            node, VX_NODE_PARAM_ALIGN_CORNERS) != 0 ||
        physical_false_or_absent(
            node, VX_NODE_PARAM_ANTIALIAS) != 0) return -1;
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
    if (!node || !input || !output || !descriptor || node->nin != 1 ||
        input->numel != output->numel) return -1;
    if (physical_same_byte_domain(input, output, &dtype) != 0) {
        /* A view can copy per-axis bytes only without changing their shape
         * or affine values. Scalar arithmetic never sees this descriptor. */
        cJSON* left = safetensors_affine_descriptor(g_graph_root, input->name);
        cJSON* right = safetensors_affine_descriptor(g_graph_root, output->name);
        int left_per_axis, right_per_axis, left_axis, right_axis;
        const char *left_scale, *right_scale, *left_zero, *right_zero;
        PerAxisScaleSource a, b;
        if ((input->dtype != T_I8 && input->dtype != T_U8) ||
            input->dtype != output->dtype || input->elem_size != 1 ||
            output->elem_size != 1 || !input->data || !output->data ||
            input->numel <= 0 || (uint64_t)input->numel > UINT32_MAX ||
            !physical_same_shape(input, output) ||
            safetensors_affine_descriptor_refs(left, &left_per_axis, &left_axis,
                &left_scale, &left_zero) ||
            safetensors_affine_descriptor_refs(right, &right_per_axis, &right_axis,
                &right_scale, &right_zero) || !left_per_axis || !right_per_axis)
            return -1;
        if (left_axis < 0) left_axis += input->ndim;
        if (right_axis < 0) right_axis += output->ndim;
        if (left_axis < 0 || left_axis >= input->ndim || left_axis != right_axis ||
            per_axis_scale_source(left, input, input->shape[left_axis], &a) ||
            per_axis_scale_source(right, output, output->shape[right_axis], &b) ||
            memcmp(a.scale->data, b.scale->data, (size_t)a.count * sizeof(float)) ||
            memcmp(a.zero_point->data, b.zero_point->data, (size_t)a.count)) return -1;
        dtype = (uint32_t)input->dtype;
    }
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
    const int32_t* declared_shape;
    int declared_rank = 0;
    uint64_t input_elements = 1;
    uint64_t output_elements = 1;
    uint32_t dtype;
    if (!node || !output || !descriptor || node->nin != 1 ||
        physical_same_byte_domain(input, output, &dtype) != 0 ||
        input->data == output->data || input->ndim <= 0 || input->ndim > 8 ||
        output->ndim < input->ndim || output->ndim > 8 ||
        input->numel <= 0 || output->numel <= 0 ||
        (unsigned long)output->numel > UINT32_MAX ||
        node->parsed_params.parsed != 1 ||
        node->parsed_params.parameter_count != 1) return -1;
    declared_shape = vx_node_param_array(
        node, VX_NODE_PARAM_SHAPE, VX_NODE_ARRAY_SHAPE, &declared_rank);
    if (!declared_shape || declared_rank != output->ndim) return -1;
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
        if (declared_shape[dimension] <= 0 ||
            declared_shape[dimension] != output->shape[dimension] ||
            (source != 1u && source != target)) return -1;
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
    uint32_t seen = 0;
    uint64_t elements = 1;
    uint32_t dtype;
    if (!node || !output || !descriptor || node->nin != 1 ||
        physical_same_byte_domain(input, output, &dtype) != 0 ||
        input->data == output->data || input->ndim <= 0 || input->ndim > 8 ||
        output->ndim != input->ndim || input->numel <= 0 ||
        input->numel != output->numel || (unsigned long)input->numel > UINT32_MAX ||
        node->parsed_params.parsed != 1 ||
        !node->parsed_params.transpose_params_valid ||
        (node->parsed_params.has_perm &&
         node->parsed_params.perm_rank != input->ndim))
        return -1;
    for (int dimension = 0; dimension < input->ndim; dimension++) {
        int source = input->ndim - 1 - dimension;
        if (input->shape[dimension] <= 0 ||
            elements > UINT32_MAX / (uint32_t)input->shape[dimension]) return -1;
        elements *= (uint32_t)input->shape[dimension];
        descriptor->input_shape[dimension] = (uint32_t)input->shape[dimension];
        if (node->parsed_params.has_perm)
            source = node->parsed_params.perm[dimension];
        if (source < 0 || source >= input->ndim) return -1;
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
    if (node->operator_kind == VX_OP_TRANSPOSE) {
        PhysicalByteTransposeDescriptor descriptor;
        return physical_shape_transpose_validate(node, output, &descriptor);
    }
    if (node->operator_kind == VX_OP_EXPAND) {
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

static int physical_cached_u32_pair(
        const Node* node, VxNodeParamKey key, VxNodeParamArraySlot slot,
        uint32_t fallback, uint32_t minimum, uint32_t output[2]) {
    int count = 0;
    const int32_t* values;
    if (!node || !output) return -1;
    values = vx_node_param_array(node, key, slot, &count);
    if (!values) {
        output[0] = fallback;
        output[1] = fallback;
        return 0;
    }
    if (count != 2 || values[0] < 0 || values[1] < 0 ||
        (uint32_t)values[0] < minimum ||
        (uint32_t)values[1] < minimum) return -1;
    output[0] = (uint32_t)values[0];
    output[1] = (uint32_t)values[1];
    return 0;
}

static int physical_cached_pads(
        const Node* node, const uint32_t padding[2], uint32_t pads[4]) {
    int count = 0;
    const int32_t* values;
    const VxCachedNodeParam* parameter;
    if (!node || !padding || !pads) return -1;
    parameter = vx_node_param(node, VX_NODE_PARAM_PADS);
    if (!parameter || parameter->kind == VX_NODE_PARAM_ABSENT ||
        parameter->kind == VX_NODE_PARAM_NULL) {
        pads[0] = padding[0];
        pads[1] = padding[1];
        pads[2] = padding[0];
        pads[3] = padding[1];
        return 0;
    }
    values = vx_node_param_array(
        node, VX_NODE_PARAM_PADS, VX_NODE_ARRAY_PADS, &count);
    if (!values || count != 4) return -1;
    for (int index = 0; index < 4; index++) {
        if (values[index] < 0) return -1;
        pads[index] = (uint32_t)values[index];
    }
    return 0;
}

static int physical_shape_maxpool_validate(Node* node, T* output,
                                           PhysicalByteMaxPoolDescriptor* descriptor) {
    static const VxNodeParamKey allowed[] = {
        VX_NODE_PARAM_KERNEL,
        VX_NODE_PARAM_STRIDE,
        VX_NODE_PARAM_PADDING,
        VX_NODE_PARAM_PADS,
        VX_NODE_PARAM_DILATION,
        VX_NODE_PARAM_CEIL_MODE,
        VX_NODE_PARAM_DATA_LAYOUT,
    };
    T* input = physical_node_data_input(node);
    uint32_t dtype;
    uint32_t dilation[2];
    uint32_t padding[2];
    uint64_t padded_height;
    uint64_t padded_width;
    uint64_t expected_height;
    uint64_t expected_width;
    const VxCachedNodeParam* kernel_value;
    if (!node || !output || !descriptor || node->nin != 1 ||
        physical_parameters_exact(
            node, allowed, sizeof(allowed) / sizeof(allowed[0])) != 0 ||
        vx_node_param_symbol(
            node, VX_NODE_PARAM_DATA_LAYOUT,
            VX_NODE_SYMBOL_NHWC) != VX_NODE_SYMBOL_NHWC ||
        physical_same_byte_domain(input, output, &dtype) != 0 ||
        qconv2d_shape4(input, descriptor->input_shape) != 0 ||
        qconv2d_shape4(output, descriptor->output_shape) != 0 ||
        descriptor->input_shape[0] != descriptor->output_shape[0] ||
        descriptor->input_shape[3] != descriptor->output_shape[3] ||
        input->data == output->data ||
        physical_false_or_absent(node, VX_NODE_PARAM_CEIL_MODE) != 0)
        return -1;
    kernel_value = vx_node_param(node, VX_NODE_PARAM_KERNEL);
    if (!kernel_value || kernel_value->kind != VX_NODE_PARAM_I32_ARRAY ||
        physical_cached_u32_pair(
            node, VX_NODE_PARAM_KERNEL, VX_NODE_ARRAY_KERNEL,
            1, 1, descriptor->kernel) != 0 ||
        physical_cached_u32_pair(
            node, VX_NODE_PARAM_STRIDE, VX_NODE_ARRAY_STRIDE,
            1, 1, descriptor->stride) != 0 ||
        physical_cached_u32_pair(
            node, VX_NODE_PARAM_DILATION, VX_NODE_ARRAY_DILATION,
            1, 1, dilation) != 0 ||
        physical_cached_u32_pair(
            node, VX_NODE_PARAM_PADDING, VX_NODE_ARRAY_PADDING,
            0, 0, padding) != 0 ||
        physical_cached_pads(node, padding, descriptor->pads) != 0 ||
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

static int physical_concat_axis(const Node* node, int rank, int* axis) {
    const VxCachedNodeParam* value = vx_node_param(node, VX_NODE_PARAM_AXIS);
    int normalized;
    if (!axis || rank <= 0) return -1;
    if (value && value->kind != VX_NODE_PARAM_ABSENT &&
        value->kind != VX_NODE_PARAM_NULL &&
        value->kind != VX_NODE_PARAM_I32) return -1;
    normalized = value && value->kind == VX_NODE_PARAM_I32
        ? value->value.i32 : 0;
    if (normalized < 0) {
        if (normalized < -rank) return -1;
        normalized += rank;
    }
    if (normalized < 0 || normalized >= rank) return -1;
    *axis = normalized;
    return 0;
}

typedef struct {
    int axis;
    uint32_t dtype;
    uint32_t outer;
    uint32_t inner;
    uint32_t output_axis;
    int source_count;
} PhysicalByteConcatDescriptor;

static int physical_shape_concat_validate(Node* node, int node_index, T* output,
                                          PhysicalByteConcatDescriptor* descriptor) {
    static const VxNodeParamKey allowed[] = {
        VX_NODE_PARAM_AXIS,
        VX_NODE_PARAM_SIGMOID,
    };
    uint64_t outer = 1;
    uint64_t inner = 1;
    uint64_t axis_sum = 0;
    int source_count = node ? node->nin : 0;
    int axis;
    if (!node || !output || !descriptor ||
        physical_parameters_exact(
            node, allowed, sizeof(allowed) / sizeof(allowed[0])) != 0 ||
        physical_byte_tensor(output, &descriptor->dtype) != 0 ||
        source_count <= 0 ||
        physical_concat_axis(node, output->ndim, &axis) != 0 ||
        physical_false_or_absent(node, VX_NODE_PARAM_SIGMOID) != 0 ||
        (node_index >= 0 && (size_t)node_index < g_node_capacity &&
         g_concat_sigmoid_fuse[node_index] >= 0)) return -1;
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
        T* source = t_find(node->ins[index].name);
        uint32_t source_dtype;
        if (physical_byte_tensor(source, &source_dtype) != 0 || source_dtype != descriptor->dtype ||
            source->dtype != output->dtype || source->quantization.scale != output->quantization.scale ||
            source->quantization.zero_point != output->quantization.zero_point ||
            source->data == output->data || source->ndim != output->ndim ||
            source->shape[axis] <= 0) return -1;
        for (int dimension = 0; dimension < output->ndim; dimension++) {
            if (dimension != axis && source->shape[dimension] != output->shape[dimension]) return -1;
        }
        uint32_t source_axis = (uint32_t)source->shape[axis];
        if ((uint64_t)outer * source_axis > UINT32_MAX / inner ||
            source->numel != (long)((uint64_t)outer * source_axis * inner) ||
            axis_sum > UINT32_MAX - source_axis) return -1;
        axis_sum += source_axis;
    }
    if (axis_sum != (uint32_t)output->shape[axis]) return -1;
    descriptor->outer = (uint32_t)outer;
    descriptor->inner = (uint32_t)inner;
    descriptor->output_axis = (uint32_t)output->shape[axis];
    descriptor->source_count = source_count;
    descriptor->axis = axis;
    return 0;
}

static int physical_shape_concat_forward(Node* node, int node_index, T* output) {
    PhysicalByteConcatDescriptor descriptor;
    if (physical_shape_concat_validate(node, node_index, output, &descriptor) != 0) return -1;
    uint32_t axis_offset = 0;
    for (int index = 0; index < descriptor.source_count; index++) {
        T* source = t_find(node->ins[index].name);
        uint32_t source_axis = (uint32_t)source->shape[descriptor.axis];
        if (concat_slice_i8u8((const uint8_t*)source->data, (uint8_t*)output->data,
                              descriptor.outer, source_axis,
                              descriptor.output_axis, descriptor.inner,
                              axis_offset, descriptor.dtype) != 1) return -1;
        axis_offset += source_axis;
    }
    return 0;
}

static int physical_shape_forward(Node* node, int node_index, T* output) {
    if (!node || !output) return -1;
    if (physical_shape_copy_op(node->operator_kind)) return physical_shape_copy_forward(node, output);
    if (node->operator_kind == VX_OP_TRANSPOSE)
        return physical_shape_transpose_forward(node, output);
    if (node->operator_kind == VX_OP_EXPAND)
        return physical_shape_expand_forward(node, output);
    if ((node->operator_kind == VX_OP_RESIZE_NEAREST_2D) || (node->operator_kind == VX_OP_RESIZE))
        return physical_shape_resize_forward(node, output);
    if (node->operator_kind == VX_OP_MAX_POOL_2D) return physical_shape_maxpool_forward(node, output);
    if (node->operator_kind == VX_OP_CONCAT) return physical_shape_concat_forward(node, node_index, output);
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
    VxConcatInputs inputs;
    if (!ops || !ops->concat_i8u8 ||
        physical_shape_concat_validate(node, node_index, output, &descriptor) != 0 ||
        !concat_inputs_build(node, output, descriptor.axis, output->dtype, &inputs)) return 0;
    size_t count = (size_t)inputs.count;
    uint32_t* elements = calloc(count, sizeof(*elements));
    float* scales = calloc(count, sizeof(*scales));
    int32_t* zeros = calloc(count, sizeof(*zeros));
    uint32_t* dtypes = calloc(count, sizeof(*dtypes));
    int ok = 0;
    if (!elements || !scales || !zeros || !dtypes) goto done;
    for (int i = 0; i < inputs.count; ++i) {
        elements[i] = (uint32_t)inputs.sizes[i];
        scales[i] = output->quantization.scale;
        zeros[i] = output->quantization.zero_point;
        dtypes[i] = descriptor.dtype;
    }
    ok = ops->concat_i8u8(inputs.raw, elements, (const uint32_t*)inputs.axes, scales, zeros,
        dtypes, (uint32_t)inputs.count, output->data, (uint32_t)output->numel,
        descriptor.output_axis, descriptor.inner, output->quantization.scale,
        output->quantization.zero_point, descriptor.dtype);
    if (ok) *node_backend = backend_label;
done:
    free(elements); free(scales); free(zeros); free(dtypes);
    concat_inputs_clear(&inputs);
    return ok;
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
    if (physical_shape_copy_op(node->operator_kind))
        return physical_shape_copy_gpu_forward(node, output, ops, alias_label, copy_label, node_backend);
    if (node->operator_kind == VX_OP_TRANSPOSE)
        return physical_shape_transpose_gpu_forward(
            node, output, ops, transpose_label, node_backend);
    if ((node->operator_kind == VX_OP_RESIZE_NEAREST_2D) || (node->operator_kind == VX_OP_RESIZE))
        return physical_shape_resize_gpu_forward(node, output, ops, alias_label, resize_label, node_backend);
    if (node->operator_kind == VX_OP_MAX_POOL_2D)
        return physical_shape_maxpool_gpu_forward(node, output, ops, maxpool_label, node_backend);
    if (node->operator_kind == VX_OP_CONCAT)
        return physical_shape_concat_gpu_forward(node, node_index, output, ops, concat_label, node_backend);
    return 0;
}
