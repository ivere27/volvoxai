/* CPU projection of the same lane rows that the WebGPU planner gathers.
 * Ordinary kernels see a dense, narrowed tensor. This keeps dtype, broadcast
 * and activation semantics in the existing C kernels. */
static int vx_decode_cpu_storage(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, T* narrow) {
    *narrow = *operand->tensor;
    narrow->shape[operand->axis] = operand->prefix ? rows->key_capacity : 1;
    uint64_t elements = (uint64_t)rows->lanes * narrow->shape[operand->axis] * operand->width;
    if (!elements || elements > INT32_MAX || elements > SIZE_MAX / narrow->elem_size) return 0;
    narrow->numel = (long)elements;
    narrow->data = malloc((size_t)elements * narrow->elem_size);
    return narrow->data != NULL;
}

static int vx_decode_cpu_transfer(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, const T* narrow, int scatter) {
    size_t row_bytes = operand->width * operand->tensor->elem_size;
    int count = rows->lanes * (operand->prefix ? rows->key_capacity : 1);
    if (count <= 0 || (size_t)count > SIZE_MAX / sizeof(int)) return 0;
    int* indices = malloc((size_t)count * sizeof(int));
    if (!indices) return 0;
    int paged = vx_paged_tensor_locked(operand->tensor);
    int status = operand->prefix
        ? vx_decode_row_set_prefix_rows(rows, operand->sequence, paged, indices)
        : vx_decode_row_set_write_rows(rows, operand->sequence, paged, indices);
    if (!operand->prefix)
        for (int lane = 0; lane < rows->lanes; lane++)
            if (rows->parked[lane]) indices[lane] = -1;
    if (status == VX_DECODE_ROW_SET_OK) {
        size_t bytes = (size_t)operand->tensor->numel * operand->tensor->elem_size;
        status = scatter
            ? vx_decode_row_scatter(operand->tensor->data, bytes, narrow->data, row_bytes, indices, count, rows->parked)
            : vx_decode_row_gather(narrow->data, operand->tensor->data, bytes, row_bytes, indices, count);
    }
    free(indices);
    return status == VX_DECODE_ROW_SET_OK;
}

static int vx_decode_cpu_attention(Node* node, const VxDecodeRowProjection* p, T* output) {
    T* q = nin(node, VX_PORT_Q), *k = nin(node, VX_PORT_K), *v = nin(node, VX_PORT_V);
    size_t mask_count = (size_t)p->rows.lanes * p->keys;
    if (mask_count > SIZE_MAX / sizeof(int32_t)) return -1;
    int32_t* mask = malloc(mask_count * sizeof(int32_t));
    if (!mask) return -1;
    int source_keys = p->mask ? p->mask->shape[p->mask->ndim - 1] : p->keys;
    for (int lane = 0; lane < p->rows.lanes; lane++)
        for (int key = 0; key < p->keys; key++)
            mask[(size_t)lane * p->keys + key] = !p->rows.parked[lane] &&
                (!p->causal || key < p->rows.kv_lengths[lane]) &&
                attention_mask_allowed(p->mask, p->mask_layout, lane, p->rows.positions[lane],
                    key, p->output.sequence, source_keys, 0);
    int result = -1, width = q->shape[q->ndim - 1];
    if (node->operator_kind == VX_OP_Q_SDPA) {
        uint32_t heads, causal;
        float scale;
        if (physical_qsdpa_parameters(node, (uint32_t)width, &heads, &causal, &scale) == 0)
            result = vx_qsdpa_i8u8_native_validated(q->data, k->data, v->data, mask,
                output->data, (uint32_t)p->rows.lanes, 1u, (uint32_t)p->keys, (uint32_t)width, heads,
                q->quantization.scale, q->quantization.zero_point,
                k->quantization.scale, k->quantization.zero_point,
                v->quantization.scale, v->quantization.zero_point,
                output->quantization.scale, output->quantization.zero_point, scale,
                physical_byte_dtype(q->dtype), physical_byte_dtype(k->dtype),
                physical_byte_dtype(v->dtype), physical_byte_dtype(output->dtype),
                0u, ATTN_MASK_BATCH_KEY) == 1 ? 0 : -1;
    } else if (q->dtype == T_F32 && k->dtype == T_F32 && v->dtype == T_F32 && output->dtype == T_F32) {
        int heads = node->parsed_params.heads;
        if (heads > 0 && width > 0 && width % heads == 0) {
            int head_width = width / heads;
            float scale = node->parsed_params.has_scale ? node->parsed_params.scale : 1.f / sqrtf((float)head_width);
            for (int lane = 0; lane < p->rows.lanes; lane++)
                cross_sdpa_f32(q->data + (size_t)lane * width,
                    k->data + (size_t)lane * p->keys * width,
                    v->data + (size_t)lane * p->keys * width,
                    output->data + (size_t)lane * width, 1, p->keys, width,
                    heads, head_width, scale, mask + (size_t)lane * p->keys, VX_ATTN_MASK_LOCAL_KEY, 0);
            result = 0;
        }
    }
    free(mask);
    return result;
}

static int vx_decode_cpu_project(Node* node, int idx, int is_last, VxDecodeRowProjection* p) {
    T saved[3], narrow[3] = {0}, output = {0};
    int result = -1;
    if (!vx_decode_cpu_storage(&p->output, &p->rows, &output)) goto done;
    for (int i = 0; i < p->input_count; i++) {
        saved[i] = *p->inputs[i].tensor;
        if (!vx_decode_cpu_storage(&p->inputs[i], &p->rows, &narrow[i]) ||
            !vx_decode_cpu_transfer(&p->inputs[i], &p->rows, &narrow[i], 0)) goto done;
    }
    for (int i = 0; i < p->input_count; i++) *p->inputs[i].tensor = narrow[i];
    T saved_output = *p->output.tensor;
    *p->output.tensor = output;
    int active_row = g_active_row, lanes = g_decode_lanes, prefix = g_prefix_rows;
    int execution_row = g_execution_row;
    int paged = vx_engine_state_current()->paged.bound;
    g_active_row = -1; g_decode_lanes = 0; g_prefix_rows = 0; g_execution_row = -1;
    vx_engine_state_current()->paged.bound = 0;
    if (p->view) {
        memcpy(output.data, narrow[0].data, (size_t)output.numel * output.elem_size);
        result = 0;
    } else if (p->attention) result = vx_decode_cpu_attention(node, p, &output);
    else result = run_node(node, idx, is_last);
    g_active_row = active_row; g_decode_lanes = lanes; g_prefix_rows = prefix;
    g_execution_row = execution_row;
    vx_engine_state_current()->paged.bound = paged;
    *p->output.tensor = saved_output;
    for (int i = 0; i < p->input_count; i++) *p->inputs[i].tensor = saved[i];
    if (result == 0 && !vx_decode_cpu_transfer(&p->output, &p->rows, &output, 1)) result = -1;
done:
    free(output.data);
    for (int i = 0; i < p->input_count; i++) free(narrow[i].data);
    return result;
}
