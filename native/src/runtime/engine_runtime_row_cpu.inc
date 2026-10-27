/* CPU projection of the same slot rows that the WebGPU planner gathers.
 * Ordinary kernels see a dense, narrowed tensor. This keeps dtype, broadcast
 * and activation semantics in the existing C kernels.
 * All temporary narrow tensors, gather/scatter index buffers, and attention
 * masks are sliced from the context's persistent decode_stage buffer so
 * multi-slot decode executes with zero per-node heap allocations. */
static int vx_decode_cpu_storage(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, T* narrow, size_t* out_bytes) {
    *narrow = *operand->tensor;
    narrow->shape[operand->axis] = operand->prefix ? rows->key_capacity : 1;
    uint64_t elements = (uint64_t)rows->slots * narrow->shape[operand->axis] * operand->width;
    if (!elements || elements > INT32_MAX ||
        elements > (SIZE_MAX - 15u) / narrow->elem_size) return 0;
    narrow->numel = (long)elements;
    narrow->data = NULL;
    *out_bytes = (((size_t)elements * narrow->elem_size) + 15u) & ~(size_t)15u;
    return 1;
}

static int vx_decode_cpu_transfer_bytes(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, size_t* out_bytes) {
    int count = rows->slots * (operand->prefix ? rows->key_capacity : 1);
    if (count <= 0 || (size_t)count > (SIZE_MAX - 15u) / sizeof(int)) return 0;
    *out_bytes = (((size_t)count * sizeof(int)) + 15u) & ~(size_t)15u;
    return 1;
}

static int vx_decode_cpu_transfer(const VxDecodeRowOperand* operand,
    const VxDecodeRowSet* rows, const T* narrow, int scatter, int* indices) {
    size_t row_bytes = operand->width * operand->tensor->elem_size;
    int count = rows->slots * (operand->prefix ? rows->key_capacity : 1);
    if (count <= 0 || !indices) return 0;
    int paged = vx_paged_tensor_locked(operand->tensor);
    int status = operand->prefix
        ? vx_decode_row_set_prefix_rows(rows, operand->sequence, paged, indices)
        : vx_decode_row_set_write_rows(rows, operand->sequence, paged, indices);
    if (!operand->prefix)
        for (int slot = 0; slot < rows->slots; slot++)
            if (rows->empty[slot]) indices[slot] = -1;
    if (status == VX_DECODE_ROW_SET_OK) {
        size_t bytes = (size_t)operand->tensor->numel * operand->tensor->elem_size;
        status = scatter
            ? vx_decode_row_scatter(operand->tensor->data, bytes, narrow->data, row_bytes, indices, count, rows->empty)
            : vx_decode_row_gather(narrow->data, operand->tensor->data, bytes, row_bytes, indices, count);
    }
    return status == VX_DECODE_ROW_SET_OK;
}

static int vx_decode_cpu_attention(Node* node, const VxDecodeRowProjection* p,
    T* output, int32_t* mask) {
    T* q = nin(node, VX_PORT_Q), *k = nin(node, VX_PORT_K), *v = nin(node, VX_PORT_V);
    if (!mask) return -1;
    int source_keys = p->mask ? p->mask->shape[p->mask->ndim - 1] : p->keys;
    for (int slot = 0; slot < p->rows.slots; slot++)
        for (int key = 0; key < p->keys; key++)
            mask[(size_t)slot * p->keys + key] = !p->rows.empty[slot] &&
                (!p->causal || key < p->rows.kv_lengths[slot]) &&
                attention_mask_allowed(p->mask, p->mask_layout, slot, p->rows.positions[slot],
                    key, p->output.sequence, source_keys, 0);
    int result = -1, width = q->shape[q->ndim - 1];
    if (node->operator_kind == VX_OP_Q_SDPA) {
        uint32_t heads, causal;
        float scale;
        if (physical_qsdpa_parameters(node, (uint32_t)width, &heads, &causal, &scale) == 0)
            result = vx_qsdpa_i8u8_native_validated(q->data, k->data, v->data, mask,
                output->data, (uint32_t)p->rows.slots, 1u, (uint32_t)p->keys, (uint32_t)width, heads,
                q->quantization.scale, q->quantization.zero_point,
                k->quantization.scale, k->quantization.zero_point,
                v->quantization.scale, v->quantization.zero_point,
                output->quantization.scale, output->quantization.zero_point, scale,
                physical_byte_dtype(q->dtype), physical_byte_dtype(k->dtype),
                physical_byte_dtype(v->dtype), physical_byte_dtype(output->dtype),
                0u, ATTN_MASK_BATCH_KEY) == 1 ? 0 : -1;
    } else if (q->dtype == T_F32 && k->dtype == T_F32 && v->dtype == T_F32 && output->dtype == T_F32) {
        int heads = node->parsed_params.heads;
        int kv_heads = node->parsed_params.kv_heads;
        int kv_width = k->shape[k->ndim - 1];
        if (heads > 0 && kv_heads > 0 && heads % kv_heads == 0 &&
            width > 0 && width % heads == 0 && kv_width > 0 &&
            kv_width % kv_heads == 0 && width / heads == kv_width / kv_heads &&
            v->shape[v->ndim - 1] == kv_width && output->shape[output->ndim - 1] == width) {
            int head_width = width / heads;
            float scale = node->parsed_params.has_scale ? node->parsed_params.scale : 1.f / sqrtf((float)head_width);
            for (int slot = 0; slot < p->rows.slots; slot++)
                cross_sdpa_gqa_f32(q->data + (size_t)slot * width,
                    k->data + (size_t)slot * p->keys * kv_width,
                    v->data + (size_t)slot * p->keys * kv_width,
                    output->data + (size_t)slot * width, 1, p->keys, width, kv_width,
                    heads, kv_heads, head_width, scale, mask + (size_t)slot * p->keys,
                    VX_ATTN_MASK_LOCAL_KEY, 0);
            result = 0;
        }
    }
    return result;
}

static int vx_decode_cpu_project(Node* node, int idx, int is_last, VxDecodeRowProjection* p) {
    T saved[3], narrow[3] = {0}, output = {0};
    size_t output_bytes = 0, input_bytes[3] = {0}, max_index_bytes = 0, index_bytes = 0, mask_bytes = 0;
    if (!vx_decode_cpu_storage(&p->output, &p->rows, &output, &output_bytes) ||
        !vx_decode_cpu_transfer_bytes(&p->output, &p->rows, &max_index_bytes))
        return -1;
    size_t total_bytes = output_bytes;
    for (int i = 0; i < p->input_count; i++) {
        saved[i] = *p->inputs[i].tensor;
        if (!vx_decode_cpu_storage(&p->inputs[i], &p->rows, &narrow[i], &input_bytes[i]) ||
            !vx_decode_cpu_transfer_bytes(&p->inputs[i], &p->rows, &index_bytes) ||
            total_bytes > SIZE_MAX - input_bytes[i])
            return -1;
        total_bytes += input_bytes[i];
        if (index_bytes > max_index_bytes) max_index_bytes = index_bytes;
    }
    if (total_bytes > SIZE_MAX - max_index_bytes) return -1;
    total_bytes += max_index_bytes;
    if (p->attention) {
        if (p->rows.slots <= 0 || p->keys <= 0 ||
            (size_t)p->rows.slots > (SIZE_MAX - 15u) / sizeof(int32_t) / (size_t)p->keys)
            return -1;
        mask_bytes = (((size_t)p->rows.slots * (size_t)p->keys * sizeof(int32_t)) + 15u) & ~(size_t)15u;
        if (total_bytes > SIZE_MAX - mask_bytes) return -1;
        total_bytes += mask_bytes;
    }
    unsigned char* stage = decode_stage_bytes(total_bytes);
    if (!stage) return -1;
    output.data = (float*)(void*)stage;
    stage += output_bytes;
    for (int i = 0; i < p->input_count; i++) {
        narrow[i].data = (float*)(void*)stage;
        stage += input_bytes[i];
    }
    int* indices = (int*)(void*)stage;
    stage += max_index_bytes;
    int32_t* mask = p->attention ? (int32_t*)(void*)stage : NULL;
    for (int i = 0; i < p->input_count; i++) {
        if (!vx_decode_cpu_transfer(&p->inputs[i], &p->rows, &narrow[i], 0, indices))
            return -1;
    }
    int result = -1;
    if (p->rope) {
        result = vx_sequence_rope_row(node, p->inputs[0].tensor, p->output.tensor,
            &p->rows, narrow[0].data, output.data) == VX_SEQUENCE_HANDLED ? 0 : -1;
        goto scatter;
    }
    for (int i = 0; i < p->input_count; i++) *p->inputs[i].tensor = narrow[i];
    T saved_output = *p->output.tensor;
    *p->output.tensor = output;
    int active_row = g_active_row, slots = g_decode_slots, prefix = g_prefix_rows;
    int execution_row = g_execution_row;
    int paged = vx_engine_state_current()->paged.bound;
    g_active_row = -1; g_decode_slots = 0; g_prefix_rows = 0; g_execution_row = -1;
    vx_engine_state_current()->paged.bound = 0;
    if (p->view) {
        memcpy(output.data, narrow[0].data, (size_t)output.numel * output.elem_size);
        result = 0;
    } else if (p->attention) result = vx_decode_cpu_attention(node, p, &output, mask);
    else result = run_node(node, idx, is_last);
    g_active_row = active_row; g_decode_slots = slots; g_prefix_rows = prefix;
    g_execution_row = execution_row;
    vx_engine_state_current()->paged.bound = paged;
    *p->output.tensor = saved_output;
    for (int i = 0; i < p->input_count; i++) *p->inputs[i].tensor = saved[i];
scatter:
    if (result == 0 && !vx_decode_cpu_transfer(&p->output, &p->rows, &output, 1, indices))
        result = -1;
    return result;
}
